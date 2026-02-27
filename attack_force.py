import os
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from torch import amp
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torchvision.transforms.functional import to_tensor, to_pil_image
from utils import *
import time
from torch.nn.utils.rnn import pad_sequence

def pgd_attack(args, model, processor, image_tensor, item_id, instruction):
    device = image_tensor.device
    img_tensor = image_tensor.clone().detach().to(device).half()  # [1,3,H,W] in [0,1]
    step_size = args.step_size / 255.
    epsilon = args.epsilon / 255.
    delta = (torch.rand_like(img_tensor) * 2 - 1) * epsilon
    iteration_time = 0
    query_time = 0

    # -------- prompt once --------
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": instruction}, {"type": "image"}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # -------- target ids once --------
    target_token_ids = processor.tokenizer.encode(args.target_text, add_special_tokens=False)
    target_ids = torch.tensor([target_token_ids], device=device)  # [1, Lt]

    # -------- masks once (list->tensor) --------
    _, _, H, W = img_tensor.shape
    masks = generate_frequency_masks(H, W, args.num_band, device=device)
    if isinstance(masks, list):
        masks = torch.stack([m.to(device) for m in masks], dim=0)
    if args.num_band > 0 and masks.dtype != torch.bool:
        masks = masks > 0

    # -------- cache prompt tokens with correct image placeholder (once) --------
    # We do this ONCE; afterwards we never call processor(text=...) inside loops.
    with torch.no_grad():
        base_inputs = processor(images=tensor_to_pil(img_tensor), text=prompt, return_tensors="pt")
        base_prompt_ids = base_inputs["input_ids"].to(device)
        base_prompt_attn = base_inputs.get("attention_mask", torch.ones_like(base_prompt_ids)).to(device)

        prompt_len = base_prompt_ids.shape[1]
        base_input_ids = torch.cat([base_prompt_ids, target_ids], dim=1)  # [1, L]
        base_attn = torch.cat([base_prompt_attn, torch.ones_like(target_ids)], dim=1)
        base_labels = torch.full_like(base_input_ids, -100)
        base_labels[0, prompt_len:] = target_ids[0]

    img_proc = processor.image_processor  # for mean/std
    total_time = 0.0

    # (optional) ensure eval for speed/stability; doesn't stop pixel_values grad
    model.eval()

    while True:
        iteration_time += 1
        time1 = time.time()

        # =========================================================
        # (A) Band selection (batch forward all bands)
        # =========================================================
        if args.num_band > 0:
            fft_result = torch.fft.fft2(delta.float(), dim=(-2, -1), norm="ortho")
            fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))

            magnitude = torch.abs(fft_shifted)[0]  # [3,H,W]
            phase = torch.angle(fft_shifted)[0]    # [3,H,W]

            with torch.no_grad():

                current_mask = masks.float().unsqueeze(1).expand(-1, 3, -1, -1)  # [B,3,H,W]
                inverted_mask = 1.0 - current_mask

                mag_b = magnitude.unsqueeze(0).expand(args.num_band, -1, -1, -1)
                pha_b = phase.unsqueeze(0).expand(args.num_band, -1, -1, -1)

                masked_fft = (mag_b * inverted_mask) * torch.exp(1j * pha_b)
                masked_fft_unshifted = torch.fft.ifftshift(masked_fft, dim=(-2, -1))

                delta_masked = torch.fft.ifft2(masked_fft_unshifted, dim=(-2, -1), norm="ortho").real
                delta_masked = delta_masked.clamp(-epsilon, epsilon)  # [B,3,H,W]

                img_b = img_tensor.expand(args.num_band, -1, -1, -1)
                x_mask = (img_b + delta_masked).clamp(0, 1)  # [B,3,H,W]

                mask_pixel_values = normalize_pixel_values_on_gpu(x_mask, img_proc, out_dtype=torch.float16)

                input_ids_b = base_input_ids.expand(args.num_band, -1)
                attn_b = base_attn.expand(args.num_band, -1)
                labels_b = base_labels.expand(args.num_band, -1)

                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    mask_out = model(
                        input_ids=input_ids_b,
                        attention_mask=attn_b,
                        pixel_values=mask_pixel_values,
                        labels=None,
                        use_cache=False,
                    )
                    band_losses = per_sample_ce_loss(mask_out.logits, labels_b).detach()  # [B]

            retained_mask = torch.zeros_like(masks[0], dtype=torch.float32)
            retained_mask[masks[0]] = 1.0
            for band_idx in range(1, args.num_band):
                if band_losses[band_idx] <= band_losses[band_idx - 1] * args.lambda_2:
                    retained_mask[masks[band_idx]] = 1.0
                else:
                    retained_mask[masks[band_idx]] = ((band_losses[band_idx - 1] * args.lambda_2) / band_losses[band_idx])

            masked_magnitude = magnitude * retained_mask
            masked_fft = masked_magnitude * torch.exp(1j * phase)
            delta = torch.fft.ifft2(torch.fft.ifftshift(masked_fft, dim=(-2, -1)), dim=(-2, -1), norm="ortho").real
            delta = delta.unsqueeze(0).clamp(-epsilon, epsilon).detach().to(torch.float16)

        # =========================================================
        # (B) Noise sampling (batch forward all noises)
        # =========================================================
        noisy_losses = None
        noisy_feats_batched = None

        if args.num_noise > 0:
            with torch.no_grad():

                delta_sample_ran = (torch.rand((args.num_noise,) + img_tensor.shape[1:], device=device, dtype=img_tensor.dtype) * 2 - 1)
                delta_sample_ran = delta_sample_ran * epsilon * args.lambda_3

                delta_noise = delta.expand(args.num_noise, -1, -1, -1) + delta_sample_ran
                img_b = img_tensor.expand(args.num_noise, -1, -1, -1)
                x_noise = (img_b + delta_noise).clamp(0, 1)

                noisy_pixel_values = normalize_pixel_values_on_gpu(x_noise, img_proc, out_dtype=torch.float16)

                input_ids_b = base_input_ids.expand(args.num_noise, -1)
                attn_b = base_attn.expand(args.num_noise, -1)
                labels_b = base_labels.expand(args.num_noise, -1)

                noisy_handles, noisy_feats = register_llm_hidden_hooks(model, require_grad=False)

                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    noisy_out = model(
                        input_ids=input_ids_b,
                        attention_mask=attn_b,
                        pixel_values=noisy_pixel_values,
                        labels=None,
                        use_cache=False,
                    )

                for h in noisy_handles:
                    h.remove()

                noisy_losses = per_sample_ce_loss(noisy_out.logits, labels_b).detach()  # [Bn]
                noisy_feats_batched = {k: v.detach() for k, v in noisy_feats.items()}   # [Bn,...] on GPU ideally

        # =========================================================
        # (C) Main adversarial forward (keep your original update rule)
        # =========================================================
        x_adv = (img_tensor + delta).clamp(0, 1).detach().requires_grad_(True)

        pv = normalize_pixel_values_on_gpu(x_adv, img_proc, out_dtype=torch.float16)
        pixel_values = pv.detach().requires_grad_(True)  # keep same semantics as you had

        input_ids = base_input_ids
        labels = base_labels
        attn = base_attn

        handles, adv_features = register_llm_hidden_hooks(model, require_grad=True)

        with amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=pixel_values,
                labels=labels,
                use_cache=False
            )
            logits = outputs.logits
            output_loss = outputs.loss

            loss_reg = torch.zeros((), device=device)
            sorted_keys = sorted(adv_features.keys())
            num_layers = len(sorted_keys)

            if args.num_noise > 0 and (noisy_losses is not None) and (noisy_feats_batched is not None):
                for idx, layer_id in enumerate(sorted_keys):
                    if layer_id not in noisy_feats_batched:
                        continue

                    adv_feat = adv_features[layer_id]              # [1,...]
                    nf = noisy_feats_batched[layer_id].to(device)  # [args.num_noise,...]

                    adv_b = adv_feat.unsqueeze(0) if adv_feat.dim() == nf.dim() - 1 else adv_feat

                    feature_diff = adv_b - nf
                    reduce_dims = tuple(range(1, feature_diff.dim()))
                    feat_num = feature_diff.norm(p=2, dim=reduce_dims)
                    feat_den = adv_b.norm(p=2, dim=reduce_dims).clamp_min(1e-8)
                    feature_l2 = (feat_num / feat_den).clamp_min(1e-12)  # [args.num_noise]

                    weight_diff = (noisy_losses / feature_l2).mean()
                    layer_weight = (max(1 - ((idx * 2) / num_layers) ** 2, 0.0)) / num_layers
                    loss_reg += args.lambda_1 * weight_diff * layer_weight

        # =========================================================
        # Match / save / query logic (unchanged)
        # =========================================================
        with torch.no_grad():
            # prompt_len cached from base_prompt_ids
            pred_token_ids = logits[:, prompt_len - 1: input_ids.shape[1] - 1, :].argmax(dim=-1)
            match = torch.equal(pred_token_ids[0], target_ids[0])
            last_token_pred = logits[:, input_ids.shape[1] - 1, :].argmax(dim=-1)

            if match and last_token_pred.item() != processor.tokenizer.eos_token_id:
                save_tensor_as_image(
                    (img_tensor + delta).clamp(0, 1).detach(),
                    os.path.join(args.output_vis_path, f"{item_id}_{query_time}_adv.png")
                )
                save_tensor_as_npy(
                    (img_tensor + delta).clamp(0, 1).detach(),
                    os.path.join(args.output_vis_path, f"{item_id}_{query_time}_adv.npy")
                )
                query_time += 1
                if query_time == args.max_query:
                    generate_output = generate_response(model, processor, x_adv, instruction)
                    for h in handles:
                        h.remove()
                    return True, generate_output, iteration_time, query_time, (img_tensor + delta).clamp(0, 1).detach(), delta.detach().clamp(0, 1)

        total_loss = output_loss + loss_reg

        total_loss.backward()

        grad = pixel_values.grad
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print("PGD grad contains NaN or Inf!")

        # same update rule
        delta = torch.clamp(delta - (step_size * grad.sign()), -epsilon, epsilon).detach()

        time2 = time.time()
        total_time += time2 - time1
        print("time", total_time / (iteration_time + 1), "output_loss", output_loss, "loss_reg", loss_reg, "total_loss", total_loss)

        model.zero_grad()
        for h in handles:
            h.remove()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_vis_path, exist_ok=True)

    success_rate = 0
    opt_rate = 0
    query_rate = 0

    model = LlavaForConditionalGeneration.from_pretrained(
        args.source_model_name,
        torch_dtype=torch.float16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.source_model_name)

    if "HADES" in args.dataset_path:
        dataset = load_dataset(args.dataset_path)['test']
        dataset = [item for item in dataset if item.get('step', None) == 5]
    else:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        dataset = [{"id": f"{dataset_name}_{i:03d}", "instruction": line.strip()} for i, line in enumerate(lines)]

    image_processor = processor.image_processor
    if "height" in image_processor.size and "width" in image_processor.size:
        h = image_processor.size["height"]
        w = image_processor.size["width"]
    else:
        h = image_processor.size["shortest_edge"]
        w = image_processor.size["shortest_edge"]
    max_dim = max(h, w)
    clamped = max_dim if max_dim < 336 else 336
    size = (clamped, clamped)

    with open(args.output_text_path, "a", encoding="utf-8") as f:
        for item in tqdm(dataset):
            item_id = item['id']
            instruction = item['instruction']

            if "HADES" in args.dataset_path:
                if args.image_input == "semantic":
                    original_image = item['image'].convert("RGB")
                else:
                    original_image = Image.new("RGB", item['image'].size, color=(128, 128, 128))
                cropped_image = original_image.crop((0, 0, original_image.width, original_image.height - 300))
            else:
                if args.image_input == "semantic":
                    original_image = Image.open(args.image_path).convert("RGB")
                else:
                    original_image = Image.new("RGB", size, color=(128, 128, 128))
                cropped_image = original_image

            resized_image = cropped_image.resize(size, Image.LANCZOS)
            tensor_image = pil_to_tensor(resized_image).unsqueeze(0).to(device)

            result, generate_output, iter, query_time, tensor_adv, tensor_noise = pgd_attack(
                args, model, processor, tensor_image, item_id, instruction
            )

            if result == True:
                success_rate +=1
            opt_rate += iter
            query_rate += query_time

            f.write(json.dumps({
                "id": item_id,
                "result": result,
                "iter": iter,
                "query_time": query_time,
                "instruction": instruction,
                "response_adv": generate_output
            }, ensure_ascii=False) + "\n")
            f.flush()

        total_len = len(dataset)
        summary = {
            "success_rate": success_rate / total_len,
            "avg_iteration": opt_rate / total_len,
            "avg_query": query_rate / total_len
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../Dataset/malicious.txt")
    parser.add_argument("--image_path", type=str, default="../Dataset/image.png")
    parser.add_argument("--source_model_name", type=str, default="../llava-v1.5-7b")
    parser.add_argument("--output_text_path", type=str, default="./output_test.jsonl")
    parser.add_argument("--output_vis_path", type=str, default="./output_test/image")

    parser.add_argument("--max_query", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=2)
    parser.add_argument("--epsilon", type=int, default=32)
    parser.add_argument("--image_input", type=str, choices=["semantic", "blank"], default="semantic")
    parser.add_argument("--target_text", type=str, default="Sure, here is")

    parser.add_argument("--lambda_1", type=float, default=0.75)
    parser.add_argument("--lambda_2", type=float, default=0.95)
    parser.add_argument("--lambda_3", type=float, default=0.125)
    parser.add_argument("--num_band", type=int, default=10)
    parser.add_argument("--num_noise", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
