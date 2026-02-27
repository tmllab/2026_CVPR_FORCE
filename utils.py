import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import base64
import io
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch import amp
from eval import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pil_to_tensor(image):
    return to_tensor(image)

def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return to_pil_image(tensor)

def save_tensor_as_image(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure dir exists
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    pil_image = to_pil_image(tensor.detach().cpu().float().clamp(0, 1))
    pil_image.save(save_path)

def save_tensor_as_npy(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, tensor.detach().cpu().numpy())

def extract_score(content):
    """Extract 0 or 1 from the content"""
    try:
        content = content.strip()
        if content[-1] in ['0', '1']:
            return int(content[-1])
        for word in content.split():
            if word == '0' or word == '1':
                return int(word)
        return None
    except:
        return None

def generate_frequency_masks(H, W, K=5, device="cpu"):
    y = torch.arange(-H // 2, H // 2, device=device).reshape(-1, 1).expand(H, W)
    x = torch.arange(-W // 2, W // 2, device=device).reshape(1, -1).expand(H, W)
    r = torch.sqrt(x**2 + y**2)
    r_max = r.max()
    bounds = torch.linspace(0, r_max, K + 1, device=device)
    return [(r >= bounds[i]) & (r < bounds[i + 1]) for i in range(K)]


def preprocess_on_gpu(x: torch.Tensor, img_proc, dtype=torch.float16):
    """
    x: [B,3,H,W], in [0,1]
    returns pixel_values: [B,3,Hp,Wp] normalized like CLIP, on GPU
    """

    size = getattr(img_proc, "size", None) or {}
    crop_size = getattr(img_proc, "crop_size", None) or {}
    do_resize = getattr(img_proc, "do_resize", True)
    do_center_crop = getattr(img_proc, "do_center_crop", True)


    if isinstance(size, dict):
        target_short = size.get("shortest_edge", None)
        target_h = size.get("height", None)
        target_w = size.get("width", None)
    else:
        target_short = None
        target_h = target_w = None

    if isinstance(crop_size, dict):
        crop_h = crop_size.get("height", None)
        crop_w = crop_size.get("width", None)
    else:
        crop_h = crop_w = None

    B, C, H, W = x.shape
    y = x

    # resize
    if do_resize and target_short is not None:
        scale = target_short / min(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        y = F.interpolate(y, size=(new_h, new_w), mode="bicubic", align_corners=False)
    elif do_resize and (target_h is not None and target_w is not None):
        y = F.interpolate(y, size=(target_h, target_w), mode="bicubic", align_corners=False)

    # center crop
    if do_center_crop and (crop_h is not None and crop_w is not None):
        _, _, H2, W2 = y.shape
        top = max((H2 - crop_h) // 2, 0)
        left = max((W2 - crop_w) // 2, 0)
        y = y[:, :, top:top + crop_h, left:left + crop_w]

    # normalize
    mean = torch.tensor(getattr(img_proc, "image_mean", [0.5, 0.5, 0.5]), device=y.device, dtype=y.dtype).view(1,3,1,1)
    std  = torch.tensor(getattr(img_proc, "image_std",  [0.5, 0.5, 0.5]), device=y.device, dtype=y.dtype).view(1,3,1,1)
    y = (y - mean) / std

    return y.to(dtype)

def per_sample_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, Tm1, V = shift_logits.shape
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(B, Tm1)

    valid = (shift_labels != ignore_index)
    denom = valid.sum(dim=1).clamp_min(1)
    return (loss_flat * valid).sum(dim=1) / denom

def normalize_pixel_values_on_gpu(x: torch.Tensor, img_proc, out_dtype=torch.float16):
    """
    x: [B,3,H,W] in [0,1], on GPU
    Return: pixel_values [B,3,H,W] normalized by image_processor's mean/std.
    NOTE: No resize/crop/rescale to keep shape identical to x.
    """
    mean = getattr(img_proc, "image_mean", [0.5, 0.5, 0.5])
    std  = getattr(img_proc, "image_std",  [0.5, 0.5, 0.5])

    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

    y = (x - mean) / std
    return y.to(out_dtype)

# def register_llm_hidden_hooks(model, require_grad=False):
#     features = {}
#     handles = []

#     def hook_fn(module, input, output):
#         hidden = output[0] if isinstance(output, tuple) else output
#         hidden_safe = hidden.detach().float().cpu()
#         if torch.isnan(hidden_safe).any() or torch.isinf(hidden_safe).any():
#             print(f"Layer {module.layer_idx} contains NaN or Inf. Skipping.")
#             return
#         if not require_grad:
#             hidden = hidden.detach().float().cpu()
#         features[module.layer_idx] = hidden

#     if hasattr(model.language_model, "model"):
#         layers = model.language_model.model.layers
#     elif hasattr(model.language_model, "layers"):
#         layers = model.language_model.layers
#     else:
#         raise AttributeError("Cannot locate transformer layers in model.language_model")

#     for i, layer in enumerate(layers):
#         layer.layer_idx = i
#         handles.append(layer.register_forward_hook(hook_fn))

#     return handles, features

def register_llm_hidden_hooks(model, require_grad=False, move_to_cpu=False, check_finite=False):
    """
    require_grad: True -> keep tensor with grad
                  False -> store detached tensor (no grad)
    move_to_cpu:  if True, move stored features to CPU (slow; default False)
    check_finite: if True, check NaN/Inf (may add overhead; default False)
    """
    features = {}
    handles = []

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output

        if check_finite:
            # avoid cpu sync; keep on device
            if not torch.isfinite(hidden).all():
                # don't print too often in hot path
                return

        if not require_grad:
            hidden = hidden.detach()

        if move_to_cpu:
            hidden = hidden.to("cpu", non_blocking=True)

        features[module.layer_idx] = hidden

    # locate layers
    if hasattr(model.language_model, "model"):
        layers = model.language_model.model.layers
    elif hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    else:
        raise AttributeError("Cannot locate transformer layers in model.language_model")

    for i, layer in enumerate(layers):
        layer.layer_idx = i
        handles.append(layer.register_forward_hook(hook_fn))

    return handles, features

def generate_response(model, processor, image_tensor, query):

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    pil_image = to_pil_image(image_tensor.cpu().float().clamp(0, 1))

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(model.device).half()
        else:
            inputs[k] = v.to(model.device)

    with amp.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1024,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=False,
        )

    input_token_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0, input_token_len:], skip_special_tokens=True
    ).strip()

    return response
