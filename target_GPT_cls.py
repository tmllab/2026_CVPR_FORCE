import torch
import numpy as np
import json
import os
import argparse
import time
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms.functional import to_pil_image
from eval import *
import io
import base64



OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)


@torch.no_grad()
def cls_evaluate(cls_model, cls_tokenizer, behavior, generation, context=None):
    if context:
        inputs = LLAMA2_CLS_PROMPT['prompt_contextual'].format(behavior=behavior, generation=generation, context=context)
    else:
        inputs = LLAMA2_CLS_PROMPT['prompt'].format(behavior=behavior, generation=generation)

    inputs = cls_tokenizer(inputs, return_tensors="pt", padding='longest').to(cls_model.device)
    outputs = cls_model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=1
    )
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    decoded = cls_tokenizer.decode(new_tokens, skip_special_tokens=False).strip().lower()
    if "yes" in decoded.split():
        return 1
    return 0

@torch.no_grad()
def generate_response_openai(image_tensor, query):
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    pil_image = to_pil_image(image_tensor.cpu().float().clamp(0, 1))

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    try:
        completion = client.chat.completions.create(
            model="gpt-5", # gpt-4-turbo # "gpt-4o"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ],
            max_completion_tokens=4096
        )
        content = completion.choices[0].message.content
        if content:
            return content.strip()
        else:
            print("empty")
            return ""
    except Exception as e:
        error_msg = f"[ERROR] {type(e).__name__}: {e}"
        return error_msg

def main(args):
    total_items = 0
    total_success = 0
    total_query_time = 0

    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_model_name, use_fast=False, truncation_side="left", padding_side="left")
    cls_model = AutoModelForCausalLM.from_pretrained(
        args.cls_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file_mode = 'a' if (os.path.exists(args.output_path) and args.start_index > 0) else 'w'
    with open(args.input_text_path, 'r', encoding='utf-8') as fin, open(args.output_path, file_mode, encoding='utf-8') as fout:
        for line_no, line in enumerate(tqdm(fin)):
            if line_no < args.start_index:
                continue
            item = json.loads(line)
            if "id" not in item:
                continue

            total_items += 1
            item_id = item['id']
            instruction = item['instruction']

            query_time = 0

            while True:
                image_filename = f"{item_id}_{query_time}_adv.npy"
                image_path = os.path.join(args.input_vis_path, image_filename)
                image_tensor = torch.from_numpy(np.load(image_path)).float()
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                response = generate_response_openai(image_tensor, instruction)

                response_top = " ".join(response.lower().split()[:20])
                if not any(prefix.lower() in response_top.lower() for prefix in test_prefixes):
                    score = cls_evaluate(cls_model, cls_tokenizer, instruction, response)
                    if score == 1:
                        total_success += 1
                        total_query_time += (query_time + 1)

                        result = {
                            "id": item_id,
                            "result": True,
                            "query_time": query_time,
                            "instruction": instruction,
                            "response": response
                        }
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        fout.flush()
                        break
                query_time += 1

                next_image_filename = f"{item_id}_{query_time}_adv.npy"
                next_image_path = os.path.join(args.input_vis_path, next_image_filename)
                if not os.path.exists(next_image_path):
                    total_query_time += (query_time)
                    result = {
                        "id": item_id,
                        "result": False,
                        "query_time": query_time - 1,
                        "instruction": instruction,
                        "response": response
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fout.flush()
                    break

        attack_success_rate = total_success / total_items if total_items > 0 else 0.0
        avg_query_time = total_query_time / total_items if total_items > 0 else 0.0

        summary_result = {
            "summary": True,
            "attack_success_rate": attack_success_rate,
            "average_query_time": avg_query_time
        }
        fout.write(json.dumps(summary_result, ensure_ascii=False) + '\n')
        fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_model_name", type=str, default="../HarmBench-Llama-2-13b-cls")
    parser.add_argument("--input_text_path", type=str, default="./input_text_path.jsonl")
    parser.add_argument("--input_vis_path", type=str, default="./input_vis_path")
    parser.add_argument("--output_path", type=str, default="./output_target.jsonl")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()
    main(args)
