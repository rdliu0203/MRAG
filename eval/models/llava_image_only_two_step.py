import argparse
import copy
import io
import json
import os
import shortuuid
import torch

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def build_choices_text(item):
    return (
        f"{item['question']}\n Choices:\n"
        f"A: {item['A']}\n"
        f"B: {item['B']}\n"
        f"C: {item['C']}\n"
        f"D: {item['D']}"
    )


def eval_model(args):
    ans_file = open(args.answers_file, "w")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device_map = "auto"
    llava_model_args = {"multimodal": True, "overwrite_config": {"image_aspect_ratio": "pad"}}
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, **llava_model_args
    )
    model = model.to("cuda")
    model.eval()

    dataset = load_dataset("uclanlp/MRAG-Bench", split="test")

    for item in tqdm(dataset):
        # Prepare images: query + retrieved
        query_image = item["image"].convert("RGB")
        retrieved_images = [
            img.convert("RGB")
            if isinstance(img, Image.Image)
            else Image.open(img["path"]).convert("RGB")
            if isinstance(img, dict) and "path" in img
            else Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            for img in item["retrieved_images"]
        ]
        if item["scenario"] == "Incomplete":
            # Only one retrieved image for this scenario
            retrieved_images = [retrieved_images[0]]
        images = [query_image] + retrieved_images

        # ---- Step 1: Describe images (no question text) ----
        placeholder_block = DEFAULT_IMAGE_TOKEN * len(images)
        describe_prompt = (
            "Describe the key objects and relationships in these images. "
            "Be concise and focus on visible evidence. "
            f"{placeholder_block}\n"
        )
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], describe_prompt)
        conv.append_message(conv.roles[1], None)
        prompt1 = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_img.to(dtype=torch.float16, device=model.device) for _img in image_tensors]
        image_sizes = [img.size for img in images]

        with torch.inference_mode():
            desc_tokens = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
            )
        description = tokenizer.batch_decode(desc_tokens, skip_special_tokens=True)[0].strip()

        # ---- Step 2: Answer question using description (text-only) ----
        qa_prompt = (
            "Use ONLY the description to answer the multiple-choice question. "
            "Return just the option letter (A/B/C/D).\n"
            f"Description: {description}\n\n"
            f"{build_choices_text(item)}"
        )
        conv2 = copy.deepcopy(conv_templates["qwen_1_5"])
        conv2.append_message(conv2.roles[0], qa_prompt)
        conv2.append_message(conv2.roles[1], None)
        prompt2 = conv2.get_prompt()

        qa_input_ids = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        with torch.inference_mode():
            qa_tokens = model.generate(
                qa_input_ids,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
            )
        answer_text = tokenizer.batch_decode(qa_tokens, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "qs_id": item["id"],
                    "prompt": qa_prompt,
                    "output": answer_text,
                    "gt_answer": item["answer"],
                    "shortuuid": ans_id,
                    "model_id": "llava_one_vision_qwen_7b_two_step_image_only",
                    "gt_choice": item["answer_choice"],
                    "scenario": item["scenario"],
                    "aspect": item["aspect"],
                    "description": description,
                }
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="llava_one_vision_image_only_results.jsonl")
    parser.add_argument("--seed", type=int, default=None, help="Torch RNG seed for generation")

    args = parser.parse_args()
    eval_model(args)
