import torch
import os
import json
import argparse
from datasets import load_dataset, Dataset
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from src.eval import initialize_models, judge
import random
from tqdm import tqdm

logging.set_verbosity_error()

DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

def run_chat_model(prompt: str, model, tokenizer, max_new_tokens=512) -> str:
    """Return model-generated rewritten text for a given prompt."""
    messages = [
        {"role": "system", "content": "You rewrite prompts to make them safer and more harmless. Do not change the original intent of the prompt. Directly output the modified version of prompt."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def parse_args():
    parser = argparse.ArgumentParser("Generate Reward Model Dataset")

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--guard-model", type=str, default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--usefulness-model", type=str, default=DEFAULT_USEFULNESS_MODEL)
    parser.add_argument("--chat-model", type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--output", type=str, default="reward_dataset.jsonl")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of rewriting + evaluation cycles per prompt")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for debugging")

    return parser.parse_args()


def load_original_dataset(path: str):
    print(f"Loading dataset {path} ...")
    ds = load_dataset(path)
    split = list(ds.keys())[0]
    ds = ds[split]
    if 'prompt' not in ds.column_names:
        raise ValueError("Dataset must contain column 'prompt'")
    return ds


def run_single_iteration(prompt: str, chat_model, tokenizer) -> Dict:
    """
    1. 用 chat-model 重寫 prompt
    2. 用 judge() 評分
    """
    rewritten = run_chat_model(prompt, chat_model, tokenizer)   # 你需要確保 inference function 回傳 string

    eval_result = judge(rewritten, prompt)

    reward = float(eval_result["safety_score"]) * float(eval_result["relevance_score"])

    return {
        "prompt": prompt,
        "response": rewritten,
        "reward": reward,
        "safety": eval_result["safety_score"],
        "relevance": eval_result["relevance_score"]
    }


def main():
    args = parse_args()

    # Initialize all models
    print("Initializing models ...")
    initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
    chat_model = AutoModelForCausalLM.from_pretrained(
            args.chat_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.chat_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load dataset
    ds = load_original_dataset(args.dataset)

    # Limit dataset size if needed
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Saving reward model data to {output_path} ...")

    with open(output_path, "w", encoding="utf-8") as f:

        for idx, row in tqdm(enumerate(ds), total=len(ds), desc="Building Reward Dataset"):
            prompt = row["prompt"]

            for it in range(args.iterations):

                result = run_single_iteration(prompt, chat_model, tokenizer)
                result["id"] = idx
                result["iteration"] = it

                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1} prompts")

    print(f"\nReward dataset created at: {output_path}")
    print("Format: {prompt, response, reward, safety, relevance}")


if __name__ == "__main__":
    main()
