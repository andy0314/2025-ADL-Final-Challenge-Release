import torch
import os
import json
import argparse
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, AutoModelForSeq2SeqLM # ADDED AutoModelForSeq2SeqLM
from src.eval import initialize_models, judge
import random
from tqdm import tqdm
from itertools import combinations # 引入 combinations 進行兩兩配對

logging.set_verbosity_error()

DEFAULT_DATASET_PATH = "theblackcat102/ADL_Final_25W_part1_with_cost"
DEFAULT_SAFETY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
DEFAULT_USEFULNESS_MODEL = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
DEFAULT_CHAT_MODEL = "unsloth/Llama-3.2-3B-Instruct" # RESTORED: 用於評估（Judge）的模型
DEFAULT_REWRITE_MODEL = "google/flan-t5-base"       # NEW: 用於生成改寫 Prompt 的模型

def run_rewrite_model(prompt: str, model, tokenizer, max_new_tokens=512) -> str:
    """Return model-generated rewritten text for a given prompt using T5 (Seq2Seq) format."""
    
    # T5/Flan-T5 處理指令的方式是將指令和輸入結合在一個字串中。
    # 這裡我們使用一個簡單的 Instruction Template。
    instruction_prefix = "TASK: Rewrite the input prompt below. The goal is to transform the potentially unsafe input into a completely harmless, non-executable, and ethically safe query while retaining the user's original informational goal. Always output the modified prompt directly and nothing else. INPUT: "
    input_text = instruction_prefix + prompt
    
    encoded = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    
    # Seq2Seq 模型只需要 generate 輸出序列
    # T5 使用 max_length 來控制輸出長度
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        # T5 通常不需要 pad_token_id，但保留以防萬一
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Seq2Seq 模型的 outputs[0] 已經只包含生成的 tokens
    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 清理輸出行首尾的空白或特殊符號
    return rewritten_text.strip()

def parse_args():
    parser = argparse.ArgumentParser("Generate Reward Model Dataset")

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--guard-model", type=str, default=DEFAULT_SAFETY_MODEL)
    parser.add_argument("--usefulness-model", type=str, default=DEFAULT_USEFULNESS_MODEL)
    parser.add_argument("--chat-model", type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--rewrite-model", type=str, default=DEFAULT_REWRITE_MODEL) # NEW ARGUMENT
    parser.add_argument("--output", type=str, default="reward_dataset_pairs.jsonl",
                        help="Output file for the final preference pairs.")
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


def run_single_iteration(prompt: str, rewrite_model, rewrite_tokenizer, iteration_idx: int, sample_id: int) -> Dict:
    """
    1. 用 rewrite-model 重寫 prompt
    2. 用 judge() 評分
    """
    # 使用獨立的重寫模型和分詞器
    rewritten = run_rewrite_model(prompt, rewrite_model, rewrite_tokenizer) 

    # 假設 judge 函式存在並能正確工作
    eval_result = judge(rewritten, prompt) 

    # 結合 safety 和 relevance 作為 Reward
    reward = float(eval_result["safety_score"]) * float(eval_result["relevance_score"])

    return {
        "id": sample_id,          # 原始 prompt 的 ID
        "iteration": iteration_idx, # 該 prompt 的第幾次生成
        "prompt": prompt,
        "response": rewritten,
        "reward": reward,
        "safety": eval_result["safety_score"],
        "relevance": eval_result["relevance_score"]
    }

def create_preference_pairs(raw_data: List[Dict]) -> List[Dict]:
    """
    將單一評分的資料轉換為 (prompt, chosen_response, rejected_response) 的偏好對。
    
    1. 根據 prompt 內容分組。
    2. 在每個組內，對所有回應進行兩兩比較。
    3. 根據 'reward' 分數判斷哪個回應是 'chosen' (較高分數) 哪個是 'rejected' (較低分數)。
    """
    
    # 根據 prompt 分組資料
    grouped_data: Dict[str, List[Tuple[str, float]]] = {}
    for item in raw_data:
        prompt = item['prompt']
        if prompt not in grouped_data:
            grouped_data[prompt] = []
        # 儲存 (response 內容, reward 分數)
        grouped_data[prompt].append((item['response'], item['reward']))

    preference_pairs: List[Dict] = []
    
    # 對每個 prompt 組建立偏好對
    print("\n--- 步驟 3: 建立偏好對 (Preference Pairs) ---")
    for prompt, response_list in tqdm(grouped_data.items(), desc="Creating Pairs", ncols=100, dynamic_ncols=False, ascii=True):
        if len(response_list) < 2:
            # 只有一個回應的 prompt 無法形成偏好對
            continue

        # 使用 itertools.combinations 產生所有可能的兩兩組合
        for response_a, response_b in combinations(response_list, 2):
            text_a, reward_a = response_a
            text_b, reward_b = response_b

            # 只有在 reward 分數不相等時才形成有效的偏好對
            if reward_a != reward_b:
                if reward_a > reward_b:
                    chosen = text_a
                    rejected = text_b
                else:
                    chosen = text_b
                    rejected = text_a
                    
                preference_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    # 可以選擇保留分數差異作為參考 (reward_margin)
                    "reward_margin": abs(reward_a - reward_b)
                })

    return preference_pairs


def main():
    args = parse_args()

    # --- 步驟 1: 初始化評估 (Judge) 模型 ---
    print("--- 步驟 1: 初始化評估模型 ---")
    # 使用 args.chat_model (用於評估) 初始化 Judge 相關模型
    initialize_models(args.guard_model, args.usefulness_model, args.chat_model)
    
    # --- 步驟 2: 載入重寫 (Rewrite) 模型 ---
    print(f"--- 步驟 2: 載入重寫模型 ({args.rewrite_model}) ---")
    
    # T5 作為重寫模型，使用 AutoModelForSeq2SeqLM
    rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.rewrite_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    rewrite_tokenizer = AutoTokenizer.from_pretrained(args.rewrite_model)
    
    # Load dataset
    ds = load_original_dataset(args.dataset)

    # Limit dataset size if needed
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    # 用於儲存所有生成的單一評分數據
    all_generated_data: List[Dict] = [] 

    print("--- 步驟 3: 生成回應並評分 (Scoring) ---")
    
    # 生成並評分循環
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="Generating Responses"):
        prompt = row["prompt"]
        
        for it in range(args.iterations):
            # 傳遞 rewrite_model 和 rewrite_tokenizer 給 run_single_iteration
            result = run_single_iteration(prompt, rewrite_model, rewrite_tokenizer, it, idx)
            all_generated_data.append(result)

    print(f"\n✅ 完成生成。共 {len(all_generated_data)} 個單一評分樣本。")
    
    # --- 步驟 4: 轉換為偏好對 (Preference Pairs) ---
    
    preference_pairs = create_preference_pairs(all_generated_data)
    
    # --- 步驟 5: 儲存結果 ---

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"\n✅ 成功創建 {len(preference_pairs)} 個偏好對。")
    print(f"--- 步驟 5: 儲存數據到 {output_path} ---")

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in tqdm(preference_pairs, desc="Writing Pairs"):
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n🔥 獎勵模型數據集 (Preference Pairs) 已創建完成。")
    print("輸出格式: {prompt, chosen, rejected, reward_margin}")


if __name__ == "__main__":
    # 注意: 運行此腳本需要正確配置 GPU 和安裝 'src.eval' 模組。
    # 這裡假設環境中已安裝 'torch', 'transformers', 'datasets', 'tqdm'。
    try:
        main()
    except Exception as e:
        print(f"\n致命錯誤: {e}")
        print("請確保 'src.eval' 模組中的 'initialize_models' 和 'judge' 函數已正確定義並可用。")
