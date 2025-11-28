import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    logging,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
from typing import Dict, List, Any

logging.set_verbosity_error()

# --- 1. 配置 ---
MODEL_NAME = "./models/flan_t5_sft_rewriter_2"
DATASET_PATH = "./data/merged_3.jsonl" 
OUTPUT_DIR = "./models/flan_t5_sft_rewriter_3"
MAX_LENGTH = 512 # 最大序列長度

def load_and_tokenize_data(tokenizer: AutoTokenizer) -> Dataset:
    """
    載入偏好對資料集，並只保留 'prompt' 和 'chosen' 欄位進行分詞。
    """
    print(f"Loading dataset from {DATASET_PATH}...")
    
    data_files = {"train": DATASET_PATH}
    dataset = load_dataset("json", data_files=data_files)["train"]
    
    # SFT 只需使用 'prompt' 和 'chosen'
    if not all(col in dataset.column_names for col in ['prompt', 'chosen']):
        raise ValueError("Dataset must contain 'prompt' and 'chosen' columns.")

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, List[int]]:
        """將文本轉換為 tokens，並設置標籤。"""
        
        # 使用我們在 build_reward_dataset.py 中定義的指令前綴來處理輸入 Prompt
        instruction_prefix = "TASK: Rewrite the input prompt below. The goal is to transform the potentially unsafe input into a completely harmless, non-executable, and ethically safe query while retaining the user's original informational goal. Always output the modified prompt directly and nothing else. INPUT: "
        
        # 1. 輸入 (Prompt) 分詞
        inputs = [instruction_prefix + p for p in examples["prompt"]]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

        # 2. 標籤 (Chosen Response) 分詞
        # 對於 Seq2Seq 模型，標籤 (labels) 必須是目標序列的 input_ids
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["chosen"], max_length=MAX_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names, # 移除原始文本欄位
    )
    return tokenized_datasets

def train_sft_rewriter():
    # --- 2. 載入模型和分詞器 ---
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 設置 pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入 Flan-T5 作為 Seq2Seq 模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        # 使用 bfloat16 或 float32
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # --- 3. 載入並預處理資料 ---
    train_dataset = load_and_tokenize_data(tokenizer)
    
    # --- 4. 配置 SFT 訓練參數 ---
    
    # 判斷使用哪種優化器：如果 CUDA 可用，使用 'adamw_torch' (標準 AdamW)；否則使用 'adamw_hf'
    optimizer_type = "adamw_torch" if torch.cuda.is_available() else "adamw_hf"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,        # 調整此值以適應您的 GPU VRAM
        gradient_accumulation_steps=1,
        num_train_epochs=10,                   # 訓練輪數
        learning_rate=5e-5,
        optim=optimizer_type,                 # CHANGED: 使用 adamw_torch 避免 bitsandbytes 依賴
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        remove_unused_columns=True,           # SFT Trainer 允許移除未使用的欄位
        bf16=torch.cuda.is_available(),       # 啟用 BF16 (如果您的 GPU 支援)
        load_best_model_at_end=False,         # 只是為了簡化
        predict_with_generate=True,           # 允許在評估時進行生成
    )

    # --- 5. 初始化 Data Collator 和 Trainer ---
    
    # Seq2Seq 的 Data Collator 負責將批次數據中的標籤進行特殊處理（遮罩 -100）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. 啟動訓練 ---
    print("\nStarting SFT fine-tuning on Chosen responses...")
    trainer.train()

    # --- 7. 儲存模型 ---
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ SFT Fine-tuning finished. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}.")
        print("請先執行 'build_reward_dataset.py' 腳本來生成偏好對數據。")
    else:
        try:
            train_sft_rewriter()
        except Exception as e:
            print(f"An error occurred during SFT training: {e}")
            print("請檢查您的環境設置和 GPU 配置。")
