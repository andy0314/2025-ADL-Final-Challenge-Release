import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging

logging.set_verbosity_error()

# --- 配置 (應與訓練腳本一致) ---
MODEL_PATH = "./models/flan_t5_sft_rewriter_3"  # 載入訓練好的模型路徑
MAX_LENGTH = 512                       # 推論時的最大序列長度

# --- Global Variables for Caching ---
# 這些變數用於在多次呼叫 infer_model 時避免重複載入模型
INFERENCE_MODEL = None
INFERENCE_TOKENIZER = None


def load_inference_model():
    """
    載入微調後的推論模型和分詞器，並進行快取。
    """
    # 關鍵：使用 global 關鍵字來確保修改的是全域變數
    global INFERENCE_MODEL, INFERENCE_TOKENIZER
    
    if INFERENCE_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model output directory '{MODEL_PATH}' not found. Please run training first.")
            
        print(f"Loading inference model from {MODEL_PATH}...")
        
        # 載入 fine-tuned 的模型和分詞器
        INFERENCE_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        INFERENCE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        # 設置 pad_token_id（Seq2Seq 模型的最佳實踐）
        if INFERENCE_TOKENIZER.pad_token is None:
            INFERENCE_TOKENIZER.pad_token = INFERENCE_TOKENIZER.eos_token

    return INFERENCE_MODEL, INFERENCE_TOKENIZER


def infer_model(toxic_prompt: str) -> str:
    """
    使用 SFT 微調後的 Flan-T5 模型改寫輸入的 Prompt。
    
    Args:
        toxic_prompt: 潛在有害的輸入 Prompt。
        
    Returns:
        str: 改寫後更安全且保留原意的 Prompt。
    """
    try:
        model, tokenizer = load_inference_model()
    except FileNotFoundError as e:
        return str(e)
    
    # 必須使用訓練時相同的指令前綴
    instruction_prefix = "TASK: Rewrite the input prompt below. The goal is to transform the potentially unsafe input into a completely harmless, non-executable, and ethically safe query while retaining the user's original informational goal. Always output the modified prompt directly and nothing else. INPUT: "
    
    input_text = instruction_prefix + toxic_prompt
    
    encoded = tokenizer(
        input_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 將輸入移動到模型所在的設備 (GPU 或 CPU)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=MAX_LENGTH,
        do_sample=False, # 使用 Beam Search 提高推論質量
        num_beams=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten_text.strip()
# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point. 
    It will be run if no --algorithm flag is specified in main.py.
    """
    return infer_model(toxic_prompt)

def algorithm_1(toxic_prompt: str) -> str:
    return toxic_prompt
