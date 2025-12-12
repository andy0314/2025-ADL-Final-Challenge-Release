import os
import pickle
import logging

from dotenv import load_dotenv

from autodan_turbo.framework import Attacker, Scorer, TrueScorer, Summarizer, Retrieval, Target
from autodan_turbo.llm import HuggingFaceLanguageModel, HuggingFaceEmbeddingModel
from autodan_turbo.pipeline import AutoDANTurbo

logger = logging.getLogger(__name__)


# --- MANDATORY ENTRY POINT for Official Evaluation ---
def evaluate_rewrite(toxic_prompt: str) -> str:
    """
    [MANDATORY] This is the final, official algorithm entry point.
    It will be run if no --algorithm flag is specified in main.py.
    """
    return algo_autodan_turbo(toxic_prompt)


# --- Config ---
strategy_library = "logs/lifelong_strategy_library.pkl"
epochs = 2
break_score= 1.0

embedding_model = "Qwen/Qwen3-Embedding-0.6B"
pooling_strategy = "last"
embed_max_length = 8192
embed_instruction = "Instruct: Retrieve semantically similar text.\nQuery:{query}"
quantize_embedder = False

language_model = "google/gemma-1.1-2b-it"
lm_combine_system_user = True
quantize_lm = False

target_model = "unsloth/Llama-3.2-3B-Instruct"
use_llm_scorer = False
guard_model = "Qwen/Qwen3Guard-Gen-0.6B"
usefulness_model = "theblackcat102/Qwen3-1.7B-Usefulness-Judge"
# --------------

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model = HuggingFaceLanguageModel(target_model, token=hf_token)
target = Target(model)

language_model = HuggingFaceLanguageModel(
    repo_name=language_model,
    should_quantize=quantize_lm,
    token=hf_token,
    combine_system_user=lm_combine_system_user,
)
attacker = Attacker(language_model)
summarizer = Summarizer(language_model)

if use_llm_scorer:
    scorer = Scorer(language_model)
else:
    scorer = TrueScorer(guard_model, usefulness_model)

text_embedding_model = HuggingFaceEmbeddingModel(
    repo_name=embedding_model,
    should_quantize=quantize_embedder,
    token=hf_token,
    pooling_strategy=pooling_strategy,
    max_length=embed_max_length,
    embed_instruction=embed_instruction,
)

with open(strategy_library, 'rb') as f:
    lifelong_strategy_library = pickle.load(f)

retrival = Retrieval(
    text_embedding_model,
    logger,
    scorer.get_decent_improvement(),
    scorer.get_large_improvement(),
    lifelong_strategy_library,
)

attack_kit = {
    'attacker': attacker,
    'scorer': scorer,
    'summarizer': summarizer,
    'retrival': retrival,
    'logger': logger,
}

autodan_turbo_pipeline = AutoDANTurbo(
    turbo_framework=attack_kit,
    data=None,
    target=target,
    epochs=epochs,
    break_score=break_score,
    warm_up_iterations=None,
    lifelong_iterations=None
)


def algo_autodan_turbo(toxic_prompt: str) -> str:
    return autodan_turbo_pipeline.test(toxic_prompt, lifelong_strategy_library)["jailbreak_prompt"]
