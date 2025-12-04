from dotenv import load_dotenv
from framework import Attacker, Scorer, TrueScorer, Summarizer, Retrieval, Target
from llm import HuggingFaceLanguageModel, HuggingFaceEmbeddingModel, OpenAIEmbeddingModel
import argparse
import logging
import os
from pipeline import AutoDANTurbo
import numpy as np
import json
import pickle
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


def config():
    config = argparse.ArgumentParser()

    config.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    config.add_argument("--data", type=str, default="theblackcat102/ADL_Final_25W_part1_with_cost")
    config.add_argument("--split", type=str, default="test")
    config.add_argument("--strategy_library", type=str, default="./logs/lifelong_strategy_library.pkl")
    config.add_argument("--epochs", type=int, default=150)
    config.add_argument("--break_score", type=float, default=1.0)
    config.add_argument("--output_file", type=Path, default="../results/autodan-turbo/prompts.jsonl")

    config.add_argument('--azure', action='store_true', help='Use azure')
    config.add_argument("--azure_endpoint", type=str, default="your_azure_endpoint")
    config.add_argument("--azure_api_version", type=str, default="2024-02-01")
    config.add_argument("--azure_deployment_name", type=str, default="your_azure_deployment_name")
    config.add_argument("--azure_api_key", type=str, default="your_azure_api_key")

    config.add_argument("--openai_api_key", type=str, default="your_openai_api_key")
    config.add_argument("--openai_embedding_model", type=str)

    config.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    config.add_argument("--pooling_strategy", type=str, default="last")
    config.add_argument("--embed_max_length", type=int, default=8192)
    config.add_argument("--embed_instruction", type=str, default="Instruct: Retrieve semantically similar text.\nQuery:{query}")

    config.add_argument("--language_model", type=str, default="google/gemma-1.1-2b-it")
    config.add_argument("--lm_combine_system_user", action="store_true")

    config.add_argument("--use_llm_scorer", action="store_true")
    config.add_argument("--guard_model", type=str, default="Qwen/Qwen3Guard-Gen-0.6B")
    config.add_argument("--usefulness_model", type=str, default="theblackcat102/Qwen3-1.7B-Usefulness-Judge")

    return config


if __name__ == '__main__':
    args = config().parse_args()

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    output_dir = args.output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "running.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    model = HuggingFaceLanguageModel(args.model, token=hf_token)
    # configure your own base model here

    language_model = HuggingFaceLanguageModel(
        args.language_model, args.lm_combine_system_user, hf_token
    )
    attacker = Attacker(language_model)
    summarizer = Summarizer(language_model)

    if args.use_llm_scorer:
        scorer = Scorer(language_model)
    else:
        scorer = TrueScorer(args.guard_model, args.usefulness_model)

    if args.azure:
        text_embedding_model = OpenAIEmbeddingModel(azure=True,
                                                    azure_endpoint=args.azure_endpoint,
                                                    azure_api_version=args.azure_api_version,
                                                    azure_deployment_name=args.azure_deployment_name,
                                                    azure_api_key=args.azure_api_key,
                                                    logger=logger)
    elif args.openai_embedding_model:
        text_embedding_model = OpenAIEmbeddingModel(azure=False,
                                                    openai_api_key=args.openai_api_key,
                                                    embedding_model=args.openai_embedding_model)
    else:
        text_embedding_model = HuggingFaceEmbeddingModel(
            args.embedding_model, args.pooling_strategy, args.embed_max_length, args.embed_instruction, hf_token
        )
    retrival = Retrieval(
        text_embedding_model,
        logger,
        scorer.get_decent_improvement(),
        scorer.get_large_improvement(),
    )

    target = Target(model)
    # configure your own target model here

    init_library, init_attack_log, init_summarizer_log = {}, [], []
    attack_kit = {
        'attacker': attacker,
        'scorer': scorer,
        'summarizer': summarizer,
        'retrival': retrival,
        'logger': logger
    }
    autodan_turbo_pipeline = AutoDANTurbo(turbo_framework=attack_kit,
                                          data=None,
                                          target=target,
                                          epochs=args.epochs,
                                          break_score=args.break_score,
                                          warm_up_iterations=None,
                                          lifelong_iterations=None)

    with open(args.strategy_library, 'rb') as f:
        lifelong_strategy_library = pickle.load(f)

    dataset = load_dataset(args.data, split=args.split)

    # --- Resume support by line count ---
    start_index = 0
    if os.path.exists(args.output_file):
        print(f"Detected existing results file at {args.output_file}.")
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                start_index = len(f.readlines())
        except Exception as e:
            print(f"Warning: Could not parse existing JSONL file to resume: {e}")

    total = len(dataset)
    print(f"Resuming processing from index {start_index}/{total} (skipping {start_index} items already completed).")

    # Use 'a' (append) mode for resilient, incremental writing
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for index, record in tqdm(enumerate(dataset)):
            # Skip already processed samples
            if index < start_index:
                continue

            toxic_prompt = record["prompt"]
            rewritten_prompt = autodan_turbo_pipeline.test(
                toxic_prompt, lifelong_strategy_library
            )

            # Save as a JSON-encoded string, as requested
            try:
                f.write(json.dumps(rewritten_prompt, ensure_ascii=False) + '\n')
            except Exception as e:
                rec_id = record.get('id', index)
                print(f"Error writing record {rec_id} to JSONL: {e}")
