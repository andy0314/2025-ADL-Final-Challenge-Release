from dotenv import load_dotenv
from framework import Attacker, Scorer, TrueScorer, Summarizer, Retrieval, Target
from llm import HuggingFaceLanguageModel, HuggingFaceEmbeddingModel, OpenAIEmbeddingModel
import argparse
import logging
import os
from pipeline import AutoDANTurbo
import wandb
import numpy as np
import json
import pickle
from pathlib import Path


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--model", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    config.add_argument("--data", type=str, default="data/public.json")
    config.add_argument("--epochs", type=int, default=150)
    config.add_argument("--warm_up_iterations", type=int, default=1)
    config.add_argument("--lifelong_iterations", type=int, default=4)
    config.add_argument('--hot', action='store_true', help='hot start')
    config.add_argument("--hot_lifelong", action='store_true', help='hot start lifelong')
    config.add_argument("--break_score", type=float, default=1.0)

    config.add_argument("--output_dir", type=Path, default="logs")
    config.add_argument("--project_name", type=str, default="AutoDAN-Turbo")
    config.add_argument("--run_name", type=str, default="run")

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
    config.add_argument("--scorer_epsilon", type=float, default="0.1")

    config.add_argument("--debug", action='store_true', help='debug')
    return config


def save_data(
        strategy_library,
        attack_log,
        summarizer_log,
        strategy_library_file,
        strategy_library_pkl,
        attack_log_file,
        summarizer_log_file
):
    strategy_library_json = {
        s_name: {
            "Strategy": s_info["Strategy"],
            "Definition": s_info["Definition"],
            "Example": s_info["Example"],
        }
        for s_name, s_info in strategy_library.items()
    }

    try:
        with open(strategy_library_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_library_json, f, ensure_ascii=False, indent=4)
        logger.info(f"Strategy library has been saved to {strategy_library_file}")
    except Exception as e:
        logger.error(f"Saving strategy library to {strategy_library_file} failed: {e}")

    try:
        with open(strategy_library_pkl, 'wb') as f:
            pickle.dump(strategy_library, f)
        logger.info(f"Strategy library has been saved to {strategy_library_pkl}")
    except Exception as e:
        logger.error(f"Saving strategy library to {strategy_library_pkl} failed: {e}")

    try:
        with open(attack_log_file, 'w', encoding='utf-8') as f:
            json.dump(attack_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Attack log has been saved to {attack_log_file}")
    except Exception as e:
        logger.error(f"Saving attack log to {attack_log_file} failed: {e}")

    try:
        with open(summarizer_log_file, 'w', encoding='utf-8') as f:
            json.dump(summarizer_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Summarizer log has been saved to {summarizer_log_file}")
    except Exception as e:
        logger.error(f"Saving summarizer log to {summarizer_log_file} failed: {e}")


if __name__ == '__main__':
    args = config().parse_args()

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    output_dir = args.output_dir / args.run_name
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

    wandb.init(project=args.project_name, name=args.run_name)

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
        scorer = TrueScorer(args.guard_model, args.usefulness_model, args.scorer_epsilon)

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
                                                    embedding_model=args.embedding_model)
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

    data = json.load(open(args.data, 'r'))
    if args.debug:
        for stage, prompts in data.items():
            data[stage] = prompts[:3]

    target = Target(model)
    # configure your own target model here

    attack_kit = {
        'attacker': attacker,
        'scorer': scorer,
        'summarizer': summarizer,
        'retrival': retrival,
        'logger': logger
    }
    autodan_turbo_pipeline = AutoDANTurbo(turbo_framework=attack_kit,
                                          data=data,
                                          target=target,
                                          epochs=args.epochs,
                                          break_score=args.break_score,
                                          warm_up_iterations=args.warm_up_iterations,
                                          lifelong_iterations=1)
    # We placed the iterations afterward to ensure the program saves the running results after each iteration. Alternatively, you can set lifelong_iterations using args.lifelong_iterations.

    if args.debug:
        suffix = "_debug"
    else:
        suffix = ''

    warm_up_strategy_library_file = output_dir / f'warm_up_strategy_library{suffix}.json'
    warm_up_strategy_library_pkl = output_dir / f'warm_up_strategy_library{suffix}.pkl'
    warm_up_attack_log_file = output_dir / f'warm_up_attack_log{suffix}.json'
    warm_up_summarizer_log_file = output_dir / f'warm_up_summarizer_log{suffix}.json'

    lifelong_strategy_library_file = output_dir / f'lifelong_strategy_library{suffix}.json'
    lifelong_strategy_library_pkl = output_dir / f'lifelong_strategy_library{suffix}.pkl'
    lifelong_attack_log_file = output_dir / f'lifelong_attack_log{suffix}.json'
    lifelong_summarizer_log_file = output_dir / f'lifelong_summarizer_log{suffix}.json'

    if args.hot_lifelong:
        warm_up_strategy_library = pickle.load(open(warm_up_strategy_library_pkl, 'rb'))
        warm_up_attack_log = json.load(open(warm_up_attack_log_file, 'r'))
        warm_up_summarizer_log = json.load(open(warm_up_summarizer_log_file, 'r'))
        autodan_turbo_pipeline.hot_start_lifelong(warm_up_attack_log)
    else:
        if args.hot:
            warm_up_attack_log = json.load(open(warm_up_attack_log_file, 'r'))
        else:
            warm_up_attack_log = autodan_turbo_pipeline.warm_up([])
            save_data({}, warm_up_attack_log, [], warm_up_strategy_library_file, warm_up_strategy_library_pkl, warm_up_attack_log_file, warm_up_summarizer_log_file)

        warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log = autodan_turbo_pipeline.hot_start(warm_up_attack_log)
        save_data(warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log, warm_up_strategy_library_file, warm_up_strategy_library_pkl, warm_up_attack_log_file, warm_up_summarizer_log_file)

    for i in range(args.lifelong_iterations):
        logger.info(f"##########Lifelong iteration {i}##########")
        if i == 0:
            lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log = autodan_turbo_pipeline.lifelong_redteaming(warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log)
        else:
            lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log = autodan_turbo_pipeline.lifelong_redteaming(lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log)
        save_data(lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log, lifelong_strategy_library_file, lifelong_strategy_library_pkl, lifelong_attack_log_file, lifelong_summarizer_log_file)
