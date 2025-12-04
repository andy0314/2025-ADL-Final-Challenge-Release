import argparse
import json
import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
from tqdm import tqdm

from llm import HuggingFaceEmbeddingModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_log", default="logs_r/lifelong_attack_log.json")
    parser.add_argument("--summarizer_log", default="logs_r/lifelong_summarizer_log.json")
    parser.add_argument("--input_strategy_library", default="logs_r/lifelong_strategy_library.pkl")
    parser.add_argument("--output_strategy_library", type=Path, default="logs_r/Qwen3-Embedding-0.6B/lifelong_strategy_library.pkl")

    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--pooling_strategy", default="last")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--embed_instruction", default="Instruct: Retrieve semantically similar text.\nQuery:{query}")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    model = HuggingFaceEmbeddingModel(
        args.embedding_model,
        args.pooling_strategy,
        args.max_length,
        args.embed_instruction,
        hf_token,
    )

    with open(args.attack_log, "r") as f:
        attack_log = json.load(f)
    prompt2response = {}
    for log in attack_log:
        prompt2response[log["prompt"]] = log["response"]
    del attack_log

    with open(args.summarizer_log, "r") as f:
        summarizer_log = json.load(f)
    strong2weak_prompt = {}
    for log in summarizer_log:
        strong2weak_prompt[log["strong_prompt"]] = log["weak_prompt"]
    del summarizer_log

    with open(args.input_strategy_library, "rb") as f:
        library = pickle.load(f)

    for strategy in tqdm(library.values()):
        weak_responses = []
        for i in range(len(strategy["Example"])):
            strong_prompt = strategy["Example"][i]
            weak_prompt = strong2weak_prompt.get(strong_prompt)
            weak_response = prompt2response.get(weak_prompt)

            if weak_response is None:
                weak_response = "Sorry, I cannot help you with that."

            weak_responses.append(weak_response)

        embeddings = []
        for i in range(0, len(weak_responses), args.batch_size):
            batch = weak_responses[i : i + args.batch_size]
            batch_embeddings = model.encode(batch)
            for embed in batch_embeddings:
                embeddings.append(embed)

        strategy["Embeddings"] = embeddings

    args.output_strategy_library.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_strategy_library, "wb") as f:
        pickle.dump(library, f)


if __name__ == "__main__":
    main()
