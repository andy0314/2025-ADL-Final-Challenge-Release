import argparse
import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
from tqdm import tqdm

from llm import HuggingFaceEmbeddingModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    with open(args.input, "rb") as f:
        library = pickle.load(f)

    for strategy in tqdm(library.values()):
        examples = strategy["Example"]
        embeddings = []
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            batch_embeddings = model.encode(batch)
            for embed in batch_embeddings:
                embeddings.append(embed)

        strategy["Embeddings"] = embeddings

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(library, f)


if __name__ == "__main__":
    main()
