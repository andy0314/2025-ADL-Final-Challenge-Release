import argparse
from pathlib import Path
import pickle

from framework import Scorer, TrueScorer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_strategy_library", type=Path, default="logs_r/lifelong_strategy_library.pkl")
    parser.add_argument("--output_strategy_library", type=Path)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.input_strategy_library, "rb") as f:
        library = pickle.load(f)

    for strategy in library.values():
        for i, score in enumerate(strategy["Score"]):
            if score >= Scorer.get_large_improvement():
                strategy["Score"][i] = TrueScorer.get_large_improvement()
            elif score >= Scorer.get_decent_improvement():
                strategy["Score"][i] = TrueScorer.get_decent_improvement()
            else:
                strategy["Score"][i] = 0.0

    output_strategy_library = args.output_strategy_library or args.input_strategy_library
    output_strategy_library.parent.mkdir(parents=True, exist_ok=True)
    with open(output_strategy_library, "wb") as f:
        pickle.dump(library, f)


if __name__ == "__main__":
    main()
