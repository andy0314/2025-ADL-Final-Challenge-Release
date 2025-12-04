import argparse
from pathlib import Path
import json

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="theblackcat102/ADL_Final_25W_part1_with_cost")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_file", type=Path, default="data/public.json")
    parser.add_argument("--warm_up_strategy", nargs="+", default=["cost_le", "1"])
    parser.add_argument("--lifelong_strategy", nargs="+", default=["all"])

    args = parser.parse_args()
    return args


STRATEGIES = {}

def strategy(name_or_func=None):
    def decorator(func):
        if isinstance(name_or_func, str):
            STRATEGIES[name_or_func] = func
        else:
            STRATEGIES[func.__name__] = func
        return func

    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator


@strategy("all")
def all_(dataset):
    return list(dataset["prompt"])


@strategy
def cost_le(dataset, n):
    return [data["prompt"] for data in dataset if data["cost"] <= int(n)]


@strategy
def cost_ge(dataset, n):
    return [data["prompt"] for data in dataset if data["cost"] >= int(n)]


def main():
    args = parse_args()

    raw_dataset = load_dataset(args.data, split=args.split)

    dataset = {}
    func = STRATEGIES.get(args.warm_up_strategy[0])
    if func is None:
        raise ValueError(
            f"Unknown warm-up filtering strategy: {args.warm_up_strategy[0]}."
            f" Available strategies: {', '.join(STRATEGIES.keys())}"
        )
    dataset["warm_up"] = func(raw_dataset, *args.warm_up_strategy[1:])

    func = STRATEGIES.get(args.lifelong_strategy[0])
    if func is None:
        raise ValueError(
            f"Unknown lifelong filtering strategy: {args.lifelong_strategy[0]}."
            f" Available strategies: {', '.join(STRATEGIES.keys())}"
        )
    dataset["lifelong"] = func(raw_dataset, *args.lifelong_strategy[1:])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
