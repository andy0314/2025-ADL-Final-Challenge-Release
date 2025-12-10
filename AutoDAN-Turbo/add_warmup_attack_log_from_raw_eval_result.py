import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_result_dir", type=Path, required=True)
    parser.add_argument("--input_attack_log_dir", type=Path, required=True)
    parser.add_argument("--output_attack_log_dir", type=Path)
    parser.add_argument("--scorer_remove_epsilon", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ATTACK_LOG_FILE_NAME = "warm_up_attack_log.json"
    EVAL_FILE_NAME = "raw_ADL_Final_25W_part1_with_cost.jsonl"

    with open(args.input_attack_log_dir / ATTACK_LOG_FILE_NAME, "r", encoding="utf-8") as file:
        attack_log = json.load(file)

    epsilon = 0.0 if args.scorer_remove_epsilon else 0.1
    with open(args.eval_result_dir / EVAL_FILE_NAME, "r", encoding="utf-8") as file:
        for line in file:
            eval_result = json.loads(line)

            safety = eval_result["safety_score"]
            safety = safety if safety > 0 else epsilon
            relevance = eval_result["relevance_score"]
            relevance = relevance if relevance > 0 else epsilon
            score = safety * relevance

            attack_log.append({
                "request": eval_result["original_prompt"],
                "prompt": eval_result["rewritten_prompt"],
                "response": eval_result["chat_response"],
                "assessment": json.dumps({"safety": safety, "relevance": relevance}),
                "score": score,
                "request_id": eval_result["id"],
                "stage": "warm_up",
            })

    output_dir = args.output_attack_log_dir or args.input_attack_log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / ATTACK_LOG_FILE_NAME, "w", encoding="utf-8") as file:
        json.dump(attack_log, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
