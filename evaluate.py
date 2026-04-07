from src import evaluate_atk_results, evaluate_atk_results_rougeL
from pathlib import Path
import argparse


def parse_setting_from_path(save_path: str):
    p = Path(save_path)

    dataset = p.parts[1]
    generator = p.parts[2]

    filename = p.stem

    parts = filename.split("_")

    attack = parts[0].lower()

    flag_map = {
        "RW": False,
        "RR": False,
        "EX": False,
    }

    for part in parts:
        for key in flag_map:
            if part.startswith(f"{key}-"):
                flag_map[key] = part.endswith("1")

    setting_now = [
        attack,
        None,
        generator,
        flag_map["RW"],
        flag_map["RR"],
        flag_map["EX"],
        dataset,
    ]

    return setting_now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate attack results from a JSONL result file.")
    parser.add_argument("save_path", type=str, help="Path to the result .jsonl file, e.g. results/fiqa/gpt-4o/.../TGTB_....jsonl")
    parser.add_argument("--num_records", type=int, default=200, help="Number of records to evaluate (default: 200)")
    parser.add_argument("--all_thresholds", action="store_true", help="Evaluate with all ROUGE-L thresholds (0.3, 0.5, 0.7, 0.9)")
    args = parser.parse_args()

    setting_now = parse_setting_from_path(args.save_path)

    unique_chunks, ly_05, as_num05, rq = evaluate_atk_results(args.save_path, num_records=args.num_records)

    print("Final Results:")
    print("setting_now =", setting_now)
    print(f"RougeL@0.5: {ly_05:.4f} ({as_num05}), unique_chunks: {unique_chunks}, queries: {rq}")

    if args.all_thresholds:
        ly_03, as_num03, ly_07, as_num07, ly_09, as_num09 = evaluate_atk_results_rougeL(args.save_path, num_records=args.num_records)
        print(f"RougeL@0.3: {ly_03:.4f} ({as_num03}), @0.7: {ly_07:.4f} ({as_num07}), @0.9: {ly_09:.4f} ({as_num09})")
