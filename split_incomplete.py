"""
Extract only the 'Incomplete' scenario entries into a separate JSONL.

Usage:
    conda activate mrag
    python split_incomplete.py \
        --source ground_truth_answers.jsonl \
        --output ground_truth_incomplete.jsonl
"""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Split out Incomplete scenario answers")
    parser.add_argument(
        "--source",
        "-s",
        default="ground_truth_answers.jsonl",
        help="Path to the full answers JSONL",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ground_truth_incomplete.jsonl",
        help="Path to write the filtered JSONL",
    )
    args = parser.parse_args()

    kept = 0
    with open(args.source, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            item = json.loads(line)
            if item.get("scenario") == "Incomplete":
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Wrote {kept} entries to {args.output}")


if __name__ == "__main__":
    main()
