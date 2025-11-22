"""
Export the ground-truth answers for every MRAG-Bench test question.

Usage:
    conda activate mrag
    python make_answers.py [--output ground_truth_answers.jsonl]

Requires the `datasets` package and network access to fetch the MRAG-Bench
dataset from Hugging Face.
"""

import argparse
import json

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MRAG-Bench ground-truth answers")
    parser.add_argument(
        "--output",
        "-o",
        default="ground_truth_answers.jsonl",
        help="Path to write JSONL with answers (default: ground_truth_answers.jsonl)",
    )
    args = parser.parse_args()

    mrag = load_dataset("uclanlp/MRAG-Bench", split="test")

    records = []
    for ex in mrag:
        opt = ex["answer_choice"]
        records.append(
            {
                "id": ex["id"],
                "scenario": ex["scenario"],
                "question": ex["question"],
                "A": ex["A"],
                "B": ex["B"],
                "C": ex["C"],
                "D": ex["D"],
                "correct_option": opt,
                "correct_answer_text": ex[opt],
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} entries to {args.output}")


if __name__ == "__main__":
    main()
