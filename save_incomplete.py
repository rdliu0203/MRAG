import json
from pathlib import Path
from datasets import load_dataset

# IDs you want to visualize (here: all incorrect Incomplete cases)
with open("gt_rag_incomplete_incorrect.jsonl", "r", encoding="utf-8") as f:
    target_ids = {json.loads(line)["id"] for line in f}

out_dir = Path("viz_incomplete")
out_dir.mkdir(exist_ok=True, parents=True)

ds = load_dataset("uclanlp/MRAG-Bench", split="test")

for ex in ds:
    if ex["id"] not in target_ids:
        continue

    qid = ex["id"]
    # Input image
    ex["image"].save(out_dir / f"{qid}_input.png")

    # Ground-truth RAG examples
    for i, img in enumerate(ex["gt_images"]):
        img.save(out_dir / f"{qid}_gt_{i}.png")

    # Retrieved examples (if you want them)
    if "retrieved_images" in ex and ex["retrieved_images"] is not None:
        for i, img in enumerate(ex["retrieved_images"]):
            img.save(out_dir / f"{qid}_retrieved_{i}.png")

print(f"Saved images for {len(target_ids)} questions to {out_dir}/")