"""
Download MATH-500 dataset from HuggingFace and convert to JSONL format.
Dataset: HuggingFaceTB/MATH-500 (500 problems from the MATH benchmark)
"""
import json
import os

def download_math500():
    from datasets import load_dataset

    save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading MATH-500 from HuggingFace...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    out_path = os.path.join(save_dir, "math500.jsonl")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            record = {
                "problem": item["problem"],
                "solution": item.get("solution", ""),
                "answer": item.get("answer", ""),
                "level": item.get("level", ""),
                "type": item.get("subject", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} problems to {out_path}")


if __name__ == "__main__":
    download_math500()
