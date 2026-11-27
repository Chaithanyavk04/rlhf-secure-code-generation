import os
import json
import random
from datasets import load_dataset

os.environ["HF_HOME"] = "/tmp/hf_cache"
DATA_DIR = os.path.abspath("data")
os.makedirs(DATA_DIR, exist_ok=True)


def extract_fields(row):
    """Normalize column names across CodeSecEval tasks."""
    prompt = (
        row.get("prompt")
        or row.get("problem")
        or row.get("description")
        or row.get("instruction")
        or ""
    )
    secure_code = (
        row.get("secure_code")
        or row.get("patched_code")
        or row.get("solution")
        or row.get("canonical_solution")
        or ""
    )
    insecure_code = (
        row.get("insecure_code")
        or row.get("vulnerable_code")
        or row.get("buggy_code")
        or ""
    )
    test_code = row.get("test") or row.get("unittest") or row.get("tests") or ""
    cwe = row.get("cwe") or row.get("cwe_id") or row.get("vulnerability_type") or "CWE-General"
    return {
        "prompt": str(prompt).strip(),
        "secure_code": str(secure_code).strip(),
        "insecure_code": str(insecure_code).strip(),
        "test_code": str(test_code).strip(),
        "cwe": str(cwe).strip(),
    }


def main():
    print("Downloading hao-li/CodeSecEval...")
    ds = load_dataset("hao-li/CodeSecEval")

    raw_records = []
    for split in ds.keys():
        for row in ds[split]:
            item = extract_fields(row)
            if item["prompt"] and item["secure_code"]:
                raw_records.append(item)

    print(f"Total valid tasks extracted: {len(raw_records)}")
    random.seed(42)
    random.shuffle(raw_records)

    split_idx = int(0.7 * len(raw_records))
    train_records = raw_records[:split_idx]
    eval_records = raw_records[split_idx:]

    sft_data = []
    for r in train_records:
        formatted_prompt = (
            f"<s>[INST] Write secure, production-grade Python code for the following task. "
            f"Avoid all security vulnerabilities ({r['cwe']}).\n\nTask:\n{r['prompt']} [/INST]\n"
            f"```python\n{r['secure_code']}\n``` </s>"
        )
        sft_data.append({"text": formatted_prompt, **r})

    rm_pairs = []
    for r in train_records:
        if r["insecure_code"]:
            rm_pairs.append({
                "prompt": r["prompt"],
                "chosen": f"```python\n{r['secure_code']}\n```",
                "rejected": f"```python\n{r['insecure_code']}\n```",
                "cwe": r["cwe"],
            })

    with open(os.path.join(DATA_DIR, "sft_train.json"), "w") as f:
        json.dump(sft_data, f, indent=2)

    with open(os.path.join(DATA_DIR, "rm_pairs.json"), "w") as f:
        json.dump(rm_pairs, f, indent=2)

    with open(os.path.join(DATA_DIR, "eval_test.json"), "w") as f:
        json.dump(eval_records, f, indent=2)

    print(f"Saved SFT train samples : {len(sft_data)}")
    print(f"Saved RM preference pairs : {len(rm_pairs)}")
    print(f"Saved Held-out Eval tasks : {len(eval_records)}")


if __name__ == "__main__":
    main()
