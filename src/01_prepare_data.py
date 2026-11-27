import os
import json
import random
from datasets import load_dataset

os.environ["HF_HOME"] = "/tmp/hf_cache"
DATA_DIR = os.path.abspath("data")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    train_pool = []

    # 1. Load Python pairs from CyberNative/Code_Vulnerability_Security_DPO
    print("Loading CyberNative/Code_Vulnerability_Security_DPO...")
    dpo_ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
    for row in dpo_ds:
        if "python" not in str(row.get("lang", "")).lower():
            continue
        prompt = str(row.get("question") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        cwe = str(row.get("vulnerability") or "CWE-General").strip()
        if prompt and chosen and rejected:
            train_pool.append({
                "prompt": prompt,
                "secure_code": chosen,
                "insecure_code": rejected,
                "test_code": "",
                "cwe": cwe,
            })
    print(f"Loaded {len(train_pool)} Python pairs from CyberNative DPO.")

    # 2. Load JasonWang1/CodeSecEval (includes executable functional & security unit tests)
    print("Loading JasonWang1/CodeSecEval...")
    cse_ds = load_dataset("JasonWang1/CodeSecEval", "CodeSecEval")
    cse_count = 0
    for split in cse_ds.keys():
        for row in cse_ds[split]:
            prompt = str(row.get("Problem") or "").strip()
            secure = str(row.get("Secure Code") or "").strip()
            insecure = str(row.get("Insecure Code") or "").strip()
            test_code = str(row.get("Test") or row.get("Test-FP") or "").strip()
            entry = str(row.get("Entry_Point") or "").strip()
            if test_code and entry and "def check(" in test_code:
                test_code = f"{test_code}\n\ncheck({entry})\n"
            cwe = str(row.get("ID") or "CWE-General").strip()
            if prompt and secure:
                train_pool.append({
                    "prompt": prompt,
                    "secure_code": secure,
                    "insecure_code": insecure,
                    "test_code": test_code,
                    "cwe": cwe,
                })
                cse_count += 1
    print(f"Loaded {cse_count} unit-tested Python tasks from JasonWang1/CodeSecEval.")

    # 3. Load s2e-lab/SecurityEval (121 CWE-categorized Python prompts) as held-out evaluation benchmark
    print("Loading s2e-lab/SecurityEval...")
    se_ds = load_dataset("s2e-lab/SecurityEval")
    eval_records = []
    for split in se_ds.keys():
        for row in se_ds[split]:
            prompt = str(row.get("Prompt") or "").strip()
            insecure = str(row.get("Insecure_code") or "").strip()
            cwe = str(row.get("ID") or "CWE-General").strip()
            if prompt:
                eval_records.append({
                    "prompt": prompt,
                    "secure_code": "",
                    "insecure_code": insecure,
                    "test_code": "",
                    "cwe": cwe,
                })
    print(f"Loaded {len(eval_records)} held-out evaluation tasks from s2e-lab/SecurityEval.")

    random.seed(42)
    random.shuffle(train_pool)

    # Format SFT dataset
    sft_data = []
    for r in train_pool:
        code_block = r["secure_code"]
        if not code_block.startswith("```"):
            code_block = f"```python\n{code_block}\n```"
        formatted_prompt = (
            f"<s>[INST] Write secure, production-grade Python code for the following task. "
            f"Avoid all security vulnerabilities ({r['cwe']}).\n\nTask:\n{r['prompt']} [/INST]\n"
            f"{code_block} </s>"
        )
        sft_data.append({"text": formatted_prompt, **r})

    # Format Reward Model preference pairs
    rm_pairs = [
        {
            "prompt": r["prompt"],
            "chosen": r["secure_code"],
            "rejected": r["insecure_code"],
            "cwe": r["cwe"],
        }
        for r in train_pool
        if r["secure_code"] and r["insecure_code"]
    ]

    with open(os.path.join(DATA_DIR, "sft_train.json"), "w") as f:
        json.dump(sft_data, f, indent=2)
    with open(os.path.join(DATA_DIR, "rm_pairs.json"), "w") as f:
        json.dump(rm_pairs, f, indent=2)
    with open(os.path.join(DATA_DIR, "eval_test.json"), "w") as f:
        json.dump(eval_records, f, indent=2)

    print("\n=== DATASET PREPARATION COMPLETE ===")
    print(f"SFT Train Samples   : {len(sft_data)}")
    print(f"RM Preference Pairs : {len(rm_pairs)}")
    print(f"Held-out Eval Tasks : {len(eval_records)}")


if __name__ == "__main__":
    main()
