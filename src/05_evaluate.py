import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import importlib.util
spec = importlib.util.spec_from_file_location("reward_engine", "src/03_reward_engine.py")
reward_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_engine)

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
SFT_ADAPTER = os.path.abspath("checkpoints/sft/final_adapter")
PPO_ADAPTER = os.path.abspath("checkpoints/ppo/final_adapter")


def evaluate_model(model, tokenizer, eval_tasks, scorer, label: str):
    model.eval()
    results = []

    for item in tqdm(eval_tasks, desc=f"Evaluating {label}"):
        prompt = (
            f"<s>[INST] Write secure, production-grade Python code for the following task. "
            f"Avoid all security vulnerabilities ({item['cwe']}).\n\nTask:\n{item['prompt']} [/INST]\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        gen_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        metrics = scorer.score(item["prompt"], gen_code, item.get("test_code", ""))
        results.append({
            "cwe": item["cwe"],
            "prompt": item["prompt"][:120],
            "generated_code": gen_code,
            **metrics,
        })

    n = len(results)
    summary = {
        "model": label,
        "num_tasks": n,
        "avg_total_reward": round(sum(r["total_reward"] for r in results) / n, 4),
        "avg_correctness": round(sum(r["correctness"] for r in results) / n, 4),
        "avg_security_score": round(sum(r["security"] for r in results) / n, 4),
        "zero_vulnerability_rate": round(sum(1 for r in results if r["static_security"] == 1.0) / n, 4),
    }
    return summary, results


def main():
    with open("data/eval_test.json") as f:
        eval_tasks = json.load(f)[:30]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    scorer = reward_engine.DualRewardScorer(alpha=0.5, beta=0.5, use_neural_rm=False)

    summaries = []
    detailed = {}

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    s_base, r_base = evaluate_model(base_model, tokenizer, eval_tasks, scorer, "Base_CodeLlama_7B")
    summaries.append(s_base)
    detailed["Base_CodeLlama_7B"] = r_base

    if os.path.exists(SFT_ADAPTER):
        sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
        s_sft, r_sft = evaluate_model(sft_model, tokenizer, eval_tasks, scorer, "SFT_CodeLlama_7B")
        summaries.append(s_sft)
        detailed["SFT_CodeLlama_7B"] = r_sft
        base_model = sft_model.unload()

    if os.path.exists(PPO_ADAPTER):
        ppo_model = PeftModel.from_pretrained(base_model, PPO_ADAPTER)
        s_ppo, r_ppo = evaluate_model(ppo_model, tokenizer, eval_tasks, scorer, "PPO_RLHF_CodeLlama_7B")
        summaries.append(s_ppo)
        detailed["PPO_RLHF_CodeLlama_7B"] = r_ppo

    with open("results/benchmark_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open("results/benchmark_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2)

    print("\n=== FINAL BENCHMARK COMPARISON ===")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
