import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import importlib.util
spec = importlib.util.spec_from_file_location("reward_engine", "src/03_reward_engine.py")
reward_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_engine)

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
SFT_ADAPTER_PATH = os.path.abspath("checkpoints/sft/final_adapter")
PPO_OUTPUT_DIR = os.path.abspath("checkpoints/ppo")


def main():
    with open("data/sft_train.json") as f:
        train_tasks = json.load(f)[:120]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading SFT policy model with ValueHead for PPO...")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        SFT_ADAPTER_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )

    ppo_config = PPOConfig(
        model_name=MODEL_ID,
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target_kl=6.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        seed=42,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,
        tokenizer=tokenizer,
    )

    scorer = reward_engine.DualRewardScorer(alpha=0.5, beta=0.5, use_neural_rm=True)

    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print("Starting Full PPO Reinforcement Learning Loop...")
    history = []
    batch_size = ppo_config.batch_size

    for step in tqdm(range(0, len(train_tasks), batch_size), desc="PPO Steps"):
        batch = train_tasks[step : step + batch_size]
        if len(batch) < batch_size:
            break

        query_tensors = []
        raw_prompts = []
        test_codes = []

        for item in batch:
            inst_prompt = (
                f"<s>[INST] Write secure, production-grade Python code for the following task. "
                f"Avoid all security vulnerabilities ({item['cwe']}).\n\nTask:\n{item['prompt']} [/INST]\n"
            )
            tokens = tokenizer.encode(inst_prompt, return_tensors="pt", truncation=True, max_length=384).squeeze(0)
            query_tensors.append(tokens.to(ppo_trainer.accelerator.device))
            raw_prompts.append(item["prompt"])
            test_codes.append(item.get("test_code", ""))

        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        rewards = []
        batch_metrics = []
        for p, resp, t_code in zip(raw_prompts, responses, test_codes):
            res = scorer.score(p, resp, t_code)
            scaled_r = (res["total_reward"] * 3.0) - 1.0
            rewards.append(torch.tensor(scaled_r, dtype=torch.float32))
            batch_metrics.append(res)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        mean_r = sum(m["total_reward"] for m in batch_metrics) / len(batch_metrics)
        mean_sec = sum(m["security"] for m in batch_metrics) / len(batch_metrics)
        mean_cor = sum(m["correctness"] for m in batch_metrics) / len(batch_metrics)

        log_entry = {
            "step": step // batch_size,
            "mean_reward": round(mean_r, 4),
            "mean_security": round(mean_sec, 4),
            "mean_correctness": round(mean_cor, 4),
            "kl": round(float(stats.get("objective/kl", 0.0)), 4),
        }
        history.append(log_entry)
        print(f"Step {log_entry['step']} | Reward: {mean_r:.3f} | Security: {mean_sec:.3f} | Correctness: {mean_cor:.3f}")

    final_ppo_path = os.path.join(PPO_OUTPUT_DIR, "final_adapter")
    ppo_trainer.model.pretrained_model.save_pretrained(final_ppo_path)
    tokenizer.save_pretrained(final_ppo_path)

    with open("results/ppo_training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"PPO training complete! Final adapter saved to {final_ppo_path}")


if __name__ == "__main__":
    main()
