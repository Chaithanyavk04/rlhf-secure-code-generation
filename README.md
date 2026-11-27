# RLHF-Based LLM Fine-Tuning for Secure and Correct Code Generation

**Nitte Meenakshi Institute of Technology (NMIT) — Department of AI & Data Science (22ADP03)**

## Architecture Overview
1. **Base Model:** `codellama/CodeLlama-7b-Instruct-hf` (4-bit NF4 QLoRA)
2. **Stage 1 (`src/01_prepare_data.py`):** Loads `hao-li/CodeSecEval` and constructs SFT, Reward Model preference pairs, and RL/Eval splits.
3. **Stage 2 (`src/02_train_sft.py`):** Supervised Fine-Tuning (SFT) on secure Python implementations.
4. **Stage 3 (`src/03_reward_engine.py`):** Trains a pairwise Security/Preference Reward Model (`AutoModelForSequenceClassification`) and integrates sandboxed **Unit Tests (Correctness)** + **Bandit & Semgrep static analysis (Security)** into $R = \alpha \cdot \text{Correctness} + \beta \cdot \text{Security}$.
5. **Stage 4 (`src/04_train_ppo.py`):** Full Proximal Policy Optimization (`trl.PPOTrainer`) with `AutoModelForCausalLMWithValueHead`.
6. **Stage 5 (`src/05_evaluate.py`):** Benchmarks Base vs. SFT vs. SFT+PPO across pass rate, vulnerability count, and security score.

## Execution Order
```bash
./setup_env.sh
/tmp/rlhf_env/bin/python src/01_prepare_data.py
/tmp/rlhf_env/bin/python src/02_train_sft.py
/tmp/rlhf_env/bin/python src/03_reward_engine.py
/tmp/rlhf_env/bin/python src/04_train_ppo.py
/tmp/rlhf_env/bin/python src/05_evaluate.py
Your terminal is waiting at the `>` prompt because `README.md` contains an inner triple-backtick code block (`````) that didn't close the outer `cat << 'EOF' > README.md` heredoc—you just need to send `EOF` and press **Enter**!

Run these three lines in that terminal right now to finish closing `README.md` and verify the files:

```bash

```

