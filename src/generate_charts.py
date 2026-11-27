import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# 1. Benchmark Comparison Chart (Base vs SFT vs PPO)
with open("results/benchmark_summary.json") as f:
    summary = json.load(f)

models = ["Base CodeLlama-7B", "SFT CodeLlama-7B", "PPO RLHF CodeLlama-7B"]
correctness = [r["avg_correctness"] * 100 for r in summary]
security = [r["avg_security_score"] * 100 for r in summary]
zero_vuln = [r["zero_vulnerability_rate"] * 100 for r in summary]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
b1 = ax.bar(x - width, correctness, width, label="Functional/Syntax Correctness (%)", color="#4C72B0")
b2 = ax.bar(x, security, width, label="Avg Static Security Score (%)", color="#55A868")
b3 = ax.bar(x + width, zero_vuln, width, label="Zero-Vulnerability Rate (%)", color="#C44E52")

ax.set_ylabel("Score (%)")
ax.set_title("SecurityEval Held-Out Benchmark: Base vs. SFT vs. PPO RLHF", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(loc="upper right", frameon=True)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig("results/figures/benchmark_comparison.png")
plt.close()

# 2. PPO Training Curves
with open("results/ppo_training_log.json") as f:
    logs = json.load(f)

steps = [row["step"] for row in logs]
rewards = [row["mean_reward"] for row in logs]
sec_scores = [row["mean_security"] for row in logs]
corr_scores = [row["mean_correctness"] for row in logs]

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.plot(steps, rewards, marker="o", linewidth=2, label="Total Dual Reward", color="#2B5C8F")
ax.plot(steps, sec_scores, marker="s", linestyle="--", linewidth=1.8, label="Security Score", color="#2E8B57")
ax.plot(steps, corr_scores, marker="^", linestyle=":", linewidth=1.8, label="Correctness Score", color="#D95F02")

ax.set_xlabel("PPO Training Step (Batch Size = 4)")
ax.set_ylabel("Mean Batch Score")
ax.set_title("PPO Reinforcement Learning Trajectory Across 30 Rollout Steps", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("results/figures/ppo_training_curves.png")
plt.close()
print("Saved charts to results/figures/")
