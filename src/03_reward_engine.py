import os
import re
import json
import tempfile
import subprocess
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import RewardTrainer, RewardConfig

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RM_BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
RM_OUTPUT_DIR = os.path.abspath("checkpoints/rm")
BANDIT_BIN = "/tmp/rlhf_env/bin/bandit"
SEMGREP_BIN = "/tmp/rlhf_env/bin/semgrep"


def extract_python_code(text: str) -> str:
    """Extract clean Python code from markdown fences if present."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_static_analysis_score(code: str) -> tuple[float, list[str]]:
    """
    Runs Bandit + Semgrep on generated Python code.
    Penalties: HIGH = -0.4, MEDIUM = -0.2, LOW = -0.1. Clamped to [0.0, 1.0].
    """
    clean_code = extract_python_code(code)
    if not clean_code:
        return 0.0, ["Empty code output"]

    penalty = 0.0
    findings = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(clean_code)
        tmp_path = tmp.name

    try:
        b_proc = subprocess.run(
            [BANDIT_BIN, "-f", "json", "-q", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if b_proc.stdout.strip():
            b_data = json.loads(b_proc.stdout)
            for issue in b_data.get("results", []):
                sev = issue.get("issue_severity", "LOW").upper()
                test_id = issue.get("test_id", "B000")
                msg = issue.get("issue_text", "")
                findings.append(f"[Bandit {test_id} | {sev}] {msg}")
                if sev == "HIGH":
                    penalty += 0.4
                elif sev == "MEDIUM":
                    penalty += 0.2
                else:
                    penalty += 0.1

        s_proc = subprocess.run(
            [SEMGREP_BIN, "--config", "p/python", "--json", "--quiet", tmp_path],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if s_proc.stdout.strip():
            s_data = json.loads(s_proc.stdout)
            for err in s_data.get("results", []):
                sev = err.get("extra", {}).get("severity", "WARNING").upper()
                rule_id = err.get("check_id", "semgrep-rule")
                findings.append(f"[Semgrep {rule_id} | {sev}]")
                if sev in ("ERROR", "HIGH"):
                    penalty += 0.4
                elif sev in ("WARNING", "MEDIUM"):
                    penalty += 0.2
                else:
                    penalty += 0.1
    except Exception as e:
        findings.append(f"[Analyzer Error] {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    score = max(0.0, min(1.0, 1.0 - penalty))
    return score, findings


def run_correctness_score(code: str, test_code: str) -> float:
    """
    Executes generated code + unit tests in a time-bounded subprocess.
    Returns 1.0 if syntax is valid & tests pass, 0.8 if syntax is valid without tests, 0.0 on failure.
    """
    clean_code = extract_python_code(code)
    if not clean_code:
        return 0.0

    try:
        compile(clean_code, "<generated>", "exec")
    except SyntaxError:
        return 0.0

    if not test_code.strip():
        return 0.8

    full_script = f"{clean_code}\n\n{test_code}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(full_script)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["/tmp/rlhf_env/bin/python", tmp_path],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return 1.0 if proc.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def train_reward_model():
    """Trains a pairwise Security Reward Model on (chosen=secure, rejected=insecure) code pairs."""
    with open("data/rm_pairs.json") as f:
        pairs = json.load(f)

    if len(pairs) < 4:
        print("Insufficient preference pairs to train neural RM; skipping neural RM training.")
        return

    tokenizer = AutoTokenizer.from_pretrained(RM_BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def format_pair(ex):
        chosen_text = f"Task: {ex['prompt']}\nCode:\n{ex['chosen']}"
        rejected_text = f"Task: {ex['prompt']}\nCode:\n{ex['rejected']}"
        c_tok = tokenizer(chosen_text, truncation=True, max_length=512)
        r_tok = tokenizer(rejected_text, truncation=True, max_length=512)
        return {
            "input_ids_chosen": c_tok["input_ids"],
            "attention_mask_chosen": c_tok["attention_mask"],
            "input_ids_rejected": r_tok["input_ids"],
            "attention_mask_rejected": r_tok["attention_mask"],
        }

    rm_dataset = Dataset.from_list(pairs).map(format_pair, remove_columns=list(pairs[0].keys()))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE_MODEL,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    reward_args = RewardConfig(
        output_dir=RM_OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=512,
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_args,
        tokenizer=tokenizer,
        train_dataset=rm_dataset,
        peft_config=peft_config,
    )

    print("Training Security Reward Model...")
    trainer.train()

    final_rm_path = os.path.join(RM_OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_rm_path)
    tokenizer.save_pretrained(final_rm_path)
    print(f"Reward Model adapter saved to {final_rm_path}")


class DualRewardScorer:
    """Combines Unit-Test Correctness, Bandit+Semgrep Static Analysis, and Neural RM Score."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, use_neural_rm: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.rm_model = None
        self.rm_tokenizer = None

        rm_adapter = os.path.join(RM_OUTPUT_DIR, "final_adapter")
        if use_neural_rm and os.path.exists(rm_adapter):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_rm = AutoModelForSequenceClassification.from_pretrained(
                RM_BASE_MODEL,
                num_labels=1,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.rm_model = PeftModel.from_pretrained(base_rm, rm_adapter)
            self.rm_model.eval()
            self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_adapter)
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
            self.rm_model.config.pad_token_id = self.rm_tokenizer.pad_token_id

    @torch.no_grad()
    def neural_rm_score(self, prompt: str, code: str) -> float:
        if self.rm_model is None:
            return 0.5
        text = f"Task: {prompt}\nCode:\n{code}"
        inputs = self.rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        logit = self.rm_model(**inputs).logits.squeeze().item()
        return float(torch.sigmoid(torch.tensor(logit)).item())

    def score(self, prompt: str, generated_code: str, test_code: str = "") -> dict:
        correctness = run_correctness_score(generated_code, test_code)
        static_sec, findings = run_static_analysis_score(generated_code)
        rm_sec = self.neural_rm_score(prompt, generated_code)

        security = 0.7 * static_sec + 0.3 * rm_sec if self.rm_model is not None else static_sec
        total_reward = self.alpha * correctness + self.beta * security

        return {
            "total_reward": round(total_reward, 4),
            "correctness": round(correctness, 4),
            "static_security": round(static_sec, 4),
            "neural_rm": round(rm_sec, 4),
            "security": round(security, 4),
            "findings": findings,
        }


if __name__ == "__main__":
    train_reward_model()
