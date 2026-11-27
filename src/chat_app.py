import os
import re
import torch
import gradio as gr
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

spec = importlib.util.spec_from_file_location("reward_engine", "src/03_reward_engine.py")
reward_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward_engine)

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
PPO_ADAPTER = os.path.abspath("checkpoints/ppo/final_adapter")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, PPO_ADAPTER)
model.eval()
scorer = reward_engine.DualRewardScorer(use_neural_rm=True)

SYSTEM_PROMPT = (
    "You are SecureCode AI, a helpful Python coding assistant aligned with RLHF. "
    "Write clean, secure, production-ready Python code and explain key points clearly."
)


def format_history(message: str, history: list) -> str:
    turns = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            text = re.sub(r"\n\n> \*\*Security Check:.*$", "", str(item.get("content", "")), flags=re.S)
            if role == "user":
                turns.append([text, None])
            elif role == "assistant" and turns:
                turns[-1][1] = text
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            clean_bot = re.sub(r"\n\n> \*\*Security Check:.*$", "", str(item[1] or ""), flags=re.S)
            turns.append([str(item[0]), clean_bot])

    if not turns:
        return f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{message.strip()} [/INST]\n"

    parts = [f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{turns[0][0].strip()} [/INST] {turns[0][1].strip()} </s>"]
    for u_msg, b_msg in turns[1:]:
        if b_msg is not None:
            parts.append(f"<s>[INST] {u_msg.strip()} [/INST] {b_msg.strip()} </s>")
    parts.append(f"<s>[INST] {message.strip()} [/INST]\n")
    return "".join(parts)


def respond(message: str, history: list):
    if not message or not message.strip():
        return ""

    prompt = format_history(message, history)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if "def " in reply or "import " in reply or "```" in reply:
        audit = scorer.score(message, reply)
        findings = audit.get("findings", [])
        status = "Passed (0 vulnerabilities)" if not findings else f"{len(findings)} warning(s)"
        footer = f"\n\n> **Security Check:** {status} · Static `{audit['static_security']:.2f}` · RM `{audit['neural_rm']:.2f}`"
        return reply + footer

    return reply


CLEAN_CSS = """
body, .gradio-container {
    background: #0f1117 !important;
    color: #ececf1 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container {
    max-width: 840px !important;
    margin: 0 auto !important;
    padding-top: 24px !important;
}
.header {
    text-align: center;
    margin-bottom: 12px;
}
.header h1 {
    font-size: 22px;
    font-weight: 600;
    color: #f3f4f6;
    margin: 0;
}
.header p {
    font-size: 13px;
    color: #9ca3af;
    margin: 4px 0 0 0;
}
footer { display: none !important; }
"""

with gr.Blocks(title="SecureCode AI") as demo:
    gr.HTML("""
    <div class="header">
      <h1>SecureCode AI</h1>
      <p>PPO-Aligned CodeLlama-7B · Secure Python Assistant</p>
    </div>
    """)

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=540),
        textbox=gr.Textbox(
            placeholder="Ask me to write or review Python code...",
            container=False,
            scale=7,
        ),
        examples=[
            "Write a safe Python function to load a YAML file.",
            "Write a SQLite login query protected against SQL injection.",
            "How do I run a ping command safely in Python?",
            "Write a Flask download route safe from path traversal.",
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="zinc"),
        css=CLEAN_CSS,
    )
