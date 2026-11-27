#!/usr/bin/env bash
set -e

echo "Creating isolated Python environment in /tmp/rlhf_env..."
rm -rf /tmp/rlhf_env
/home/nmit/anaconda3/bin/python3 -m venv /tmp/rlhf_env

/tmp/rlhf_env/bin/pip install --upgrade pip
/tmp/rlhf_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
/tmp/rlhf_env/bin/pip install -r requirements.txt

echo "Verifying GPU and packages..."
/tmp/rlhf_env/bin/python -c "import torch, transformers, peft, trl; print('GPU:', torch.cuda.get_device_name(0), '| TRL:', trl.__version__)"
/tmp/rlhf_env/bin/bandit --version | head -n 1
/tmp/rlhf_env/bin/semgrep --version
echo "Environment setup complete!"
