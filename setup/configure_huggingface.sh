#!/bin/bash
set -euo pipefail

# =========================================
# Hugging Face 認証設定
# =========================================

echo "=== Configuring Hugging Face Authentication ==="
echo ""
echo "This script will help you set up Hugging Face authentication for accessing"
echo "private models like Llama-2."
echo ""

# HF_TOKEN環境変数の確認
if [ -n "${HF_TOKEN:-}" ]; then
    echo "✓ HF_TOKEN environment variable is already set"
    CURRENT_TOKEN=$HF_TOKEN
else
    echo "HF_TOKEN environment variable is not set."
    echo ""
    echo "To get your token:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'read' permission"
    echo "3. Copy the token"
    echo ""
    read -p "Enter your Hugging Face token (or press Enter to skip): " HF_TOKEN
    CURRENT_TOKEN=$HF_TOKEN
fi

if [ -n "$CURRENT_TOKEN" ]; then
    # トークンを環境変数ファイルに保存
    echo "export HF_TOKEN='$CURRENT_TOKEN'" > ~/.hf_token
    echo "✓ Token saved to ~/.hf_token"
    
    # .bashrcに追加
    if ! grep -q "source ~/.hf_token" ~/.bashrc 2>/dev/null; then
        echo "source ~/.hf_token" >> ~/.bashrc
        echo "✓ Added to ~/.bashrc"
    fi
    
    # 全ノードに配布（PDSHを使用）
    if [ -f /root/.pdshrc ]; then
        source /root/.pdshrc
        echo ""
        echo "Distributing token to all nodes..."
        for node in $(echo $NODES | tr ',' ' '); do
            scp -q ~/.hf_token $node:~/ 2>/dev/null && echo "✓ $node" || echo "✗ $node (failed)"
            ssh $node "grep -q 'source ~/.hf_token' ~/.bashrc || echo 'source ~/.hf_token' >> ~/.bashrc" 2>/dev/null
        done
    fi
    
    # huggingface-cliでログイン
    echo ""
    echo "Configuring huggingface-cli..."
    pip install -q huggingface_hub
    python3 -c "from huggingface_hub import login; login(token='$CURRENT_TOKEN', add_to_git_credential=False)" 2>/dev/null && \
        echo "✓ Successfully logged in to Hugging Face" || \
        echo "⚠ Could not verify login (this is normal if offline)"
    
else
    echo ""
    echo "⚠ Warning: No token provided."
    echo "You will only be able to access public models."
    echo "Private models like meta-llama/Llama-2-* will not be accessible."
fi

echo ""
echo "=== Testing Model Access ==="

# テスト用のPythonスクリプト
cat > /tmp/test_hf_access.py << 'EOF'
import os
from transformers import AutoConfig
import sys

token = os.environ.get("HF_TOKEN", None)

# テストするモデルリスト
models_to_test = [
    ("gpt2", "Public"),
    ("meta-llama/Llama-2-7b-hf", "Private (Llama-2)"),
    ("mistralai/Mistral-7B-v0.1", "Public"),
]

print("\nModel accessibility check:")
print("-" * 50)

for model_id, model_type in models_to_test:
    try:
        config = AutoConfig.from_pretrained(model_id, token=token)
        print(f"✓ {model_id:<30} [{model_type}] - Accessible")
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            print(f"✗ {model_id:<30} [{model_type}] - Requires authentication")
        elif "404" in str(e):
            print(f"✗ {model_id:<30} [{model_type}] - Not found")
        else:
            print(f"⚠ {model_id:<30} [{model_type}] - Error: {str(e)[:30]}...")

print("-" * 50)
EOF

source ~/.hf_token 2>/dev/null || true
python3 /tmp/test_hf_access.py

echo ""
echo "=== Configuration Complete ==="
echo ""
echo "Next steps:"
echo "1. If you need access to Llama-2 models:"
echo "   - Go to https://huggingface.co/meta-llama"
echo "   - Request access to the model you want to use"
echo "   - Wait for approval (usually within 24 hours)"
echo ""
echo "2. To use your token in scripts:"
echo "   source ~/.hf_token"
echo ""
echo "3. To change your token later:"
echo "   ./setup/configure_huggingface.sh"