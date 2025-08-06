#!/bin/bash
set -euo pipefail

# =========================================
# SpectrumX H200 クラスタ - 完全自動セットアップ
# =========================================

echo "======================================"
echo " SpectrumX H200 Cluster Setup"
echo " Starting at: $(date)"
echo "======================================"
echo ""

# カラー出力の設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 成功/失敗メッセージ
success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }
warning() { echo -e "${YELLOW}!${NC} $1"; }

# Step 1: システム依存関係のインストール
echo "Step 1/5: Installing system dependencies..."
if ./setup/install_dependencies.sh; then
    success "System dependencies installed successfully"
else
    error "Failed to install system dependencies"
fi

echo ""

# Step 2: Python環境のセットアップ
echo "Step 2/5: Installing Python packages..."
if ./setup/install_python_packages.sh; then
    success "Python packages installed successfully"
else
    error "Failed to install Python packages"
fi

echo ""

# Step 3: NCCL設定
echo "Step 3/5: Configuring NCCL..."
if ./setup/configure_nccl.sh; then
    success "NCCL configured successfully"
else
    error "Failed to configure NCCL"
fi

echo ""

# Step 4: クラスタ検証
echo "Step 4/5: Verifying cluster..."
if ./setup/verify_cluster.sh; then
    success "Cluster verification passed"
else
    warning "Cluster verification showed some issues - please review"
fi

echo ""

# Step 5: データセット準備の確認
echo "Step 5/5: Checking dataset preparation..."
if [ -f "datasets/prepare_dataset.py" ]; then
    success "Dataset preparation script found"
    echo "  You can prepare datasets using: python3 datasets/prepare_dataset.py"
else
    warning "Dataset preparation script not found"
fi

echo ""
echo "======================================"
echo " Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Prepare dataset: python3 datasets/prepare_dataset.py"
echo "2. Run benchmark: ./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2"
echo "3. Check results: cat results/*/training.log"
echo ""
echo "For detailed instructions, see README.md"