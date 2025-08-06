#!/bin/bash
set -euo pipefail

# =========================================
# システム依存関係のインストール
# =========================================

# クラスタ設定の読み込み
source "$(dirname "$0")/cluster_config.sh"

echo "=== Installing system dependencies on all nodes ==="

# ノードの自動検出
detect_available_nodes || exit 1
show_cluster_summary

echo "Target nodes: $PDSH_NODES"

# CUDA 12.5とDriver 560.xxの確認
echo "Checking NVIDIA drivers..."
pdsh -w "$PDSH_NODES" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1" 2>/dev/null || {
    echo "Warning: Could not verify GPU drivers on all nodes"
}

# 基本的なシステムパッケージのインストール
echo "Installing system packages..."
pdsh -w "$PDSH_NODES" << 'REMOTE_EOF'
# 必要なシステムパッケージのインストール
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    iftop \
    sysstat \
    numactl \
    infiniband-diags \
    rdma-core \
    perftest \
    python3-pip \
    python3-dev

# Miniconda環境の作成（存在しない場合）
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # .bashrcに追加
    echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# H200ベンチマーク用のconda環境作成
source $HOME/miniconda3/bin/activate
if ! conda env list | grep -q "h200-bench"; then
    conda create -n h200-bench python=3.10 -y
fi
conda activate h200-bench

# 基本的なPythonパッケージ
pip install --upgrade pip setuptools wheel
REMOTE_EOF

echo ""
echo "=== Installing NCCL ==="

# NCCL 2.26.2のインストール（H200対応バージョン、Sharp対応）
pdsh -w "$PDSH_NODES" << 'REMOTE_EOF'
# NCCLのインストール確認
if ! ldconfig -p | grep -q libnccl.so; then
    echo "Installing NCCL..."
    cd /tmp
    wget -q https://github.com/NVIDIA/nccl/archive/v2.26.2.tar.gz
    tar xzf v2.26.2.tar.gz
    cd nccl-2.26.2
    make -j src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
    sudo make install
    sudo ldconfig
    cd /
    rm -rf /tmp/nccl*
fi
REMOTE_EOF

echo ""
echo "=== Installing NCCL Tests ==="

# NCCL Testsのインストール
pdsh -w "$PDSH_NODES" << 'REMOTE_EOF'
if [ ! -d "/opt/nccl-tests" ]; then
    echo "Installing NCCL tests..."
    cd /opt
    sudo git clone https://github.com/NVIDIA/nccl-tests.git
    cd nccl-tests
    sudo make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
fi
REMOTE_EOF

echo ""
echo "=== Verifying installations ==="

# インストール確認
pdsh -w "$PDSH_NODES" "which python3 && python3 --version" 2>/dev/null | sort | uniq
pdsh -w "$PDSH_NODES" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1" 2>/dev/null | sort | uniq

echo ""
echo "=== System dependencies installation completed! ==="
echo "Next: Run ./setup/install_python_packages.sh for Python packages"