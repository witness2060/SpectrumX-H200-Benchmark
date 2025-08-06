#!/bin/bash
# SpectrumX H200 Benchmark - 環境変数エクスポートスクリプト
# このスクリプトをsourceして環境変数を設定: source setup/export_env_vars.sh

echo "=== SpectrumX H200 最適化環境変数を設定中 ==="

# ========== NCCL環境変数（SpectrumX RoCEv2最適化） ==========
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_BUFFSIZE=8388608
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple,LL128
export NCCL_COLLNET_ENABLE=0
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TOPO_FILE=/etc/nccl-topo.xml

# ========== GPU/CUDA環境変数 ==========
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_NVLS_ENABLE=1

# ========== InfiniBand/RoCEv2環境変数 ==========
export NCCL_IB_ECN_ENABLE=1
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_IB_SL=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GDR_LEVEL=5
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=22

# ========== DeepSpeed環境変数 ==========
export DEEPSPEED_LAUNCHER=pdsh
export DEEPSPEED_SSH_PORT=44222
export DS_COMM_BACKEND=nccl
export DS_ACCELERATOR=cuda
export DS_ZERO_STAGE=3
export DS_ZERO_REDUCE_SCATTER=true
export DS_ZERO_ALLGATHER_PARTITIONS=true
export DS_ZERO_ROUND_ROBIN_GRADIENTS=true
export DS_ZERO_OFFLOAD_OPTIMIZER_DEVICE=none
export DS_ZERO_OFFLOAD_PARAM_DEVICE=none
export DS_ACTIVATION_CHECKPOINTING_PARTITION=true

# ========== PyTorch分散環境変数 ==========
export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_FUSION=1
export TORCH_CUDNN_BENCHMARK=1

# ========== システム環境変数 ==========
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NCCL_IB_DISABLE_HUGEPAGES=0

# ========== 監視・デバッグ環境変数 ==========
export NCCL_DEBUG_FILE=/tmp/nccl-debug-%h-%p.log
export TORCH_PROFILER_ENABLED=0
export NCCL_LOG_LEVEL=WARN
export GLOO_LOG_LEVEL=WARNING
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ========== アプリケーション固有環境変数 ==========
# Hugging Face
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-~/.cache/huggingface/transformers}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Weights & Biases
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}

# ========== カスタム設定の読み込み ==========
if [ -f ".env" ]; then
    echo "カスタム設定を.envから読み込み中..."
    set -a
    source .env
    set +a
fi

echo "=== 環境変数の設定が完了しました ==="
echo ""
echo "主要な設定:"
echo "  - NCCL_IB_HCA: $NCCL_IB_HCA"
echo "  - NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "  - NCCL_BUFFSIZE: $NCCL_BUFFSIZE"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - DEEPSPEED_SSH_PORT: $DEEPSPEED_SSH_PORT"
echo ""
echo "詳細は docs/SPECTRUMX_ENV_VARIABLES.md を参照してください"