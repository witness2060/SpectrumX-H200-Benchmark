#!/bin/bash
# 環境変数読み込みスクリプト
# 他のスクリプトから source されることを想定

# .envファイルが存在する場合は読み込む
if [ -f "$(dirname "$0")/../.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a  # 自動的にexport
    source "$(dirname "$0")/../.env"
    set +a
elif [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
fi

# 環境変数のデフォルト値設定（.envで設定されていない場合）
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CUSTOM_NODES="${CUSTOM_NODES:-}"
export MASTER_IP="${MASTER_IP:-}"
export SKIP_CONFIRM="${SKIP_CONFIRM:-false}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-meta-llama/Llama-2-7b-hf}"
export DEFAULT_NODES="${DEFAULT_NODES:-2}"
export DEFAULT_TEST_TYPE="${DEFAULT_TEST_TYPE:-sft}"
export DATASET_NUM_SAMPLES="${DATASET_NUM_SAMPLES:-10000}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-2048}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export PRECISION="${PRECISION:-bf16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export RESULTS_DIR="${RESULTS_DIR:-results}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export REPORT_TO="${REPORT_TO:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

# 環境変数の検証関数
validate_env() {
    local has_error=0
    
    # HF_TOKENの検証（Llamaモデル使用時）
    if [[ "${DEFAULT_MODEL}" == *"llama"* ]] && [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN is not set. You may not be able to access Llama models."
        echo "Please set HF_TOKEN in .env file or export it manually."
    fi
    
    # ノード数の検証
    if ! [[ "$DEFAULT_NODES" =~ ^[0-9]+$ ]] || [ "$DEFAULT_NODES" -lt 1 ] || [ "$DEFAULT_NODES" -gt 8 ]; then
        echo "Error: DEFAULT_NODES must be between 1 and 8"
        has_error=1
    fi
    
    # バッチサイズの検証
    if ! [[ "$BATCH_SIZE_PER_GPU" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE_PER_GPU" -lt 1 ]; then
        echo "Error: BATCH_SIZE_PER_GPU must be a positive integer"
        has_error=1
    fi
    
    return $has_error
}

# 環境変数の表示関数（デバッグ用）
show_env() {
    echo "=== Current Environment Variables ==="
    echo "HF_TOKEN: ${HF_TOKEN:+[SET]}"
    echo "CUSTOM_NODES: ${CUSTOM_NODES:-[AUTO-DETECT]}"
    echo "DEFAULT_MODEL: $DEFAULT_MODEL"
    echo "DEFAULT_NODES: $DEFAULT_NODES"
    echo "DEFAULT_TEST_TYPE: $DEFAULT_TEST_TYPE"
    echo "BATCH_SIZE_PER_GPU: $BATCH_SIZE_PER_GPU"
    echo "LEARNING_RATE: $LEARNING_RATE"
    echo "PRECISION: $PRECISION"
    echo "===================================="
}