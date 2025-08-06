#!/bin/bash
set -euo pipefail

# 環境変数の読み込み
source "$(dirname "$0")/../setup/load_env.sh"

# クラスタ設定の読み込み
source "$(dirname "$0")/../setup/cluster_config.sh"

# デフォルト値（環境変数から取得、引数で上書き可能）
MODEL_NAME="${1:-$DEFAULT_MODEL}"
NUM_NODES="${2:-$DEFAULT_NODES}"
TEST_TYPE="${3:-$DEFAULT_TEST_TYPE}"  # sft, allreduce, full
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)_${TEST_TYPE}_${NUM_NODES}node"

# ノードの自動検出
echo "=== ノード検出とクラスタ準備 ==="
detect_available_nodes || exit 1

# 指定されたノード数の検証
if [ $NUM_NODES -gt $NODE_COUNT ]; then
    echo "エラー: 要求されたノード数 ($NUM_NODES) が利用可能なノード数 ($NODE_COUNT) を超えています"
    exit 1
fi

# 使用するノードの選択
SELECTED_NODES=("${AVAILABLE_NODES[@]:0:$NUM_NODES}")
SELECTED_PDSH=$(IFS=,; echo "${SELECTED_NODES[*]}")

echo "=== ベンチマーク開始 ==="
echo "モデル: $MODEL_NAME"
echo "使用ノード数: $NUM_NODES"
echo "選択されたノード: ${SELECTED_NODES[*]}"
echo "テストタイプ: $TEST_TYPE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 出力ディレクトリの作成
mkdir -p "$OUTPUT_DIR"

# 環境変数の設定
export MODEL_NAME="$MODEL_NAME"
export OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
export MASTER_ADDR="${MASTER_IP}"
export MASTER_PORT=29500
export WORLD_SIZE=$((NUM_NODES * 8))

# ログファイルの設定
BENCHMARK_LOG="$OUTPUT_DIR/benchmark.log"
exec > >(tee -a "$BENCHMARK_LOG")
exec 2>&1

# GPUの状態確認
verify_gpus() {
    echo "=== GPU状態の確認 ==="
    pdsh -w "$SELECTED_PDSH" "nvidia-smi -L" | sort
    echo ""
}

# NCCLテストの実行
run_nccl_test() {
    echo "=== NCCLベンチマーク実行 ==="
    local nccl_log="$OUTPUT_DIR/nccl_test.log"
    
    # hostfileの生成
    echo "Generating hostfile..."
    > "$OUTPUT_DIR/hostfile"
    for node in "${SELECTED_NODES[@]}"; do
        echo "$node slots=8" >> "$OUTPUT_DIR/hostfile"
    done
    
    # NCCLテストの実行（存在する場合）
    if command -v /opt/nccl-tests/build/all_reduce_perf &> /dev/null; then
        mpirun -np $WORLD_SIZE \
            --hostfile "$OUTPUT_DIR/hostfile" \
            --mca btl_tcp_if_include bond0 \
            --mca btl ^openib \
            -x NCCL_DEBUG=INFO \
            -x NCCL_IB_HCA=mlx5 \
            -x NCCL_IB_GID_INDEX=3 \
            /opt/nccl-tests/build/all_reduce_perf \
            -b 8 -e 8G -f 2 -g 1 2>&1 | tee "$nccl_log"
    else
        echo "NCCLテストがインストールされていません。スキップします。"
    fi
    echo ""
}

# 簡易的な分散PyTorchテスト
run_pytorch_test() {
    echo "=== PyTorch分散テスト実行 ==="
    
    # テストスクリプトの生成
    cat > "$OUTPUT_DIR/pytorch_test.py" << 'EOF'
import os
import torch
import torch.distributed as dist
import time
import json

def main():
    # 環境変数から設定を取得
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # CUDAデバイスの設定
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    # 分散環境の初期化
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # デバイス情報の表示
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # All-reduceベンチマーク
    sizes = [1e6, 1e7, 1e8, 1e9]  # 1MB, 10MB, 100MB, 1GB
    results = []
    
    for size in sizes:
        tensor_size = int(size / 4)  # float32
        tensor = torch.rand(tensor_size).cuda()
        
        # ウォームアップ
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
        # 計測
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        end = time.time()
        
        # 結果の計算
        time_per_iter = (end - start) / iterations
        bandwidth = (size * 2 * (world_size - 1) / world_size) / time_per_iter / 1e9  # GB/s
        
        if rank == 0:
            print(f"Size: {size/1e9:.1f}GB, Time: {time_per_iter*1e3:.2f}ms, Bandwidth: {bandwidth:.2f}GB/s")
            results.append({
                "size_gb": size/1e9,
                "time_ms": time_per_iter*1e3,
                "bandwidth_gbps": bandwidth * 8  # Gbps
            })
    
    # 結果の保存
    if rank == 0:
        with open(f"{os.environ.get('OUTPUT_DIR', '.')}/pytorch_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # クリーンアップ
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
EOF
    
    # 分散実行
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$NUM_NODES \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        "$OUTPUT_DIR/pytorch_test.py"
    
    echo ""
}

# SFT訓練の実行
run_sft_training() {
    echo "=== SFT訓練ベンチマーク実行 ==="
    
    # データセットの準備確認
    DATASET_PATH="datasets/sft_${MODEL_NAME##*/}"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "データセットを準備中..."
        cd datasets && python prepare_dataset.py && cd ..
    fi
    
    # DeepSpeed設定の選択
    if [[ "$MODEL_NAME" == *"70b"* ]]; then
        DS_CONFIG="configs/ds_config_70b.json"
    elif [[ "$MODEL_NAME" == *"13b"* ]]; then
        DS_CONFIG="configs/ds_config_13b.json"
    else
        DS_CONFIG="configs/ds_config_7b.json"
    fi
    
    # SFT訓練スクリプトが存在するか確認
    if [ -f "scripts/train_sft.py" ]; then
        # DeepSpeedランチャーで実行
        deepspeed --hostfile "$OUTPUT_DIR/hostfile" \
            --num_nodes $NUM_NODES \
            --num_gpus $((NUM_NODES * 8)) \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            scripts/train_sft.py \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --deepspeed "$DS_CONFIG" \
            2>&1 | tee "$OUTPUT_DIR/sft_training.log"
    else
        echo "train_sft.pyが見つかりません。PyTorchテストのみ実行します。"
        run_pytorch_test
    fi
    echo ""
}

# メトリクス収集の開始
start_metrics_collection() {
    echo "=== メトリクス収集開始 ==="
    scripts/collect_metrics.sh "$OUTPUT_DIR" &
    METRICS_PID=$!
    echo "メトリクス収集PID: $METRICS_PID"
}

# メトリクス収集の停止
stop_metrics_collection() {
    if [ -n "${METRICS_PID:-}" ]; then
        echo "=== メトリクス収集停止 ==="
        kill $METRICS_PID 2>/dev/null || true
        wait $METRICS_PID 2>/dev/null || true
    fi
}

# クリーンアップ処理
cleanup() {
    stop_metrics_collection
    echo "クリーンアップ完了"
}
trap cleanup EXIT

# メイン実行
main() {
    verify_gpus
    
    case "$TEST_TYPE" in
        nccl)
            run_nccl_test
            ;;
        pytorch)
            start_metrics_collection
            run_pytorch_test
            ;;
        sft)
            start_metrics_collection
            run_sft_training
            ;;
        full)
            start_metrics_collection
            run_nccl_test
            run_pytorch_test
            run_sft_training
            ;;
        *)
            echo "不明なテストタイプ: $TEST_TYPE"
            echo "使用可能: nccl, pytorch, sft, full"
            exit 1
            ;;
    esac
    
    # 結果サマリーの生成
    generate_summary
}

# 結果サマリーの生成
generate_summary() {
    echo "=== 結果サマリー生成 ==="
    
    cat > "$OUTPUT_DIR/summary.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL_NAME",
  "nodes": $NUM_NODES,
  "gpus": $WORLD_SIZE,
  "test_type": "$TEST_TYPE",
  "selected_nodes": $(printf '%s\n' "${SELECTED_NODES[@]}" | jq -R . | jq -s .),
  "output_dir": "$OUTPUT_DIR"
}
EOF
    
    echo "ベンチマーク完了！"
    echo "結果: $OUTPUT_DIR"
}

# 使用方法の表示
if [ "$#" -eq 0 ]; then
    echo "使用方法: $0 [モデル名] [ノード数] [テストタイプ]"
    echo ""
    echo "例:"
    echo "  $0 meta-llama/Llama-2-7b-hf 2 sft"
    echo "  $0 meta-llama/Llama-2-13b-hf 4 full"
    echo ""
    echo "テストタイプ:"
    echo "  nccl    - NCCLベンチマークのみ"
    echo "  pytorch - PyTorch分散テストのみ"
    echo "  sft     - SFT訓練ベンチマーク"
    echo "  full    - 全てのテストを実行"
    exit 1
fi

# メイン処理の実行
main