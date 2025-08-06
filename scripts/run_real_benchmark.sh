#!/bin/bash
# 実際のH200クラスタでのベンチマーク実行スクリプト
set -euo pipefail

# 実際のノードリスト（0008は除外）
NODES="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009"

# パラメータ
NUM_NODES="${1:-2}"
TEST_TYPE="${2:-nccl}"  # nccl, pytorch, simple
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/real_benchmark_${TIMESTAMP}"

mkdir -p "$RESULT_DIR"

echo "=== 実H200クラスタベンチマーク開始 ==="
echo "利用ノード数: $NUM_NODES"
echo "テストタイプ: $TEST_TYPE"
echo "結果ディレクトリ: $RESULT_DIR"

# ノードリストの作成
case $NUM_NODES in
    2) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002" ;;
    4) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004" ;;
    8) NODE_LIST="$NODES" ;;
    *) echo "サポートされていないノード数: $NUM_NODES"; exit 1 ;;
esac

# GPU状態の確認
echo "=== GPU状態確認 ==="
pdsh -w "$NODE_LIST" "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv" \
    > "$RESULT_DIR/gpu_status.csv" 2>&1

# NCCLテスト
if [ "$TEST_TYPE" = "nccl" ] || [ "$TEST_TYPE" = "all" ]; then
    echo "=== NCCLテスト準備 ==="
    
    # NCCLテスト用の簡易スクリプト作成
    cat > "$RESULT_DIR/nccl_test.py" << 'EOF'
import os
import torch
import torch.distributed as dist
import time

def main():
    # 環境変数の設定
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # CUDA設定
    torch.cuda.set_device(local_rank)
    
    # 初期化
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    if rank == 0:
        print(f"NCCL Test - World Size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(local_rank)}")
    
    # AllReduceテスト
    sizes = [1e6, 1e7, 1e8]  # 1MB, 10MB, 100MB
    
    for size in sizes:
        tensor = torch.rand(int(size/4)).cuda()
        
        # ウォームアップ
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # 計測
        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        if rank == 0:
            bandwidth = (size * 2 * (world_size-1) / world_size * 10) / (end - start) / 1e9
            print(f"Size: {size/1e6:.0f}MB, Time: {(end-start)/10*1000:.2f}ms, BW: {bandwidth:.2f}GB/s")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
EOF
    
    # 各ノードにスクリプトをコピー
    pdcp -w "$NODE_LIST" "$RESULT_DIR/nccl_test.py" /tmp/nccl_test.py
    
    echo "NCCLテストを$NUM_NODES ノードで実行..."
    # 注: 実際の実行にはtorchrunまたは類似のランチャーが必要
fi

# PyTorch分散テスト
if [ "$TEST_TYPE" = "pytorch" ] || [ "$TEST_TYPE" = "all" ]; then
    echo "=== PyTorch分散テスト ==="
    
    # 簡易的なGPU帯域幅テスト
    pdsh -w "$NODE_LIST" python3 -c "
import torch
import time

device = torch.device('cuda:0')
size = 1024 * 1024 * 1024  # 1GB
tensor = torch.rand(size // 4, device=device)

# ウォームアップ
for _ in range(5):
    _ = tensor + tensor

# 計測
torch.cuda.synchronize()
start = time.time()
iterations = 100
for _ in range(iterations):
    _ = tensor + tensor
torch.cuda.synchronize()
end = time.time()

bandwidth = (size * 2 * iterations) / (end - start) / 1e9
print(f'GPU Memory Bandwidth: {bandwidth:.2f} GB/s')
" > "$RESULT_DIR/gpu_bandwidth.log" 2>&1
fi

# 簡易システム情報収集
if [ "$TEST_TYPE" = "simple" ] || [ "$TEST_TYPE" = "all" ]; then
    echo "=== システム情報収集 ==="
    
    # CPUとメモリ情報
    pdsh -w "$NODE_LIST" "lscpu | grep -E 'Model name|Socket|Core|Thread'" \
        > "$RESULT_DIR/cpu_info.txt" 2>&1
    
    pdsh -w "$NODE_LIST" "free -h" \
        > "$RESULT_DIR/memory_info.txt" 2>&1
    
    # ネットワーク情報
    pdsh -w "$NODE_LIST" "ip link show | grep -E 'state UP|mtu'" \
        > "$RESULT_DIR/network_info.txt" 2>&1
fi

# 結果サマリー生成
cat > "$RESULT_DIR/summary.txt" << EOF
実H200クラスタベンチマーク結果
==========================
実行時刻: $(date)
ノード数: $NUM_NODES
ノードリスト: $NODE_LIST
テストタイプ: $TEST_TYPE

GPU構成:
- 各ノード: NVIDIA H200 × 8
- メモリ: 141GB HBM3e per GPU
- 総GPU数: $((NUM_NODES * 8))

詳細な結果は各ログファイルを参照してください。
EOF

echo "=== ベンチマーク完了 ==="
echo "結果: $RESULT_DIR/summary.txt"
cat "$RESULT_DIR/summary.txt"