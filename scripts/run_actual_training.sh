#!/bin/bash
# 実際のH200クラスタでの学習ベンチマーク
set -euo pipefail

# パラメータ
NUM_NODES="${1:-2}"
MODEL_SIZE="${2:-7b}"  # 7b, 13b, 70b
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/actual_training_${TIMESTAMP}"

mkdir -p "$RESULT_DIR"

# ノードリスト設定
case $NUM_NODES in
    2) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002" ;;
    4) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004" ;;
    8) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009" ;;
esac

echo "=== 実H200クラスタ学習ベンチマーク ==="
echo "ノード数: $NUM_NODES"
echo "モデルサイズ: $MODEL_SIZE"
echo "ノードリスト: $NODE_LIST"

# 簡易的な分散学習テストスクリプト作成
cat > "$RESULT_DIR/distributed_test.py" << 'EOF'
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class SimpleModel(nn.Module):
    def __init__(self, size="7b"):
        super().__init__()
        # モデルサイズに応じた簡易的なレイヤー
        if size == "7b":
            hidden_size = 4096
            num_layers = 32
        elif size == "13b":
            hidden_size = 5120
            num_layers = 40
        else:  # 70b
            hidden_size = 8192
            num_layers = 80
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.size = size
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'fukushimadc-02-hgx-0001')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size, model_size):
    setup(rank, world_size)
    
    # モデル作成
    model = SimpleModel(model_size).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # ダミーデータ
    batch_size = 4 if model_size == "70b" else 8 if model_size == "13b" else 16
    data = torch.randn(batch_size, 4096).to(rank)
    
    # ウォームアップ
    for _ in range(5):
        output = ddp_model(data)
        loss = output.sum()
        loss.backward()
    
    # ベンチマーク
    torch.cuda.synchronize()
    start = time.time()
    
    iterations = 10
    for _ in range(iterations):
        output = ddp_model(data)
        loss = output.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    end = time.time()
    
    if rank == 0:
        throughput = iterations / (end - start)
        print(f"Model: {model_size}")
        print(f"World Size: {world_size}")
        print(f"Throughput: {throughput:.2f} iterations/sec")
        print(f"Time per iteration: {(end-start)/iterations*1000:.2f} ms")
    
    cleanup()

if __name__ == "__main__":
    import sys
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    model_size = sys.argv[1] if len(sys.argv) > 1 else "7b"
    
    run_training(rank, world_size, model_size)
EOF

# 各ノードにスクリプトをコピー
echo "スクリプトを各ノードにコピー中..."
pdcp -w "$NODE_LIST" "$RESULT_DIR/distributed_test.py" /tmp/distributed_test.py

# 単一ノードでのテスト（動作確認）
echo "=== 単一ノードテスト実行 ==="
pdsh -w fukushimadc-02-hgx-0001 "cd /tmp && CUDA_VISIBLE_DEVICES=0 python3 distributed_test.py $MODEL_SIZE" \
    2>&1 | tee "$RESULT_DIR/single_node_test.log"

# マルチノードテストの準備（実際の実行にはMPIやtorchrunが必要）
echo ""
echo "=== マルチノード実行のための設定 ==="
cat > "$RESULT_DIR/run_multinode.sh" << EOF
#!/bin/bash
# マルチノード実行用スクリプト
# 注: 実際の実行にはtorchrunまたはmpirunの設定が必要

export MASTER_ADDR=fukushimadc-02-hgx-0001
export MASTER_PORT=12355
export WORLD_SIZE=$((NUM_NODES * 8))

# 各ノードで実行するコマンド例
# torchrun --nproc_per_node=8 \\
#          --nnodes=$NUM_NODES \\
#          --node_rank=\$NODE_RANK \\
#          --master_addr=\$MASTER_ADDR \\
#          --master_port=\$MASTER_PORT \\
#          /tmp/distributed_test.py $MODEL_SIZE
EOF

# 結果サマリー
cat > "$RESULT_DIR/summary.md" << EOF
# 実H200クラスタ学習ベンチマーク結果

## 構成情報
- 実行時刻: $(date)
- ノード数: $NUM_NODES
- モデルサイズ: $MODEL_SIZE
- 各ノード: NVIDIA H200 × 8 (141GB HBM3e)

## 実行ノード
$NODE_LIST

## テスト結果
単一ノードでの動作確認結果は single_node_test.log を参照

## マルチノード実行
マルチノード実行には追加の設定が必要です：
1. SSHキーの配布
2. torchrunまたはmpirunの設定
3. ファイアウォール設定（必要に応じて）

詳細は run_multinode.sh を参照してください。
EOF

echo ""
echo "=== ベンチマーク準備完了 ==="
echo "結果: $RESULT_DIR/"
echo ""
cat "$RESULT_DIR/summary.md"