#!/bin/bash
# H200クラスタでのマルチノード分散学習ベンチマーク
set -euo pipefail

# パラメータ
NUM_NODES="${1:-2}"
GPUS_PER_NODE=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/multinode_benchmark_${TIMESTAMP}"

# ノードリスト設定
MASTER_ADDR="fukushimadc-02-hgx-0001"
MASTER_PORT="29500"

case $NUM_NODES in
    2) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002" ;;
    4) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004" ;;
    8) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009" ;;
esac

mkdir -p "$RESULT_DIR"

echo "========================================"
echo "H200 マルチノード分散学習ベンチマーク"
echo "========================================"
echo "ノード数: $NUM_NODES"
echo "GPU/ノード: $GPUS_PER_NODE"
echo "総GPU数: $((NUM_NODES * GPUS_PER_NODE))"
echo "マスターノード: $MASTER_ADDR"
echo ""

# 分散学習スクリプト作成
cat > "$RESULT_DIR/multinode_test.py" << 'EOF'
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

def setup():
    """分散環境の初期化"""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    """分散環境のクリーンアップ"""
    dist.destroy_process_group()

class TransformerModel(nn.Module):
    """実用的なサイズのTransformerモデル"""
    def __init__(self, vocab_size=32000, hidden_size=2048, num_layers=16, num_heads=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.ln(x)
        return self.output(x)

def main():
    """メイン実行関数"""
    setup()
    
    # ランク情報
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # モデル作成
    model = TransformerModel(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=16,
        num_heads=16
    ).to(local_rank)
    
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    # パラメータ
    batch_size = 4
    seq_length = 256
    num_steps = 20
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("分散学習ベンチマーク開始")
        print(f"{'='*60}")
        print(f"World Size: {world_size} GPUs")
        print(f"Nodes: {world_size // 8}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Batch Size per GPU: {batch_size}")
        print(f"Global Batch Size: {batch_size * world_size}")
        print(f"Sequence Length: {seq_length}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'='*60}\n")
    
    # ウォームアップ
    for _ in range(3):
        data = torch.randint(0, 32000, (batch_size, seq_length)).to(local_rank)
        output = ddp_model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # ベンチマーク実行
    if rank == 0:
        print("ベンチマーク実行中...")
    
    start_time = time.time()
    total_loss = 0.0
    
    for step in range(num_steps):
        data = torch.randint(0, 32000, (batch_size, seq_length)).to(local_rank)
        labels = torch.randint(0, 32000, (batch_size, seq_length)).to(local_rank)
        
        output = ddp_model(data)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            labels.reshape(-1)
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if rank == 0 and (step + 1) % 5 == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()
    
    # 結果集計
    if rank == 0:
        total_time = end_time - start_time
        avg_loss = total_loss / num_steps
        throughput = num_steps / total_time
        samples_per_sec = (batch_size * world_size * num_steps) / total_time
        tokens_per_sec = samples_per_sec * seq_length
        
        # メモリ使用量
        gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        result = {
            "benchmark_type": "multinode_distributed",
            "nodes": world_size // 8,
            "world_size": world_size,
            "model_parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
            "batch_size_per_gpu": batch_size,
            "global_batch_size": batch_size * world_size,
            "sequence_length": seq_length,
            "num_steps": num_steps,
            "average_loss": avg_loss,
            "total_time_seconds": total_time,
            "throughput_steps_per_sec": throughput,
            "samples_per_second": samples_per_sec,
            "tokens_per_second": tokens_per_sec,
            "gpu_memory_allocated_gb": gpu_memory_gb,
            "efficiency_metrics": {
                "time_per_step_ms": (total_time / num_steps) * 1000,
                "scaling_efficiency": "calculated_separately"
            }
        }
        
        print(f"\n{'='*60}")
        print("ベンチマーク結果:")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2))
        
        # 結果保存
        with open(f"multinode_result_{world_size}gpu.json", "w") as f:
            json.dump(result, f, indent=2)
    
    cleanup()

if __name__ == "__main__":
    main()
EOF

# 各ノードにスクリプトをコピー
echo "スクリプトを各ノードにコピー中..."
for node in $(echo $NODE_LIST | tr ',' ' '); do
    scp -P 44222 "$RESULT_DIR/multinode_test.py" "${node}:/tmp/" 2>/dev/null || \
    echo "Note: コピーできませんでした - $node"
done

# torchrun実行スクリプト作成
cat > "$RESULT_DIR/run_torchrun.sh" << EOF
#!/bin/bash
# torchrunでの分散実行

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0

cd /tmp

# ノード0（マスター）での実行コマンド
torchrun \\
    --nproc_per_node=$GPUS_PER_NODE \\
    --nnodes=$NUM_NODES \\
    --node_rank=0 \\
    --master_addr=\$MASTER_ADDR \\
    --master_port=\$MASTER_PORT \\
    multinode_test.py

# 他のノードでの実行例（node_rankを変更）
# torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=1 ...
EOF

chmod +x "$RESULT_DIR/run_torchrun.sh"

# 単一ノードでのマルチGPUテスト
echo ""
echo "=== 単一ノード・マルチGPUテスト実行 ==="
cd "$RESULT_DIR"
torchrun --nproc_per_node=2 --nnodes=1 multinode_test.py 2>&1 | tee single_node_multi_gpu.log || {
    echo "torchrunが利用できません。代替方法を使用します。"
    python3 -m torch.distributed.run --nproc_per_node=2 multinode_test.py 2>&1 | tee single_node_multi_gpu.log
}

# 結果サマリー
echo ""
echo "=== 実行サマリー ==="
echo "結果ディレクトリ: $RESULT_DIR"
echo ""
echo "マルチノード実行方法:"
echo "1. 各ノードで以下を実行:"
echo "   Node 0: cd /tmp && torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT multinode_test.py"
echo "   Node 1: cd /tmp && torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT multinode_test.py"
echo "   ..."
echo ""
echo "注意: 全ノードで同時に実行する必要があります"