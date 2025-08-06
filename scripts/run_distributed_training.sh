#!/bin/bash
# 実H200クラスタでの分散学習実行スクリプト
set -euo pipefail

# パラメータ
NUM_NODES="${1:-2}"
MODEL_SIZE="${2:-7b}"
BATCH_SIZE="${3:-4}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/distributed_training_${TIMESTAMP}"

# ノードリスト
case $NUM_NODES in
    2) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002" ;;
    4) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004" ;;
    8) NODE_LIST="fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009" ;;
esac

mkdir -p "$RESULT_DIR"

echo "====================================="
echo "実H200クラスタ分散学習ベンチマーク"
echo "====================================="
echo "ノード数: $NUM_NODES"
echo "モデルサイズ: $MODEL_SIZE"
echo "バッチサイズ/GPU: $BATCH_SIZE"
echo "ノードリスト: $NODE_LIST"
echo ""

# 簡易的なDeepSpeedテストスクリプト作成
cat > "$RESULT_DIR/test_deepspeed.py" << 'EOF'
import os
import time
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader
import json

# 環境情報の表示
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if rank == 0:
    print(f"PyTorch version: {torch.__version__}")
    print(f"DeepSpeed version: {deepspeed.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"World Size: {world_size}")

# 簡易モデル（メモリ使用量をシミュレート）
class SimpleModel(nn.Module):
    def __init__(self, model_size="7b"):
        super().__init__()
        if model_size == "7b":
            hidden_size = 4096
            num_layers = 32
        elif model_size == "13b":
            hidden_size = 5120
            num_layers = 40
        else:  # 70b
            hidden_size = 8192
            num_layers = 80
        
        self.embedding = nn.Embedding(50000, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=32,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 50000)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# ダミーデータセット
class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_len=512):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 50000, (self.seq_len,)),
            'labels': torch.randint(0, 50000, (self.seq_len,))
        }

# DeepSpeed設定
ds_config = {
    "train_batch_size": int(os.environ.get("BATCH_SIZE", 4)) * world_size,
    "train_micro_batch_size_per_gpu": int(os.environ.get("BATCH_SIZE", 4)),
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "wall_clock_breakdown": False
}

# メイン実行
def main():
    # モデルとデータローダー
    model = SimpleModel(os.environ.get("MODEL_SIZE", "7b"))
    dataset = DummyDataset(size=100)
    
    # DeepSpeed初期化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu,
        shuffle=True
    )
    
    # 訓練ループ
    model_engine.train()
    
    if rank == 0:
        print("\nStarting training...")
    
    start_time = time.time()
    total_loss = 0.0
    num_steps = 10  # ベンチマーク用に短縮
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        input_ids = batch['input_ids'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)
        
        outputs = model_engine(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1)
        )
        
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        
        if rank == 0 and step % 2 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # 結果の集計
    torch.cuda.synchronize()
    end_time = time.time()
    
    if rank == 0:
        avg_loss = total_loss / num_steps
        throughput = num_steps / (end_time - start_time)
        
        result = {
            "model_size": os.environ.get("MODEL_SIZE", "7b"),
            "world_size": world_size,
            "batch_size_per_gpu": model_engine.train_micro_batch_size_per_gpu,
            "num_steps": num_steps,
            "average_loss": avg_loss,
            "throughput_steps_per_sec": throughput,
            "total_time": end_time - start_time,
            "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
        
        print("\n" + "="*50)
        print("Training Results:")
        print(json.dumps(result, indent=2))
        
        # 結果を保存
        with open("training_result.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
EOF

# 各ノードにスクリプトをコピー
echo "スクリプトを各ノードにコピー中..."
pdcp -w "$NODE_LIST" "$RESULT_DIR/test_deepspeed.py" /tmp/test_deepspeed.py

# DeepSpeed起動スクリプト作成
cat > "$RESULT_DIR/launch_deepspeed.sh" << EOF
#!/bin/bash
cd /tmp

export MASTER_ADDR=fukushimadc-02-hgx-0001
export MASTER_PORT=29500
export BATCH_SIZE=$BATCH_SIZE
export MODEL_SIZE=$MODEL_SIZE

# DeepSpeed hostfileの作成
cat > hostfile << HOSTFILE
fukushimadc-02-hgx-0001 slots=8
fukushimadc-02-hgx-0002 slots=8
HOSTFILE

# 2ノードでの実行
deepspeed --hostfile hostfile \
          --num_nodes $NUM_NODES \
          --num_gpus $((NUM_NODES * 8)) \
          test_deepspeed.py
EOF

# 実行権限を付与
chmod +x "$RESULT_DIR/launch_deepspeed.sh"

# 各ノードにDeepSpeedがインストールされているか確認
echo ""
echo "DeepSpeedインストール状況確認..."
pdsh -w "$NODE_LIST" "python3 -c 'import deepspeed; print(f\"DeepSpeed {deepspeed.__version__} installed\")'" 2>&1 | grep -v Warning || {
    echo "DeepSpeedがインストールされていません。インストール中..."
    pdsh -w "$NODE_LIST" "pip install deepspeed==0.14.0" 2>&1 | grep -v Warning
}

# シングルGPUテスト（動作確認）
echo ""
echo "=== シングルGPUテスト実行 ==="
CUDA_VISIBLE_DEVICES=0 python3 "$RESULT_DIR/test_deepspeed.py" 2>&1 | tee "$RESULT_DIR/single_gpu_test.log"

# マルチノード実行の準備
echo ""
echo "=== マルチノード実行準備 ==="
echo "実行コマンド例:"
echo "cd $RESULT_DIR && ./launch_deepspeed.sh"
echo ""
echo "注意: マルチノード実行にはノード間の通信設定が必要です"

# 結果サマリー
cat > "$RESULT_DIR/summary.md" << EOF
# 分散学習ベンチマーク準備完了

## 構成
- ノード数: $NUM_NODES
- モデルサイズ: $MODEL_SIZE
- バッチサイズ/GPU: $BATCH_SIZE
- 総GPU数: $((NUM_NODES * 8))

## ノードリスト
$NODE_LIST

## 実行方法
\`\`\`bash
cd $RESULT_DIR
./launch_deepspeed.sh
\`\`\`

## 結果
シングルGPUテストの結果は single_gpu_test.log を参照
EOF

echo ""
echo "準備完了: $RESULT_DIR/summary.md"