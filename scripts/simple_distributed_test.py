#!/usr/bin/env python3
"""
シンプルなPyTorch分散学習テスト
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json

def setup(rank, world_size):
    """分散環境の初期化"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # NCCLバックエンドで初期化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """分散環境のクリーンアップ"""
    dist.destroy_process_group()

class SimpleTransformer(nn.Module):
    """簡易的なTransformerモデル"""
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=32,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

def run_benchmark(rank, world_size, model_size="7b"):
    """ベンチマーク実行"""
    setup(rank, world_size)
    
    # モデルサイズの設定
    if model_size == "7b":
        hidden_size = 4096
        num_layers = 32
        batch_size = 4
    elif model_size == "13b":
        hidden_size = 5120
        num_layers = 40
        batch_size = 2
    else:  # 70b
        hidden_size = 8192
        num_layers = 80
        batch_size = 1
    
    # モデル作成とDDP設定
    model = SimpleTransformer(
        vocab_size=50000,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(rank)
    
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    
    # ダミーデータ
    seq_length = 512
    
    if rank == 0:
        print(f"Model: {model_size}")
        print(f"World Size: {world_size}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Num Layers: {num_layers}")
        print(f"Batch Size per GPU: {batch_size}")
        print(f"Total Batch Size: {batch_size * world_size}")
        print("Starting benchmark...")
    
    # ウォームアップ
    for _ in range(3):
        data = torch.randint(0, 50000, (batch_size, seq_length)).to(rank)
        output = ddp_model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # ベンチマーク実行
    num_steps = 10
    total_loss = 0.0
    
    for step in range(num_steps):
        data = torch.randint(0, 50000, (batch_size, seq_length)).to(rank)
        labels = torch.randint(0, 50000, (batch_size, seq_length)).to(rank)
        
        output = ddp_model(data)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            labels.reshape(-1)
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if rank == 0 and step % 2 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 結果の集計
    if rank == 0:
        total_time = end_time - start_time
        throughput = num_steps / total_time
        samples_per_sec = (batch_size * world_size * num_steps) / total_time
        tokens_per_sec = samples_per_sec * seq_length
        
        # GPU メモリ使用量
        gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        result = {
            "model_size": model_size,
            "world_size": world_size,
            "batch_size_per_gpu": batch_size,
            "total_batch_size": batch_size * world_size,
            "num_steps": num_steps,
            "average_loss": total_loss / num_steps,
            "total_time": total_time,
            "throughput_steps_per_sec": throughput,
            "samples_per_second": samples_per_sec,
            "tokens_per_second": tokens_per_sec,
            "gpu_memory_allocated_gb": gpu_memory_gb,
            "gpu_utilization": "N/A"  # 実際の測定には nvidia-ml-py が必要
        }
        
        print("\n" + "="*50)
        print("Benchmark Results:")
        print(json.dumps(result, indent=2))
        
        # 結果を保存
        with open(f"benchmark_result_{model_size}_{world_size}gpu.json", "w") as f:
            json.dump(result, f, indent=2)
    
    cleanup()

def main():
    """メイン関数"""
    # 環境変数から設定を取得
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    model_size = os.environ.get("MODEL_SIZE", "7b")
    
    if world_size > 1:
        # マルチプロセス実行
        mp.spawn(
            run_benchmark,
            args=(world_size, model_size),
            nprocs=world_size,
            join=True
        )
    else:
        # シングルプロセス実行
        run_benchmark(0, 1, model_size)

if __name__ == "__main__":
    # 単一GPUでのテスト
    if not dist.is_initialized():
        print("Running single GPU test...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        run_benchmark(0, 1, os.environ.get("MODEL_SIZE", "7b"))