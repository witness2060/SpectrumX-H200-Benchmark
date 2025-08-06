#!/usr/bin/env python3
"""
軽量な分散学習テスト（H200用に最適化）
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

class LightweightModel(nn.Module):
    """メモリ効率的な軽量モデル"""
    def __init__(self, vocab_size=10000, hidden_size=1024, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # residual connection
            x = torch.relu(x)
        x = self.norm(x)
        return self.output(x)

def run_test(rank=0, world_size=1):
    """分散学習テスト実行"""
    
    # 分散環境の初期化（マルチGPUの場合）
    if world_size > 1:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # モデルとオプティマイザー
    model = LightweightModel(
        vocab_size=10000,
        hidden_size=1024,
        num_layers=8
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # テストパラメータ
    batch_size = 2
    seq_length = 128
    num_steps = 10
    
    if rank == 0:
        print(f"Device: {device}")
        print(f"World Size: {world_size}")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Batch Size: {batch_size}")
        print(f"Sequence Length: {seq_length}")
        
        # GPU情報
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # ウォームアップ
    for _ in range(2):
        data = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # メモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # ベンチマーク開始
    if rank == 0:
        print("\nStarting benchmark...")
    
    start_time = time.time()
    losses = []
    
    for step in range(num_steps):
        # データ生成
        data = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
        labels = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
        
        # フォワードパス
        output = model(data)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.size(-1)),
            labels.reshape(-1)
        )
        
        # バックワードパス
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        if rank == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 結果の集計
    if rank == 0:
        total_time = end_time - start_time
        avg_loss = sum(losses) / len(losses)
        throughput = num_steps / total_time
        samples_per_sec = (batch_size * world_size * num_steps) / total_time
        
        result = {
            "test_type": "lightweight_distributed",
            "world_size": world_size,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_steps": num_steps,
            "average_loss": avg_loss,
            "total_time": total_time,
            "throughput_steps_per_sec": throughput,
            "samples_per_second": samples_per_sec,
            "model_parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6
        }
        
        if torch.cuda.is_available():
            result["gpu_memory_used_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            result["gpu_name"] = torch.cuda.get_device_name(device)
        
        print("\n" + "="*50)
        print("Test Results:")
        print(json.dumps(result, indent=2))
        
        # 結果を保存
        with open("lightweight_test_result.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # クリーンアップ
    if world_size > 1:
        dist.destroy_process_group()

def main():
    """メイン関数"""
    # 単一GPU実行
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        # 最初のGPUで実行
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        run_test(rank=0, world_size=1)
    else:
        print("CUDA not available, running on CPU")
        run_test(rank=0, world_size=1)

if __name__ == "__main__":
    main()