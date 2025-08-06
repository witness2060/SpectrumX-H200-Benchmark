#!/usr/bin/env python3
"""
実環境での動作テスト用スクリプト
小規模なモデルとデータで実際の学習を実行
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import json

def setup_distributed():
    """分散環境のセットアップ"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Not running in distributed mode")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

class SimpleModel(nn.Module):
    """テスト用の簡単なモデル"""
    def __init__(self, input_size=1024, hidden_size=2048, output_size=1000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_step(model, data, target, optimizer, criterion):
    """1ステップの訓練"""
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # 分散設定
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"Process {rank}/{world_size} using GPU {local_rank}")
    
    # モデル作成
    model = SimpleModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # オプティマイザと損失関数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # テストデータ生成
    batch_size = 32
    input_size = 1024
    num_steps = 100
    
    # 訓練ループ
    start_time = time.time()
    losses = []
    
    for step in range(num_steps):
        # ダミーデータ生成
        data = torch.randn(batch_size, input_size).to(device)
        target = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # 訓練ステップ
        loss = train_step(model, data, target, optimizer, criterion)
        losses.append(loss)
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss:.4f}")
    
    # 結果まとめ
    total_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    throughput = (num_steps * batch_size) / total_time
    
    if rank == 0:
        print(f"\n=== Training Complete ===")
        print(f"World Size: {world_size}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        # 結果保存
        results = {
            "world_size": world_size,
            "total_time": total_time,
            "avg_loss": avg_loss,
            "throughput": throughput,
            "gpu_count": world_size
        }
        
        with open(f"/tmp/test_results_{world_size}.json", "w") as f:
            json.dump(results, f, indent=2)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()