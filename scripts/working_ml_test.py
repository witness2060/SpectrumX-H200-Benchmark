#!/usr/bin/env python3
"""
実環境で動作する機械学習テスト
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import json

class SimpleLM(nn.Module):
    """シンプルな言語モデル"""
    def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=1024, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch_size, seq_length]
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        output = self.fc(lstm_out)
        return output

def create_dataset(num_samples=1000, seq_length=64, vocab_size=10000):
    """テスト用データセット作成"""
    # ランダムな入力とターゲット
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    targets = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    dataset = TensorDataset(input_ids, targets)
    return dataset

def train_model():
    """モデルの学習"""
    print("=== H200 GPU Training Test ===")
    
    # デバイス設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # メモリクリア
        torch.cuda.empty_cache()
    
    # モデル作成
    model = SimpleLM(vocab_size=10000, embed_dim=512, hidden_dim=1024, num_layers=4)
    model = model.to(device)
    
    # モデル情報
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # データセット
    print("\nPreparing dataset...")
    dataset = create_dataset(num_samples=2000, seq_length=64, vocab_size=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 最適化
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 学習
    print("Starting training...")
    model.train()
    
    num_epochs = 3
    start_time = time.time()
    all_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss計算
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                current_loss = loss.item()
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {current_loss:.4f}")
                all_losses.append(current_loss)
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
    
    # 結果集計
    total_time = time.time() - start_time
    final_loss = all_losses[-1] if all_losses else 0
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    
    # メモリ使用量
    if device.type == "cuda":
        max_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        current_memory_gb = torch.cuda.memory_allocated() / 1e9
    else:
        max_memory_gb = 0
        current_memory_gb = 0
    
    # 結果表示
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Device: {device}")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Throughput: {len(dataset) * num_epochs / total_time:.2f} samples/sec")
    
    if device.type == "cuda":
        print(f"Peak GPU memory usage: {max_memory_gb:.2f} GB")
        print(f"Current GPU memory usage: {current_memory_gb:.2f} GB")
    
    # 結果をJSONに保存
    results = {
        "device": str(device),
        "gpu_name": gpu_name if device.type == "cuda" else "CPU",
        "model_parameters": total_params,
        "training_time": total_time,
        "final_loss": final_loss,
        "average_loss": avg_loss,
        "throughput": len(dataset) * num_epochs / total_time,
        "peak_memory_gb": max_memory_gb,
        "num_epochs": num_epochs,
        "batch_size": 32,
        "dataset_size": len(dataset)
    }
    
    output_file = "/tmp/ml_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(min(2, torch.cuda.device_count())):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("")
    
    # 学習実行
    results = train_model()
    
    print("\n✅ ML training test completed successfully!")
    
    return results

if __name__ == "__main__":
    main()