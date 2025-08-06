#!/usr/bin/env python3
"""
Production-ready SFT benchmark optimized for H200 GPUs
"""
import torch
import torch.nn as nn
import time
import os
import json
from datetime import datetime
import sys

class SimplifiedTransformer(nn.Module):
    """Optimized transformer for benchmarking"""
    def __init__(self, vocab_size=50000, d_model=2048, n_heads=16, n_layers=12, seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,  # Reduced for memory
                batch_first=True,
                dropout=0.1
            ) for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.output(x)

def run_production_benchmark(duration=60):
    """Production SFT benchmark"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hostname = os.environ.get('HOSTNAME', 'unknown')
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return None
    
    print(f"=== Production SFT Benchmark ===")
    print(f"Host: {hostname}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Duration: {duration}s")
    print("=" * 40)
    
    # Model configuration (7B equivalent, memory optimized)
    batch_size = 4
    seq_length = 1024
    vocab_size = 50000
    
    # Create model
    model = SimplifiedTransformer(
        vocab_size=vocab_size,
        d_model=2048,
        n_heads=16,
        n_layers=12,
        seq_len=seq_length
    ).to(device)
    
    # Use mixed precision
    model = model.half()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_length}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        with torch.cuda.amp.autocast():
            output = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark
    print("Starting benchmark...")
    start_time = time.time()
    iterations = 0
    total_tokens = 0
    losses = []
    
    while time.time() - start_time < duration:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        iterations += 1
        total_tokens += batch_size * seq_length
        losses.append(loss.item())
        
        if iterations % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            print(f"Step {iterations:4d} | Loss: {loss.item():.4f} | "
                  f"Tokens/s: {tokens_per_sec:.0f} | Memory: {mem_gb:.1f}GB")
    
    # Final results
    total_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    tokens_per_sec = total_tokens / total_time
    samples_per_sec = (iterations * batch_size) / total_time
    
    # Calculate effective TFLOPS
    # Approximate FLOPs per token for transformer
    flops_per_token = 6 * sum(p.numel() for p in model.parameters())
    total_flops = flops_per_token * total_tokens
    tflops = total_flops / total_time / 1e12
    
    # GPU utilization estimate (based on achieved performance)
    # H200 theoretical: ~67 TFLOPS FP16
    theoretical_tflops = 67
    gpu_utilization = min(100, (tflops / theoretical_tflops) * 100)
    
    results = {
        "hostname": hostname,
        "device": torch.cuda.get_device_name(0),
        "duration": total_time,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_sec,
        "samples_per_second": samples_per_sec,
        "avg_loss": avg_loss,
        "tflops": tflops,
        "gpu_utilization": gpu_utilization,
        "memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n" + "=" * 40)
    print("BENCHMARK COMPLETE")
    print("=" * 40)
    print(f"Duration: {total_time:.1f}s")
    print(f"Iterations: {iterations}")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"Token Rate: {tokens_per_sec:.0f} tokens/sec")
    print(f"Performance: {tflops:.2f} TFLOPS")
    print(f"GPU Utilization: {gpu_utilization:.1f}%")
    print(f"Peak Memory: {results['memory_gb']:.1f} GB")
    
    # Save results
    output_file = f"/tmp/sft_benchmark_{hostname}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_production_benchmark(duration=duration)