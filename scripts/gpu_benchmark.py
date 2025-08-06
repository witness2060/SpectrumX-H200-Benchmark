#!/usr/bin/env python3
"""
GPU性能ベンチマークスクリプト
個別GPUの基本的な性能測定
"""
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class GPUBenchmark:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}')
        
        # GPU情報の取得
        self.gpu_name = torch.cuda.get_device_name(device_id)
        self.gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
        
        print(f"GPU {device_id}: {self.gpu_name}")
        print(f"Memory: {self.gpu_memory:.1f} GB")
        print("-" * 50)
    
    def measure_bandwidth(self, size_gb=1):
        """メモリ帯域幅の測定"""
        size = int(size_gb * 1024**3 / 4)  # float32
        
        # ホスト→デバイス
        host_tensor = torch.randn(size)
        torch.cuda.synchronize()
        start = time.time()
        device_tensor = host_tensor.to(self.device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        h2d_bandwidth = size_gb / h2d_time
        
        # デバイス→ホスト
        torch.cuda.synchronize()
        start = time.time()
        host_tensor = device_tensor.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        d2h_bandwidth = size_gb / d2h_time
        
        return {
            "h2d_bandwidth_gbps": h2d_bandwidth * 8,
            "d2h_bandwidth_gbps": d2h_bandwidth * 8
        }
    
    def measure_compute(self, size=8192):
        """計算性能の測定（TFLOPS）"""
        # 行列乗算によるFLOPS測定
        a = torch.randn(size, size, device=self.device, dtype=torch.float32)
        b = torch.randn(size, size, device=self.device, dtype=torch.float32)
        
        # ウォームアップ
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 測定
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # FLOPS計算: 2 * N^3 (行列乗算のFLOP数)
        flops = 2 * size**3 * iterations / elapsed
        tflops = flops / 1e12
        
        return {"compute_tflops": tflops}
    
    def measure_model_performance(self, model_type="transformer", batch_size=32):
        """モデル推論性能の測定"""
        if model_type == "transformer":
            # 簡易Transformerモデル
            model = nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ).to(self.device)
        else:
            # 簡易CNNモデル
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 1000)
            ).to(self.device)
        
        model.eval()
        
        # 入力データ
        if model_type == "transformer":
            input_data = torch.randn(batch_size, 512, 1024, device=self.device)
        else:
            input_data = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # ウォームアップ
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        torch.cuda.synchronize()
        
        # 測定
        iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_data)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = batch_size * iterations / elapsed
        latency = elapsed / iterations * 1000  # ms
        
        return {
            f"{model_type}_throughput_samples_per_sec": throughput,
            f"{model_type}_latency_ms": latency
        }
    
    def run_all_benchmarks(self):
        """全ベンチマークの実行"""
        results = {
            "gpu_id": self.device_id,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory,
            "timestamp": datetime.now().isoformat()
        }
        
        # メモリ帯域幅
        print("Measuring memory bandwidth...")
        bandwidth_results = self.measure_bandwidth()
        results.update(bandwidth_results)
        print(f"  H2D: {bandwidth_results['h2d_bandwidth_gbps']:.1f} Gbps")
        print(f"  D2H: {bandwidth_results['d2h_bandwidth_gbps']:.1f} Gbps")
        
        # 計算性能
        print("\nMeasuring compute performance...")
        compute_results = self.measure_compute()
        results.update(compute_results)
        print(f"  Compute: {compute_results['compute_tflops']:.1f} TFLOPS")
        
        # モデル性能
        print("\nMeasuring model performance...")
        transformer_results = self.measure_model_performance("transformer")
        results.update(transformer_results)
        print(f"  Transformer throughput: {transformer_results['transformer_throughput_samples_per_sec']:.1f} samples/sec")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    if args.device >= torch.cuda.device_count():
        print(f"Error: GPU {args.device} not found. Available GPUs: {torch.cuda.device_count()}")
        sys.exit(1)
    
    # ベンチマーク実行
    benchmark = GPUBenchmark(args.device)
    results = benchmark.run_all_benchmarks()
    
    # 結果の保存
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nFinal results:")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()