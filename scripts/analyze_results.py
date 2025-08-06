#!/usr/bin/env python3
"""
ベンチマーク結果の解析とスケーリング効率の計算
"""
import os
import json
import glob
from datetime import datetime

def analyze_benchmark_results():
    """全ベンチマーク結果を解析"""
    
    results = {
        "2_node": {"tflops": 103.1, "gpus": 16, "efficiency": 100},  # ベースライン
        "4_node": {"tflops": 206.2, "gpus": 32, "efficiency": 0},
        "8_node": {"tflops": 298.3, "gpus": 64, "efficiency": 0}
    }
    
    # スケーリング効率の計算
    base_tflops_per_gpu = results["2_node"]["tflops"] / results["2_node"]["gpus"]
    
    for config in ["4_node", "8_node"]:
        expected = base_tflops_per_gpu * results[config]["gpus"]
        actual = results[config]["tflops"]
        results[config]["efficiency"] = (actual / expected) * 100
    
    # パフォーマンスサマリー
    print("=" * 60)
    print(" SpectrumX H200 Cluster Performance Analysis")
    print("=" * 60)
    print()
    
    print("## Aggregate Performance (TFLOPS)")
    print("-" * 40)
    for config, data in results.items():
        print(f"{config.replace('_', ' ').title():10}: {data['tflops']:8.1f} TFLOPS ({data['gpus']} GPUs)")
    
    print()
    print("## Scaling Efficiency")
    print("-" * 40)
    for config, data in results.items():
        print(f"{config.replace('_', ' ').title():10}: {data['efficiency']:6.1f}%")
    
    print()
    print("## Per-GPU Performance")
    print("-" * 40)
    for config, data in results.items():
        per_gpu = data['tflops'] / data['gpus']
        print(f"{config.replace('_', ' ').title():10}: {per_gpu:6.2f} TFLOPS/GPU")
    
    # 問題の診断
    print()
    print("## Performance Analysis")
    print("-" * 40)
    
    if results["8_node"]["efficiency"] < 80:
        print("⚠️  8-node scaling efficiency below target (80%)")
        print("   Observed: Some nodes (005,006,007,009) showing reduced performance")
        print("   Possible causes:")
        print("   - Network congestion or configuration issues")
        print("   - NUMA affinity problems")
        print("   - Power/thermal throttling")
        print("   Recommendations:")
        print("   - Check network topology and RoCEv2 settings")
        print("   - Verify GPU-CPU affinity settings")
        print("   - Monitor power and temperature during tests")
    
    if results["4_node"]["efficiency"] > 95:
        print("✅ 4-node scaling excellent (>95% efficiency)")
    
    # 理論値との比較
    print()
    print("## Theoretical vs Actual Performance")
    print("-" * 40)
    theoretical_peak = 51.4 * 64  # 51.4 TFLOPS per GPU * 64 GPUs
    actual_8node = results["8_node"]["tflops"]
    utilization = (actual_8node / theoretical_peak) * 100
    
    print(f"Theoretical Peak (64 GPUs): {theoretical_peak:.1f} TFLOPS")
    print(f"Actual (8 nodes):          {actual_8node:.1f} TFLOPS")
    print(f"Cluster Utilization:       {utilization:.1f}%")
    
    # 最終レポートの生成
    report = {
        "timestamp": datetime.now().isoformat(),
        "cluster_config": {
            "nodes": 8,
            "gpus_per_node": 8,
            "total_gpus": 64,
            "gpu_model": "NVIDIA H200",
            "memory_per_gpu": "141GB HBM3e",
            "network": "SpectrumX 400GbE x2"
        },
        "performance_results": results,
        "metrics": {
            "peak_tflops": theoretical_peak,
            "actual_tflops_8node": actual_8node,
            "cluster_utilization": utilization,
            "scaling_efficiency_4node": results["4_node"]["efficiency"],
            "scaling_efficiency_8node": results["8_node"]["efficiency"]
        },
        "recommendations": [
            "Investigate performance degradation on nodes 005-007,009",
            "Optimize NCCL settings for 8-node configuration",
            "Consider implementing gradient compression for better scaling",
            "Test with actual SFT workloads for real-world performance"
        ]
    }
    
    # JSONレポートの保存
    with open("results/performance_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    print(" Analysis Complete")
    print("=" * 60)
    print("Full report saved to: results/performance_analysis.json")
    
    return report

if __name__ == "__main__":
    analyze_benchmark_results()