#!/bin/bash
set -euo pipefail

# パラメータ
NUM_NODES="${1:-2}"
TEST_DURATION="${2:-30}"  # 秒

echo "==========================================="
echo " Parallel GPU Performance Test"
echo " Nodes: $NUM_NODES | Duration: ${TEST_DURATION}s"
echo "==========================================="

# PDSHのセットアップ
source /root/.pdshrc

# ノードリストの生成
case $NUM_NODES in
    2)
        NODES="node001,node002"
        ;;
    4)
        NODES="node001,node002,node003,node004"
        ;;
    8)
        NODES="node001,node002,node003,node004,node005,node006,node007,node009"
        ;;
    *)
        echo "Invalid number of nodes. Use 2, 4, or 8."
        exit 1
        ;;
esac

OUTPUT_DIR="results/parallel_test_${NUM_NODES}node_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Testing nodes: $NODES"
echo ""

# 簡単なGPUストレステストスクリプトの作成
cat > /tmp/gpu_stress.py << 'EOF'
import torch
import time
import sys
import os

def gpu_stress_test(duration=30):
    """GPU stress test for specified duration"""
    hostname = os.environ.get('HOSTNAME', 'unknown')
    
    if not torch.cuda.is_available():
        print(f"{hostname}: No CUDA available")
        return
    
    # 全GPUを使用
    num_gpus = torch.cuda.device_count()
    print(f"{hostname}: Starting stress test on {num_gpus} GPUs for {duration}s")
    
    # 各GPUで大きな行列乗算を実行
    tensors = []
    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        # 8GB程度のテンソルを作成
        size = 16384
        a = torch.randn(size, size // 2, device=device, dtype=torch.float32)
        b = torch.randn(size // 2, size, device=device, dtype=torch.float32)
        tensors.append((a, b, device))
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        for a, b, device in tensors:
            with torch.cuda.device(device):
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
        iterations += 1
    
    elapsed = time.time() - start_time
    
    # 結果を出力
    total_flops = 2 * (16384 ** 2) * (16384 // 2) * iterations * num_gpus
    tflops = total_flops / elapsed / 1e12
    
    print(f"{hostname}: Completed {iterations} iterations in {elapsed:.2f}s")
    print(f"{hostname}: Aggregate performance: {tflops:.2f} TFLOPS")
    
    # GPU使用率を確認
    for gpu_id in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"{hostname}: GPU{gpu_id} memory: {mem_alloc:.1f}/{mem_total:.1f} GB")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    gpu_stress_test(duration)
EOF

# 全ノードにスクリプトをコピー
echo "Distributing test script..."
for node in $(echo $NODES | tr ',' ' '); do
    scp -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no /tmp/gpu_stress.py $node:/tmp/ 2>/dev/null
done

# メトリクス収集開始
echo "Starting metrics collection..."
./scripts/collect_metrics.sh "$OUTPUT_DIR" 5 $((TEST_DURATION + 10)) &
METRICS_PID=$!

sleep 2

# 並列GPUストレステストの実行
echo ""
echo "=== Running Parallel GPU Stress Test ==="
echo "Duration: ${TEST_DURATION} seconds"
echo ""

pdsh -w "$NODES" "HOSTNAME=\$(hostname) python3 /tmp/gpu_stress.py $TEST_DURATION" 2>/dev/null &
TEST_PID=$!

# プログレスバーの表示
echo -n "Progress: "
for i in $(seq 1 $TEST_DURATION); do
    echo -n "."
    sleep 1
done
echo " Done!"

# テストの完了を待つ
wait $TEST_PID 2>/dev/null || true

# メトリクス収集を停止
sleep 5
kill $METRICS_PID 2>/dev/null || true

# 結果の集約
echo ""
echo "=== Test Results Summary ==="

# GPU使用率の確認
echo "Average GPU Utilization:"
if [ -f "$OUTPUT_DIR/gpu_metrics.csv" ]; then
    tail -100 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F',' '{sum+=$5; count++} END {if(count>0) printf "  %.1f%%\n", sum/count}'
fi

# ネットワーク使用量の確認
echo "Network Traffic:"
if [ -f "$OUTPUT_DIR/network_metrics.csv" ]; then
    echo "  Data collected in $OUTPUT_DIR/network_metrics.csv"
fi

# サマリーの生成
cat > "$OUTPUT_DIR/summary.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "num_nodes": $NUM_NODES,
  "total_gpus": $((NUM_NODES * 8)),
  "test_duration": $TEST_DURATION,
  "test_type": "parallel_gpu_stress",
  "output_dir": "$OUTPUT_DIR"
}
EOF

echo ""
echo "==========================================="
echo " Test Completed Successfully!"
echo "==========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""