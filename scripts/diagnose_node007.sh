#!/bin/bash
set -euo pipefail

# ============================================
# Node007性能問題診断スクリプト
# ============================================
# Node007が他のノードの約50%の性能しか出ない問題を調査

echo "============================================"
echo " Node007 Performance Diagnostic Tool"
echo " Target: fukushimadc-02-hgx-0007"
echo " Date: $(date)"
echo "============================================"
echo ""

# ターゲットノード
TARGET_NODE="fukushimadc-02-hgx-0007"
REFERENCE_NODE="fukushimadc-02-hgx-0002"  # 比較用の正常ノード
OUTPUT_DIR="results/node007_diagnosis_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

# ログ関数
log_info() { echo "[INFO] $1" | tee -a "$OUTPUT_DIR/diagnosis.log"; }
log_warn() { echo "[WARN] $1" | tee -a "$OUTPUT_DIR/diagnosis.log"; }
log_error() { echo "[ERROR] $1" | tee -a "$OUTPUT_DIR/diagnosis.log"; }

# 1. 基本的なシステム情報の収集
log_info "=== 1. システム基本情報の収集 ==="

# CPUとメモリ情報
log_info "CPU情報を収集しています..."
ssh -p 44222 "$TARGET_NODE" "lscpu" > "$OUTPUT_DIR/node007_cpu_info.txt" 2>&1
ssh -p 44222 "$REFERENCE_NODE" "lscpu" > "$OUTPUT_DIR/node002_cpu_info.txt" 2>&1

ssh -p 44222 "$TARGET_NODE" "free -h" > "$OUTPUT_DIR/node007_memory.txt" 2>&1
ssh -p 44222 "$REFERENCE_NODE" "free -h" > "$OUTPUT_DIR/node002_memory.txt" 2>&1

# 2. GPU詳細情報の収集
log_info "=== 2. GPU詳細情報の収集 ==="

# nvidia-smi詳細クエリ
cat > /tmp/gpu_query.sh << 'EOF'
#!/bin/bash
echo "=== GPU Overview ==="
nvidia-smi

echo -e "\n=== GPU Detailed Query ==="
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics,clocks.max.sm,clocks.max.memory,temperature.gpu,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv

echo -e "\n=== GPU Performance State ==="
for i in {0..7}; do
    echo "GPU $i:"
    nvidia-smi -i $i --query-gpu=pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost --format=csv,noheader
done

echo -e "\n=== PCIe Information ==="
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

echo -e "\n=== ECC Errors ==="
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
EOF

chmod +x /tmp/gpu_query.sh

# 両ノードでGPU情報を収集
log_info "Node007のGPU情報を収集しています..."
scp -P 44222 /tmp/gpu_query.sh "$TARGET_NODE:/tmp/" 2>/dev/null
ssh -p 44222 "$TARGET_NODE" "/tmp/gpu_query.sh" > "$OUTPUT_DIR/node007_gpu_detail.txt" 2>&1

log_info "Node002（参照）のGPU情報を収集しています..."
scp -P 44222 /tmp/gpu_query.sh "$REFERENCE_NODE:/tmp/" 2>/dev/null
ssh -p 44222 "$REFERENCE_NODE" "/tmp/gpu_query.sh" > "$OUTPUT_DIR/node002_gpu_detail.txt" 2>&1

# 3. 熱制限とパワー制限の確認
log_info "=== 3. 熱制限とパワー制限の確認 ==="

# 温度とパワーの監視
cat > /tmp/monitor_thermal.sh << 'EOF'
#!/bin/bash
echo "Monitoring GPU thermal and power for 30 seconds..."
echo "Time,GPU,Temp(C),Power(W),Util(%),Clock(MHz)"

for i in {1..30}; do
    timestamp=$(date +%s)
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,utilization.gpu,clocks.sm --format=csv,noheader | \
    while read line; do
        echo "$timestamp,$line"
    done
    sleep 1
done
EOF

chmod +x /tmp/monitor_thermal.sh

log_info "30秒間の熱・電力監視を実行しています..."
scp -P 44222 /tmp/monitor_thermal.sh "$TARGET_NODE:/tmp/" 2>/dev/null
ssh -p 44222 "$TARGET_NODE" "/tmp/monitor_thermal.sh" > "$OUTPUT_DIR/node007_thermal_monitor.csv" 2>&1 &
MONITOR_PID=$!

# 4. 簡単なGPUベンチマーク
log_info "=== 4. GPUベンチマーク実行 ==="

# PyTorchベンチマークスクリプト
cat > /tmp/gpu_benchmark.py << 'EOF'
import torch
import time
import sys

def benchmark_gpu(gpu_id=0, duration=10):
    """単一GPUのベンチマーク"""
    device = torch.device(f'cuda:{gpu_id}')
    
    # GPUのウォームアップ
    print(f"Warming up GPU {gpu_id}...")
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # ベンチマーク実行
    print(f"Running benchmark on GPU {gpu_id} for {duration} seconds...")
    size = 8192
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        iterations += 1
    
    elapsed = time.time() - start_time
    
    # パフォーマンス計算
    flops_per_matmul = 2 * size ** 3
    total_flops = flops_per_matmul * iterations
    tflops = total_flops / elapsed / 1e12
    
    # メモリ情報
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    
    print(f"\nGPU {gpu_id} Results:")
    print(f"  Iterations: {iterations}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    print(f"  Memory allocated: {allocated:.1f} GB")
    print(f"  Memory reserved: {reserved:.1f} GB")
    
    return tflops

if __name__ == "__main__":
    # 全GPUでベンチマーク
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    results = []
    for gpu_id in range(num_gpus):
        tflops = benchmark_gpu(gpu_id, duration=5)
        results.append(tflops)
    
    # 結果のサマリー
    print("\n=== Summary ===")
    avg_tflops = sum(results) / len(results)
    print(f"Average performance: {avg_tflops:.2f} TFLOPS")
    print(f"Min performance: {min(results):.2f} TFLOPS")
    print(f"Max performance: {max(results):.2f} TFLOPS")
    
    # 性能のばらつきをチェック
    variation = (max(results) - min(results)) / avg_tflops * 100
    if variation > 10:
        print(f"WARNING: Large performance variation detected: {variation:.1f}%")
EOF

log_info "Node007でGPUベンチマークを実行しています..."
scp -P 44222 /tmp/gpu_benchmark.py "$TARGET_NODE:/tmp/" 2>/dev/null
ssh -p 44222 "$TARGET_NODE" "cd /tmp && python3 gpu_benchmark.py" > "$OUTPUT_DIR/node007_benchmark.txt" 2>&1

log_info "Node002（参照）でGPUベンチマークを実行しています..."
scp -P 44222 /tmp/gpu_benchmark.py "$REFERENCE_NODE:/tmp/" 2>/dev/null
ssh -p 44222 "$REFERENCE_NODE" "cd /tmp && python3 gpu_benchmark.py" > "$OUTPUT_DIR/node002_benchmark.txt" 2>&1

# 監視の終了
wait $MONITOR_PID 2>/dev/null || true

# 5. システムログの確認
log_info "=== 5. システムログの確認 ==="

# dmesgからGPU関連のエラーを抽出
ssh -p 44222 "$TARGET_NODE" "dmesg | grep -iE 'nvidia|gpu|error|fail' | tail -100" > "$OUTPUT_DIR/node007_dmesg.txt" 2>&1

# 6. 結果の解析とレポート生成
log_info "=== 6. 診断結果の解析 ==="

cat > "$OUTPUT_DIR/diagnosis_report.md" << EOF
# Node007 性能問題診断レポート

生成日時: $(date)

## 1. 問題の概要
- **対象ノード**: $TARGET_NODE
- **症状**: 他のノードと比較して約50%の性能低下
- **参照ノード**: $REFERENCE_NODE

## 2. ベンチマーク結果比較

### Node007 性能
\`\`\`
$(grep -A5 "Summary" "$OUTPUT_DIR/node007_benchmark.txt" 2>/dev/null || echo "ベンチマーク結果なし")
\`\`\`

### Node002（参照）性能
\`\`\`
$(grep -A5 "Summary" "$OUTPUT_DIR/node002_benchmark.txt" 2>/dev/null || echo "ベンチマーク結果なし")
\`\`\`

## 3. GPU状態の確認

### 温度とクロック
$(grep "temperature.gpu" "$OUTPUT_DIR/node007_gpu_detail.txt" | head -5 2>/dev/null || echo "GPU情報取得失敗")

### スロットリング状態
$(grep -A8 "Performance State" "$OUTPUT_DIR/node007_gpu_detail.txt" 2>/dev/null || echo "性能状態情報なし")

## 4. 推奨される対処法

EOF

# 結果に基づく推奨事項を追加
if grep -q "hw_thermal_slowdown" "$OUTPUT_DIR/node007_gpu_detail.txt" 2>/dev/null; then
    echo "- **熱制限が検出されました**: 冷却システムの確認が必要です" >> "$OUTPUT_DIR/diagnosis_report.md"
fi

if grep -q "sw_power_cap" "$OUTPUT_DIR/node007_gpu_detail.txt" 2>/dev/null; then
    echo "- **電力制限が検出されました**: 電源設定の確認が必要です" >> "$OUTPUT_DIR/diagnosis_report.md"
fi

# パフォーマンス差の計算
NODE007_PERF=$(grep "Average performance" "$OUTPUT_DIR/node007_benchmark.txt" 2>/dev/null | awk '{print $3}')
NODE002_PERF=$(grep "Average performance" "$OUTPUT_DIR/node002_benchmark.txt" 2>/dev/null | awk '{print $3}')

if [ -n "$NODE007_PERF" ] && [ -n "$NODE002_PERF" ]; then
    PERF_RATIO=$(echo "scale=2; $NODE007_PERF / $NODE002_PERF * 100" | bc 2>/dev/null || echo "計算失敗")
    echo "" >> "$OUTPUT_DIR/diagnosis_report.md"
    echo "### 性能比較" >> "$OUTPUT_DIR/diagnosis_report.md"
    echo "- Node007性能: $NODE007_PERF TFLOPS" >> "$OUTPUT_DIR/diagnosis_report.md"
    echo "- Node002性能: $NODE002_PERF TFLOPS" >> "$OUTPUT_DIR/diagnosis_report.md"
    echo "- 性能比: ${PERF_RATIO}%" >> "$OUTPUT_DIR/diagnosis_report.md"
fi

echo "" >> "$OUTPUT_DIR/diagnosis_report.md"
echo "## 5. 詳細ログ" >> "$OUTPUT_DIR/diagnosis_report.md"
echo "診断結果の詳細は以下のファイルを参照してください：" >> "$OUTPUT_DIR/diagnosis_report.md"
echo "- GPU詳細: node007_gpu_detail.txt" >> "$OUTPUT_DIR/diagnosis_report.md"
echo "- 熱監視: node007_thermal_monitor.csv" >> "$OUTPUT_DIR/diagnosis_report.md"
echo "- システムログ: node007_dmesg.txt" >> "$OUTPUT_DIR/diagnosis_report.md"

# 完了メッセージ
echo ""
echo "============================================"
log_info "診断が完了しました！"
echo "============================================"
echo ""
echo "診断結果: $OUTPUT_DIR/diagnosis_report.md"
echo ""

# 主要な問題点のサマリー
if [ -n "$PERF_RATIO" ] && [ $(echo "$PERF_RATIO < 80" | bc) -eq 1 ] 2>/dev/null; then
    log_warn "性能が${PERF_RATIO}%に低下しています（期待値の80%未満）"
fi

if grep -q "hw_thermal_slowdown\|hw_power_brake_slowdown" "$OUTPUT_DIR/node007_gpu_detail.txt" 2>/dev/null; then
    log_warn "ハードウェアによるスロットリングが検出されました"
fi

echo "推奨アクション:"
echo "1. sudo nvidia-smi -pm 1  # パーシステンスモードを有効化"
echo "2. sudo nvidia-smi -pl 700  # 電力制限を700Wに設定（H200の場合）"
echo "3. 冷却システムとエアフローの確認"
echo "4. PCIeリンク速度の確認"