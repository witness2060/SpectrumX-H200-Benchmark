#!/bin/bash
set -euo pipefail

# ============================================
# NCCL Benchmark Test Script
# ============================================
# SpectrumX H200クラスタでNCCL通信性能を測定

echo "============================================"
echo " NCCL Benchmark Test for SpectrumX H200"
echo " Date: $(date)"
echo "============================================"
echo ""

# パラメータ
NUM_NODES="${1:-2}"
MIN_SIZE="${2:-8}"           # 最小サイズ (bytes)
MAX_SIZE="${3:-8G}"          # 最大サイズ
OUTPUT_DIR="results/nccl_benchmark_$(date +%Y%m%d_%H%M%S)"

# クラスタ設定の読み込み
source /root/test/spectrumx-h200-benchmark/setup/cluster_config.sh

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ログ関数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 出力ディレクトリの作成
mkdir -p "$OUTPUT_DIR"

# 利用可能なノードを検出
log_info "利用可能なノードを検出しています..."
detect_available_nodes

if [ $NODE_COUNT -lt $NUM_NODES ]; then
    log_error "要求されたノード数 ($NUM_NODES) が利用可能なノード数 ($NODE_COUNT) を超えています"
    exit 1
fi

# テスト対象ノードの選択
TEST_NODES=(${AVAILABLE_NODES[@]:0:$NUM_NODES})
log_info "テスト対象ノード: ${TEST_NODES[*]}"

# NCCL環境変数の設定
cat > "$OUTPUT_DIR/nccl_env.sh" << 'EOF'
# NCCL Configuration for SpectrumX 400GbE
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SL=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_ALGO=Ring,Tree
export NCCL_BUFFSIZE=8388608
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EOF

# ホストファイルの生成
echo "# NCCL Test Hostfile" > "$OUTPUT_DIR/hostfile"
for node in "${TEST_NODES[@]}"; do
    echo "$node slots=8" >> "$OUTPUT_DIR/hostfile"
done

# 1. 単一ノード内のNVLink性能テスト
if [ $NUM_NODES -ge 1 ]; then
    log_info "=== 1. 単一ノード内NVLink性能テスト ==="
    
    cat > "$OUTPUT_DIR/single_node_test.sh" << 'EOF'
#!/bin/bash
source /root/test/spectrumx-h200-benchmark/setup/cluster_config.sh
cd /opt/nccl-tests/build

echo "Testing NVLink performance within single node..."
echo "============================================"

# All-Reduce test
./all_reduce_perf -b 8 -e 1G -f 2 -g 8 2>&1 | tee single_node_allreduce.log

# All-Gather test
./all_gather_perf -b 8 -e 512M -f 2 -g 8 2>&1 | tee single_node_allgather.log

# Broadcast test
./broadcast_perf -b 8 -e 512M -f 2 -g 8 2>&1 | tee single_node_broadcast.log
EOF
    
    chmod +x "$OUTPUT_DIR/single_node_test.sh"
    
    log_info "単一ノードテストを実行中..."
    ssh -p 44222 "${TEST_NODES[0]}" "bash -s" < "$OUTPUT_DIR/single_node_test.sh" > "$OUTPUT_DIR/single_node_results.log" 2>&1
fi

# 2. マルチノードNCCL通信テスト
if [ $NUM_NODES -ge 2 ]; then
    log_info "=== 2. マルチノード通信性能テスト ($NUM_NODES nodes) ==="
    
    # MPI環境の設定（OpenMPIを使用）
    cat > "$OUTPUT_DIR/run_multinode_nccl.sh" << EOF
#!/bin/bash
set -euo pipefail

# NCCL環境変数の読み込み
source "$OUTPUT_DIR/nccl_env.sh"

# テスト実行ディレクトリ
cd /opt/nccl-tests/build

# ノード数とGPU数の設定
NNODES=$NUM_NODES
NGPUS_PER_NODE=8
TOTAL_GPUS=\$((NNODES * NGPUS_PER_NODE))

echo "============================================"
echo "Running NCCL tests with:"
echo "  Nodes: \$NNODES"
echo "  GPUs per node: \$NGPUS_PER_NODE"
echo "  Total GPUs: \$TOTAL_GPUS"
echo "  Min size: $MIN_SIZE"
echo "  Max size: $MAX_SIZE"
echo "============================================"

# All-Reduce Performance Test
echo -e "\n=== All-Reduce Performance Test ==="
mpirun --mca btl_tcp_if_include bond0 \
       --mca oob_tcp_if_include bond0 \
       -np \$TOTAL_GPUS \
       --hostfile "$OUTPUT_DIR/hostfile" \
       --bind-to none \
       -x NCCL_DEBUG=INFO \
       -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_IB_TC \
       -x NCCL_IB_QPS_PER_CONNECTION -x NCCL_NET_GDR_LEVEL \
       -x NCCL_SOCKET_IFNAME -x NCCL_ALGO -x NCCL_BUFFSIZE \
       -x LD_LIBRARY_PATH \
       ./all_reduce_perf -b $MIN_SIZE -e $MAX_SIZE -f 2 -g 1

# Reduce-Scatter Performance Test
echo -e "\n=== Reduce-Scatter Performance Test ==="
mpirun --mca btl_tcp_if_include bond0 \
       --mca oob_tcp_if_include bond0 \
       -np \$TOTAL_GPUS \
       --hostfile "$OUTPUT_DIR/hostfile" \
       --bind-to none \
       -x NCCL_DEBUG=WARN \
       -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_IB_TC \
       -x NCCL_SOCKET_IFNAME -x NCCL_ALGO \
       -x LD_LIBRARY_PATH \
       ./reduce_scatter_perf -b $MIN_SIZE -e $MAX_SIZE -f 2 -g 1

# All-Gather Performance Test
echo -e "\n=== All-Gather Performance Test ==="
mpirun --mca btl_tcp_if_include bond0 \
       --mca oob_tcp_if_include bond0 \
       -np \$TOTAL_GPUS \
       --hostfile "$OUTPUT_DIR/hostfile" \
       --bind-to none \
       -x NCCL_DEBUG=WARN \
       -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_IB_TC \
       -x NCCL_SOCKET_IFNAME -x NCCL_ALGO \
       -x LD_LIBRARY_PATH \
       ./all_gather_perf -b $MIN_SIZE -e $MAX_SIZE -f 2 -g 1
EOF
    
    chmod +x "$OUTPUT_DIR/run_multinode_nccl.sh"
    
    # マルチノードテストの実行
    log_info "マルチノードNCCLテストを実行中..."
    scp -P 44222 "$OUTPUT_DIR/run_multinode_nccl.sh" "${TEST_NODES[0]}:/tmp/" 2>/dev/null
    scp -P 44222 "$OUTPUT_DIR/hostfile" "${TEST_NODES[0]}:/tmp/" 2>/dev/null
    scp -P 44222 "$OUTPUT_DIR/nccl_env.sh" "${TEST_NODES[0]}:/tmp/" 2>/dev/null
    
    ssh -p 44222 "${TEST_NODES[0]}" "cd /tmp && bash run_multinode_nccl.sh" 2>&1 | tee "$OUTPUT_DIR/multinode_results.log"
fi

# 3. 結果の解析とレポート生成
log_info "=== 3. 結果の解析 ==="

# Pythonスクリプトで結果を解析
cat > "$OUTPUT_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
import re
import sys
import json

def parse_nccl_output(filename):
    """NCCLテスト出力から性能データを抽出"""
    results = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # データ行のパターン
        data_pattern = r'^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        
        for line in lines:
            match = re.match(data_pattern, line)
            if match:
                size = int(match.group(1))
                count = int(match.group(2))
                time_us = float(match.group(5))
                algo_bw = float(match.group(6))
                
                results.append({
                    'size_bytes': size,
                    'time_us': time_us,
                    'bandwidth_gbps': algo_bw
                })
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
    
    return results

def generate_summary(results, test_name):
    """結果のサマリーを生成"""
    if not results:
        return f"No results for {test_name}"
    
    bandwidths = [r['bandwidth_gbps'] for r in results]
    avg_bw = sum(bandwidths) / len(bandwidths)
    max_bw = max(bandwidths)
    
    # 最大帯域を達成したサイズを見つける
    max_entry = max(results, key=lambda x: x['bandwidth_gbps'])
    
    summary = f"""
### {test_name}
- Average Bandwidth: {avg_bw:.2f} Gbps
- Peak Bandwidth: {max_bw:.2f} Gbps
- Peak at size: {max_entry['size_bytes'] / (1024**3):.2f} GB
- Efficiency vs 400GbE x2: {(max_bw / 800) * 100:.1f}%
"""
    return summary

# メイン処理
if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("# NCCL Benchmark Analysis Report\n")
    
    # 各テスト結果を解析
    test_files = {
        "All-Reduce": f"{output_dir}/multinode_results.log",
        "Single Node": f"{output_dir}/single_node_results.log"
    }
    
    for test_name, filepath in test_files.items():
        try:
            results = parse_nccl_output(filepath)
            if results:
                print(generate_summary(results, test_name))
        except FileNotFoundError:
            print(f"### {test_name}: File not found")
    
    # JSON形式でも保存
    all_results = {}
    for test_name, filepath in test_files.items():
        try:
            all_results[test_name] = parse_nccl_output(filepath)
        except:
            pass
    
    with open(f"{output_dir}/nccl_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
EOF

chmod +x "$OUTPUT_DIR/analyze_results.py"

# 結果の解析
python3 "$OUTPUT_DIR/analyze_results.py" "$OUTPUT_DIR" > "$OUTPUT_DIR/analysis_report.md"

# 4. 最終レポートの生成
cat > "$OUTPUT_DIR/benchmark_report.md" << EOF
# NCCL Benchmark Report - SpectrumX H200 Cluster

生成日時: $(date)

## テスト構成
- ノード数: $NUM_NODES
- 総GPU数: $((NUM_NODES * 8))
- ネットワーク: SpectrumX 400GbE x2 (RoCEv2)
- テストサイズ: $MIN_SIZE - $MAX_SIZE

## テスト対象ノード
$(printf '%s\n' "${TEST_NODES[@]}" | sed 's/^/- /')

## 環境設定
\`\`\`bash
$(cat "$OUTPUT_DIR/nccl_env.sh")
\`\`\`

## 性能結果

$(cat "$OUTPUT_DIR/analysis_report.md" 2>/dev/null || echo "解析結果なし")

## 推奨事項

### 現在の性能が期待値以下の場合：
1. **ネットワーク設定の確認**
   - RoCEv2が正しく設定されているか確認
   - ECN/PFCが有効になっているか確認
   
2. **GPU-NIC アフィニティ**
   - GPUとNICの物理的な配置を最適化
   - NUMA配置の確認

3. **NCCL環境変数の調整**
   - NCCL_IB_QPS_PER_CONNECTION を増やす
   - NCCL_BUFFSIZE を調整

### ログファイル
- 単一ノード結果: single_node_results.log
- マルチノード結果: multinode_results.log
- 解析データ: nccl_results.json
EOF

# 完了メッセージ
echo ""
echo "============================================"
log_info "NCCLベンチマークが完了しました！"
echo "============================================"
echo ""
echo "結果ディレクトリ: $OUTPUT_DIR"
echo "レポート: $OUTPUT_DIR/benchmark_report.md"
echo ""

# 主要な結果の表示
if [ -f "$OUTPUT_DIR/multinode_results.log" ]; then
    PEAK_BW=$(grep "Avg bus bandwidth" "$OUTPUT_DIR/multinode_results.log" | tail -1 | awk '{print $NF}' 2>/dev/null)
    if [ -n "$PEAK_BW" ]; then
        log_info "ピーク帯域: $PEAK_BW Gbps"
        EFFICIENCY=$(echo "scale=1; $PEAK_BW / 800 * 100" | bc 2>/dev/null)
        log_info "ネットワーク効率: ${EFFICIENCY}% (理論値比)"
    fi
fi