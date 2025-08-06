#!/bin/bash
set -euo pipefail

# クラスタ設定の読み込み
source "$(dirname "$0")/../setup/cluster_config.sh"

OUTPUT_DIR="${1:-results/metrics}"
INTERVAL="${2:-5}"
DURATION="${3:-300}"  # デフォルト5分間

mkdir -p "$OUTPUT_DIR"

echo "Starting metrics collection to $OUTPUT_DIR"
echo "Interval: ${INTERVAL}s, Duration: ${DURATION}s"

# 利用可能なノードを使用（既に検出済みの場合）
if [ -z "${PDSH_NODES:-}" ]; then
    detect_available_nodes || exit 1
fi
NODES="$PDSH_NODES"

# バックグラウンドでメトリクス収集を開始
END_TIME=$(($(date +%s) + DURATION))

# GPU使用率の収集
(
    echo "timestamp,node,gpu_index,gpu_name,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,temperature" > "$OUTPUT_DIR/gpu_metrics.csv"
    while [ $(date +%s) -lt $END_TIME ]; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        pdsh -w "$NODES" "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null | \
        while read line; do
            node=$(echo $line | cut -d: -f1)
            data=$(echo $line | cut -d: -f2-)
            echo "$timestamp,$node,$data" >> "$OUTPUT_DIR/gpu_metrics.csv"
        done
        sleep $INTERVAL
    done
) &
GPU_PID=$!

# ネットワーク使用率の収集
(
    echo "timestamp,node,interface,rx_bytes,tx_bytes,rx_packets,tx_packets" > "$OUTPUT_DIR/network_metrics.csv"
    while [ $(date +%s) -lt $END_TIME ]; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        pdsh -w "$NODES" "cat /sys/class/net/bond0/statistics/rx_bytes /sys/class/net/bond0/statistics/tx_bytes /sys/class/net/bond0/statistics/rx_packets /sys/class/net/bond0/statistics/tx_packets | xargs echo" 2>/dev/null | \
        while read line; do
            node=$(echo $line | cut -d: -f1)
            data=$(echo $line | cut -d: -f2)
            echo "$timestamp,$node,bond0,$data" | tr ' ' ',' >> "$OUTPUT_DIR/network_metrics.csv"
        done
        sleep $INTERVAL
    done
) &
NET_PID=$!

# CPU使用率の収集
(
    echo "timestamp,node,cpu_usage,load_1min,load_5min,load_15min" > "$OUTPUT_DIR/cpu_metrics.csv"
    while [ $(date +%s) -lt $END_TIME ]; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        pdsh -w "$NODES" "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - \$1}' && cat /proc/loadavg | awk '{print \$1,\$2,\$3}'" 2>/dev/null | \
        while read line; do
            node=$(echo $line | cut -d: -f1)
            cpu_usage=$(echo $line | cut -d: -f2 | head -1)
            load=$(echo $line | cut -d: -f2 | tail -1)
            echo "$timestamp,$node,$cpu_usage,$load" | tr ' ' ',' >> "$OUTPUT_DIR/cpu_metrics.csv"
        done
        sleep $INTERVAL
    done
) &
CPU_PID=$!

# メモリ使用率の収集
(
    echo "timestamp,node,total_mb,used_mb,free_mb,available_mb" > "$OUTPUT_DIR/memory_metrics.csv"
    while [ $(date +%s) -lt $END_TIME ]; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        pdsh -w "$NODES" "free -m | grep '^Mem:' | awk '{print \$2,\$3,\$4,\$7}'" 2>/dev/null | \
        while read line; do
            node=$(echo $line | cut -d: -f1)
            data=$(echo $line | cut -d: -f2)
            echo "$timestamp,$node,$data" | tr ' ' ',' >> "$OUTPUT_DIR/memory_metrics.csv"
        done
        sleep $INTERVAL
    done
) &
MEM_PID=$!

# InfiniBand統計の収集（利用可能な場合）
(
    echo "timestamp,node,port,port_xmit_data,port_rcv_data,port_xmit_packets,port_rcv_packets" > "$OUTPUT_DIR/ib_metrics.csv"
    while [ $(date +%s) -lt $END_TIME ]; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        # perfqueryコマンドが存在するかチェック
        pdsh -w "$NODES" "which perfquery >/dev/null 2>&1 && perfquery -x 2>/dev/null | grep -E 'PortXmitData|PortRcvData|PortXmitPkts|PortRcvPkts' | awk '{print \$NF}' | paste -sd ',' || echo 'N/A,N/A,N/A,N/A'" 2>/dev/null | \
        while read line; do
            node=$(echo $line | cut -d: -f1)
            data=$(echo $line | cut -d: -f2)
            if [ "$data" != "N/A,N/A,N/A,N/A" ]; then
                echo "$timestamp,$node,1,$data" >> "$OUTPUT_DIR/ib_metrics.csv"
            fi
        done
        sleep $INTERVAL
    done
) &
IB_PID=$!

echo "Metrics collection started with PIDs:"
echo "  GPU: $GPU_PID"
echo "  Network: $NET_PID"
echo "  CPU: $CPU_PID"
echo "  Memory: $MEM_PID"
echo "  InfiniBand: $IB_PID"

# 待機またはCtrl+Cで終了
trap "kill $GPU_PID $NET_PID $CPU_PID $MEM_PID $IB_PID 2>/dev/null; echo 'Metrics collection stopped'" EXIT

if [ "$DURATION" -gt 0 ]; then
    echo "Collecting metrics for $DURATION seconds..."
    sleep $DURATION
else
    echo "Press Ctrl+C to stop metrics collection"
    wait
fi

echo "Metrics collection completed. Files saved to $OUTPUT_DIR"