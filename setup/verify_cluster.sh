#!/bin/bash
set -euo pipefail

# クラスタ設定の読み込み
source "$(dirname "$0")/cluster_config.sh"

echo "=== SpectrumX H200 Cluster Verification ==="
echo "Date: $(date)"
echo "Master Node: $MASTER_NODE ($MASTER_IP)"
echo ""

# ノードの自動検出
echo "=== Node Connectivity Check ==="
detect_available_nodes || exit 1

echo ""
echo "Connected nodes: ${AVAILABLE_NODES[*]}"

echo ""
echo "=== GPU Status Check ==="
pdsh -w "$PDSH_NODES" "nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,cuda_version --format=csv,noheader | head -1" 2>/dev/null | sort

echo ""
echo "=== Network Interface Check ==="
pdsh -w "$PDSH_NODES" "ip link show | grep -E 'mlx5|bond0' | head -2" 2>/dev/null | sort | head -10

echo ""
echo "=== Python Environment Check ==="
pdsh -w "$PDSH_NODES" "python3 --version 2>/dev/null || echo 'Python3 not found'" 2>/dev/null | sort | uniq

echo ""
echo "=== CUDA Check ==="
pdsh -w "$PDSH_NODES" "nvcc --version 2>/dev/null | grep release || echo 'NVCC not found'" 2>/dev/null | sort | uniq

echo ""
echo "=== Cluster Summary ==="
TOTAL_GPUS=$(pdsh -w "$PDSH_NODES" "nvidia-smi --query-gpu=count --format=csv,noheader" 2>/dev/null | wc -l)
echo "Total GPUs in cluster: $TOTAL_GPUS"
echo "Total nodes: 8"
echo "GPUs per node: 8"
echo "Total theoretical GPUs: 64"

echo ""
echo "=== Verification Complete ==="