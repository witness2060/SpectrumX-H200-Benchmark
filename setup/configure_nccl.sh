#!/bin/bash
set -euo pipefail

# クラスタ設定の読み込み
source "$(dirname "$0")/cluster_config.sh"

echo "=== Configuring NCCL for SpectrumX RoCEv2 ==="

# ノードの自動検出
detect_available_nodes || exit 1

# NCCL環境変数の設定
cat > /tmp/nccl_env.sh << 'EOF'
# SpectrumX 400GbE x2 最適化設定
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SL=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFFSIZE=8388608
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
EOF

# 全ノードに配布
echo "Distributing NCCL configuration to all nodes..."
for node in "${AVAILABLE_NODES[@]}"; do
    scp -o StrictHostKeyChecking=no /tmp/nccl_env.sh $node:/tmp/ 2>/dev/null
done

# 環境変数を.bashrcに追加
pdsh -w "$PDSH_NODES" "cat /tmp/nccl_env.sh >> ~/.bashrc" 2>/dev/null

# GPU設定の最適化
echo "Optimizing GPU settings..."
pdsh -w "$PDSH_NODES" "sudo nvidia-smi -pm 1" 2>/dev/null
pdsh -w "$PDSH_NODES" "sudo nvidia-smi -ac 2619,1980" 2>/dev/null || echo "Clock setting not supported"

# ネットワークバッファの最適化
echo "Optimizing network buffers..."
pdsh -w "$PDSH_NODES" "sudo sysctl -w net.core.rmem_max=134217728" 2>/dev/null
pdsh -w "$PDSH_NODES" "sudo sysctl -w net.core.wmem_max=134217728" 2>/dev/null
pdsh -w "$PDSH_NODES" "sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 134217728'" 2>/dev/null
pdsh -w "$PDSH_NODES" "sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 134217728'" 2>/dev/null

echo "NCCL configuration completed!"