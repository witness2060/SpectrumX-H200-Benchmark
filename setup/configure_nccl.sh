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

# RoCEv2 QoS設定
echo "Configuring RoCEv2 QoS settings..."
pdsh -w "$PDSH_NODES" << 'REMOTE_EOF'
# RoCEv2 QoS設定（インターフェースが存在する場合のみ実行）
if [ -d "/sys/class/net/mlx5_0" ]; then
    sudo mlnx_qos -i mlx5_0 --trust dscp 2>/dev/null || echo "mlnx_qos not available for mlx5_0"
    sudo cma_roce_mode -d mlx5_0 -m 2 2>/dev/null || echo "cma_roce_mode not available for mlx5_0"
fi

if [ -d "/sys/class/net/mlx5_1" ]; then
    sudo mlnx_qos -i mlx5_1 --trust dscp 2>/dev/null || echo "mlnx_qos not available for mlx5_1"
    sudo cma_roce_mode -d mlx5_1 -m 2 2>/dev/null || echo "cma_roce_mode not available for mlx5_1"
fi

# Bonding設定確認
if [ -f "/proc/net/bonding/bond0" ]; then
    echo "=== Bond0 Configuration ==="
    cat /proc/net/bonding/bond0 | grep -E "Bonding Mode|MII Status|Slave Interface" | head -10
fi
REMOTE_EOF

# NCCL設定ファイルの作成（オプション）
echo "Creating NCCL configuration file..."
pdsh -w "$PDSH_NODES" << 'REMOTE_EOF'
sudo tee /etc/nccl.conf << 'NCCL_EOF'
# SpectrumX 400GbE x2 設定
NCCL_IB_HCA=mlx5_0,mlx5_1
NCCL_SOCKET_IFNAME=bond0
NCCL_IB_GID_INDEX=3
NCCL_IB_TC=106
NCCL_IB_QPS_PER_CONNECTION=4
NCCL_BUFFSIZE=8388608
NCCL_ALGO=Ring,Tree
NCCL_COLLNET_ENABLE=0
NCCL_DEBUG=WARN
NCCL_TOPO_FILE=/etc/nccl-topo.xml
NCCL_EOF
REMOTE_EOF

echo "NCCL configuration completed!"