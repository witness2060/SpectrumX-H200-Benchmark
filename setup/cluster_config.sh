#!/bin/bash
# Cluster Configuration - 動的にノード情報を管理

# マスターノード情報
export MASTER_NODE=$(hostname)
export MASTER_IP=$(hostname -i | awk '{print $1}')

# 現在利用可能なノード（動的に検出または手動設定）
# デフォルトのノードリスト
export DEFAULT_NODES=(
    "fukushimadc-02-hgx-0001:10.2.201.1"
    "fukushimadc-02-hgx-0002:10.2.201.2"
    "fukushimadc-02-hgx-0003:10.2.201.3"
    "fukushimadc-02-hgx-0004:10.2.201.4"
    "fukushimadc-02-hgx-0005:10.2.201.5"
    "fukushimadc-02-hgx-0006:10.2.201.6"
    "fukushimadc-02-hgx-0007:10.2.201.7"
    "fukushimadc-02-hgx-0009:10.2.201.9"
)

# ノードの動的検出関数
detect_available_nodes() {
    local available_nodes=()
    echo "=== ノードの可用性を確認中 ==="
    
    for node_info in "${DEFAULT_NODES[@]}"; do
        local hostname=$(echo "$node_info" | cut -d: -f1)
        local ip=$(echo "$node_info" | cut -d: -f2)
        
        # SSH接続テスト（タイムアウト3秒、ポート44222）
        if timeout 3 ssh -p 44222 -o StrictHostKeyChecking=no -o ConnectTimeout=2 "$hostname" "exit" 2>/dev/null; then
            available_nodes+=("$hostname")
            echo "✓ $hostname ($ip) - 利用可能"
        else
            echo "✗ $hostname ($ip) - 利用不可"
        fi
    done
    
    # 利用可能なノードを環境変数に設定
    export AVAILABLE_NODES=("${available_nodes[@]}")
    export NODE_COUNT=${#AVAILABLE_NODES[@]}
    
    # PDSHフォーマットのノードリスト生成
    if [ ${#AVAILABLE_NODES[@]} -gt 0 ]; then
        export PDSH_NODES=$(IFS=,; echo "${AVAILABLE_NODES[*]}")
        echo ""
        echo "=== 利用可能なノード数: $NODE_COUNT ==="
        echo "PDSH_NODES=$PDSH_NODES"
    else
        echo "エラー: 利用可能なノードが見つかりません"
        return 1
    fi
}

# ノード設定ファイルの生成
generate_node_files() {
    local node_file="/tmp/available_nodes.txt"
    local hostfile="/tmp/hostfile"
    
    # 通常のノードリスト
    printf "%s\n" "${AVAILABLE_NODES[@]}" > "$node_file"
    
    # DeepSpeed用hostfile
    > "$hostfile"
    for node in "${AVAILABLE_NODES[@]}"; do
        echo "$node slots=8" >> "$hostfile"
    done
    
    export NODE_FILE="$node_file"
    export HOSTFILE="$hostfile"
}

# カスタムノードリストの設定（オプション）
set_custom_nodes() {
    local custom_nodes="$1"
    if [ -n "$custom_nodes" ]; then
        IFS=',' read -ra AVAILABLE_NODES <<< "$custom_nodes"
        export NODE_COUNT=${#AVAILABLE_NODES[@]}
        export PDSH_NODES="$custom_nodes"
        generate_node_files
        echo "カスタムノードリストを使用: $PDSH_NODES"
    fi
}

# GPU情報の取得
get_gpu_info() {
    local node="${1:-$MASTER_NODE}"
    ssh "$node" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1" 2>/dev/null || echo "Unknown GPU"
}

# クラスタ情報のサマリー表示
show_cluster_summary() {
    echo ""
    echo "==============================================="
    echo "           クラスタ構成サマリー"
    echo "==============================================="
    echo "マスターノード: $MASTER_NODE ($MASTER_IP)"
    echo "利用可能ノード数: $NODE_COUNT"
    echo "合計GPU数: $((NODE_COUNT * 8))"
    echo ""
    echo "GPU情報:"
    local gpu_info=$(get_gpu_info)
    echo "  $gpu_info"
    echo ""
    echo "ノードリスト:"
    for node in "${AVAILABLE_NODES[@]}"; do
        echo "  - $node"
    done
    echo "==============================================="
}

# 環境変数のエクスポート（他のスクリプトから使用）
export -f detect_available_nodes
export -f generate_node_files
export -f set_custom_nodes
export -f get_gpu_info
export -f show_cluster_summary