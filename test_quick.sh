#!/bin/bash
set -euo pipefail

# ==============================================================================
# 簡易動作確認スクリプト
# ==============================================================================

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== SpectrumX H200 クラスタ簡易テスト ===${NC}"
echo ""

# クラスタ設定の読み込み
source setup/cluster_config.sh

# テスト結果
TESTS_PASSED=0
TESTS_FAILED=0

# テスト関数
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "テスト: $test_name ... "
    
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# 詳細テスト関数
run_detailed_test() {
    local test_name="$1"
    shift
    local test_commands=("$@")
    
    echo ""
    echo -e "${YELLOW}=== $test_name ===${NC}"
    
    for cmd in "${test_commands[@]}"; do
        echo "実行: $cmd"
        eval "$cmd" || true
        echo ""
    done
}

# 1. ノード接続性テスト
echo -e "${YELLOW}1. ノード接続性テスト${NC}"
detect_available_nodes
echo "利用可能なノード: ${AVAILABLE_NODES[*]}"
echo "ノード数: $NODE_COUNT"
echo ""

# 2. 基本的な動作確認
echo -e "${YELLOW}2. 基本動作確認${NC}"
run_test "cluster_config.sh の読み込み" "[ -n \"\$PDSH_NODES\" ]"
run_test "PDSHコマンドの存在" "command -v pdsh"
run_test "SSHキー認証" "ssh -o BatchMode=yes -o ConnectTimeout=3 ${AVAILABLE_NODES[0]} exit"
run_test "GPUドライバの確認" "pdsh -w \${AVAILABLE_NODES[0]} nvidia-smi"
echo ""

# 3. Python環境テスト
echo -e "${YELLOW}3. Python環境テスト${NC}"
if [ ${#AVAILABLE_NODES[@]} -gt 0 ]; then
    TEST_NODE="${AVAILABLE_NODES[0]}"
    
    run_test "Miniconda環境" "ssh $TEST_NODE '[ -f \$HOME/miniconda3/bin/conda ]'"
    run_test "h200-bench環境" "ssh $TEST_NODE 'source \$HOME/miniconda3/bin/activate h200-bench && python --version'"
    
    # パッケージの確認
    echo ""
    echo "インストール済みパッケージ確認 (${TEST_NODE}):"
    ssh "$TEST_NODE" "source \$HOME/miniconda3/bin/activate h200-bench && pip list | grep -E '(torch|deepspeed|transformers)' | head -5" 2>/dev/null || echo "パッケージ未インストール"
fi
echo ""

# 4. GPU情報の表示
if [ $NODE_COUNT -gt 0 ]; then
    run_detailed_test "GPU情報" \
        "pdsh -w \"\${AVAILABLE_NODES[0]}\" 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1'"
fi

# 5. ネットワーク接続テスト
if [ $NODE_COUNT -gt 1 ]; then
    echo -e "${YELLOW}5. ノード間ネットワークテスト${NC}"
    
    # ping テスト
    echo "ノード間のping応答時間:"
    for i in {0..2}; do
        if [ $i -lt $NODE_COUNT ]; then
            node="${AVAILABLE_NODES[$i]}"
            ping_time=$(ssh "${AVAILABLE_NODES[0]}" "ping -c 1 -W 1 $node 2>/dev/null | grep 'time=' | sed 's/.*time=//'" 2>/dev/null || echo "N/A")
            echo "  ${AVAILABLE_NODES[0]} → $node: $ping_time"
        fi
    done
    echo ""
fi

# 6. 簡易PyTorchテスト
if [ $NODE_COUNT -gt 0 ] && [ -f "scripts/run_benchmark_new.sh" ]; then
    echo -e "${YELLOW}6. PyTorch動作確認${NC}"
    
    # 単一ノードでの簡易テスト
    cat > /tmp/torch_test.py << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # 簡単な計算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU演算: 成功")
EOF
    
    echo "PyTorchテストを実行中..."
    scp /tmp/torch_test.py "${AVAILABLE_NODES[0]}:/tmp/" &>/dev/null
    ssh "${AVAILABLE_NODES[0]}" "source \$HOME/miniconda3/bin/activate h200-bench && python /tmp/torch_test.py" || echo "PyTorchテスト失敗"
    echo ""
fi

# 結果サマリー
echo -e "${YELLOW}=== テスト結果サマリー ===${NC}"
echo -e "成功: ${GREEN}$TESTS_PASSED${NC}"
echo -e "失敗: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}全てのテストが成功しました！${NC}"
    echo ""
    echo "次のステップ:"
    echo "  1. 完全なセットアップ: ./run_all.sh setup"
    echo "  2. ベンチマーク実行: ./run_all.sh bench"
    echo "  3. 全ステップ実行: ./run_all.sh all"
else
    echo -e "${RED}一部のテストが失敗しました。${NC}"
    echo "セットアップが必要な場合は './run_all.sh setup' を実行してください。"
fi

# クリーンアップ
rm -f /tmp/torch_test.py