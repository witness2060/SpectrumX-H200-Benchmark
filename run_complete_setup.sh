#!/bin/bash
set -euo pipefail

# ============================================
# SpectrumX H200 完全セットアップスクリプト
# ============================================
# このスクリプトは、マスターノードから実行して
# クラスタ全体を自動的にセットアップします

echo "============================================"
echo " SpectrumX H200 Complete Setup"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo "============================================"
echo ""

# プロジェクトディレクトリ
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# ログ関数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 実行前確認
echo "このスクリプトは以下の処理を実行します："
echo "  1. マスターノードの前提条件インストール"
echo "  2. PDSHの全ノード設定"
echo "  3. 全ノードへの依存関係インストール"
echo "  4. NCCLとネットワーク設定"
echo "  5. クラスタ検証テスト"
echo ""

if [ "${SKIP_CONFIRM:-false}" != "true" ]; then
    read -p "続行しますか? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "中止しました"
        exit 0
    fi
fi

# ログディレクトリの作成
LOG_DIR="$PROJECT_DIR/logs/setup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 実行ログの記録開始
exec > >(tee -a "$LOG_DIR/setup.log")
exec 2>&1

# 1. マスターノードの前提条件インストール
log_step "1/6 マスターノードの前提条件をインストールしています..."
if [ -x "$PROJECT_DIR/setup/install_master_prerequisites.sh" ]; then
    bash "$PROJECT_DIR/setup/install_master_prerequisites.sh" 2>&1 | tee "$LOG_DIR/master_prerequisites.log"
else
    log_error "install_master_prerequisites.sh が見つかりません"
    exit 1
fi

# 環境の再読み込み
source ~/.bashrc 2>/dev/null || true
if [ -f ~/.pdshrc ]; then
    source ~/.pdshrc
fi

# 2. PDSHの全ノード設定
log_step "2/6 PDSHを全ノードに設定しています..."
if [ -x "$PROJECT_DIR/setup/setup_pdsh_all_nodes.sh" ]; then
    bash "$PROJECT_DIR/setup/setup_pdsh_all_nodes.sh" 2>&1 | tee "$LOG_DIR/pdsh_setup.log"
else
    log_error "setup_pdsh_all_nodes.sh が見つかりません"
    exit 1
fi

# PDSHの設定を再読み込み
source ~/.pdshrc 2>/dev/null || true

# 3. 全ノードへの依存関係インストール
log_step "3/6 全ノードに依存関係をインストールしています..."
if [ -x "$PROJECT_DIR/setup/install_dependencies.sh" ]; then
    bash "$PROJECT_DIR/setup/install_dependencies.sh" 2>&1 | tee "$LOG_DIR/dependencies.log"
else
    log_warn "install_dependencies.sh が見つかりません - スキップします"
fi

# 4. Pythonパッケージのインストール
log_step "4/6 Pythonパッケージをインストールしています..."
if [ -x "$PROJECT_DIR/setup/install_python_packages.sh" ]; then
    bash "$PROJECT_DIR/setup/install_python_packages.sh" 2>&1 | tee "$LOG_DIR/python_packages.log"
else
    log_warn "install_python_packages.sh が見つかりません - スキップします"
fi

# 5. NCCLとネットワーク設定
log_step "5/6 NCCLとネットワークを設定しています..."
if [ -x "$PROJECT_DIR/setup/configure_nccl.sh" ]; then
    bash "$PROJECT_DIR/setup/configure_nccl.sh" 2>&1 | tee "$LOG_DIR/nccl_config.log"
else
    log_warn "configure_nccl.sh が見つかりません - スキップします"
fi

# 6. クラスタ検証
log_step "6/6 クラスタを検証しています..."
if [ -x "$PROJECT_DIR/setup/verify_cluster.sh" ]; then
    bash "$PROJECT_DIR/setup/verify_cluster.sh" 2>&1 | tee "$LOG_DIR/verification.log"
else
    log_warn "verify_cluster.sh が見つかりません"
fi

# Node007の性能問題チェック（オプション）
if [ "${CHECK_NODE007:-false}" == "true" ]; then
    log_info "Node007の診断を実行しています..."
    if [ -x "$PROJECT_DIR/scripts/diagnose_node007.sh" ]; then
        bash "$PROJECT_DIR/scripts/diagnose_node007.sh" 2>&1 | tee "$LOG_DIR/node007_diagnosis.log"
    fi
fi

# セットアップサマリーの生成
cat > "$LOG_DIR/setup_summary.md" << EOF
# SpectrumX H200 Setup Summary

Date: $(date)
Host: $(hostname)

## Setup Status

| Component | Status | Log File |
|-----------|--------|----------|
| Master Prerequisites | ✓ | master_prerequisites.log |
| PDSH Setup | ✓ | pdsh_setup.log |
| Dependencies | ✓ | dependencies.log |
| Python Packages | ✓ | python_packages.log |
| NCCL Config | ✓ | nccl_config.log |
| Cluster Verification | ✓ | verification.log |

## Quick Test Commands

\`\`\`bash
# PDSHテスト
source ~/.pdshrc
pdsh_all hostname

# GPU確認
pdsh_all nvidia-smi -L

# NCCLテスト（2ノード）
./scripts/run_nccl_benchmark.sh 2

# SFTベンチマーク（2ノード）
./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2
\`\`\`

## Environment

- SSH Port: 44222
- Python: Miniconda h200-bench environment
- NCCL: Optimized for SpectrumX 400GbE
- Nodes: 8 (fukushimadc-02-hgx-0001~0007, 0009)

## Next Steps

1. Run performance benchmarks
2. Monitor with \`gpustat\` or \`nvitop\`
3. Check logs in: $LOG_DIR
EOF

# 完了メッセージ
echo ""
echo "============================================"
log_info "セットアップが完了しました！"
echo "============================================"
echo ""
echo "セットアップログ: $LOG_DIR"
echo "サマリー: $LOG_DIR/setup_summary.md"
echo ""
echo "次のステップ:"
echo "  1. source ~/.bashrc && source ~/.pdshrc"
echo "  2. pdsh_all hostname  # 接続確認"
echo "  3. ./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2"
echo ""

# 最終チェック
if command -v pdsh &>/dev/null && [ -f ~/.pdshrc ]; then
    source ~/.pdshrc
    log_info "PDSHの動作確認..."
    if pdsh -w "${AVAILABLE_NODES[1]:-fukushimadc-02-hgx-0002}" hostname 2>/dev/null; then
        log_info "✓ PDSHは正常に動作しています"
    else
        log_warn "PDSHの接続に問題がある可能性があります"
        log_warn "SSH鍵の配布を確認してください"
    fi
else
    log_error "PDSHのセットアップに問題があります"
fi