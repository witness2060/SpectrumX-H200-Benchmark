#!/bin/bash
set -euo pipefail

# ==============================================================================
# SpectrumX H200 Benchmark - 統合実行スクリプト
# ==============================================================================

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ロゴ表示
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                   SpectrumX H200 Benchmark Suite                      ║
║                    Multi-Node SFT Training Benchmark                  ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 基本設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 環境変数の読み込み
source setup/load_env.sh

# クラスタ設定の読み込み
source setup/cluster_config.sh

# デフォルト値（環境変数から取得、引数で上書き可能）
ACTION="${1:-help}"
MODEL="${2:-$DEFAULT_MODEL}"
NODES="${3:-$DEFAULT_NODES}"
TEST_TYPE="${4:-$DEFAULT_TEST_TYPE}"

# ログ関数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
使用方法: $0 [アクション] [オプション]

アクション:
  setup     - 環境のセットアップ（依存関係のインストール）
  verify    - クラスタの検証
  bench     - ベンチマークの実行
  clean     - 結果のクリーンアップ
  report    - レポートの生成
  all       - setup → verify → bench → report の順に全て実行

ベンチマークオプション:
  $0 bench [モデル名] [ノード数] [テストタイプ]
  
  モデル名:
    - meta-llama/Llama-2-7b-hf (デフォルト)
    - meta-llama/Llama-2-13b-hf
    - meta-llama/Llama-2-70b-hf
    
  ノード数: 1-8 (デフォルト: 2)
  
  テストタイプ:
    - nccl    : NCCLベンチマークのみ
    - pytorch : PyTorch分散テストのみ
    - sft     : SFT訓練ベンチマーク
    - full    : 全てのテストを実行 (デフォルト)

例:
  $0 setup                                    # 環境セットアップ
  $0 verify                                   # クラスタ検証
  $0 bench                                    # デフォルト設定でベンチマーク実行
  $0 bench meta-llama/Llama-2-13b-hf 4 sft   # 13Bモデルで4ノードSFT
  $0 all                                      # 全ステップを実行

環境変数:
  CUSTOM_NODES  - カスタムノードリスト (例: export CUSTOM_NODES="node1,node2")
  SKIP_CONFIRM  - 確認プロンプトをスキップ
EOF
}

# 確認プロンプト
confirm() {
    local message="$1"
    if [ "${SKIP_CONFIRM:-}" != "true" ]; then
        read -p "$message [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    return 0
}

# 環境セットアップ
run_setup() {
    log_info "環境セットアップを開始します"
    
    # ノード検出
    detect_available_nodes || exit 1
    show_cluster_summary
    
    if ! confirm "上記のノードで環境をセットアップしますか？"; then
        log_warn "セットアップをキャンセルしました"
        return 1
    fi
    
    # 依存関係のインストール
    log_info "システム依存関係をインストール中..."
    setup/install_dependencies.sh || {
        log_error "依存関係のインストールに失敗しました"
        return 1
    }
    
    # Python環境のセットアップ
    log_info "Python環境をセットアップ中..."
    setup/install_python_packages.sh || {
        log_error "Python環境のセットアップに失敗しました"
        return 1
    }
    
    # NCCL設定
    log_info "NCCLを設定中..."
    setup/configure_nccl.sh || {
        log_error "NCCL設定に失敗しました"
        return 1
    }
    
    log_info "セットアップが完了しました！"
}

# クラスタ検証
run_verify() {
    log_info "クラスタの検証を開始します"
    
    setup/verify_cluster.sh || {
        log_error "クラスタ検証に失敗しました"
        return 1
    }
    
    log_info "クラスタ検証が完了しました"
}

# ベンチマーク実行
run_benchmark() {
    local model="$1"
    local nodes="$2"
    local test_type="$3"
    
    log_info "ベンチマークを開始します"
    log_info "モデル: $model"
    log_info "ノード数: $nodes"
    log_info "テストタイプ: $test_type"
    
    # ノード検出
    detect_available_nodes || exit 1
    
    if [ $nodes -gt $NODE_COUNT ]; then
        log_error "要求されたノード数 ($nodes) が利用可能なノード数 ($NODE_COUNT) を超えています"
        return 1
    fi
    
    if ! confirm "ベンチマークを開始しますか？"; then
        log_warn "ベンチマークをキャンセルしました"
        return 1
    fi
    
    # ベンチマーク実行
    scripts/run_benchmark.sh "$model" "$nodes" "$test_type" || {
        log_error "ベンチマーク実行に失敗しました"
        return 1
    }
    
    log_info "ベンチマークが完了しました"
}

# レポート生成
run_report() {
    log_info "レポート生成を開始します"
    
    if [ ! -d "results" ]; then
        log_error "結果ディレクトリが見つかりません"
        return 1
    fi
    
    python scripts/generate_report.py || {
        log_error "レポート生成に失敗しました"
        return 1
    }
    
    log_info "レポートが生成されました: docs/benchmark_report.md"
}

# クリーンアップ
run_clean() {
    log_info "結果のクリーンアップを開始します"
    
    if confirm "results/ディレクトリを削除しますか？"; then
        rm -rf results/*
        log_info "クリーンアップが完了しました"
    else
        log_warn "クリーンアップをキャンセルしました"
    fi
}

# 全ステップ実行
run_all() {
    log_info "全ステップを実行します"
    
    # カスタムノードリストの確認
    if [ -n "${CUSTOM_NODES:-}" ]; then
        log_info "カスタムノードリストを使用: $CUSTOM_NODES"
        set_custom_nodes "$CUSTOM_NODES"
    fi
    
    # 各ステップの実行
    run_setup || return 1
    echo ""
    
    run_verify || return 1
    echo ""
    
    run_benchmark "$MODEL" "$NODES" "$TEST_TYPE" || return 1
    echo ""
    
    run_report || return 1
    
    log_info "全ステップが完了しました！"
}

# メイン処理
main() {
    case "$ACTION" in
        setup)
            run_setup
            ;;
        verify)
            run_verify
            ;;
        bench|benchmark)
            run_benchmark "$MODEL" "$NODES" "$TEST_TYPE"
            ;;
        report)
            run_report
            ;;
        clean)
            run_clean
            ;;
        all)
            run_all
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            log_error "不明なアクション: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# エラーハンドリング
trap 'log_error "エラーが発生しました (行 $LINENO)"' ERR

# 実行
main