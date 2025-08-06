#!/bin/bash
set -euo pipefail

# クラスタ設定の読み込み
source "$(dirname "$0")/cluster_config.sh"

echo "=== Python環境のセットアップ ==="

# ノードの自動検出
detect_available_nodes || exit 1

# Conda環境名
CONDA_ENV="h200-bench"

# Python環境のセットアップ関数
setup_python_env() {
    echo "=== 全ノードでPython環境を構築中 ==="
    
    pdsh -w "$PDSH_NODES" << 'EOF'
# Miniconda環境の確認と有効化
if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
    echo "エラー: Minicondaが見つかりません。install_dependencies.shを先に実行してください。"
    exit 1
fi

source $HOME/miniconda3/bin/activate

# 既存の環境を削除（クリーンインストール）
conda env remove -n h200-bench -y 2>/dev/null || true

# 新しい環境を作成
conda create -n h200-bench python=3.10 -y

# 環境を有効化
source $HOME/miniconda3/bin/activate h200-bench

# 基本パッケージのインストール
echo "=== 基本パッケージをインストール中 ==="
pip install --upgrade pip setuptools wheel

# PyTorch 2.3.0 with CUDA 12.1
echo "=== PyTorchをインストール中 ==="
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# DeepSpeed と関連パッケージ
echo "=== DeepSpeedをインストール中 ==="
pip install deepspeed==0.14.0

# Transformers関連
echo "=== Transformersをインストール中 ==="
pip install transformers==4.41.0 accelerate>=0.30.0 datasets peft trl

# 追加ツール
echo "=== 追加パッケージをインストール中 ==="
pip install wandb scipy scikit-learn pandas matplotlib seaborn
pip install sentencepiece protobuf

# Flash Attention 2のインストール（エラーを許容）
echo "=== Flash Attention 2をインストール中 ==="
pip install ninja packaging
MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation 2>/dev/null || echo "Flash Attention 2のインストールをスキップ"

# インストール確認
echo "=== インストール確認 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "Python環境のセットアップが完了しました！"
EOF
}

# 環境確認関数
verify_python_env() {
    echo ""
    echo "=== Python環境の確認 ==="
    
    pdsh -w "$PDSH_NODES" "source \$HOME/miniconda3/bin/activate h200-bench && python -c 'import torch; print(f\"{torch.cuda.get_device_name(0)} - PyTorch {torch.__version__}\")'" 2>/dev/null || echo "確認エラー"
    
    echo ""
    echo "=== インストール済みパッケージのサマリー ==="
    ssh "${AVAILABLE_NODES[0]}" "source \$HOME/miniconda3/bin/activate h200-bench && pip list | grep -E '(torch|deepspeed|transformers|accelerate)'"
}

# メイン処理
main() {
    echo "利用可能なノード: $PDSH_NODES"
    echo ""
    
    read -p "Python環境をセットアップしますか？ [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_python_env
        verify_python_env
        
        echo ""
        echo "=== セットアップ完了 ==="
        echo "以下のコマンドで環境を有効化できます："
        echo "source \$HOME/miniconda3/bin/activate h200-bench"
    else
        echo "セットアップをキャンセルしました"
        exit 0
    fi
}

# 引数処理
if [ $# -eq 0 ]; then
    main
else
    case "$1" in
        --verify)
            verify_python_env
            ;;
        --nodes)
            set_custom_nodes "$2"
            main
            ;;
        *)
            echo "使用方法: $0 [--verify] [--nodes ノードリスト]"
            echo "  --verify: インストール済み環境の確認のみ"
            echo "  --nodes: カスタムノードリストを指定 (例: node001,node002)"
            exit 1
            ;;
    esac
fi