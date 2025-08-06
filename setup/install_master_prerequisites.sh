#!/bin/bash
set -euo pipefail

# =========================================
# マスターノード前提条件の自動インストール
# =========================================

# カラー出力の設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

echo "==========================================="
echo " マスターノード前提条件セットアップ"
echo "==========================================="
echo ""

# OSチェック
if ! grep -qiE 'ubuntu|debian' /etc/os-release; then
    log_error "このスクリプトはUbuntu/Debian専用です"
    exit 1
fi

# root権限チェック
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log_info "sudo権限が必要です。パスワードを入力してください："
        sudo -v
    fi
}

# 1. システムパッケージの更新
log_step "1/5 システムパッケージを更新しています..."
check_sudo
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# 2. PDSHのインストール
log_step "2/5 PDSHをインストールしています..."
if ! command -v pdsh &> /dev/null; then
    log_info "PDSHが見つかりません。インストールします..."
    sudo apt-get install -y pdsh pdsh-rcmd-ssh
    
    # PDSHのデフォルトリモートコマンドをSSHに設定
    echo "export PDSH_RCMD_TYPE=ssh" >> ~/.bashrc
    export PDSH_RCMD_TYPE=ssh
    
    log_info "PDSHがインストールされました: $(which pdsh)"
else
    log_info "PDSHは既にインストールされています: $(which pdsh)"
fi

# PDSHモジュールの確認
pdsh -V | grep -E "available rcmd modules|loaded modules" || true

# 3. 必須ツールのインストール
log_step "3/5 必須ツールをインストールしています..."
REQUIRED_PACKAGES=(
    "openssh-server"
    "openssh-client"
    "sshpass"
    "expect"
    "git"
    "wget"
    "curl"
    "python3"
    "python3-pip"
    "jq"
    "bc"
    "htop"
    "iftop"
    "screen"
    "tmux"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii.*$package"; then
        log_info "$package をインストールしています..."
        sudo apt-get install -y "$package"
    else
        log_info "$package は既にインストールされています"
    fi
done

# 4. SSH設定の最適化
log_step "4/5 SSH設定を最適化しています..."
SSH_CONFIG="/etc/ssh/sshd_config"
SSH_CONFIG_BACKUP="${SSH_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"

# バックアップ作成
sudo cp "$SSH_CONFIG" "$SSH_CONFIG_BACKUP"
log_info "SSH設定のバックアップを作成しました: $SSH_CONFIG_BACKUP"

# SSH設定の最適化
configure_ssh() {
    local key="$1"
    local value="$2"
    
    if sudo grep -q "^#*${key}" "$SSH_CONFIG"; then
        sudo sed -i "s/^#*${key}.*/${key} ${value}/" "$SSH_CONFIG"
    else
        echo "${key} ${value}" | sudo tee -a "$SSH_CONFIG" > /dev/null
    fi
}

# 高速化とセキュリティのための設定
configure_ssh "MaxSessions" "100"
configure_ssh "MaxStartups" "100:30:200"
configure_ssh "ClientAliveInterval" "60"
configure_ssh "ClientAliveCountMax" "3"
configure_ssh "UseDNS" "no"
configure_ssh "GSSAPIAuthentication" "no"

# SSH再起動
sudo systemctl restart sshd
log_info "SSH設定が最適化されました"

# 5. Pythonツールのインストール
log_step "5/8 Python管理ツールをインストールしています..."
pip3 install --user --upgrade pip
pip3 install --user ansible fabric3 paramiko

# 6. 監視ツールのインストール
log_step "6/8 GPU監視ツールをインストールしています..."
pip3 install --user nvidia-ml-py gpustat nvitop

# 7. NCCL Testsのビルド
log_step "7/8 NCCL Testsをビルドしています..."
if [ ! -d "/opt/nccl-tests" ]; then
    log_info "NCCL Testsをビルドします..."
    cd /tmp
    git clone https://github.com/NVIDIA/nccl-tests.git
    cd nccl-tests
    make -j$(nproc) MPI=0 CUDA_HOME=/usr/local/cuda || {
        log_warn "NCCL Testsのビルドに失敗しました。CUDAが必要です。"
    }
    if [ -f "build/all_reduce_perf" ]; then
        sudo mkdir -p /opt
        sudo mv /tmp/nccl-tests /opt/
        log_info "NCCL Testsが /opt/nccl-tests にインストールされました"
    fi
else
    log_info "NCCL Testsは既にインストールされています"
fi

# 8. クラスタ設定ファイルの作成
log_step "8/8 クラスタ設定ファイルを作成しています..."
# PDSHホストファイルの作成
cat > /tmp/hosts.txt << 'EOF'
fukushimadc-02-hgx-0001
fukushimadc-02-hgx-0002
fukushimadc-02-hgx-0003
fukushimadc-02-hgx-0004
fukushimadc-02-hgx-0005
fukushimadc-02-hgx-0006
fukushimadc-02-hgx-0007
fukushimadc-02-hgx-0009
EOF

# .pdshrcの作成（存在しない場合）
if [ ! -f ~/.pdshrc ]; then
    cat > ~/.pdshrc << 'EOF'
# PDSH Configuration for SpectrumX H200 Cluster
export PDSH_SSH_ARGS="-p 44222 -o StrictHostKeyChecking=no -o ConnectTimeout=10"
export PDSH_RCMD_TYPE=ssh
export WCOLL=/tmp/hosts.txt

# Node groups
all_nodes() {
    echo "fukushimadc-02-hgx-0001,fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009"
}

compute_nodes() {
    echo "fukushimadc-02-hgx-0002,fukushimadc-02-hgx-0003,fukushimadc-02-hgx-0004,fukushimadc-02-hgx-0005,fukushimadc-02-hgx-0006,fukushimadc-02-hgx-0007,fukushimadc-02-hgx-0009"
}
EOF
    log_info ".pdshrcファイルを作成しました"
fi

# 環境変数の設定
if ! grep -q "export PATH=\$HOME/.local/bin:\$PATH" ~/.bashrc; then
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
fi

# インストールサマリーの生成
cat > /tmp/master_prerequisites_summary.txt << EOF
SpectrumX H200 Master Node Prerequisites Installation Summary
============================================================
Date: $(date)
Host: $(hostname)

Installed Components:
- System packages: pdsh, ssh tools, monitoring utilities
- Python packages: ansible, fabric3, paramiko, gpustat, nvitop
- NCCL tests: /opt/nccl-tests (if CUDA available)
- Configuration files: ~/.pdshrc, /tmp/hosts.txt

Cluster Nodes:
$(cat /tmp/hosts.txt | sed 's/^/  - /')

Next Steps:
1. source ~/.bashrc && source ~/.pdshrc
2. Test PDSH connectivity: pdsh -w fukushimadc-02-hgx-0002 hostname
3. Run cluster verification: ./setup/verify_cluster.sh
4. Start benchmarks: ./scripts/run_benchmark.sh
EOF

# 完了メッセージ
echo ""
echo "==========================================="
log_info "マスターノードの前提条件インストールが完了しました！"
echo "==========================================="
echo ""
echo "インストールされたツール:"
echo "  - PDSH: $(pdsh -V 2>&1 | head -n1)"
echo "  - SSH: $(ssh -V 2>&1)"
echo "  - Python3: $(python3 --version)"
echo "  - 監視ツール: gpustat, nvitop"
if [ -d "/opt/nccl-tests" ]; then
    echo "  - NCCL Tests: /opt/nccl-tests"
fi
echo ""
echo "設定ファイル:"
echo "  - ~/.pdshrc: PDSH設定"
echo "  - /tmp/hosts.txt: ノードリスト"
echo ""
echo "次のステップ:"
echo "  1. source ~/.bashrc && source ~/.pdshrc  # 環境変数を再読み込み"
echo "  2. pdsh -w fukushimadc-02-hgx-0002 hostname  # 接続テスト"
echo "  3. ./setup/verify_cluster.sh  # クラスタ検証"
echo ""
echo "サマリーファイル: /tmp/master_prerequisites_summary.txt"