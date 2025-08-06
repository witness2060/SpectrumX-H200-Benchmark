#!/bin/bash
set -euo pipefail

# =========================================
# Slurm ジョブスケジューラーのセットアップ
# =========================================

echo "=== Setting up Slurm for H200 cluster ==="

# 権限チェック
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo or as root"
    exit 1
fi

# Slurmがインストールされているか確認
if ! command -v slurmctld &> /dev/null; then
    echo "Installing Slurm..."
    apt-get update
    apt-get install -y slurm-wlm slurm-client slurmctld slurmd
fi

# Slurm設定ファイルの作成
echo "Creating Slurm configuration..."
cat > /etc/slurm/slurm.conf << 'EOF'
# Slurm configuration for H200 cluster
ClusterName=h200-cluster
SlurmctldHost=fukushimadc-02-hgx-0001

# スケジューリング設定
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory,CR_CORE_DEFAULT_DIST_BLOCK

# プロセス設定
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity

# ログ設定
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmdDebug=3

# タイムアウト設定
SlurmdTimeout=300
MessageTimeout=60
KillWait=30
MinJobAge=300
SlurmctldTimeout=120

# GPU設定
GresTypes=gpu
AccountingStorageTRES=gres/gpu

# ノード設定（8ノード × 8GPU）
NodeName=fukushimadc-02-hgx-0001 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0002 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0003 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0004 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0005 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0006 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0007 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN
NodeName=fukushimadc-02-hgx-0009 Gres=gpu:8 CPUs=128 RealMemory=512000 State=UNKNOWN

# パーティション設定
PartitionName=h200-bench Nodes=fukushimadc-02-hgx-000[1-7],fukushimadc-02-hgx-0009 Default=YES MaxTime=INFINITE State=UP
EOF

# GPU設定ファイルの作成
echo "Creating GPU configuration..."
cat > /etc/slurm/gres.conf << 'EOF'
# GPU configuration for H200
NodeName=fukushimadc-02-hgx-0001 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0002 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0003 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0004 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0005 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0006 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0007 Name=gpu Type=h200 File=/dev/nvidia[0-7]
NodeName=fukushimadc-02-hgx-0009 Name=gpu Type=h200 File=/dev/nvidia[0-7]
EOF

# cgroupの設定
echo "Configuring cgroups..."
cat > /etc/slurm/cgroup.conf << 'EOF'
# Cgroup configuration
CgroupAutomount=yes
CgroupReleaseAgentDir="/etc/slurm/cgroup"
ConstrainCores=yes
ConstrainDevices=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
TaskAffinity=yes
EOF

# プロローグスクリプトの作成（GPU初期化）
echo "Creating prolog script..."
cat > /etc/slurm/prolog.sh << 'EOF'
#!/bin/bash
# GPU初期化とパフォーマンス設定

# GPUパーシステンスモードを有効化
nvidia-smi -pm 1

# GPUクロックを最大に設定（H200用）
nvidia-smi -ac 2619,1980 2>/dev/null || true

# Huge Pagesの設定
echo 128 > /proc/sys/vm/nr_hugepages

# ネットワークバッファの最適化
echo 134217728 > /proc/sys/net/core/rmem_max
echo 134217728 > /proc/sys/net/core/wmem_max

# NUMA最適化
numactl --hardware > /tmp/numa_info.txt
EOF

chmod +x /etc/slurm/prolog.sh

# エピローグスクリプトの作成（クリーンアップ）
echo "Creating epilog script..."
cat > /etc/slurm/epilog.sh << 'EOF'
#!/bin/bash
# ジョブ終了後のクリーンアップ

# GPUメモリのクリア
nvidia-smi --gpu-reset 2>/dev/null || true

# 一時ファイルのクリーンアップ
rm -f /tmp/nccl-* 2>/dev/null
rm -f /dev/shm/nccl-* 2>/dev/null
EOF

chmod +x /etc/slurm/epilog.sh

# ログディレクトリの作成
mkdir -p /var/log/slurm
chown -R slurm:slurm /var/log/slurm

# Munge（認証）の設定
echo "Setting up Munge authentication..."
if ! command -v munge &> /dev/null; then
    apt-get install -y munge
fi

# Mungeキーの生成（存在しない場合）
if [ ! -f /etc/munge/munge.key ]; then
    dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi

# サービスの起動
echo "Starting services..."
systemctl enable munge
systemctl start munge
systemctl enable slurmctld
systemctl start slurmctld
systemctl enable slurmd
systemctl start slurmd

# 状態確認
echo ""
echo "=== Slurm Status ==="
scontrol show partition
sinfo -N

echo ""
echo "=== Slurm setup completed! ==="
echo ""
echo "Usage examples:"
echo "  Submit job: sbatch job_script.sh"
echo "  Interactive: srun -N2 --gres=gpu:8 --pty bash"
echo "  Check queue: squeue"
echo "  Node info: sinfo -N"