# SpectrumX H200 Benchmark - 最適化環境変数一覧

## 1. NCCL環境変数（SpectrumX RoCEv2最適化）

```bash
# ========== 基本設定 ==========
# InfiniBandデバイスの指定（SpectrumX dual-port）
export NCCL_IB_HCA=mlx5_0,mlx5_1

# ネットワークインターフェース（bonding構成）
export NCCL_SOCKET_IFNAME=bond0

# RoCEv2 GIDインデックス
export NCCL_IB_GID_INDEX=3

# トラフィッククラス（QoS用）
export NCCL_IB_TC=106

# ========== パフォーマンス最適化 ==========
# QPs per connection（並列度向上）
export NCCL_IB_QPS_PER_CONNECTION=4

# バッファサイズ（8MB）
export NCCL_BUFFSIZE=8388608

# アルゴリズム選択
export NCCL_ALGO=Ring,Tree

# プロトコル選択（Simple for small, LL128 for large）
export NCCL_PROTO=Simple,LL128

# ========== SpectrumX固有設定 ==========
# SHARP collective無効化（SpectrumXでは未対応）
export NCCL_COLLNET_ENABLE=0

# Adaptive Routing有効化
export NCCL_IB_AR_THRESHOLD=0

# PCI relaxed ordering
export NCCL_IB_PCI_RELAXED_ORDERING=1

# ========== デバッグ・監視 ==========
# デバッグレベル
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET,COLL

# トポロジファイル
export NCCL_TOPO_FILE=/etc/nccl-topo.xml

# グラフダンプ（デバッグ用）
export NCCL_GRAPH_DUMP_FILE=/tmp/nccl-graph.xml
```

## 2. GPU/CUDA環境変数

```bash
# ========== GPU設定 ==========
# CUDA visible devices（全GPU使用）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU persistence mode
export CUDA_LAUNCH_BLOCKING=0

# ========== メモリ最適化 ==========
# H200 HBM3e最適化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# TensorCore使用
export CUDA_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1

# メモリプール設定
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9

# ========== NVLINK最適化 ==========
# P2P通信有効化
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# NVSwitchトポロジ最適化
export NCCL_NVLS_ENABLE=1
```

## 3. InfiniBand/RoCEv2環境変数

```bash
# ========== RoCEv2設定 ==========
# ECN (Explicit Congestion Notification)
export NCCL_IB_ECN_ENABLE=1

# RoCEv2 mode
export NCCL_IB_ROCE_VERSION_NUM=2

# ========== QoS設定 ==========
# Service Level
export NCCL_IB_SL=3

# DSCP値（Differentiated Services Code Point）
export NCCL_NET_GDR_LEVEL=5

# ========== パフォーマンスチューニング ==========
# Receive queue size
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=22

# GPU Direct RDMA
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GDR_LEVEL=5
```

## 4. DeepSpeed環境変数

```bash
# ========== 基本設定 ==========
# ランチャー設定
export DEEPSPEED_LAUNCHER=pdsh
export DEEPSPEED_SSH_PORT=44222

# ========== 通信最適化 ==========
# 通信バックエンド
export DS_COMM_BACKEND=nccl
export DS_ACCELERATOR=cuda

# ZeRO最適化
export DS_ZERO_STAGE=3
export DS_ZERO_REDUCE_SCATTER=true
export DS_ZERO_ALLGATHER_PARTITIONS=true
export DS_ZERO_ROUND_ROBIN_GRADIENTS=true

# ========== メモリ最適化 ==========
# CPU offload設定
export DS_ZERO_OFFLOAD_OPTIMIZER_DEVICE=none
export DS_ZERO_OFFLOAD_PARAM_DEVICE=none

# Activation checkpointing
export DS_ACTIVATION_CHECKPOINTING_PARTITION=true
```

## 5. PyTorch分散環境変数

```bash
# ========== 分散設定 ==========
# バックエンド
export TORCH_DISTRIBUTED_BACKEND=nccl

# タイムアウト（2時間）
export TORCH_DISTRIBUTED_TIMEOUT=7200

# ========== NCCL統合 ==========
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# ========== パフォーマンス ==========
# Fusion最適化
export TORCH_FUSION=1

# Cudnn benchmark
export TORCH_CUDNN_BENCHMARK=1
```

## 6. システム環境変数

```bash
# ========== ネットワークバッファ ==========
# TCPバッファサイズ（128MB）
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4

# ========== CPU親和性 ==========
# OpenMPスレッド
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ========== Huge Pages ==========
export NCCL_IB_DISABLE_HUGEPAGES=0
```

## 7. 監視・デバッグ環境変数

```bash
# ========== プロファイリング ==========
# NCCLプロファイリング
export NCCL_DEBUG_FILE=/tmp/nccl-debug-%h-%p.log

# PyTorchプロファイリング
export TORCH_PROFILER_ENABLED=0

# ========== ログ設定 ==========
# ログレベル
export NCCL_LOG_LEVEL=WARN
export GLOO_LOG_LEVEL=WARNING

# ========== メトリクス収集 ==========
# GPU使用率追跡
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## 8. アプリケーション固有環境変数

```bash
# ========== Hugging Face ==========
# トークン
export HF_TOKEN=your-token-here

# キャッシュディレクトリ
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# オフラインモード（必要時）
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# ========== Weights & Biases ==========
export WANDB_MODE=offline
export WANDB_DIR=/tmp/wandb
```

## 使用例：完全な環境設定

```bash
#!/bin/bash
# SpectrumX H200クラスタ用の完全な環境設定

# NCCLコア設定
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_BUFFSIZE=8388608
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple,LL128
export NCCL_COLLNET_ENABLE=0
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_DEBUG=WARN

# GPU/CUDA設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# RoCEv2最適化
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_IB_ECN_ENABLE=1
export NCCL_IB_SL=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=5

# DeepSpeed設定
export DS_COMM_BACKEND=nccl
export DS_ACCELERATOR=cuda
export DEEPSPEED_SSH_PORT=44222

# PyTorch設定
export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_BENCHMARK=1

# システム最適化
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close

echo "SpectrumX H200環境変数の設定が完了しました"
```

## パフォーマンスへの影響

| 環境変数カテゴリ | 期待される改善 |
|-----------------|--------------|
| NCCL最適化 | 通信遅延 20-30% 削減 |
| GPU/CUDA設定 | メモリ効率 15-20% 向上 |
| RoCEv2チューニング | 帯域幅利用率 90%+ |
| DeepSpeed統合 | スケーリング効率 95%+ |

## 注意事項

1. **ハードウェア依存**: これらの設定はSpectrumX 400GbE + H200 GPUに最適化されています
2. **検証必須**: 本番環境適用前に必ずテスト環境で検証してください
3. **モニタリング**: `NCCL_DEBUG=INFO`で詳細ログを確認できます
4. **調整**: ワークロードに応じて`NCCL_BUFFSIZE`や`NCCL_IB_QPS_PER_CONNECTION`の調整が必要な場合があります