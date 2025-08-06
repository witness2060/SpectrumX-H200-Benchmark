# SpectrumX + H200 × 8ノード環境
# 最適化ベンチマーク実行ガイド

## 📋 目次

1. [概要](#概要)
2. [環境構築](#環境構築)
3. [ベンチマーク実行](#ベンチマーク実行)
4. [結果確認](#結果確認)
5. [トラブルシューティング](#トラブルシューティング)

---

## 概要

本ドキュメントは、SpectrumX + H200 × 8ノード環境で、Llama-2モデル（7B/13B/70B）のSFTベンチマークを実行し、営業資料用のレポートを生成するまでの完全な手順を説明します。

### 目標性能
- **GPU利用率**: 90%以上
- **スケーリング効率**: 95%以上（4ノードまで）
- **通信帯域**: 920 Gbps（理論値の92%）

---

## 環境構築

### 1. 事前確認

```bash
# マスターノードで実行
cd /root/test/spectrumx-h200-benchmark

# ノード接続確認
pdsh -w "node[001-008]" hostname

# GPU確認
pdsh -w "node[001-008]" "nvidia-smi -L" | head -16

# 期待される出力:
# GPU 0: NVIDIA H200 SXM (UUID: xxx)
# GPU 1: NVIDIA H200 SXM (UUID: xxx)
# ...（各ノード8GPU）
```

### 2. ソフトウェア環境セットアップ

#### 2.1 基本環境構築

```bash
# 一括セットアップスクリプト実行
./setup/install_all.sh

# または個別実行
./setup/install_dependencies.sh  # システム依存関係
./setup/install_python_packages.sh  # Python環境
./setup/configure_nccl.sh  # NCCL最適化
```

#### 2.2 NCCL環境変数の設定（重要）

```bash
# 全ノードに配布
cat > /tmp/nccl_env.sh << 'EOF'
# SpectrumX RoCEv2最適化設定
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_BUFFSIZE=8388608
export NCCL_ALGO=Ring,Tree
export NCCL_COLLNET_ENABLE=0
export NCCL_DEBUG=WARN
EOF

# 各ノードにコピー
for node in node001 node002 node003 node004 node005 node006 node007 node008; do
    scp /tmp/nccl_env.sh $node:/tmp/
    ssh $node "source /tmp/nccl_env.sh && echo 'source /tmp/nccl_env.sh' >> ~/.bashrc"
done
```

### 3. データセット準備（ローカルRAID配置）

```bash
# データセット作成
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source alpaca \
    --num-samples 50000 \
    --output-dir /raid/datasets/alpaca_tokenized

# 各ノードのローカルRAIDに配布
pdsh -w "node[001-008]" "mkdir -p /raid/datasets"

for node in node001 node002 node003 node004 node005 node006 node007 node008; do
    rsync -avz /raid/datasets/alpaca_tokenized $node:/raid/datasets/
    echo "✓ Data copied to $node"
done
```

### 4. Hugging Face認証（Llama-2使用時）

```bash
# トークン設定
./setup/configure_huggingface.sh

# Llama-2へのアクセス確認
python3 -c "
from transformers import AutoConfig
import os
config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                    token=os.environ.get('HF_TOKEN'))
print('✓ Llama-2 access confirmed')
"
```

---

## ベンチマーク実行

### 方法1: 完全自動実行（推奨）

```bash
# 全モデル・全ノード構成で自動実行
chmod +x scripts/spectrum_x_benchmark.sh
./scripts/spectrum_x_benchmark.sh

# 実行内容:
# - 2/4/8ノードでの実行
# - Llama-2 7B/13B/70Bの3モデル
# - 自動メトリクス収集
# - レポート自動生成
```

### 方法2: 個別実行

#### Step 1: NCCLベースラインテスト

```bash
# 2ノードテスト
mpirun -np 16 -H node001,node002 \
    --map-by ppr:8:node \
    /opt/nccl-tests/build/all_reduce_perf \
    -b 256 -e 32G -f 2 -g 1

# 期待される結果:
# 32 GB: ~920 Gbps (理論値の92%)
```

#### Step 2: 単一モデルベンチマーク

```bash
# Llama-2 7B, 2ノード
deepspeed --hostfile hostfile_2nodes \
    --num_nodes 2 \
    --num_gpus 16 \
    scripts/optimized_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_path /raid/datasets/alpaca_tokenized \
    --output_dir results/7b_2nodes \
    --deepspeed configs/ds_config_7b_optimized.json \
    --max_steps 700 \
    --warmup_steps 200 \
    --bf16 \
    --gradient_checkpointing
```

#### Step 3: GPU/ネットワークモニタリング

```bash
# 別ターミナルで実行

# GPU使用率モニタリング
pdsh -w "node[001-002]" \
    "nvidia-smi dmon -s pucvmt -i 0,1,2,3,4,5,6,7" &

# ネットワーク帯域モニタリング
pdsh -w "node[001-002]" \
    "sar -n DEV 1 | grep bond0" &
```

### 実行時間の目安

| モデル | ノード数 | 実行時間 |
|--------|---------|----------|
| Llama-2-7B | 2 | 約10分 |
| Llama-2-7B | 4 | 約8分 |
| Llama-2-7B | 8 | 約6分 |
| Llama-2-13B | 4 | 約15分 |
| Llama-2-70B | 8 | 約30分 |

**総実行時間**: 全組み合わせで約2-3時間

---

## 結果確認

### 1. パフォーマンスサマリー確認

```bash
# 最新の結果ディレクトリ
LATEST_RESULTS=$(ls -td results/spectrumx_benchmark_* | head -1)

# パフォーマンスサマリー表示
cat $LATEST_RESULTS/metrics/performance_summary.csv | column -t -s','

# 期待される出力:
# model_size  nodes  samples_per_sec  gpu_utilization
# 7           2      15234.5          92.3
# 7           4      29876.2          91.8
# 7           8      57234.1          90.5
```

### 2. スケーリング効率確認

```bash
# スケーリング分析結果
cat $LATEST_RESULTS/reports/scaling_analysis.json | python3 -m json.tool

# 主要指標の抽出
python3 << EOF
import json
with open('$LATEST_RESULTS/reports/scaling_analysis.json') as f:
    data = json.load(f)
    for model, results in data.items():
        print(f"\n{model}:")
        for config, metrics in results.items():
            print(f"  {config}: {metrics['efficiency']:.1f}% efficiency")
EOF
```

### 3. 営業資料の確認

```bash
# Markdownレポート表示
cat $LATEST_RESULTS/reports/SpectrumX_H200_Benchmark_Report.md

# HTML版を生成（オプション）
pandoc $LATEST_RESULTS/reports/SpectrumX_H200_Benchmark_Report.md \
    -o $LATEST_RESULTS/reports/report.html \
    --standalone --toc

# PDFに変換（要wkhtmltopdf）
wkhtmltopdf $LATEST_RESULTS/reports/report.html \
    $LATEST_RESULTS/reports/SpectrumX_H200_Benchmark.pdf
```

---

## トラブルシューティング

### 問題1: GPU利用率が低い（<90%）

```bash
# 原因調査
nvidia-smi -q -d UTILIZATION | grep -A 5 "GPU Utilization"

# 解決策
# 1. バッチサイズを増やす
sed -i 's/"train_micro_batch_size_per_gpu": 8/"train_micro_batch_size_per_gpu": 16/' \
    configs/ds_config_7b_optimized.json

# 2. Gradient Accumulation を減らす
sed -i 's/"gradient_accumulation_steps": 16/"gradient_accumulation_steps": 8/' \
    configs/ds_config_7b_optimized.json
```

### 問題2: NCCL通信エラー

```bash
# デバッグモード有効化
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# ネットワーク確認
pdsh -w "node[001-008]" "ibstatus" | grep -E "state|rate"
pdsh -w "node[001-008]" "ip link show bond0"

# RoCEv2設定確認
pdsh -w "node[001-008]" "cma_roce_mode -d mlx5_0"
```

### 問題3: Out of Memory

```bash
# 70Bモデル用の調整
# CPU Offloadを有効化
sed -i 's/"device": "none"/"device": "cpu"/' \
    configs/ds_config_70b_optimized.json

# Activation Checkpointingの調整
sed -i 's/"number_checkpoints": null/"number_checkpoints": 4/' \
    configs/ds_config_70b_optimized.json
```

### 問題4: スケーリング効率が低い

```bash
# ネットワーク最適化の再確認
# 1. Multi-rail設定
export NCCL_IB_HCA=mlx5_0,mlx5_1

# 2. QoS設定
sudo mlnx_qos -i mlx5_0 --trust dscp
sudo mlnx_qos -i mlx5_1 --trust dscp

# 3. ECN有効化
echo 1 | sudo tee /sys/class/net/mlx5_0/ecn/enable
echo 1 | sudo tee /sys/class/net/mlx5_1/ecn/enable
```

---

## ベストプラクティスまとめ

### ✅ 必須設定

1. **NCCL環境変数**: 必ず設定する
2. **ローカルRAID**: NFSは使わない
3. **BF16精度**: H200では必須
4. **Flash Attention 2**: 有効化必須

### 📊 推奨構成

| モデル | ノード数 | バッチサイズ | Gradient Accumulation |
|--------|---------|--------------|---------------------|
| 7B | 2-4 | 8-16 | 16 |
| 13B | 4 | 4-8 | 16 |
| 70B | 8 | 1-2 | 32 |

### 🎯 期待される結果

- **GPU利用率**: 92-95%
- **スケーリング効率**: 
  - 2→4ノード: 96%
  - 4→8ノード: 93%
- **通信帯域**: 920 Gbps（ピーク）

---

## サポート

技術的な問題が発生した場合は、以下の情報を含めてお問い合わせください：

1. 実行ログ: `results/*/benchmark.log`
2. GPU状態: `nvidia-smi -q`の出力
3. ネットワーク状態: `ibstatus`の出力
4. NCCL設定: `env | grep NCCL`の出力

---

*最終更新: 2025年8月*