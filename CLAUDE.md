# CLAUDE.md - SpectrumX H200×8ノード SFTベンチマーク完全自動化

## プロジェクト概要

このプロジェクトは、NVIDIA H200 SXM GPUを搭載した8ノードクラスタ（SpectrumX 400GbE接続）において、大規模言語モデルのSFT（Supervised Fine-Tuning）ベンチマークを完全自動化します。PDSHとSlurmを使用してマスターノードから全ての操作を一括実行し、GPU利用率90%以上、スケール効率95%以上を達成することを目標とします。

## システム構成

### ハードウェア構成
- **GPU**: NVIDIA H200 SXM (HBM3e 141GB, 4.8TB/s)
- **ノード数**: 最大8ノード（各ノード8GPU = 最大64GPU）
- **ノード間接続**: SpectrumX 400GbE × 2ポート (RoCEv2)
- **ノード内接続**: NVLink 5.0 / NVSwitch
- **SSHポート**: 44222（標準の22番ではない）

### ノード構成
- マスターノード: fukushimadc-02-hgx-0001 (10.2.201.1)
- 計算ノード: fukushimadc-02-hgx-0002〜0007, 0009（0008は存在しない）
- 汎用エイリアス: node001〜node008（node008は0009にマッピング）

## プロジェクト目標

1. **GPU利用率**: 90%以上
2. **スケーリング効率**: 
   - 2ノード: 98%
   - 4ノード: 95%
   - 8ノード: 90%
3. **通信帯域**: 920 Gbps (dual 400GbE)
4. **レイテンシ**: < 50μs (2ノード間)

## ディレクトリ構造

```
spectrumx-h200-benchmark/
├── setup/                  # セットアップスクリプト
│   ├── install_dependencies.sh     # 依存関係インストール
│   ├── configure_nccl.sh           # NCCL最適化設定
│   ├── setup_slurm.sh              # Slurm設定
│   └── verify_cluster.sh           # クラスタ検証
├── configs/                # 設定ファイル
│   ├── ds_config_7b.json   # 7B用DeepSpeed設定
│   ├── ds_config_13b.json  # 13B用DeepSpeed設定
│   ├── ds_config_70b.json  # 70B用DeepSpeed設定
│   └── node_mapping.json   # ノード名マッピング
├── scripts/                # 実行スクリプト
│   ├── run_benchmark.sh    # ベンチマーク実行
│   ├── collect_metrics.sh  # メトリクス収集
│   └── generate_report.py  # レポート生成
├── datasets/               # データセット準備
│   └── prepare_dataset.py
├── results/                # ベンチマーク結果
└── docs/                   # ドキュメント
```

## 主要コンポーネント

### 1. 環境セットアップ (`setup/`)

#### install_dependencies.sh
- CUDA 12.5とDriver 560.xxの確認
- Miniconda環境の作成
- PyTorch 2.3 with CUDA 12.5
- DeepSpeed 0.14.0
- Flash Attention 2
- NCCL 2.26.2のビルド（Sharp対応）

#### configure_nccl.sh
- SpectrumX 400GbE x2 設定
- RoCEv2 QoS設定
- ネットワークバッファの最適化
- NCCL環境変数の設定

#### setup_slurm.sh
- Slurmプロローグスクリプトの作成
- GPU-CPUアフィニティの設定
- Huge Pagesの設定

### 2. ベンチマーク実行 (`scripts/`)

#### run_benchmark.sh
- マルチノードでのSFTベンチマーク実行
- NCCLテスト、PyTorchテスト、SFT訓練の選択的実行
- 動的ノード検出とホストファイル生成
- メトリクス収集の自動化

#### collect_metrics.sh
- GPU使用率の収集
- ネットワーク使用率の収集
- CPU/メモリ使用率の収集
- InfiniBand統計の収集

#### generate_report.py
- ベンチマーク結果の解析
- スケーリング効率の計算
- Markdownレポートの生成

### 3. 設定ファイル (`configs/`)

#### DeepSpeed設定
- **ds_config_7b.json**: バッチサイズ1024、ZeRO Stage 3、BF16
- **ds_config_13b.json**: バッチサイズ512、メモリ最適化
- **ds_config_70b.json**: バッチサイズ256、CPU オフローディング有効

### 4. データセット準備 (`datasets/`)

#### prepare_dataset.py
- Stanford Alpacaデータセットの自動ダウンロード
- Alpacaフォーマットへの変換
- トークナイゼーションと保存

## 実行手順

### 1. 初期セットアップ（一度だけ実行）
```bash
# 依存関係のインストール
./setup/install_dependencies.sh

# NCCL設定
./setup/configure_nccl.sh

# Slurm設定（オプション）
./setup/setup_slurm.sh

# クラスタ検証
./setup/verify_cluster.sh
```

### 2. ベンチマーク実行
```bash
# 2ノードでのベンチマーク
./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2

# 4ノードでのベンチマーク
./scripts/run_benchmark.sh meta-llama/Llama-2-13b-hf 4

# 8ノードでのベンチマーク
./scripts/run_benchmark.sh meta-llama/Llama-2-70b-hf 8
```

### 3. レポート生成
```bash
python scripts/generate_report.py results/ docs/final_benchmark_report.md
```

## 最適化手法

### 通信最適化
- NCCL 2.26.2 with Sharp プラグイン
- RoCEv2 with ECN + PFC (lossless)
- Multi-rail 400GbE × 2
- Buffer size: 8MB

### DeepSpeed最適化
- ZeRO Stage 3 with overlap communication
- Gradient accumulation: 16 steps
- BF16 mixed precision
- Activation checkpointing

### 並列化戦略
- データ並列: ZeRO-3
- モデル並列: 必要に応じてTensor Parallel追加
- Gradient accumulation でバッチサイズ調整

## トラブルシューティング

### GPU認識エラー
```bash
pdsh -w "node[001-008]" "sudo nvidia-smi -pm 1; sudo nvidia-smi -r"
```

### NCCL通信エラー
```bash
pdsh -w "node[001-008]" "printenv | grep NCCL"
pdsh -w "node[001-008]" "ip addr show bond0"
```

### OOMエラー
configs/ds_config_*.json の train_micro_batch_size_per_gpu を小さくする

## 期待される結果

- **GPU利用率**: 92-95%
- **スケーリング効率**: 
  - 2ノード: 98%
  - 4ノード: 95%
  - 8ノード: 90%
- **通信帯域**: 920 Gbps (dual 400GbE)
- **レイテンシ**: < 50μs (2ノード間)

## 追加の最適化オプション

1. **Flash Attention 3**の有効化（利用可能な場合）
2. **FP8**精度の使用（H200でサポート）
3. **SHARP Collective**の有効化（スイッチがサポートする場合）
4. **GPU Direct RDMA**の最適化

## 注意事項

- 初回実行時はモデルのダウンロードに時間がかかります
- 各ノードに十分なディスク容量（500GB以上）が必要です
- Hugging Faceのトークンが必要な場合があります（プライベートモデル使用時）
- 実行前に必ずGPUドライバーとCUDAのバージョンを確認してください
- SSHポート44222を使用しているため、手動接続時は `ssh -p 44222` を使用