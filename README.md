# SpectrumX H200 Benchmark

Automated SFT (Supervised Fine-Tuning) benchmark suite for NVIDIA H200 GPU clusters with SpectrumX 400GbE interconnect. Achieves 90%+ GPU utilization and 95%+ scaling efficiency.

## 🚀 特徴

- **動的ノード検出**: SSH接続可能なノードを自動的に検出
- **柔軟な構成**: ノード数、モデルサイズ、テストタイプを動的に指定可能
- **完全自動化**: 環境セットアップからベンチマーク実行、レポート生成まで一括実行
- **包括的なテスト**: NCCL通信、PyTorch分散処理、SFT訓練の各種ベンチマーク
- **プロダクション対応**: エラーハンドリング、ログ記録、メトリクス収集を完備

## 📋 前提条件

### ハードウェア要件
- NVIDIA H200 GPU搭載ノード（各ノード8GPU構成推奨）
- ノード間高速ネットワーク（InfiniBand または 400GbE以上推奨）
- 各ノード最低500GB以上のディスク容量

### ソフトウェア要件
- Ubuntu 20.04以上
- CUDA 12.1以上（ドライバー530.xx以上）
- Python 3.10
- SSH鍵認証設定済み（パスワードなしでノード間通信可能）
- PDSHインストール済み

## 🏃 クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-org/spectrumx-h200-benchmark.git
cd spectrumx-h200-benchmark
```

### 2. 動作確認

```bash
# クラスタの基本的な動作確認（SSHやGPUの確認）
./test_quick.sh
```

### 3. 自動セットアップと実行

```bash
# 全ての処理を自動実行（推奨）
# セットアップ → 検証 → ベンチマーク → レポート生成
./run_all.sh all
```

## 📖 詳細な使用方法

### 個別ステップの実行

```bash
# 1. 環境セットアップ（初回のみ必要）
./run_all.sh setup

# 2. クラスタ検証
./run_all.sh verify

# 3. ベンチマーク実行
./run_all.sh bench [モデル名] [ノード数] [テストタイプ]

# 4. レポート生成
./run_all.sh report
```

### ベンチマークオプション

#### モデルサイズ
- `meta-llama/Llama-2-7b-hf` - 7Bパラメータモデル（デフォルト）
- `meta-llama/Llama-2-13b-hf` - 13Bパラメータモデル
- `meta-llama/Llama-2-70b-hf` - 70Bパラメータモデル（要大容量メモリ）

#### テストタイプ
- `nccl` - NCCL通信ベンチマークのみ
- `pytorch` - PyTorch分散テストのみ
- `sft` - SFT訓練ベンチマーク（推奨）
- `full` - 全てのテストを実行（デフォルト）

### 実行例

```bash
# 2ノードで7Bモデルの完全なベンチマーク
./run_all.sh bench meta-llama/Llama-2-7b-hf 2 full

# 4ノードで13BモデルのSFT訓練のみ
./run_all.sh bench meta-llama/Llama-2-13b-hf 4 sft

# 8ノードで70BモデルのNCCL通信テストのみ
./run_all.sh bench meta-llama/Llama-2-70b-hf 8 nccl
```

### 環境変数による制御

#### 方法1: .envファイルを使用（推奨）
```bash
# サンプルファイルをコピー
cp .env.example .env

# .envファイルを編集
nano .env

# 主要な設定項目：
# HF_TOKEN=your-hugging-face-token
# DEFAULT_MODEL=meta-llama/Llama-2-7b-hf
# DEFAULT_NODES=4
# BATCH_SIZE_PER_GPU=8
# LEARNING_RATE=2e-4
```

#### 方法2: 環境変数を直接設定
```bash
# カスタムノードリストを指定
export CUSTOM_NODES="node001,node002,node003,node004"

# 確認プロンプトをスキップ（自動化用）
export SKIP_CONFIRM=true

# Hugging Faceトークン（プライベートモデル用）
export HF_TOKEN="your-token-here"

# デフォルトのモデルとパラメータ
export DEFAULT_MODEL="meta-llama/Llama-2-13b-hf"
export DEFAULT_NODES=4
export BATCH_SIZE_PER_GPU=4
export LEARNING_RATE=1e-4

# 実行
./run_all.sh all
```

#### 主要な環境変数一覧

| 環境変数 | 説明 | デフォルト値 |
|---------|------|------------|
| `HF_TOKEN` | Hugging Faceトークン | なし |
| `CUSTOM_NODES` | カスタムノードリスト | 自動検出 |
| `DEFAULT_MODEL` | デフォルトモデル | meta-llama/Llama-2-7b-hf |
| `DEFAULT_NODES` | デフォルトノード数 | 2 |
| `DEFAULT_TEST_TYPE` | デフォルトテストタイプ | sft |
| `BATCH_SIZE_PER_GPU` | GPUあたりバッチサイズ | 8 |
| `LEARNING_RATE` | 学習率 | 2e-4 |
| `NUM_EPOCHS` | エポック数 | 3 |
| `GRADIENT_ACCUMULATION_STEPS` | 勾配累積ステップ | 16 |
| `PRECISION` | 精度設定 | bf16 |
| `NCCL_DEBUG` | NCCLデバッグレベル | WARN |

詳細な設定は`.env.example`を参照してください。

## 📁 プロジェクト構造

```
spectrumx-h200-benchmark/
├── run_all.sh              # メイン実行スクリプト
├── test_quick.sh           # 簡易動作確認スクリプト
├── setup/                  # セットアップスクリプト
│   ├── cluster_config.sh   # クラスタ設定・ノード検出
│   ├── install_dependencies.sh     # システム依存関係
│   ├── install_python_packages.sh  # Python環境構築
│   ├── configure_nccl.sh   # NCCL最適化設定
│   └── verify_cluster.sh   # クラスタ検証
├── scripts/                # 実行スクリプト
│   ├── run_benchmark.sh    # ベンチマーク実行
│   ├── train_sft.py        # SFT訓練実装
│   ├── collect_metrics.sh  # メトリクス収集
│   ├── generate_report.py  # レポート生成
│   └── gpu_benchmark.py    # GPU性能測定
├── configs/                # 設定ファイル
│   ├── ds_config_7b.json   # 7B用DeepSpeed設定
│   ├── ds_config_13b.json  # 13B用DeepSpeed設定
│   ├── ds_config_70b.json  # 70B用DeepSpeed設定
│   └── nccl_config.conf    # NCCL設定
├── datasets/               # データセット準備
│   ├── prepare_dataset.py  # デフォルトデータセット準備
│   └── prepare_custom_dataset.py  # カスタムデータセット対応
├── results/                # ベンチマーク結果
└── docs/                   # ドキュメント
```

## 📊 データセットとモデルの管理

### データセットの準備

#### デフォルトデータセット（Stanford Alpaca）
自動的にダウンロードされます：
```bash
# ベンチマーク実行時に自動準備
./run_all.sh bench
```

#### カスタムデータセットの使用
```bash
# JSONL形式
python datasets/prepare_custom_dataset.py \
    --input your_data.jsonl \
    --input-type jsonl \
    --model meta-llama/Llama-2-7b-hf

# Hugging Faceから
python datasets/prepare_custom_dataset.py \
    --input "OpenAssistant/oasst1" \
    --input-type huggingface \
    --format chatgpt
```

### モデルのダウンロード

モデルは初回実行時に自動的にダウンロードされます。プライベートモデルの場合：
```bash
# Hugging Faceトークンを設定
export HF_TOKEN="your-hugging-face-token"

# Llama 2モデルを使用
./run_all.sh bench meta-llama/Llama-2-7b-hf 2 sft
```

モデルは`~/.cache/huggingface/`にキャッシュされ、2回目以降は高速に読み込まれます。

## 🔧 トラブルシューティング

### ノードが検出されない

```bash
# SSH接続を手動で確認
ssh node001 hostname

# ノードリストを手動で指定
export CUSTOM_NODES="node1,node2,node3"
./run_all.sh verify
```

### GPUが認識されない

```bash
# GPUドライバとCUDAの確認
nvidia-smi
nvcc --version

# 全ノードでGPU状態確認
pdsh -w "node1,node2" nvidia-smi
```

### Python環境エラー

```bash
# Conda環境のリセット
conda env remove -n h200-bench -y
./run_all.sh setup
```

### メモリ不足エラー

DeepSpeed設定ファイルのバッチサイズを調整：
- `configs/ds_config_*.json`の`train_micro_batch_size_per_gpu`を減らす
- gradient_accumulation_stepsを増やして同等の有効バッチサイズを維持

### NCCL通信エラー

```bash
# NCCL環境変数の確認
printenv | grep NCCL

# ネットワークインターフェースの確認
ip addr show
ibstat  # InfiniBandの場合
```

## 📊 結果の確認

ベンチマーク実行後、以下の場所に結果が保存されます：

- `results/YYYYMMDD_HHMMSS_*/` - 各実行の詳細ログとメトリクス
  - `benchmark.log` - 実行ログ
  - `summary.json` - 実行サマリー
  - `gpu_metrics.csv` - GPU使用率の時系列データ
  - `network_metrics.csv` - ネットワーク使用率
  - `pytorch_test_results.json` - PyTorchベンチマーク結果
  - `sft_training.log` - SFT訓練ログ（該当する場合）

レポート生成後：
- `docs/benchmark_report.md` - 統合レポート（Markdown形式）

## 🎯 パフォーマンス目標

このベンチマークスイートは以下の性能目標を想定しています：

- **GPU利用率**: 90%以上
- **スケーリング効率**: 
  - 2ノード: 95%以上
  - 4ノード: 90%以上
  - 8ノード: 85%以上
- **通信帯域**: 理論値の80%以上
- **メモリ効率**: OOMなしで最大バッチサイズを実現

## 🤝 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- NVIDIA社のH200 GPUおよびNCCLライブラリ
- PyTorchおよびDeepSpeedコミュニティ
- Hugging Face Transformersライブラリ

## 📞 サポート

問題が発生した場合や質問がある場合は、以下の方法でサポートを受けられます：

1. [Issues](https://github.com/your-org/spectrumx-h200-benchmark/issues)でバグ報告や機能要望
2. [Discussions](https://github.com/your-org/spectrumx-h200-benchmark/discussions)で質問や議論
3. プロジェクトのWikiでより詳細なドキュメント

## 📋 更新履歴

### 最新の改善点 (2024-01)
- **動的ノード検出**: SSH接続テストによる自動ノード検出機能
- **DeepSpeed統合改善**: コマンドライン引数の修正
- **GPU性能測定**: gpu_benchmark.pyの追加
- **データセット管理**: カスタムデータセット対応の強化
- **セットアップ改善**: install_all.shの実行順序を最適化

---

**注意**: このベンチマークスイートは大量の計算リソースを使用します。実行前に十分なリソースが利用可能であることを確認してください。