# SpectrumX H200 SFTベンチマーク 完全セットアップガイド

## 🚀 クイックスタート

```bash
# 1. プロジェクトディレクトリに移動
cd /root/test/spectrumx-h200-benchmark

# 2. 完全自動セットアップの実行
./run_complete_setup.sh

# 3. 環境の再読み込み
source ~/.bashrc && source ~/.pdshrc

# 4. 接続テスト
pdsh_all hostname

# 5. ベンチマーク実行（例：2ノード）
./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2
```

## 📋 前提条件チェックリスト

### マスターノード要件
- [ ] Ubuntu 22.04 LTS
- [ ] CUDA 12.1以上
- [ ] SSH ポート 44222 でのアクセス
- [ ] sudo権限
- [ ] インターネット接続

### クラスタ構成
- [ ] マスターノード: fukushimadc-02-hgx-0001 (10.2.201.1)
- [ ] 計算ノード: fukushimadc-02-hgx-0002〜0007, 0009
- [ ] 各ノード: NVIDIA H200 GPU × 8
- [ ] ネットワーク: SpectrumX 400GbE × 2 (RoCEv2)

## 🔧 手動セットアップ手順

### Step 1: マスターノードの準備

```bash
# 前提条件のインストール
./setup/install_master_prerequisites.sh

# 環境の読み込み
source ~/.bashrc
```

### Step 2: PDSHの設定

```bash
# 全ノードにPDSHを設定
./setup/setup_pdsh_all_nodes.sh

# 設定の読み込み
source ~/.pdshrc

# 接続テスト
pdsh_all hostname
```

### Step 3: 全ノードへの依存関係インストール

```bash
# システムパッケージ
./setup/install_dependencies.sh

# Pythonパッケージ
./setup/install_python_packages.sh

# NCCL設定
./setup/configure_nccl.sh
```

### Step 4: クラスタ検証

```bash
# 基本的な検証
./setup/verify_cluster.sh

# Node007の診断（必要な場合）
./scripts/diagnose_node007.sh
```

## 🏃 ベンチマーク実行

### 基本的なSFTベンチマーク

```bash
# 2ノードでLlama-7Bのテスト
./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2

# 4ノードでLlama-13Bのテスト
./scripts/run_benchmark.sh meta-llama/Llama-2-13b-hf 4

# 8ノードでLlama-70Bのテスト
./scripts/run_benchmark.sh meta-llama/Llama-2-70b-hf 8
```

### NCCLベンチマーク

```bash
# 2ノードでのNCCLテスト
./scripts/run_nccl_benchmark.sh 2

# 4ノードでのNCCLテスト（詳細モード）
./scripts/run_nccl_benchmark.sh 4 8 8G
```

## 🔍 トラブルシューティング

### PDSHが動作しない

```bash
# SSH鍵の確認
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 各ノードに鍵をコピー
for node in fukushimadc-02-hgx-{0002..0007} fukushimadc-02-hgx-0009; do
    ssh-copy-id -p 44222 $node
done

# PDSHの再設定
./setup/setup_pdsh_all_nodes.sh
```

### GPUが認識されない

```bash
# 全ノードでGPUステータス確認
pdsh_all nvidia-smi -L

# ドライバーの再初期化
pdsh_all "sudo nvidia-smi -pm 1; sudo nvidia-smi -r"
```

### Node007の性能問題

```bash
# 診断スクリプトの実行
./scripts/diagnose_node007.sh

# 電力制限の調整
ssh -p 44222 fukushimadc-02-hgx-0007 "sudo nvidia-smi -pl 700"
```

### HF_TOKEN エラー

```bash
# .envファイルの作成
cat > .env << EOF
HF_TOKEN=your_huggingface_token_here
EOF

# または環境変数として設定
export HF_TOKEN=your_huggingface_token_here
```

## 📊 期待される性能

### 単一ノード（8 GPU）
| モデル | バッチサイズ/GPU | 期待スループット |
|--------|-----------------|----------------|
| Llama-7B | 16 | ~1,280 samples/sec |
| Llama-13B | 8 | ~640 samples/sec |
| Llama-70B | 2 | ~160 samples/sec |

### スケーリング効率
| ノード数 | 期待効率 | 実効倍率 |
|---------|---------|---------|
| 1 | 100% | 1.0x |
| 2 | 95% | 1.9x |
| 4 | 90% | 3.6x |
| 8 | 80% | 6.4x |

### NCCL通信性能
- ピーク帯域: ~720 Gbps (理論値800Gbpsの90%)
- レイテンシ: <50μs (2ノード間)

## 📁 プロジェクト構造

```
spectrumx-h200-benchmark/
├── setup/                    # セットアップスクリプト
│   ├── install_master_prerequisites.sh  # マスターノード設定
│   ├── setup_pdsh_all_nodes.sh         # PDSH一括設定
│   ├── install_dependencies.sh          # 依存関係
│   ├── configure_nccl.sh               # NCCL最適化
│   └── verify_cluster.sh               # クラスタ検証
├── scripts/                  # 実行スクリプト
│   ├── run_benchmark.sh                # メインベンチマーク
│   ├── run_nccl_benchmark.sh          # NCCL性能測定
│   ├── diagnose_node007.sh            # Node007診断
│   └── generate_report.py             # レポート生成
├── configs/                  # 設定ファイル
│   ├── ds_config_*.json               # DeepSpeed設定
│   └── sft_config.yaml                # SFT訓練設定
├── results/                  # 実行結果
└── docs/                     # ドキュメント

```

## 🔐 セキュリティ考慮事項

1. **SSH設定**
   - ポート44222を使用（標準の22ではない）
   - StrictHostKeyCheckingは開発環境のため無効化
   - 本番環境では適切に設定すること

2. **認証情報**
   - HF_TOKENは.envファイルで管理
   - .envファイルは.gitignoreに追加推奨

## 🆘 サポート

問題が発生した場合：

1. ログファイルを確認: `logs/` ディレクトリ
2. プロジェクトステータスを確認: `PROJECT_STATUS_REPORT.md`
3. Node007の問題: `results/node007_diagnosis.md`

## ✅ 最終チェックリスト

- [ ] 全ノードでPDSHが動作する
- [ ] 全ノードでGPUが8個認識される
- [ ] NCCLテストが成功する
- [ ] 2ノードでのSFTベンチマークが動作する
- [ ] Node007の性能が他ノードの80%以上

これで準備完了です！🎉