# SpectrumX H200 Benchmark - Migration Guide

## プロジェクト統合に関する変更点

### 2024年1月の大規模リファクタリング

プロジェクトの保守性と使いやすさを向上させるため、以下の変更を行いました。

## 1. 重複スクリプトの統合

### 廃止されたスクリプト
以下のスクリプトは`run_benchmark.sh`と`run_all.sh`に統合されました：

| 廃止されたスクリプト | 新しいコマンド |
|---------------------|---------------|
| `scripts/run_actual_training.sh` | `./run_all.sh bench MODEL NODES sft` |
| `scripts/run_distributed_training.sh` | `./run_all.sh bench MODEL NODES full` |
| `scripts/run_h200_benchmark.sh` | `./run_all.sh bench MODEL NODES full` |
| `scripts/run_multinode_benchmark.sh` | `./run_all.sh bench MODEL NODES full` |
| `scripts/run_real_benchmark.sh` | `./run_all.sh bench MODEL NODES sft` |

### 移行例

```bash
# 以前のコマンド
./scripts/run_actual_training.sh

# 新しいコマンド
./run_all.sh bench meta-llama/Llama-2-7b-hf 2 sft
```

```bash
# 以前のコマンド
./scripts/run_multinode_benchmark.sh 4

# 新しいコマンド
./run_all.sh bench meta-llama/Llama-2-7b-hf 4 full
```

## 2. 新しい設定ファイル

### `.env.example`
環境変数の管理を簡単にするため、`.env.example`ファイルを追加しました：

```bash
# .envファイルを作成
cp .env.example .env

# 必要な設定を編集
nano .env
```

主要な設定項目：
- `HF_TOKEN`: Hugging Faceトークン
- `SSH_PORT`: SSH接続ポート（デフォルト: 44222）
- `DEFAULT_MODEL`: デフォルトのモデル
- `DEFAULT_NODES`: デフォルトのノード数

### `configs/node_mapping.json`
ノード名のマッピングを明確にするため、対応表を作成しました：

```json
{
  "node001": "fukushimadc-02-hgx-0001",
  "node002": "fukushimadc-02-hgx-0002",
  ...
  "node008": "fukushimadc-02-hgx-0009"  // 0008は利用不可
}
```

## 3. 統合されたメインコマンド

### `run_all.sh` - 推奨される実行方法

```bash
# 完全な自動実行
./run_all.sh all

# 個別ステップの実行
./run_all.sh setup    # 環境セットアップ
./run_all.sh verify   # クラスタ検証
./run_all.sh bench    # ベンチマーク実行
./run_all.sh report   # レポート生成
```

### `scripts/run_benchmark.sh` - 詳細制御用

```bash
# 直接実行する場合
./scripts/run_benchmark.sh MODEL NODES TEST_TYPE

# 例：4ノードで13BモデルのSFT訓練
./scripts/run_benchmark.sh meta-llama/Llama-2-13b-hf 4 sft
```

## 4. ドキュメントの改善

### README.mdの更新
- SSHポート44222の明記
- ノード構成セクションの追加
- 環境変数設定の詳細説明

### CLAUDE.mdとの整合性
CLAUDE.mdは初期設計を示すリファレンスとして保持されています。
実際の使用方法はREADME.mdを参照してください。

## 5. ディレクトリ構造

```
scripts/
├── run_benchmark.sh        # メインベンチマークスクリプト
├── train_sft.py           # SFT訓練実装
├── collect_metrics.sh     # メトリクス収集
├── generate_report.py     # レポート生成
└── DEPRECATED/            # 廃止されたスクリプト（参照用）
    └── README.md          # 廃止理由の説明
```

## トラブルシューティング

### Q: 以前のスクリプトが動作しない
A: `scripts/DEPRECATED/`に移動されています。新しいコマンドを使用してください。

### Q: 環境変数が認識されない
A: `.env`ファイルを作成し、必要な変数を設定してください。

### Q: ノードに接続できない
A: SSHポート44222が使用されていることを確認してください。

## サポート

問題が発生した場合は、以下を確認してください：

1. このマイグレーションガイド
2. README.mdの最新版
3. `.env.example`の設定例

---

最終更新: 2024年1月