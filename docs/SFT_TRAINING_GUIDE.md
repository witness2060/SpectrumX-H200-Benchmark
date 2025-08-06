# SFT (Supervised Fine-Tuning) 完全ガイド

## 📚 目次
1. [概要](#概要)
2. [Hugging Faceモデルの準備](#hugging-faceモデルの準備)
3. [データセットの準備](#データセットの準備)
4. [SFT訓練の実行](#sft訓練の実行)
5. [ベンチマークテスト](#ベンチマークテスト)
6. [トラブルシューティング](#トラブルシューティング)

## 概要

このガイドでは、H200 GPUクラスタで大規模言語モデル（LLM）のSFT（教師あり微調整）を実行する完全な手順を説明します。

### SFTとは？
- **Supervised Fine-Tuning**: 事前学習済みモデルを特定のタスクやドメインに適応させる手法
- **メリット**: 少ないデータで高品質なカスタムモデルを作成可能
- **用途**: チャットボット、コード生成、専門分野の質問応答など

## Hugging Faceモデルの準備

### Step 1: Hugging Faceトークンの取得

1. **アカウント作成**
   - https://huggingface.co/join にアクセス
   - アカウントを作成（無料）

2. **トークン生成**
   - https://huggingface.co/settings/tokens にアクセス
   - "New token"をクリック
   - Token name: `h200-benchmark`
   - Role: `read`を選択
   - "Generate token"をクリック
   - トークンをコピー（一度だけ表示されます）

3. **トークンの設定**
   ```bash
   # 自動設定スクリプトを実行
   ./setup/configure_huggingface.sh
   
   # プロンプトが表示されたらトークンを貼り付け
   # Enter your Hugging Face token: hf_xxxxxxxxxxxxx
   ```

### Step 2: Llama-2モデルへのアクセス申請（必要な場合）

Llama-2などのゲート付きモデルを使用する場合：

1. **モデルページにアクセス**
   - https://huggingface.co/meta-llama/Llama-2-7b-hf
   - "Request access"をクリック

2. **利用規約に同意**
   - Meta社の利用規約を確認
   - 必要事項を入力
   - Submit

3. **承認待ち**
   - 通常24時間以内に承認
   - メールで通知が届きます

### Step 3: 利用可能なモデル

#### 公開モデル（トークン不要）
```python
# すぐに使えるモデル
models = [
    "gpt2",                          # 124M params
    "EleutherAI/gpt-neo-125M",      # 125M params
    "microsoft/phi-2",               # 2.7B params
    "mistralai/Mistral-7B-v0.1",   # 7B params
    "tiiuae/falcon-7b",            # 7B params
]
```

#### ゲート付きモデル（トークン必要）
```python
# アクセス申請が必要
gated_models = [
    "meta-llama/Llama-2-7b-hf",    # 7B params
    "meta-llama/Llama-2-13b-hf",   # 13B params
    "meta-llama/Llama-2-70b-hf",   # 70B params
    "google/gemma-7b",              # 7B params
]
```

## データセットの準備

### 方法1: 公開データセットを使用

```bash
# Alpacaデータセット（推奨）
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source alpaca \
    --num-samples 10000 \
    --format alpaca

# Dollyデータセット
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source dolly \
    --num-samples 5000 \
    --format alpaca

# WizardLMデータセット
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source wizardlm \
    --num-samples 20000 \
    --format alpaca
```

### 方法2: カスタムデータセットを使用

#### JSONフォーマット
```json
// data.json
[
    {
        "instruction": "東京の人口を教えてください",
        "input": "",
        "output": "東京都の人口は約1400万人です（2023年時点）"
    },
    {
        "instruction": "次の文章を要約してください",
        "input": "人工知能（AI）は...",
        "output": "AIは人間の知能を模倣する..."
    }
]
```

```bash
# カスタムJSONデータセットの準備
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source json \
    --source-path ./data.json \
    --format alpaca
```

#### CSVフォーマット
```csv
instruction,input,output
"東京の人口を教えてください","","東京都の人口は約1400万人です"
"次の文章を要約してください","人工知能は...","AIは..."
```

```bash
# カスタムCSVデータセットの準備
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source csv \
    --source-path ./data.csv \
    --format alpaca
```

### データセットフォーマットの選択

```bash
# Alpaca形式（デフォルト）
--format alpaca

# ChatML形式（GPT系モデル用）
--format chatml

# Llama-2 Chat形式
--format llama2
```

## SFT訓練の実行

### 基本的な訓練

```bash
# Step 1: データセット準備
python3 datasets/prepare_custom_dataset.py \
    --model meta-llama/Llama-2-7b-hf \
    --source alpaca \
    --num-samples 10000

# Step 2: SFT訓練実行
python3 scripts/train_sft.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --dataset-path datasets/alpaca_Llama-2-7b-hf_10000 \
    --output-dir ./outputs/llama2-7b-alpaca \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --use-lora \
    --gradient-checkpointing \
    --bf16
```

### パラメータ効率的な訓練（LoRA）

```bash
# LoRAを使用したメモリ効率的な訓練
python3 scripts/train_sft.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --dataset-path datasets/alpaca_Llama-2-7b-hf_10000 \
    --output-dir ./outputs/llama2-7b-lora \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --bf16
```

### 量子化を使用した訓練（大規模モデル用）

```bash
# 4-bit量子化 + LoRA（70Bモデルでも1GPU可能）
python3 scripts/train_sft.py \
    --model-name meta-llama/Llama-2-70b-hf \
    --dataset-path datasets/alpaca_Llama-2-70b-hf_10000 \
    --output-dir ./outputs/llama2-70b-qlora \
    --use-4bit \
    --use-lora \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --bf16
```

### マルチノード分散訓練

```bash
# DeepSpeedを使用した8ノード訓練
deepspeed --num_nodes 8 \
    --num_gpus 64 \
    --hostfile hostfile \
    scripts/train_sft.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --dataset-path datasets/alpaca_Llama-2-7b-hf_10000 \
    --output-dir ./outputs/llama2-7b-distributed \
    --deepspeed configs/ds_config_7b.json \
    --batch-size 16 \
    --gradient-accumulation-steps 1 \
    --bf16
```

## ベンチマークテスト

### 完全なベンチマーク実行

```bash
# Step 1: Hugging Face認証
./setup/configure_huggingface.sh

# Step 2: データセット準備（複数サイズ）
for size in 1000 5000 10000; do
    python3 datasets/prepare_custom_dataset.py \
        --model meta-llama/Llama-2-7b-hf \
        --source alpaca \
        --num-samples $size
done

# Step 3: スケーリングテスト
for nodes in 2 4 8; do
    echo "Testing with $nodes nodes..."
    ./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf $nodes
done

# Step 4: レポート生成
python3 scripts/generate_report.py results/
```

### パフォーマンス測定

```bash
# GPU利用率のモニタリング
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# 訓練速度の確認
tail -f outputs/*/trainer_state.json | grep loss

# メトリクス収集
./scripts/collect_metrics.sh results/current/
```

## トラブルシューティング

### よくある問題

#### 1. Hugging Faceモデルがダウンロードできない

```bash
# トークンが設定されているか確認
echo $HF_TOKEN

# 再設定
source ~/.hf_token

# モデルアクセスのテスト
python3 -c "
from transformers import AutoModel
import os
model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                  token=os.environ.get('HF_TOKEN'))
"
```

#### 2. Out of Memory (OOM)エラー

```bash
# 解決策1: バッチサイズを減らす
--batch-size 2
--gradient-accumulation-steps 8

# 解決策2: LoRAを使用
--use-lora
--lora-r 8  # rankを下げる

# 解決策3: 量子化を使用
--use-8bit  # または --use-4bit

# 解決策4: gradient checkpointingを有効化
--gradient-checkpointing
```

#### 3. データセットエラー

```bash
# データセットの検証
python3 -c "
from datasets import load_from_disk
ds = load_from_disk('datasets/alpaca_Llama-2-7b-hf_10000')
print(f'Train samples: {len(ds[\"train\"])}')
print(f'Columns: {ds[\"train\"].column_names}')
print(f'First sample: {ds[\"train\"][0]}')
"
```

#### 4. 訓練が遅い

```bash
# Flash Attention 2を有効化
--use-flash-attention

# Mixed precisionを確認
--bf16  # H200では推奨

# DataLoaderワーカー数を増やす
export OMP_NUM_THREADS=4
```

### デバッグモード

```bash
# 小規模データで動作確認
python3 datasets/prepare_custom_dataset.py \
    --model gpt2 \
    --source alpaca \
    --num-samples 100

python3 scripts/train_sft.py \
    --model-name gpt2 \
    --dataset-path datasets/alpaca_gpt2_100 \
    --output-dir ./test_output \
    --num-epochs 1 \
    --batch-size 2 \
    --logging-steps 1 \
    --save-steps 10
```

## ベストプラクティス

### 推奨設定

| モデルサイズ | バッチサイズ | LoRA rank | 量子化 | ノード数 |
|------------|-------------|-----------|--------|---------|
| 7B | 4-8 | 16 | なし | 1-2 |
| 13B | 2-4 | 16 | 8-bit | 2-4 |
| 70B | 1 | 8 | 4-bit | 8 |

### 訓練のコツ

1. **段階的なスケールアップ**
   - まず小規模データで動作確認
   - 徐々にデータ量とモデルサイズを増やす

2. **チェックポイント活用**
   - `--save-steps`を適切に設定
   - 中断しても再開可能

3. **メトリクス監視**
   - 訓練損失の推移を確認
   - GPU利用率を90%以上に維持

4. **データ品質**
   - 高品質なデータセットを使用
   - 重複データの除去
   - 適切なフォーマット選択

## まとめ

このガイドに従えば、H200クラスタで効率的にLLMのSFTを実行できます。
質問や問題がある場合は、GitHubイシューまたはドキュメントを参照してください。

---
最終更新: 2025年8月