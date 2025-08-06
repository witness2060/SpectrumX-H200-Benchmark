# DEPRECATED Scripts

このディレクトリには、統合のため廃止されたスクリプトが含まれています。

## 廃止理由

プロジェクトの保守性向上のため、以下のスクリプトは`run_benchmark.sh`と`run_all.sh`に統合されました：

- `run_actual_training.sh` - `run_benchmark.sh`の`sft`モードに統合
- `run_distributed_training.sh` - `run_benchmark.sh`に統合
- `run_h200_benchmark.sh` - `run_benchmark.sh`に統合  
- `run_multinode_benchmark.sh` - `run_benchmark.sh`に統合
- `run_real_benchmark.sh` - `run_benchmark.sh`に統合

## 移行ガイド

### 以前のコマンド → 新しいコマンド

```bash
# 以前
./scripts/run_actual_training.sh

# 現在
./run_all.sh bench meta-llama/Llama-2-7b-hf 2 sft
# または
./scripts/run_benchmark.sh meta-llama/Llama-2-7b-hf 2 sft
```

```bash
# 以前
./scripts/run_multinode_benchmark.sh 4

# 現在
./run_all.sh bench meta-llama/Llama-2-7b-hf 4 full
```

## Python訓練スクリプトの統合

以下のPythonスクリプトも`train_sft.py`に統合されました：

- `full_sft_benchmark.py` - `train_sft.py`に統合
- `optimized_train.py` - `train_sft.py`に統合
- `production_sft_benchmark.py` - `train_sft.py`に統合
- `production_training.py` - `train_sft.py`に統合

`train_sft.py`は全ての機能を含み、環境変数で動作を制御できます。

## 注意

これらのスクリプトは参照用に保存されていますが、今後は使用しないでください。
全ての機能は統合されたスクリプトで利用可能です。