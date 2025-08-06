#!/usr/bin/env python3
"""
SFT用データセットの準備スクリプト
"""
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
import json

def prepare_sft_dataset(model_name="meta-llama/Llama-2-7b-hf", max_length=2048, num_samples=10000):
    """
    SFT用のデータセットを準備
    """
    print(f"Preparing SFT dataset for {model_name}")
    print(f"Max length: {max_length}, Samples: {num_samples}")
    
    # トークナイザーのロード（ローカルで利用可能な代替モデルを使用）
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    except:
        print(f"Cannot load tokenizer for {model_name}, using default GPT2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
    
    # データセットのロード（Alpaca形式の代替データセット）
    try:
        print("Loading dataset...")
        dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_samples}]")
    except:
        # フォールバック: ダミーデータセットを作成
        print("Creating dummy dataset for benchmarking...")
        dummy_data = []
        for i in range(num_samples):
            dummy_data.append({
                "instruction": f"Instruction {i}: Complete this task.",
                "input": f"Input data {i}" if i % 2 == 0 else "",
                "output": f"Output response {i}. " * 50  # 長めのテキストで実際の使用に近づける
            })
        
        import pandas as pd
        dataset = pd.DataFrame(dummy_data)
    
    def format_instruction(example):
        """Alpacaフォーマットに変換"""
        if example.get('input', ''):
            text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        else:
            text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": text}
    
    # フォーマット変換
    if hasattr(dataset, 'map'):
        dataset = dataset.map(format_instruction)
    else:
        # DataFrameの場合
        formatted_data = []
        for _, row in dataset.iterrows():
            formatted_data.append(format_instruction(row))
        dataset = formatted_data
    
    # 簡易的なトークナイズ（メモリ効率重視）
    def tokenize_function(examples):
        if isinstance(examples, dict) and "text" in examples:
            texts = examples["text"]
            if not isinstance(texts, list):
                texts = [texts]
        else:
            texts = examples if isinstance(examples, list) else [examples]
        
        result = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        return result
    
    print("Tokenizing dataset...")
    if hasattr(dataset, 'map'):
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None
        )
    else:
        # リスト形式の場合
        tokenized_dataset = []
        for i in range(0, len(dataset), 100):
            batch = dataset[i:i+100]
            tokenized_batch = tokenize_function(batch)
            for j in range(len(batch)):
                item = {}
                for key in tokenized_batch:
                    item[key] = tokenized_batch[key][j]
                tokenized_dataset.append(item)
    
    # 保存
    output_dir = f"datasets/sft_{model_name.split('/')[-1]}_{num_samples}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 簡易保存（JSONファイル形式）
    output_file = os.path.join(output_dir, "dataset.json")
    
    if hasattr(tokenized_dataset, 'save_to_disk'):
        tokenized_dataset.save_to_disk(output_dir)
    else:
        # リスト形式の場合はJSON保存
        with open(output_file, 'w') as f:
            json.dump(tokenized_dataset[:1000], f)  # メモリ節約のため最初の1000件のみ
    
    print(f"Dataset saved to {output_dir}")
    
    # メタデータの保存
    metadata = {
        "model_name": model_name,
        "max_length": max_length,
        "num_samples": min(num_samples, len(tokenized_dataset) if hasattr(tokenized_dataset, '__len__') else 1000),
        "tokenizer": tokenizer.__class__.__name__
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir

if __name__ == "__main__":
    # デフォルトパラメータまたはコマンドライン引数から取得
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    prepare_sft_dataset(model_name=model_name, num_samples=num_samples)