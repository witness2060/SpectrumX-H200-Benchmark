#!/usr/bin/env python3
"""
カスタムSFTデータセットの準備スクリプト
様々な形式のデータセットに対応
"""
import os
import json
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import argparse

class CustomDatasetPreparer:
    def __init__(self, model_name, max_length=2048):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer()
    
    def _load_tokenizer(self):
        """トークナイザーのロード"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=os.environ.get("HF_TOKEN")
            )
        except:
            print(f"Using fallback tokenizer for {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<pad>"
        
        return tokenizer
    
    def load_from_jsonl(self, file_path):
        """JSONL形式のデータセットを読み込み"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def load_from_csv(self, file_path):
        """CSV形式のデータセットを読み込み"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def load_from_huggingface(self, dataset_name, split="train"):
        """Hugging Faceからデータセットをロード"""
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    
    def format_for_sft(self, data, format_type="alpaca"):
        """データをSFT用にフォーマット"""
        formatted_data = []
        
        for item in data:
            if format_type == "alpaca":
                # Alpaca形式: instruction, input, output
                text = self._format_alpaca(item)
            elif format_type == "chatgpt":
                # ChatGPT形式: messages リスト
                text = self._format_chatgpt(item)
            elif format_type == "simple":
                # シンプル形式: question, answer
                text = self._format_simple(item)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            formatted_data.append({"text": text})
        
        return formatted_data
    
    def _format_alpaca(self, item):
        """Alpaca形式のフォーマット"""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', item.get('response', ''))
        
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    def _format_chatgpt(self, item):
        """ChatGPT形式のフォーマット"""
        messages = item.get('messages', [])
        text = ""
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                text += f"### System:\n{content}\n\n"
            elif role == 'user':
                text += f"### User:\n{content}\n\n"
            elif role == 'assistant':
                text += f"### Assistant:\n{content}\n\n"
        
        return text.strip()
    
    def _format_simple(self, item):
        """シンプルな質問応答形式"""
        question = item.get('question', item.get('prompt', ''))
        answer = item.get('answer', item.get('response', ''))
        
        return f"""### Question:
{question}

### Answer:
{answer}"""
    
    def tokenize_dataset(self, formatted_data):
        """データセットをトークナイズ"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
        
        # Dataset形式に変換
        dataset = Dataset.from_list(formatted_data)
        
        # トークナイズ
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def save_dataset(self, dataset, output_dir):
        """データセットを保存"""
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        
        # メタデータ保存
        metadata = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_samples": len(dataset),
            "tokenizer": self.tokenizer.__class__.__name__
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Total samples: {len(dataset)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare custom SFT dataset")
    parser.add_argument("--input", required=True, help="Input file path or HF dataset name")
    parser.add_argument("--input-type", choices=["jsonl", "csv", "huggingface"], 
                       default="jsonl", help="Input data format")
    parser.add_argument("--format", choices=["alpaca", "chatgpt", "simple"], 
                       default="alpaca", help="Output format type")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", 
                       help="Model name for tokenizer")
    parser.add_argument("--max-length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--output", default=None, 
                       help="Output directory (default: datasets/sft_MODEL)")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    if args.output is None:
        model_short = args.model.split('/')[-1]
        args.output = f"datasets/sft_{model_short}_custom"
    
    # データセット準備
    preparer = CustomDatasetPreparer(args.model, args.max_length)
    
    # データ読み込み
    print(f"Loading data from {args.input}")
    if args.input_type == "jsonl":
        data = preparer.load_from_jsonl(args.input)
    elif args.input_type == "csv":
        data = preparer.load_from_csv(args.input)
    elif args.input_type == "huggingface":
        data = preparer.load_from_huggingface(args.input)
    else:
        raise ValueError(f"Unknown input type: {args.input_type}")
    
    # フォーマット変換
    print(f"Formatting data in {args.format} format")
    formatted_data = preparer.format_for_sft(data, args.format)
    
    # トークナイズ
    print("Tokenizing dataset")
    tokenized_dataset = preparer.tokenize_dataset(formatted_data)
    
    # 保存
    preparer.save_dataset(tokenized_dataset, args.output)

if __name__ == "__main__":
    main()