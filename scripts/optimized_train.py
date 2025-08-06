#!/usr/bin/env python3
"""
SpectrumX + H200環境用に最適化されたSFT訓練スクリプト
営業ベンチマーク用に特化
"""

import os
import sys
import time
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
import numpy as np
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedTrainer:
    """H200最適化トレーナー"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.metrics = {
            "samples_processed": 0,
            "tokens_processed": 0,
            "gpu_memory_allocated": [],
            "step_times": [],
            "throughput_history": []
        }
        
        # H200特有の最適化
        self._setup_h200_optimizations()
        
    def _setup_h200_optimizations(self):
        """H200専用の最適化設定"""
        
        # TF32を有効化（H200で高速）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Flash Attention設定
        os.environ["FLASH_ATTENTION_SKIP_SOFTMAX"] = "FALSE"
        
        # CUDA Graph最適化
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            torch.cuda.empty_cache()
            
        # メモリプール設定（HBM3e最適化）
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーのロード"""
        
        logger.info(f"Loading model: {self.args.model_name_or_path}")
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            use_fast=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデル設定
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.args.bf16 else torch.float16,
            "trust_remote_code": True,
        }
        
        # Flash Attention 2の有効化
        if hasattr(self.args, 'use_flash_attention_2'):
            model_kwargs["use_flash_attention_2"] = True
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # モデルロード
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            **model_kwargs
        )
        
        # Gradient Checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # モデル情報
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def load_dataset(self):
        """データセットのロード（ローカルRAID対応）"""
        
        dataset_path = self.args.dataset_path
        
        # ローカルRAIDパスのチェック
        if dataset_path.startswith("/raid/"):
            logger.info(f"Loading dataset from local RAID: {dataset_path}")
        
        try:
            dataset = load_from_disk(dataset_path)
            
            if "train" in dataset:
                self.train_dataset = dataset["train"]
            else:
                self.train_dataset = dataset
                
            logger.info(f"Dataset loaded: {len(self.train_dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # ダミーデータセットの作成（ベンチマーク用）
            logger.info("Creating dummy dataset for benchmarking")
            self.train_dataset = self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """ベンチマーク用ダミーデータセット"""
        from datasets import Dataset
        
        num_samples = 10000
        max_length = 2048
        
        # ランダムトークン生成
        data = []
        for i in range(num_samples):
            tokens = torch.randint(0, 32000, (max_length,)).tolist()
            data.append({
                "input_ids": tokens,
                "attention_mask": [1] * max_length,
                "labels": tokens
            })
        
        return Dataset.from_list(data)
    
    def train(self):
        """最適化された訓練ループ"""
        
        # モデルとデータセットのロード
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # DeepSpeed設定
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            max_steps=self.args.max_steps,
            learning_rate=5e-5,
            bf16=self.args.bf16,
            fp16=not self.args.bf16,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            deepspeed=self.args.deepspeed,
            gradient_checkpointing=self.args.gradient_checkpointing,
            remove_unused_columns=False,
            report_to=self.args.report_to.split(",") if self.args.report_to != "none" else [],
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            tf32=True,  # H200最適化
        )
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # テンソルコア最適化
        )
        
        # カスタムトレーナー（メトリクス収集付き）
        class BenchmarkTrainer(Trainer):
            def __init__(self, parent, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent = parent
                self.step_start_time = None
                
            def training_step(self, model, inputs):
                # ステップ開始時刻
                if self.step_start_time is None:
                    self.step_start_time = time.time()
                
                # 通常の訓練ステップ
                loss = super().training_step(model, inputs)
                
                # メトリクス収集
                if self.state.global_step % 10 == 0:
                    step_time = time.time() - self.step_start_time
                    self.parent.metrics["step_times"].append(step_time)
                    
                    # GPU使用率
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                        self.parent.metrics["gpu_memory_allocated"].append(gpu_mem)
                    
                    # スループット計算
                    batch_size = inputs["input_ids"].shape[0]
                    samples_per_sec = batch_size / step_time
                    self.parent.metrics["throughput_history"].append(samples_per_sec)
                    
                    # ログ出力
                    if self.state.global_step % self.args.logging_steps == 0:
                        avg_throughput = np.mean(self.parent.metrics["throughput_history"][-100:])
                        logger.info(f"Step {self.state.global_step}: "
                                  f"Loss={loss:.4f}, "
                                  f"Throughput={avg_throughput:.2f} samples/sec, "
                                  f"GPU Mem={gpu_mem:.1f}GB")
                    
                    self.step_start_time = time.time()
                
                return loss
        
        # トレーナー初期化
        trainer = BenchmarkTrainer(
            parent=self,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 訓練実行
        logger.info("Starting optimized training...")
        train_result = trainer.train()
        
        # 最終メトリクス
        total_time = time.time() - self.start_time
        avg_throughput = np.mean(self.metrics["throughput_history"])
        avg_gpu_mem = np.mean(self.metrics["gpu_memory_allocated"]) if self.metrics["gpu_memory_allocated"] else 0
        
        # 結果の保存
        final_metrics = {
            "model": self.args.model_name_or_path,
            "total_time_seconds": total_time,
            "total_steps": trainer.state.global_step,
            "samples_per_second": avg_throughput,
            "avg_gpu_memory_gb": avg_gpu_mem,
            "final_loss": train_result.metrics.get("train_loss", 0),
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        }
        
        # メトリクスファイル保存
        metrics_file = Path(self.args.output_dir) / "training_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"Training completed!")
        logger.info(f"Average throughput: {avg_throughput:.2f} samples/sec")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        return final_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized training for H200")
    
    # モデルとデータ
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # 訓練パラメータ
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # 最適化
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    
    # ログ
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--report_to", type=str, default="none")
    
    args = parser.parse_args()
    
    # 訓練実行
    trainer = OptimizedTrainer(args)
    metrics = trainer.train()
    
    # 結果出力（営業資料用）
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Throughput: {metrics['samples_per_second']:.2f} samples/sec")
    print(f"GPU Memory: {metrics['avg_gpu_memory_gb']:.1f} GB")
    print(f"World Size: {metrics['world_size']} GPUs")
    print("="*60)

if __name__ == "__main__":
    main()