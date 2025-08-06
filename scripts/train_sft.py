#!/usr/bin/env python3
"""
SFT（Supervised Fine-Tuning）訓練スクリプト
DeepSpeedとLoRAを使用した効率的な学習
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SFTTrainer:
    """SFT訓練を管理するクラス"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hugging Faceトークンの設定
        self.hf_token = os.environ.get("HF_TOKEN", None)
        if not self.hf_token and "llama" in args.model_name.lower():
            logger.warning("HF_TOKEN not set. Private models may not be accessible.")
        
        # DeepSpeed設定の選択
        if args.deepspeed:
            self.ds_config_path = args.deepspeed
        else:
            self.ds_config_path = self._select_deepspeed_config()
        
    def _select_deepspeed_config(self) -> str:
        """モデルサイズに基づいてDeepSpeed設定を選択"""
        model_name = self.args.model_name.lower()
        
        if "70b" in model_name:
            config_file = "configs/ds_config_70b.json"
        elif "13b" in model_name:
            config_file = "configs/ds_config_13b.json"
        else:
            config_file = "configs/ds_config_7b.json"
        
        if not os.path.exists(config_file):
            logger.warning(f"DeepSpeed config {config_file} not found, using default")
            config_file = None
        
        return config_file
    
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーのロード"""
        logger.info(f"Loading model: {self.args.model_name}")
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量子化設定（メモリ節約）
        quantization_config = None
        if self.args.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        elif self.args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # モデルのロード
        model_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if not quantization_config else "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif self.args.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            **model_kwargs
        )
        
        # 勾配チェックポイントの有効化
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # LoRA設定
        if self.args.use_lora:
            logger.info("Applying LoRA configuration")
            
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=self.args.lora_target_modules.split(",") if self.args.lora_target_modules else None,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """データセットのロード"""
        logger.info(f"Loading dataset from: {self.args.dataset_path}")
        
        dataset = load_from_disk(self.args.dataset_path)
        
        if "train" in dataset:
            self.train_dataset = dataset["train"]
        else:
            self.train_dataset = dataset
        
        if "validation" in dataset:
            self.eval_dataset = dataset["validation"]
        else:
            # 検証セットがない場合は訓練セットの一部を使用
            split_dataset = self.train_dataset.train_test_split(test_size=0.1, seed=42)
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataset)}")
    
    def setup_training_args(self) -> TrainingArguments:
        """訓練引数の設定"""
        
        # 基本的な訓練引数
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            
            # 訓練パラメータ
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            
            # 最適化
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            weight_decay=self.args.weight_decay,
            max_grad_norm=self.args.max_grad_norm,
            
            # 精度
            bf16=self.args.bf16,
            fp16=self.args.fp16,
            tf32=True,  # H200はTF32をサポート
            
            # ログと保存
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # 評価
            evaluation_strategy="steps" if self.eval_dataset else "no",
            
            # DeepSpeed
            deepspeed=self.ds_config_path,
            
            # その他
            report_to=self.args.report_to.split(",") if self.args.report_to else [],
            push_to_hub=False,
            gradient_checkpointing=self.args.gradient_checkpointing,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        )
        
        return training_args
    
    def train(self):
        """訓練の実行"""
        
        # モデルとデータセットのロード
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 訓練引数
        training_args = self.setup_training_args()
        
        # Trainerの初期化
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 訓練実行
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # メトリクスの保存
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        
        # モデルの保存
        if self.args.save_model:
            logger.info(f"Saving model to {self.args.output_dir}")
            trainer.save_model()
            
            # LoRAの場合はマージも可能
            if self.args.use_lora and self.args.merge_lora:
                logger.info("Merging LoRA weights...")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(f"{self.args.output_dir}/merged")
        
        # 最終メトリクスの表示
        logger.info("Training completed!")
        logger.info(f"Final metrics: {metrics}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="SFT Training Script")
    
    # モデル設定（環境変数からデフォルト値を取得）
    parser.add_argument("--model-name", "--model_name", type=str, 
                       default=os.environ.get("DEFAULT_MODEL", "meta-llama/Llama-2-7b-hf"),
                       help="Model name from Hugging Face")
    parser.add_argument("--dataset-path", "--dataset_path", type=str, 
                       default=None, help="Path to prepared dataset")
    parser.add_argument("--output-dir", "--output_dir", type=str, default="./output",
                       help="Output directory for model and logs")
    
    # 訓練パラメータ（環境変数からデフォルト値を取得）
    parser.add_argument("--num-epochs", type=int, 
                       default=int(os.environ.get("NUM_EPOCHS", 3)),
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, 
                       default=int(os.environ.get("BATCH_SIZE_PER_GPU", 4)),
                       help="Training batch size per device")
    parser.add_argument("--eval-batch-size", type=int, 
                       default=int(os.environ.get("BATCH_SIZE_PER_GPU", 8)),
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, 
                       default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4)),
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, 
                       default=float(os.environ.get("LEARNING_RATE", 2e-4)),
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # LoRA設定
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                       help="Target modules for LoRA (comma-separated)")
    parser.add_argument("--merge-lora", action="store_true",
                       help="Merge LoRA weights after training")
    
    # 最適化設定
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--use-flash-attention", action="store_true",
                       help="Use Flash Attention 2")
    parser.add_argument("--use-8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--use-4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision")
    
    # ログ設定
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint frequency")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--save-model", action="store_true",
                       help="Save the final model")
    parser.add_argument("--report-to", type=str, default="",
                       help="Reporting tools (comma-separated): wandb,tensorboard")
    
    # DeepSpeed設定
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed config file path")
    
    args = parser.parse_args()
    
    # データセットパスのデフォルト設定
    if args.dataset_path is None:
        model_name_short = args.model_name.split('/')[-1]
        args.dataset_path = f"datasets/sft_{model_name_short}"
    
    # 環境確認
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU.")
        sys.exit(1)
    
    logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.get_device_name()}")
    
    # 訓練実行
    trainer = SFTTrainer(args)
    metrics = trainer.train()
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"📁 Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()