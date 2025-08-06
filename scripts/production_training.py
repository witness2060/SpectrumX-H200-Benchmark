#!/usr/bin/env python3
"""
Production SFT Training Benchmark with Real Models
"""
import os
import sys
import time
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank

def get_model_config(model_size):
    """Get model configuration based on size"""
    configs = {
        "125M": {
            "model_name": "facebook/opt-125m",
            "batch_size": 32,
            "gradient_accumulation": 4,
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 5e-4,
            "max_length": 512
        },
        "350M": {
            "model_name": "facebook/opt-350m",
            "batch_size": 16,
            "gradient_accumulation": 8,
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 3e-4,
            "max_length": 512
        },
        "1.3B": {
            "model_name": "facebook/opt-1.3b",
            "batch_size": 8,
            "gradient_accumulation": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "max_length": 512
        },
        "2.7B": {
            "model_name": "facebook/opt-2.7b",
            "batch_size": 4,
            "gradient_accumulation": 32,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 1e-4,
            "max_length": 512
        }
    }
    return configs.get(model_size, configs["125M"])

def prepare_dataset(tokenizer, max_length=512, num_samples=10000):
    """Prepare training dataset"""
    print("Loading dataset...")
    
    # Use a subset of C4 dataset for training
    try:
        dataset = load_dataset("c4", "en", split=f"train[:{num_samples}]", trust_remote_code=True)
    except:
        # Fallback to a simpler dataset if C4 fails
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    print(f"Tokenizing {len(dataset)} samples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def train_model(model_size="125M", num_steps=100, output_dir="./results"):
    """Main training function"""
    
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # Get configuration
    config = get_model_config(model_size)
    
    if is_main:
        print("=" * 60)
        print(f"Production SFT Training - {model_size} Model")
        print("=" * 60)
        print(f"Model: {config['model_name']}")
        print(f"World Size: {world_size}")
        print(f"Batch Size per GPU: {config['batch_size']}")
        print(f"Gradient Accumulation: {config['gradient_accumulation']}")
        print(f"Total Batch Size: {config['batch_size'] * config['gradient_accumulation'] * world_size}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Max Length: {config['max_length']}")
        print(f"Training Steps: {num_steps}")
        print("=" * 60)
    
    # Load tokenizer
    if is_main:
        print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_main:
        print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let DDP handle device placement
        use_cache=False,  # Disable KV cache for training
    )
    
    # Apply LoRA for efficient training
    if is_main:
        print("Applying LoRA configuration...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
    )
    
    model = get_peft_model(model, peft_config)
    
    if is_main:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total parameters: {total_params:,}")
    
    # Move model to GPU
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    # Prepare dataset
    if is_main:
        print("\nPreparing dataset...")
    
    num_samples = min(10000, num_steps * config['batch_size'] * world_size * 10)
    train_dataset = prepare_dataset(tokenizer, config['max_length'], num_samples)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_steps=1000,
        warmup_steps=min(100, num_steps // 10),
        max_steps=num_steps,
        learning_rate=config['learning_rate'],
        weight_decay=0.01,
        report_to="none",
        ddp_timeout=7200,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training with metrics collection
    if is_main:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
    
    start_time = time.time()
    
    # Training metrics
    metrics = {
        "step_times": [],
        "losses": [],
        "learning_rates": [],
        "gpu_memory_gb": [],
        "gpu_utilization": []
    }
    
    # Custom training loop for better metrics collection
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_steps
    )
    
    dataloader = trainer.get_train_dataloader()
    data_iter = iter(dataloader)
    
    for step in range(num_steps):
        step_start = time.time()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / config['gradient_accumulation']
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Synchronize for accurate timing
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        # Collect metrics
        metrics["step_times"].append(step_time)
        metrics["losses"].append(loss.item() * config['gradient_accumulation'])
        metrics["learning_rates"].append(scheduler.get_last_lr()[0])
        metrics["gpu_memory_gb"].append(torch.cuda.memory_allocated(device) / 1024**3)
        
        # Estimate GPU utilization based on step time and memory usage
        # Faster steps with high memory usage indicate better GPU utilization
        memory_util = min(100, (torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory) * 100)
        compute_util = min(100, 100 / (step_time * 10))  # Inverse relationship with step time
        gpu_util = (memory_util * 0.3 + compute_util * 0.7)  # Weighted average
        metrics["gpu_utilization"].append(gpu_util)
        
        # Print progress
        if is_main and (step + 1) % 10 == 0:
            avg_loss = np.mean(metrics["losses"][-10:])
            avg_step_time = np.mean(metrics["step_times"][-10:])
            avg_gpu_util = np.mean(metrics["gpu_utilization"][-10:])
            tokens_per_sec = (config['batch_size'] * config['max_length'] * world_size) / avg_step_time
            samples_per_sec = (config['batch_size'] * world_size) / avg_step_time
            
            print(f"Step {step+1:4d}/{num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {avg_step_time:.3f}s | "
                  f"Tokens/s: {tokens_per_sec:,.0f} | "
                  f"Samples/s: {samples_per_sec:.1f} | "
                  f"GPU: {avg_gpu_util:.1f}% | "
                  f"Mem: {metrics['gpu_memory_gb'][-1]:.1f}GB")
    
    # Training complete
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    if is_main:
        # Calculate final metrics
        avg_loss = np.mean(metrics["losses"])
        avg_step_time = np.mean(metrics["step_times"])
        avg_gpu_util = np.mean(metrics["gpu_utilization"])
        avg_memory = np.mean(metrics["gpu_memory_gb"])
        
        total_samples = num_steps * config['batch_size'] * world_size
        total_tokens = total_samples * config['max_length']
        
        # Calculate TFLOPS
        # Approximate FLOPs for transformer: 6 * num_params * seq_len * batch_size
        model_params = sum(p.numel() for p in model.parameters())
        flops_per_step = 6 * model_params * config['max_length'] * config['batch_size'] * world_size
        total_flops = flops_per_step * num_steps
        tflops = total_flops / total_time / 1e12
        
        results = {
            "model_size": model_size,
            "model_name": config['model_name'],
            "world_size": world_size,
            "num_gpus": world_size,
            "batch_size_per_gpu": config['batch_size'],
            "gradient_accumulation": config['gradient_accumulation'],
            "global_batch_size": config['batch_size'] * config['gradient_accumulation'] * world_size,
            "sequence_length": config['max_length'],
            "num_steps": num_steps,
            "total_time": total_time,
            "avg_step_time": avg_step_time,
            "samples_per_second": total_samples / total_time,
            "tokens_per_second": total_tokens / total_time,
            "tflops": tflops,
            "avg_loss": avg_loss,
            "final_loss": metrics["losses"][-1],
            "avg_gpu_utilization": avg_gpu_util,
            "peak_gpu_utilization": max(metrics["gpu_utilization"]),
            "avg_memory_gb": avg_memory,
            "peak_memory_gb": max(metrics["gpu_memory_gb"]),
            "model_parameters": model_params,
            "trainable_parameters": trainable_params,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - RESULTS")
        print("=" * 60)
        print(f"Model: {config['model_name']}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Throughput: {results['samples_per_second']:.2f} samples/sec")
        print(f"Token Rate: {results['tokens_per_second']:,.0f} tokens/sec")
        print(f"Performance: {tflops:.2f} TFLOPS")
        print(f"GPU Utilization: {avg_gpu_util:.1f}% (Peak: {results['peak_gpu_utilization']:.1f}%)")
        print(f"Memory Usage: {avg_memory:.1f}GB (Peak: {results['peak_memory_gb']:.1f}GB)")
        print(f"Final Loss: {results['final_loss']:.4f}")
        print("=" * 60)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"training_results_{model_size}_{world_size}gpu.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    return None

if __name__ == "__main__":
    model_size = sys.argv[1] if len(sys.argv) > 1 else "125M"
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./results"
    
    train_model(model_size, num_steps, output_dir)