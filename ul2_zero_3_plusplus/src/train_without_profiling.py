import os
import yaml
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainerCallback
)

#from transformers.integrations import DeepSpeedPlugin
import evaluate
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter   
import matplotlib.pyplot as plt 
import sys
import torch.distributed as dist

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_function(examples, tokenizer, config):
    """Preprocess the CNN/DailyMail dataset"""
    inputs = [f"summarize: {doc}" for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=config['data']['max_source_length'],
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=config['data']['max_target_length'],
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics_rouge(eval_preds, tokenizer, metric):
    """Compute ROUGE metrics for evaluation"""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    
    return {k: round(v, 4) for k, v in result.items()}

def compute_metrics(eval_preds, tokenizer, rouge_metric, meteor_metric):
    """
    Compute ROUGE and METEOR metrics with robust error handling.
    Handles invalid token IDs that can cause OverflowError.
    """
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Critical: Clip predictions to valid token ID range
    vocab_size = tokenizer.vocab_size
    preds = np.clip(preds, 0, vocab_size - 1)
    
    # Replace -100 in labels (used for padding)
    labels_mod = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Also clip labels to be safe
    labels_mod = np.clip(labels_mod, 0, vocab_size - 1)
    
    # Convert to correct dtype to prevent overflow
    preds = preds.astype(np.int64)
    labels_mod = labels_mod.astype(np.int64)
    
    try:
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_mod, skip_special_tokens=True)
        
        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # Filter out empty predictions/labels
        valid_pairs = [
            (pred, label) 
            for pred, label in zip(decoded_preds, decoded_labels) 
            if pred and label
        ]
        
        if not valid_pairs:
            # Return dummy metrics if no valid predictions
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
                "meteor": 0.0
            }
        
        decoded_preds, decoded_labels = zip(*valid_pairs)
        
        result = {}
        
        # ROUGE
        rouge_res = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        for k, v in rouge_res.items():
            result[k] = v * 100.0
        
        # METEOR
        meteor_res = meteor_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        result["meteor"] = meteor_res["meteor"] * 100.0
        
        # Round for logging
        result = {k: round(v, 4) for k, v in result.items()}
        return result
        
    except Exception as e:
        # If anything fails, return dummy metrics and log error
        print(f"Error in compute_metrics: {e}")
        print(f"Prediction shape: {preds.shape}, min: {preds.min()}, max: {preds.max()}")
        print(f"Label shape: {labels_mod.shape}, min: {labels_mod.min()}, max: {labels_mod.max()}")
        
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "meteor": 0.0
        }

class GPUMemoryCallback(TrainerCallback):   # <<< NEW
    """
    Callback to log GPU memory usage every `mem_log_steps` steps
    into TensorBoard and stdout.
    """

    def __init__(self, writer: SummaryWriter, mem_log_steps: int = 50):
        self.writer = writer
        self.mem_log_steps = mem_log_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Only log on certain steps and when CUDA is available
        if state.global_step is None or state.global_step == 0:
            return

        if state.global_step % self.mem_log_steps != 0:
            return

        if not torch.cuda.is_available():
            return

        device = torch.device("cuda")
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        max_reserved = torch.cuda.max_memory_reserved(device) / 1e9

        print(
            f"[step {state.global_step}] "
            f"GPU mem allocated: {allocated:.2f} GB "
            f"(max {max_allocated:.2f} GB), "
            f"reserved: {reserved:.2f} GB "
            f"(max {max_reserved:.2f} GB)"
        )

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("gpu/allocated_gb", allocated, state.global_step)
            self.writer.add_scalar("gpu/reserved_gb", reserved, state.global_step)
            self.writer.add_scalar("gpu/max_allocated_gb", max_allocated, state.global_step)
            self.writer.add_scalar("gpu/max_reserved_gb", max_reserved, state.global_step)

def main():
    parser = argparse.ArgumentParser(description='Train FLAN-T5 on CNN/DailyMail')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
    
    # Load tokenizer and model
    if rank == 0:
        print(f"Loading tokenizer and model: {config['model']['name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['pretrained_path'],
        cache_dir=config['paths']['cache_dir']
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config['model']['pretrained_path'],
        cache_dir=config['paths']['cache_dir']
    )

    # Optional: enable gradient checkpointing for memory savings
    if config['training'].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Load dataset
    if rank == 0:
        print(f"Loading dataset: {config['data']['dataset_name']}")
    
    dataset = load_dataset(
        config['data']['dataset_name'],
        config['data']['dataset_config'],
        cache_dir=config['paths']['cache_dir']
    )
    
    # Use subset for debugging if specified
    if config['debug'].get('use_subset', False):
        dataset['train'] = dataset['train'].select(range(config['debug']['train_subset_size']))
        dataset['validation'] = dataset['validation'].select(range(config['debug']['valid_subset_size']))
    
    # Preprocess dataset
    if rank == 0:
        print("Preprocessing dataset...")
    
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        remove_columns=dataset['train'].column_names,
        num_proc=config['data'].get('num_proc', 4),
        load_from_cache_file=True,
        desc="Tokenizing dataset"
    )
    
    # Load ROUGE metric
    rouge_metric = evaluate.load("rouge")
    # bertscore_metric = evaluate.load("bertscore") #compute heavy
    meteor_metric = evaluate.load("meteor")
    
    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if config['training'].get("fp16") or config['training'].get("bf16") else None,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['output_dir'], f"run_{timestamp}")
    #os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        #overwrite_output_dir=True,
        #do_train=True,
        #do_eval=True,
        # Training hyperparameters
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),

        # Optimization
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_strategy="steps",
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss", #rouge1
        greater_is_better=False,
        
        # Generation
        predict_with_generate=False,  #True,
        generation_max_length=config['data']['max_target_length'],
        # Add these to prevent generation issues
        #generation_config={
        #    "max_length": config['data']['max_target_length'],
        #    "min_length": 10,
        #    "num_beams": 4,
        #    "length_penalty": 0.6,
        #    "no_repeat_ngram_size": 3,
        #    "early_stopping": True,
        #    "forced_eos_token_id": tokenizer.eos_token_id
        #},

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config['training']['logging_steps'],
        report_to=["tensorboard"],
        
        # Distributed training
        ddp_find_unused_parameters=False,
        deepspeed=config['deepspeed']['deepspeed_zero2_config'] if config['deepspeed']['enabled'] else None,
        
        # Misc
        seed=config['training']['seed'],
        dataloader_num_workers=config['data'].get('dataloader_workers', 4),
        remove_unused_columns=True,
        push_to_hub=False,
    )
    
    resume_ckpt = config['paths'].get("resume_from_checkpoint", None)
    
    # ----- TensorBoard writer & GPU memory callback  -----  <<< NEW
    # Only rank 0 should write
    if rank == 0:
        writer = SummaryWriter(log_dir=training_args.logging_dir)
        mem_log_steps = config['training'].get("mem_log_steps", 5)
        gpu_mem_callback = GPUMemoryCallback(writer, mem_log_steps=mem_log_steps)
    else:
        writer = None
        mem_log_steps = config['training'].get("mem_log_steps", 5)
        gpu_mem_callback = GPUMemoryCallback(None, mem_log_steps=mem_log_steps)


    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= None,   #lambda x: compute_metrics(x, tokenizer, rouge_metric, meteor_metric ), #need to verify
        #compute_metrics=compute_metrics,
        callbacks=[gpu_mem_callback]   # <<< NEW
    )
    
    # Train
    if rank == 0:
        print("Starting training...")
    
    #if training_args.do_train:
    #    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    #else: 
    train_result = trainer.train()
    
    # Save model
    if rank == 0:
        print("Saving model...")
        trainer.save_model(config['paths']['trained_model_path'])
        tokenizer.save_pretrained(config['paths']['trained_model_path'])
        
        # Save training metrics 
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Final evaluation
    if rank == 0:
        print("Running final evaluation...")
    
    if training_args.do_eval:
        metrics = trainer.evaluate()
        if rank == 0:
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            trainer.save_state()
            print(f"Evaluation results: {metrics}")
            print("Training completed!")
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] Post-evaluation barrier passed")

    # ----- Plot train/val loss curve and save as PNG -----  <<< NEW
    if rank == 0:
        log_history = trainer.state.log_history
        train_steps = [x["step"] for x in log_history if "loss" in x]
        train_losses = [x["loss"] for x in log_history if "loss" in x]
        eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
        eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

        plt.figure(figsize=(8, 5))
        if train_steps:
            plt.plot(train_steps, train_losses, label="Train loss")
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="Eval loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Train / Eval Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(output_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved loss curve to {loss_plot_path}")
        print("Training completed!")

    # Close TensorBoard writer properly
    if rank == 0:
        try:
            if writer is not None:
                writer.flush()
                writer.close()
            print("TensorBoard writer closed")
        except Exception as e:
            print(f"Error closing writer: {e}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        print(f"[Rank {rank}] Final barrier passed")
    
    # Clean up distributed process group
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print(f"[Rank {rank}] Process group destroyed")

    # Now all ranks can exit
    if rank == 0:
        print("Training completed successfully!")

    # CRITICAL: Force exit
    sys.exit(0)

if __name__ == "__main__":
    main()
