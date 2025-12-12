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
#from transformers.deepspeed import deepspeed_zero3_checkpoint_conversion
import evaluate
import numpy as np
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter   
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import matplotlib.pyplot as plt 
import sys
import torch.distributed as dist
import json

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_function(examples, tokenizer, config):
    """Preprocess the CCDV/govreport dataset"""
    inputs = [f"[S2S] Summarize: {doc}" for doc in examples["report"]]
    model_inputs = tokenizer(
        inputs,
        max_length=config['data']['max_source_length'],
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        text_target=examples["summary"],
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
"""
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
"""
class DetailedGPUMemoryCallback(TrainerCallback):
    """
    Comprehensive callback to log GPU memory usage, including:
    - Memory allocated/reserved per GPU
    - Max memory allocated/reserved
    - Layer-wise memory consumption tracking
    - Memory snapshots at key training phases
    """

    def __init__(self, writer: SummaryWriter, mem_log_steps: int = 1,
                 output_dir: str = "./", log_layer_memory: bool = True):
        self.writer = writer
        self.mem_log_steps = mem_log_steps
        self.output_dir = output_dir
        self.log_layer_memory = log_layer_memory
        self.memory_timeline = []
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Create memory logs directory
        self.memory_log_dir = os.path.join(output_dir, "memory_logs")
        if self.rank == 0:
            os.makedirs(self.memory_log_dir, exist_ok=True)

    def _get_gpu_memory_stats(self):
        """Get comprehensive GPU memory statistics"""
        if not torch.cuda.is_available():
            return None

        device = torch.device(f"cuda:{self.local_rank}" if self.local_rank != -1 else "cuda")

        stats = {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
            "cached_gb": torch.cuda.memory_reserved(device) / 1e9 - torch.cuda.memory_allocated(device) / 1e9,
        }

        # Get memory summary if available
        try:
            stats["memory_summary"] = torch.cuda.memory_summary(device, abbreviated=True)
        except:
            stats["memory_summary"] = "Not available"

        return stats

    def _log_memory_to_tensorboard(self, step, stats, prefix="train"):
        """Log memory statistics to TensorBoard"""
        if self.writer is not None and stats is not None:
            self.writer.add_scalar(f"{prefix}/gpu_allocated_gb_rank_{self.rank}",
                                   stats["allocated_gb"], step)
            self.writer.add_scalar(f"{prefix}/gpu_reserved_gb_rank_{self.rank}",
                                   stats["reserved_gb"], step)
            self.writer.add_scalar(f"{prefix}/gpu_max_allocated_gb_rank_{self.rank}",
                                   stats["max_allocated_gb"], step)
            self.writer.add_scalar(f"{prefix}/gpu_max_reserved_gb_rank_{self.rank}",
                                   stats["max_reserved_gb"], step)
            self.writer.add_scalar(f"{prefix}/gpu_cached_gb_rank_{self.rank}",
                                   stats["cached_gb"], step)

    def _print_memory_stats(self, step, stats, phase=""):
        """Print memory statistics to console"""
        if stats is None:
            return

        print(f"\n{'='*80}")
        print(f"[Rank {self.rank} | Local Rank {self.local_rank}] GPU Memory Stats - Step {step} {phase}")
        print(f"{'='*80}")
        print(f"  Allocated:     {stats['allocated_gb']:.3f} GB (Max: {stats['max_allocated_gb']:.3f} GB)")
        print(f"  Reserved:      {stats['reserved_gb']:.3f} GB (Max: {stats['max_reserved_gb']:.3f} GB)")
        print(f"  Cached (Free): {stats['cached_gb']:.3f} GB")
        print(f"{'='*80}\n")

    def on_train_begin(self, args, state, control, **kwargs):
        """Log memory at training start"""
        stats = self._get_gpu_memory_stats()
        self._print_memory_stats(0, stats, phase="[TRAIN BEGIN]")
        self._log_memory_to_tensorboard(0, stats, prefix="train")

        if stats:
            self.memory_timeline.append({
                "step": 0,
                "phase": "train_begin",
                "stats": stats
            })
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log memory at regular intervals during training"""
        if state.global_step is None or state.global_step == 0:
            return

        if state.global_step % self.mem_log_steps != 0:
            return

        stats = self._get_gpu_memory_stats()
        #self._print_memory_stats(state.global_step, stats, phase="[STEP END]")
        self._log_memory_to_tensorboard(state.global_step, stats, prefix="train")

        if stats:
            self.memory_timeline.append({
                "step": state.global_step,
                "phase": "step_end",
                "stats": stats
            })

    def on_evaluate(self, args, state, control, **kwargs):
        """Log memory during evaluation"""
        stats = self._get_gpu_memory_stats()
        #self._print_memory_stats(state.global_step, stats, phase="[EVALUATION]")
        self._log_memory_to_tensorboard(state.global_step, stats, prefix="eval")

        if stats:
            self.memory_timeline.append({
                "step": state.global_step,
                "phase": "evaluation",
                "stats": stats
            })

    def on_save(self, args, state, control, **kwargs):
        """Log memory during checkpoint saving"""
        stats = self._get_gpu_memory_stats()
        #self._print_memory_stats(state.global_step, stats, phase="[CHECKPOINT SAVE]")

        if stats:
            self.memory_timeline.append({
                "step": state.global_step,
                "phase": "checkpoint_save",
                "stats": stats
            })

    def on_train_end(self, args, state, control, **kwargs):
        """Save complete memory timeline at end of training"""
        stats = self._get_gpu_memory_stats()
        self._print_memory_stats(state.global_step, stats, phase="[TRAIN END]")

        if stats:
            self.memory_timeline.append({
                "step": state.global_step,
                "phase": "train_end",
                "stats": stats
            })

        # Save memory timeline to JSON
        if self.rank == 0:
            timeline_file = os.path.join(self.memory_log_dir, f"memory_timeline_rank_{self.rank}.json")
            with open(timeline_file, 'w') as f:
                json.dump(self.memory_timeline, f, indent=2, default=str)
            print(f"Memory timeline saved to {timeline_file}")


class ThroughputCallback(TrainerCallback):
    """
    Callback to measure and log training throughput:
    - Tokens per second
    - Samples per second
    - Time per step
    - Time per evaluation
    """

    def __init__(self, writer: SummaryWriter, log_steps: int = 1):
        self.writer = writer
        self.log_steps = log_steps
        self.step_start_time = None
        self.eval_start_time = None
        self.step_times = []
        self.eval_times = []
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        # ============================================
        # ADD: Track total times
        # ============================================
        self.train_start_time = None
        self.train_end_time = None
        self.total_eval_time = 0
        self.num_evaluations = 0
        # ============================================

    # ============================================
    # ADD: Training start/end tracking
    # ============================================
    def on_train_begin(self, args, state, control, **kwargs):
        """Track training start"""
        self.train_start_time = time.time()
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Track training end and print summary"""
        self.train_end_time = time.time()
        total_training_time = self.train_end_time - self.train_start_time

        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"TRAINING TIME SUMMARY")
            print(f"{'='*80}")
            print(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
            print(f"                     ({total_training_time:.2f} seconds)")
            print(f"Total evaluation time: {str(timedelta(seconds=int(self.total_eval_time)))}")
            print(f"                       ({self.total_eval_time:.2f} seconds)")
            print(f"Number of evaluations: {self.num_evaluations}")
            if self.num_evaluations > 0:
                avg_eval_time = self.total_eval_time / self.num_evaluations
                print(f"Average eval time: {avg_eval_time:.2f} seconds")
            print(f"{'='*80}\n")

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("timing/total_training_seconds", total_training_time, state.global_step)
                self.writer.add_scalar("timing/total_evaluation_seconds", self.total_eval_time, state.global_step)
    # ============================================

    def on_step_begin(self, args, state, control, **kwargs):
        """Mark step start time"""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate and log step throughput"""
        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)

        # Log every log_steps
        if state.global_step % self.log_steps == 0:
            avg_step_time = np.mean(self.step_times[-self.log_steps:])

            # Calculate throughput
            # samples_per_step = batch_size * gradient_accumulation_steps * world_size
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
            samples_per_sec = samples_per_step / avg_step_time

            # Rough token estimation (assuming avg 512 tokens per sample)
            tokens_per_sec = samples_per_sec * 512

            #print(f"\n[Rank {self.rank}] Throughput at step {state.global_step}:")
            #print(f"  Time per step: {avg_step_time:.3f} sec")
            #print(f"  Samples/sec:   {samples_per_sec:.2f}")
            #print(f"  Tokens/sec:    {tokens_per_sec:.0f}")

            if self.writer is not None and self.rank == 0:
                self.writer.add_scalar("throughput/time_per_step", avg_step_time, state.global_step)
                self.writer.add_scalar("throughput/samples_per_sec", samples_per_sec, state.global_step)
                self.writer.add_scalar("throughput/tokens_per_sec", tokens_per_sec, state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        """Mark evaluation start"""
        self.eval_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log evaluation time if evaluation just completed"""
        if self.eval_start_time is not None and logs is not None and "eval_loss" in logs:
            eval_time = time.time() - self.eval_start_time
            self.eval_times.append(eval_time)
            self.total_eval_time += eval_time
            self.num_evaluations += 1

            #print(f"\n[Rank {self.rank}] Evaluation completed in {eval_time:.2f} seconds")

            if self.writer is not None and self.rank == 0:
                self.writer.add_scalar("throughput/eval_time", eval_time, state.global_step)
                self.writer.add_scalar("timing/cumulative_eval_time", self.total_eval_time, state.global_step)

            self.eval_start_time = None

class PyTorchProfilerCallback(TrainerCallback):
    """
    Callback to run PyTorch Profiler at specified intervals
    """

    def __init__(self, output_dir: str, profile_steps: int = 10,
                 wait: int = 1, warmup: int = 1, active: int = 8, repeat: int = 1, save_on_train_end: bool = True):
        self.output_dir = output_dir
        self.profile_steps = profile_steps
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # Profiler schedule
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.save_on_train_end = save_on_train_end

        self.profiler = None
        self.should_profile = False

        # Create profiler output directory
        self.profiler_dir = os.path.join(output_dir, "profiler_logs")
        if self.rank == 0:
            os.makedirs(self.profiler_dir, exist_ok=True)

    def on_step_begin(self, args, state, control, **kwargs):
        """Start profiler at specified steps"""
        # Only profile on rank 0 to avoid overwhelming logs
        if self.rank != 0:
            return

        # Profile at specific intervals
        if state.global_step % self.profile_steps == 0 and state.global_step > 0:
            self.should_profile = True

            print(f"\n{'='*80}")
            print(f"Starting PyTorch Profiler at step {state.global_step}")
            print(f"{'='*80}\n")

            # Create profiler
            try:
                self.profiler = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(
                        wait=self.wait,
                        warmup=self.warmup,
                        active=self.active,
                        repeat=self.repeat
                    ),
                    on_trace_ready=self._trace_handler,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True
                )
                self.profiler.__enter__()
                print(f"[DEBUG] Profiler initialized successfully")
            except Exception as e:
                print(f"[DEBUG] ❌ Error initializing profiler: {e}")
                import traceback
                traceback.print_exc()

    def on_step_end(self, args, state, control, **kwargs):
        """Step profiler"""
        if self.should_profile and self.profiler is not None:
            #self.profiler.step()
            try:
                self.profiler.step()
                print(f"[DEBUG] Profiler stepped at step {state.global_step}")
            except Exception as e:
                print(f"[DEBUG] ❌ Error stepping profiler: {e}")
                import traceback
                traceback.print_exc()

        # Check if profiling is complete
        if state.global_step % self.profile_steps == (self.wait + self.warmup + self.active):
            try:
                self.profiler.__exit__(None, None, None)
                print(f"[DEBUG] Profiler exited successfully")
            except Exception as e:
                print(f"[DEBUG] ❌ Error exiting profiler: {e}")
                import traceback
                traceback.print_exc()

                #self.profiler.__exit__(None, None, None)
            self.profiler = None
            self.should_profile = False
            print(f"\n{'='*80}")
            print(f"PyTorch Profiler completed at step {state.global_step}")
            print(f"{'='*80}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """
        ✅ CRITICAL: Handle profiler cleanup at end of training
        This ensures traces are saved even if training ends during profiling
        """
        if self.rank != 0:
            return

        if self.should_profile and self.profiler is not None:
            print(f"\n{'='*80}")
            print(f"⚠️  TRAINING ENDED WHILE PROFILER WAS ACTIVE")
            print(f"{'='*80}")
            print(f"Training ended at step: {state.global_step}")

            if self.save_on_train_end:
                try:
                    # Export trace manually
                    trace_file = os.path.join(
                        self.profiler_dir,
                        f"trace_rank_{self.rank}_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )

                    # Get key averages and export
                    key_averages = self.profiler.key_averages()

                    # Export Chrome trace
                    self.profiler.export_chrome_trace(trace_file)

                    if os.path.exists(trace_file):
                        file_size = os.path.getsize(trace_file) / 1024
                        print(f"\n✅ INCOMPLETE TRACE SAVED!")
                        print(f"File: {trace_file}")
                        print(f"Size: {file_size:.2f} KB")
                    else:
                            print(f"❌ Failed to save incomplete trace")
                except Exception as e:
                    print(f"❌ Error force-saving trace: {e}")
                    import traceback
                    traceback.print_exc()
                print(f"{'='*80}\n")
    
    def _trace_handler(self, prof):
        """Handle profiler trace output"""
        # Save trace
        trace_file = os.path.join(
            self.profiler_dir,
            f"trace_rank_{self.rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        prof.export_chrome_trace(trace_file)
        print(f"Profiler trace saved to {trace_file}")
    
def plot_loss_curve(log_history, output_dir):
    """Plot and save training/validation loss curves"""
    train_steps = [x["step"] for x in log_history if "loss" in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    eval_steps = [x["step"] for x in log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_losses, label="Train loss", marker='o', markersize=3)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Eval loss", marker='s', markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")

def plot_memory_timeline(memory_log_file, output_dir):
    """Plot GPU memory usage timeline"""
    try:
        with open(memory_log_file, 'r') as f:
            timeline = json.load(f)

        steps = [entry['step'] for entry in timeline]
        allocated = [entry['stats']['allocated_gb'] for entry in timeline]
        reserved = [entry['stats']['reserved_gb'] for entry in timeline]
        max_allocated = [entry['stats']['max_allocated_gb'] for entry in timeline]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, allocated, label='Allocated', marker='o', markersize=3)
        plt.plot(steps, reserved, label='Reserved', marker='s', markersize=3)
        plt.plot(steps, max_allocated, label='Max Allocated', marker='^', markersize=3, linestyle='--')
        plt.xlabel('Training Step')
        plt.ylabel('GPU Memory (GB)')
        plt.title('GPU Memory Usage Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)

        memory_plot_path = os.path.join(output_dir, "memory_timeline.png")
        plt.tight_layout()
        plt.savefig(memory_plot_path, dpi=300)
        plt.close()
        print(f"Memory timeline plot saved to {memory_plot_path}")
    except Exception as e:
        print(f"Could not plot memory timeline: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train UL2-20B on GovReport')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # ============================================
    # ADD: Start total time tracking
    # ============================================
    total_start_time = time.time()

    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print("\n" + "="*80)
        print("DISTRIBUTED TRAINING SETUP")
        print("="*80)
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print("="*80 + "\n")

    # Load tokenizer and model
    if rank == 0:
        print(f"Loading tokenizer: {config['model']['name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['pretrained_path'],
        cache_dir=config['paths']['cache_dir']
    )
    
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
        if rank == 0:
            print(f"Using subset: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")

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
    meteor_metric = evaluate.load("meteor")
    
    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if config['training'].get("fp16") or config['training'].get("bf16") else None,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['output_dir'], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
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
        save_strategy="no",
        #save_steps=config['training']['save_steps'],
        save_total_limit=None,  #   config['training']['save_total_limit'],
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss", #rouge1
        greater_is_better=False,
        
        # Generation
        predict_with_generate=False,  #True,
        generation_max_length=config['training'].get('generation_max_length', 128),

        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config['training']['logging_steps'],
        report_to=["tensorboard"],
        
        # Distributed training
        ddp_find_unused_parameters=False,
        deepspeed=config['deepspeed']['deepspeed_zero3_config'] if config['deepspeed']['enabled'] else None,
        
        # Misc
        seed=config['training']['seed'],
        dataloader_num_workers=config['data'].get('dataloader_workers', 4),
        remove_unused_columns=True,
        push_to_hub=False,

        # Important for ZeRO-3
        save_safetensors=True,
        save_on_each_node=False
    )
    
    resume_ckpt = config['paths'].get("resume_from_checkpoint", None)
    
    # ----- TensorBoard writer & GPU memory callback  -----  <<< NEW
    # Only rank 0 should write
    # Initialize TensorBoard writer and callbacks
    callbacks = []
    if rank == 0:
        writer = SummaryWriter(log_dir=training_args.logging_dir)
        print(f"\nTensorBoard logging to: {training_args.logging_dir}")
        print(f"View with: tensorboard --logdir {training_args.logging_dir}\n")
    else:
        writer = None

    # GPU Memory Callback
    if config['training'].get("enable_memory_profiler", False):
        mem_log_steps = config['training'].get("mem_log_steps", 50)
        gpu_mem_callback = DetailedGPUMemoryCallback(
            writer=writer if rank == 0 else None,
            mem_log_steps=mem_log_steps,
            output_dir=output_dir,
            log_layer_memory=True
        )
        callbacks.append(gpu_mem_callback)
        if rank == 0:
            print("GPU Memory profiling enabled")

    # Throughput Callback
    if config['training'].get("enable_throughput_profiler", False):
        throughput_callback = ThroughputCallback(
            writer=writer if rank == 0 else None,
            log_steps=config['training'].get("logging_steps", 50)
        )
        callbacks.append(throughput_callback)
        if rank == 0:
            print("Throughput profiling enabled")

    # PyTorch Profiler Callback (optional)
    #if args.enable_profiler:
    if config['training'].get("enable_pytorch_profiler", False):
        profiler_callback = PyTorchProfilerCallback(
            output_dir=output_dir,
            profile_steps=config['training'].get("profile_steps", 50),
            wait=config['training'].get("wait_steps", 50),
            warmup=config['training'].get("warmup_steps", 50),
            active=config['training'].get("active_steps", 50),
            repeat=config['training'].get("repeat_steps", 50)
        )
        callbacks.append(profiler_callback)
        if rank == 0:
            print("PyTorch Profiler enabled")

    if rank == 0:
        print(f"Loading model: {config['model']['name']}")
    #For ZeRO-3, the model initialization can be done with special handling
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config['model']['pretrained_path'],
        cache_dir=config['paths']['cache_dir']
    )
   
    # Enable gradient checkpointing if configured
    if config['training'].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("Gradient checkpointing enabled")

    # Update data collator with model
    data_collator.model = model

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
        callbacks=callbacks   # <<< NEW
    )
    
    # Calculate and print training info
    if rank == 0:
        total_batch_size = (
            training_args.per_device_train_batch_size *
            training_args.gradient_accumulation_steps *
            world_size
        )
        num_training_steps = (
            len(tokenized_dataset['train']) // total_batch_size
        ) * training_args.num_train_epochs

        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"  Total batch size:     {total_batch_size}")
        print(f"  Num training samples: {len(tokenized_dataset['train'])}")
        print(f"  Num epochs:           {training_args.num_train_epochs}")
        print(f"  Num training steps:   {num_training_steps}")
        print(f"  Learning rate:        {training_args.learning_rate}")
        print(f"  Warmup steps:         {training_args.warmup_steps}")
        print("="*80 + "\n")

    # Train
    if rank == 0:
        print("Starting training with ZERO3 ...")
    
    train_start_time = time.time()
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    else: 
        train_result = trainer.train()
    
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    # Synchronize after training
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("[All ranks] Training completed, synchronized")

    
    # Save model
    if rank == 0:
        print("Saving model...")
        trainer.save_model(config['paths']['trained_model_path'])
        tokenizer.save_pretrained(config['paths']['trained_model_path'])

        # Save training metrics
        metrics = train_result.metrics
        metrics['total_training_time_seconds'] = training_time
        metrics['total_training_time_formatted'] = str(timedelta(seconds=int(training_time)))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if rank == 0:
        print("Running final evaluation...")
    
    eval_start_time = time.time()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        eval_end_time = time.time()
        evaluation_time = eval_end_time - eval_start_time
        
        if rank == 0:
            metrics['evaluation_time_seconds'] = evaluation_time
            metrics['evaluation_time_formatted'] = str(timedelta(seconds=int(evaluation_time)))
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            trainer.save_state()
            print(f"Evaluation results: {metrics}")
            print("evaluation completed!")
        dist.barrier()
        print(f"[Rank {rank}] Post-evaluation barrier passed")

    # ----- Plot train/val loss curve and save as PNG -----  <<< NEW
    if rank == 0:
        print("\nGenerating visualizations...")

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

    if rank==0:
        # Plot memory timeline
        memory_log_file = os.path.join(output_dir, "memory_logs", f"memory_timeline_rank_0.json")
        if os.path.exists(memory_log_file):
            plot_memory_timeline(memory_log_file, output_dir)
            print(f"Plotted memory timeline to {memory_log_file}!")

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

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING SUMMARY - TIME BREAKDOWN")
        print("="*80)
        print(f"Total Training Time:    {str(timedelta(seconds=int(training_time)))}")
        print(f"                        ({training_time:.2f} seconds)")
        if training_args.do_eval:
            print(f"Final Evaluation Time:  {str(timedelta(seconds=int(evaluation_time)))}")
            print(f"                        ({evaluation_time:.2f} seconds)")
        print(f"Total Runtime:          {str(timedelta(seconds=int(total_time)))}")
        print(f"                        ({total_time:.2f} seconds)")
        print("="*80 + "\n")

        # Save to a dedicated timing file
        timing_info = {
            "training_start": datetime.fromtimestamp(train_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "training_end": datetime.fromtimestamp(train_end_time).strftime('%Y-%m-%d %H:%M:%S'),
            "training_time_seconds": training_time,
            "training_time_formatted": str(timedelta(seconds=int(training_time))),
            "evaluation_time_seconds": evaluation_time if training_args.do_eval else 0,
            "evaluation_time_formatted": str(timedelta(seconds=int(evaluation_time))) if training_args.do_eval else "N/A",
            "total_runtime_seconds": total_time,
            "total_runtime_formatted": str(timedelta(seconds=int(total_time))),
            "num_training_steps": trainer.state.global_step,
            "seconds_per_step": training_time / trainer.state.global_step if trainer.state.global_step > 0 else 0
        }

        timing_file = os.path.join(output_dir, "timing_summary.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=2)
        print(f"Timing summary saved to {timing_file}")
    # ============================================

    # CRITICAL: Force exit
    sys.exit(1)

if __name__ == "__main__":
    main()
