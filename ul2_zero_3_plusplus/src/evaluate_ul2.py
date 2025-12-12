#!/usr/bin/env python3
"""
Evaluate UL2-20B Model on GovReport Summarization Dataset
Supports both single-GPU and multi-GPU evaluation
"""

import os
import yaml
import torch
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from datetime import datetime
import sys
import gc


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_function(examples, tokenizer, config):
    """
    Preprocess the GovReport dataset
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer instance
        config: Configuration dictionary
    """
    # Add UL2 prefix for summarization
    inputs = ["[S2S] Summarize: " + doc for doc in examples["report"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=config['data']['max_source_length'],
        truncation=True,
        padding=False
    )
    
    # Tokenize targets (summaries)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=config['data']['max_target_length'],
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def evaluate_model(model, dataloader, tokenizer, device, config, rank=0):
    """
    Evaluate the model on test set
    
    Args:
        model: Model instance
        dataloader: DataLoader for evaluation
        tokenizer: Tokenizer instance
        device: Device to run on
        config: Configuration dictionary
        rank: Process rank (for distributed)
    
    Returns:
        Dictionary with ROUGE scores, predictions, and references
    """
    model.eval()
    
    # Load metrics
    rouge_metric = evaluate.load("rouge")
    
    all_predictions = []
    all_references = []
    
    print(f"[Rank {rank}] Generating predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", disable=rank!=0)):
            # Move batch to device
            torch.cuda.empty_cache()
            gc.collect()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate predictions
            try:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=config['data']['max_target_length'],
                    num_beams= 4, #config['evaluation'].get('num_beams', 4),
                    length_penalty= .6,  #config['evaluation'].get('length_penalty', 0.6),
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            except RuntimeError as e:
                print(f"Error generating batch {batch_idx}: {e}")
                continue
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get references
            labels = batch["labels"]
            # Replace -100 with pad_token_id for decoding
            labels = np.where(labels.cpu().numpy() != -100, labels.cpu().numpy(), tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            # Print sample every 10 batches
            if rank == 0 and batch_idx % 10 == 0 and batch_idx > 0:
                print(f"\n{'='*80}")
                print(f"Sample Prediction (Batch {batch_idx}):")
                print(f"{'='*80}")
                print(f"Input length: {input_ids.shape[1]}")
                print(f"Reference: {references[0][:200]}...")
                print(f"Prediction: {predictions[0][:200]}...")
                print(f"{'='*80}\n")
    
    # Compute ROUGE scores
    if rank == 0:
        print("\nComputing ROUGE scores...")
    
    rouge_results = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True
    )
    
    # Convert to percentage
    rouge_results = {k: v * 100 for k, v in rouge_results.items()}
    
    return rouge_results, all_predictions, all_references


def load_model_for_evaluation(model_path, config, device):
    """
    Load model for evaluation with proper handling for large models
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Try to load with different methods
    try:
        # Method 1: Standard loading with device_map for large models
        print("Attempting to load with device_map='auto'...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded with device_map='auto'")
        return model
        
    except Exception as e1:
        print(f"Failed with device_map: {e1}")
        
        try:
            # Method 2: Load and move to device
            print("Attempting standard loading...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            model.to(device)
            print("✅ Model loaded and moved to device")
            return model
            
        except Exception as e2:
            print(f"Failed with standard loading: {e2}")
            raise RuntimeError(f"Could not load model: {e1}, {e2}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate UL2-20B on GovReport')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, 
                       help='Path to trained model (overrides config)')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['validation', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size for evaluation (recommend 1 for UL2-20B)')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to evaluate (None for full dataset)')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save predictions to file')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda:0, cuda:1, etc.). Auto-detect if not specified')
    args = parser.parse_args()
    
    # Load configuration
    print("="*80)
    print("UL2-20B EVALUATION ON GOVREPORT")
    print("="*80)
    
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model path
    model_path = args.model_path if args.model_path else config['paths']['trained_model_path']
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nConfiguration:")
    print(f"  Model path: {model_path}")
    print(f"  Dataset split: {args.split}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"  Output directory: {args.output_dir}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")
    except:
        print("⚠️  Could not load tokenizer from model_path, trying pretrained...")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_path'])
        print("✅ Tokenizer loaded from pretrained")
    
    # Load model
    model = load_model_for_evaluation(model_path, config, device)
    
    # Set to evaluation mode
    model.eval()
    
    # Print model info
    if hasattr(model, 'hf_device_map'):
        print(f"\nModel device map:")
        for name, dev in model.hf_device_map.items():
            print(f"  {name}: {dev}")
    
    # Load dataset
    print(f"\nLoading {args.split} split of {config['data']['dataset_name']}")
    try:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=args.split,
            cache_dir=config['paths'].get('cache_dir')
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Trying alternative dataset name...")
        dataset = load_dataset(
            "ccdv/govreport-summarization",
            split=args.split,
            cache_dir=config['paths'].get('cache_dir')
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Use subset if specified
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"Evaluating on {len(dataset)} samples")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=config['data'].get('num_proc', 4),
        desc="Tokenizing dataset"
    )
    print(f"✅ Dataset preprocessed")
    
    # Create dataloader
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model if not hasattr(model, 'module') else model.module,
        padding=True
    )
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    print(f"\nDataLoader created:")
    print(f"  Total batches: {len(dataloader)}")
    print(f"  Batch size: {args.batch_size}")
    
    # Check memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory before evaluation:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    rouge_results, predictions, references = evaluate_model(
        model, dataloader, tokenizer, device, config
    )
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(predictions)}")
    print(f"Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Speed: {len(predictions)/elapsed_time:.2f} samples/second")
    print(f"{'='*80}")
    for metric, score in rouge_results.items():
        print(f"{metric}: {score:.4f}")
    print(f"{'='*80}")
    
    # Check memory after evaluation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nGPU Memory after evaluation:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {max_allocated:.2f} GB")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, f'predictions_{args.split}.txt')
        print(f"\nSaving predictions to {predictions_file}")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"Example {i+1}:\n")
                f.write(f"{'='*80}\n")
                f.write(f"Reference:\n{ref}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Prediction:\n{pred}\n")
                f.write(f"{'='*80}\n\n")
        print(f"✅ Predictions saved!")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, f'eval_results_{args.split}.json')
    results_dict = {
        'dataset': config['data']['dataset_name'],
        'split': args.split,
        'num_samples': len(predictions),
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'samples_per_second': len(predictions) / elapsed_time,
        'metrics': {k: float(v) for k, v in rouge_results.items()},
        'config': {
            'max_source_length': config['data']['max_source_length'],
            'max_target_length': config['data']['max_target_length'],
            'num_beams': config['evaluation'].get('num_beams', 4),
            'length_penalty': config['evaluation'].get('length_penalty', 0.6),
        }
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Metrics saved to {metrics_file}")
    
    # Save summary text file
    summary_file = os.path.join(args.output_dir, f'eval_summary_{args.split}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("UL2-20B EVALUATION RESULTS ON GOVREPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {config['data']['dataset_name']}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(predictions)}\n")
        f.write(f"Evaluation Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"Speed: {len(predictions)/elapsed_time:.2f} samples/second\n\n")
        f.write("ROUGE Scores:\n")
        f.write("-"*80 + "\n")
        for metric, score in rouge_results.items():
            f.write(f"{metric:15s}: {score:7.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Summary saved to {summary_file}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
