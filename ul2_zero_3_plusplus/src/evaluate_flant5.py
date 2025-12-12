import os
import yaml
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


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
    #model_inputs["references"] = examples["highlights"]
    return model_inputs


def evaluate_model(model, dataloader, tokenizer, device, config):
    """Evaluate the model on test set"""
    model.eval()
    rouge_metric = evaluate.load("rouge")
    
    all_predictions = []
    all_references = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['data']['max_target_length'],
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get references
            labels = batch["labels"]
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute ROUGE scores
    print("Computing ROUGE scores...")
    rouge_results = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True
    )
    
    # Convert to percentage
    rouge_results = {k: v * 100 for k, v in rouge_results.items()}
    
    return rouge_results, all_predictions, all_references


def main():
    parser = argparse.ArgumentParser(description='Evaluate FLAN-T5 on CNN/DailyMail')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, help='Path to trained model (overrides config)')
    parser.add_argument('--split', type=str, default='test', choices=['validation', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None, 
                        help='Number of samples to evaluate (None for full dataset)')
    parser.add_argument('--save_predictions', action='store_true', 
                        help='Save predictions to file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine model path
    model_path = args.model_path if args.model_path else config['paths']['trained_model_path']
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    
    # Load dataset
    print(f"Loading {args.split} split of {config['data']['dataset_name']}")
    dataset = load_dataset(
        config['data']['dataset_name'],
        config['data']['dataset_config'],
        split=args.split,
        cache_dir=config['paths']['cache_dir']
    )
    
    # Use subset if specified
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"Evaluating on {len(dataset)} samples")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=config['data'].get('num_proc', 4),
        desc="Tokenizing dataset"
    )
    
    # Create dataloader
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...")
    rouge_results, predictions, references = evaluate_model(
        model, dataloader, tokenizer, device, config
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, score in rouge_results.items():
        print(f"{metric}: {score:.4f}")
    print("="*50)
    
    # Save predictions if requested
    if args.save_predictions:
        output_file = f"predictions_{args.split}.txt"
        print(f"\nSaving predictions to {output_file}")
        with open(output_file, 'w') as f:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"Example {i+1}:\n")
                f.write(f"Prediction: {pred}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 80 + "\n")
        print(f"Predictions saved!")
    
    # Save metrics
    metrics_file = f"eval_results_{args.split}.txt"
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        for metric, score in rouge_results.items():
            f.write(f"{metric}: {score:.4f}\n")
        f.write("="*50 + "\n")
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
