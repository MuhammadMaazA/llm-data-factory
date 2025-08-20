#!/usr/bin/env python3
"""
Simple Fine-tuning Script for LLM Data Factory

This script fine-tunes Phi-3-mini using QLoRA on the generated synthetic data.
Optimized for quick training and achieving 80%+ accuracy.
"""

import json
import logging
import os
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFineTuner:
    """Simple fine-tuner for ticket classification."""
    
    def __init__(self):
        self.model_name = "microsoft/phi-3-mini-4k-instruct"
        self.categories = ["Authentication", "Technical", "Billing", "Feature Request", "General"]
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
    def load_data(self, data_file: str):
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Prepare the data
        texts = []
        labels = []
        
        for item in data:
            texts.append(item['customer_message'])
            labels.append(self.category_to_id[item['category']])
        
        # Create dataset
        dataset_dict = {
            'text': texts,
            'labels': labels
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Split train/validation
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} validation")
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    def setup_model(self):
        """Setup model with quantization and LoRA."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.categories),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("Model setup complete")
        self.model.print_trainable_parameters()
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
        }
    
    def train(self, dataset, output_dir="./fine_tuned_model"):
        """Train the model."""
        logger.info("Starting training...")
        # Tokenize datasets, but keep the 'labels' column
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset["train"].column_names if col != "labels"]
        )
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=400,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb for simplicity
            learning_rate=2e-4,
            fp16=True,
        )
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        # Train
        trainer.train()
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Training complete! Model saved to {output_dir}")
        return trainer
    
    def evaluate(self, trainer):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        # Get predictions
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Get detailed predictions for classification report
        predictions = trainer.predict(trainer.eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.categories,
            output_dict=True
        )
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=self.categories))
        
        return report, eval_results

def main():
    """Main training function."""
    
    # Check for data file
    data_files = [
        "data/large_synthetic_data.json",  # Our new efficient generation
        "data/synthetic_data.json",       # Original generation
        "data/test_synthetic_data.json"   # Test data
    ]
    
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            data_file = file_path
            break
    
    if not data_file:
        logger.error("No synthetic data found! Please run data generation first.")
        logger.info("Available options:")
        logger.info("1. python generate_large_dataset.py  # For 1,200 tickets")
        logger.info("2. python scripts/01_generate_synthetic_data.py  # For 1,000 tickets")
        return
    
    logger.info(f"Using data file: {data_file}")
    
    # Initialize fine-tuner
    fine_tuner = SimpleFineTuner()
    
    # Load data
    dataset = fine_tuner.load_data(data_file)
    
    # Setup model
    fine_tuner.setup_model()
    
    # Train
    trainer = fine_tuner.train(dataset)
    
    # Evaluate
    report, results = fine_tuner.evaluate(trainer)
    
    # Save results
    with open("training_results.json", "w") as f:
        json.dump({
            "classification_report": report,
            "eval_results": results,
            "model_info": {
                "base_model": fine_tuner.model_name,
                "categories": fine_tuner.categories,
                "training_data": data_file,
                "accuracy": results.get("eval_accuracy", 0.0)
            }
        }, f, indent=2)
    
    logger.info("Training complete! Results saved to training_results.json")
    logger.info(f"Final accuracy: {results.get('eval_accuracy', 0.0):.3f}")

if __name__ == "__main__":
    main()
