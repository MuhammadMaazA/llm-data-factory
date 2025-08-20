#!/usr/bin/env python3
"""
Inference Script for Fine-tuned Customer Support Ticket Classifier

This script loads the fine-tuned Phi-3-mini model and provides easy inference
for classifying customer support tickets.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketClassifier:
    """Customer support ticket classifier using fine-tuned Phi-3-mini."""
    
    def __init__(self, model_path="./fine_tuned_model"):
        self.model_path = model_path
        self.categories = ["Authentication", "Technical", "Billing", "Feature Request", "General"]
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        base_model_name = "microsoft/phi-3-mini-4k-instruct"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(self.categories),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, self.model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def predict(self, text, return_probabilities=False):
        """
        Classify a customer support ticket.
        
        Args:
            text (str): The customer message to classify
            return_probabilities (bool): Whether to return category probabilities
            
        Returns:
            dict: Classification results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_category = self.id_to_category[predicted_id]
        confidence = probabilities[0][predicted_id].item()
        
        result = {
            "text": text,
            "predicted_category": predicted_category,
            "confidence": confidence
        }
        
        if return_probabilities:
            result["all_probabilities"] = {
                self.categories[i]: probabilities[0][i].item() 
                for i in range(len(self.categories))
            }
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Classify multiple customer support tickets.
        
        Args:
            texts (list): List of customer messages to classify
            return_probabilities (bool): Whether to return category probabilities
            
        Returns:
            list: List of classification results
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probabilities)
            results.append(result)
        return results

def demo_inference():
    """Demonstrate the classifier with sample tickets."""
    
    # Sample customer support tickets for testing
    sample_tickets = [
        "I can't log into my account. It says my password is incorrect but I'm sure it's right.",
        "The application is running very slowly and sometimes crashes when I try to upload files.",
        "I was charged twice for my subscription this month. Can you please refund the duplicate charge?",
        "Can you add a feature to export data as PDF? It would be really helpful for our reports.",
        "Hi, I just wanted to say thanks for the great customer service. Keep up the good work!"
    ]
    
    # Load classifier
    classifier = TicketClassifier()
    
    print("🎯 Customer Support Ticket Classification Demo")
    print("=" * 60)
    
    # Classify each ticket
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n📧 Ticket #{i}:")
        print(f"Message: {ticket}")
        
        result = classifier.predict(ticket, return_probabilities=True)
        
        print(f"🏷️  Category: {result['predicted_category']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        
        print("📈 All Probabilities:")
        for category, prob in result['all_probabilities'].items():
            print(f"   {category}: {prob:.3f}")
        print("-" * 60)

if __name__ == "__main__":
    demo_inference()
