"""
Inference Module for LLM Data Factory

This module provides functionality to load the fine-tuned student model and make predictions
on new customer support tickets. It includes caching, batch processing, and confidence scoring.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TicketClassifier:
    """Class for making predictions with the fine-tuned student model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the classifier with the trained model."""
        self.model_path = model_path or "./final_student_model"
        self.categories = ["Urgent Bug", "Feature Request", "How-To Question"]
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.categories)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Check if model directory exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.categories),
                trust_remote_code=True,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Preprocess a single text input."""
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return encoding
    
    def predict_single(self, text: str, return_confidence: bool = True) -> Union[str, Tuple[str, float]]:
        """Make a prediction for a single text input."""
        try:
            # Preprocess
            encoding = self.preprocess_text(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
            
            # Get category
            category = self.id_to_category[prediction]
            
            if return_confidence:
                return category, confidence
            else:
                return category
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> List[Union[str, Tuple[str, float]]]:
        """Make predictions for a batch of texts."""
        try:
            results = []
            
            for text in texts:
                result = self.predict_single(text, return_confidence)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def predict_with_details(self, text: str) -> Dict:
        """Make a prediction with detailed information including all class probabilities."""
        try:
            # Preprocess
            encoding = self.preprocess_text(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
            
            # Get category
            category = self.id_to_category[prediction]
            
            # Get all probabilities
            all_probabilities = probabilities.cpu().numpy()[0]
            class_probabilities = {
                self.id_to_category[i]: float(prob)
                for i, prob in enumerate(all_probabilities)
            }
            
            return {
                'predicted_category': category,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'input_text': text,
                'model_path': self.model_path
            }
            
        except Exception as e:
            logger.error(f"Detailed prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'categories': self.categories,
            'num_categories': len(self.categories),
            'device': str(self.device),
            'model_type': type(self.model).__name__,
            'tokenizer_type': type(self.tokenizer).__name__
        }


class PredictionCache:
    """Simple cache for predictions to avoid redundant computations."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the cache."""
        self.max_size = max_size
        self.cache = {}
    
    def get(self, text: str) -> Optional[Dict]:
        """Get a cached prediction."""
        return self.cache.get(text)
    
    def set(self, text: str, prediction: Dict):
        """Cache a prediction."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = prediction
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class EnhancedTicketClassifier(TicketClassifier):
    """Enhanced classifier with caching and additional features."""
    
    def __init__(self, model_path: Optional[str] = None, use_cache: bool = True):
        """Initialize the enhanced classifier."""
        super().__init__(model_path)
        self.use_cache = use_cache
        self.cache = PredictionCache() if use_cache else None
    
    def predict_single(self, text: str, return_confidence: bool = True) -> Union[str, Tuple[str, float]]:
        """Make a prediction with caching."""
        if self.use_cache and self.cache:
            cached_result = self.cache.get(text)
            if cached_result:
                logger.debug("Using cached prediction")
                if return_confidence:
                    return cached_result['category'], cached_result['confidence']
                else:
                    return cached_result['category']
        
        # Make prediction
        result = super().predict_single(text, return_confidence)
        
        # Cache result
        if self.use_cache and self.cache:
            if return_confidence:
                category, confidence = result
                self.cache.set(text, {'category': category, 'confidence': confidence})
            else:
                self.cache.set(text, {'category': result, 'confidence': 1.0})
        
        return result
    
    def predict_with_details(self, text: str) -> Dict:
        """Make a detailed prediction with caching."""
        if self.use_cache and self.cache:
            cached_result = self.cache.get(text)
            if cached_result and 'detailed' in cached_result:
                logger.debug("Using cached detailed prediction")
                return cached_result['detailed']
        
        # Make prediction
        result = super().predict_with_details(text)
        
        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(text, {'detailed': result})
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': self.cache.size(),
            'max_cache_size': self.cache.max_size
        }


def load_classifier(model_path: Optional[str] = None, use_cache: bool = True) -> EnhancedTicketClassifier:
    """Convenience function to load the classifier."""
    try:
        classifier = EnhancedTicketClassifier(model_path, use_cache)
        logger.info("Classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        raise


def predict_ticket_category(text: str, model_path: Optional[str] = None) -> Dict:
    """Simple function to predict ticket category."""
    classifier = load_classifier(model_path)
    return classifier.predict_with_details(text)


if __name__ == "__main__":
    # Example usage
    try:
        # Load classifier
        classifier = load_classifier()
        
        # Example predictions
        test_texts = [
            "I can't log into my account. It keeps saying invalid credentials.",
            "I would love to see a dark mode option for the interface.",
            "How do I export my data to CSV format?"
        ]
        
        print("Model Info:")
        print(json.dumps(classifier.get_model_info(), indent=2))
        print("\nPredictions:")
        
        for text in test_texts:
            result = classifier.predict_with_details(text)
            print(f"\nText: {text}")
            print(f"Prediction: {result['predicted_category']}")
            print(f"Confidence: {result['confidence']:.3f}")
        
        print(f"\nCache Stats: {classifier.get_cache_stats()}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
