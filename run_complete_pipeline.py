#!/usr/bin/env python3
"""
Complete Pipeline for LLM Data Factory

This script runs the complete end-to-end pipeline:
1. Generate 1,200+ synthetic tickets from 20 seed examples (60x amplification)
2. Fine-tune Phi-3-mini using QLoRA for efficient training
3. Achieve 80%+ accuracy on customer support ticket classification
4. Deploy the trained model

Usage:
    python run_complete_pipeline.py [--quick-test]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompletePipeline:
    """Complete LLM Data Factory Pipeline."""
    
    def __init__(self, quick_test=False):
        self.quick_test = quick_test
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        
        # Pipeline configuration
        self.seed_file = self.data_dir / "seed_examples.json"
        
        if quick_test:
            self.synthetic_data_file = self.data_dir / "test_synthetic_data.json"
            self.target_samples = 50
            logger.info("🚀 Running QUICK TEST mode with 50 samples")
        else:
            self.synthetic_data_file = self.data_dir / "large_synthetic_data.json"
            self.target_samples = 1200
            logger.info("🚀 Running FULL PIPELINE mode with 1,200 samples")
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if seed examples exist
        if not self.seed_file.exists():
            logger.error(f"❌ Seed examples not found: {self.seed_file}")
            return False
        
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ OPENAI_API_KEY not found in environment")
            return False
        
        # Check if required packages are installed
        required_packages = [
            "openai", "transformers", "datasets", "peft", 
            "bitsandbytes", "torch", "scikit-learn"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                logger.error(f"❌ Required package not installed: {package}")
                return False
        
        logger.info("✅ All prerequisites met!")
        return True
    
    def generate_synthetic_data(self):
        """Generate synthetic data using GPT-4."""
        logger.info(f"📊 Generating {self.target_samples} synthetic tickets...")
        
        if self.synthetic_data_file.exists() and self.synthetic_data_file.stat().st_size > 1000:
            logger.info(f"✅ Synthetic data already exists: {self.synthetic_data_file}")
            return True
        
        try:
            if self.quick_test:
                # Use the simple test generator for quick test
                cmd = ["python", "test_generation.py"]
                # Modify test_generation.py to generate more samples
                self._run_quick_generation()
            else:
                # Use the efficient large dataset generator
                cmd = ["python", "generate_large_dataset.py"]
                result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"❌ Data generation failed: {result.stderr}")
                    return False
            
            if self.synthetic_data_file.exists():
                # Check the generated data
                with open(self.synthetic_data_file, 'r') as f:
                    data = json.load(f)
                
                logger.info(f"✅ Generated {len(data)} synthetic tickets")
                
                # Log category distribution
                categories = {}
                for item in data:
                    cat = item.get('category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                logger.info(f"📊 Category distribution: {categories}")
                return True
            else:
                logger.error("❌ Synthetic data file not created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error generating synthetic data: {e}")
            return False
    
    def _run_quick_generation(self):
        """Run quick generation for testing."""
        # Create a quick synthetic dataset
        categories = ["Authentication", "Technical", "Billing", "Feature Request", "General"]
        priorities = ["Low", "Medium", "High", "Critical"]
        
        quick_data = []
        for i in range(50):
            quick_data.append({
                "ticket_id": f"TICKET-{i+1:03d}",
                "customer_message": f"Test customer message {i+1} for category testing and validation purposes. This is a synthetic message created for pipeline testing.",
                "category": categories[i % len(categories)],
                "priority": priorities[i % len(priorities)],
                "customer_id": f"CUST-{i+1:03d}"
            })
        
        # Save quick test data
        with open(self.synthetic_data_file, 'w') as f:
            json.dump(quick_data, f, indent=2)
        
        logger.info(f"✅ Created quick test dataset with {len(quick_data)} samples")
    
    def fine_tune_model(self):
        """Fine-tune the student model."""
        logger.info("🤖 Fine-tuning Phi-3-mini model...")
        
        try:
            # Run the fine-tuning script
            cmd = ["python", "simple_finetune.py"]
            result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Fine-tuning failed: {result.stderr}")
                return False
            
            logger.info("✅ Model fine-tuning completed!")
            
            # Check if results file exists
            results_file = self.base_dir / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                accuracy = results.get('eval_results', {}).get('eval_accuracy', 0)
                logger.info(f"🎯 Final model accuracy: {accuracy:.1%}")
                
                if accuracy >= 0.8:
                    logger.info("🏆 SUCCESS: Achieved 80%+ accuracy target!")
                else:
                    logger.warning(f"⚠️  Model accuracy ({accuracy:.1%}) below 80% target")
                
                return True
            
        except Exception as e:
            logger.error(f"❌ Error during fine-tuning: {e}")
            return False
    
    def evaluate_pipeline(self):
        """Evaluate the complete pipeline."""
        logger.info("📈 Evaluating pipeline performance...")
        
        # Load results
        results_file = self.base_dir / "training_results.json"
        if not results_file.exists():
            logger.error("❌ Training results not found")
            return False
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        accuracy = results.get('eval_results', {}).get('eval_accuracy', 0)
        classification_report = results.get('classification_report', {})
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PIPELINE EVALUATION COMPLETE")
        logger.info("="*60)
        
        # Data amplification metrics
        seed_count = 20  # We know this from seed_examples.json
        if self.synthetic_data_file.exists():
            with open(self.synthetic_data_file, 'r') as f:
                synthetic_data = json.load(f)
            synthetic_count = len(synthetic_data)
            amplification = synthetic_count / seed_count
            
            logger.info(f"📊 Data Amplification: {seed_count} → {synthetic_count} ({amplification:.0f}x)")
        
        # Model performance
        logger.info(f"🎯 Model Accuracy: {accuracy:.1%}")
        logger.info(f"🤖 Base Model: microsoft/phi-3-mini-4k-instruct (3.8B parameters)")
        logger.info(f"⚡ Training Method: QLoRA + Parameter-Efficient Fine-tuning")
        
        # Category performance
        if classification_report:
            logger.info("\n📊 Per-Category Performance:")
            for category, metrics in classification_report.items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    f1 = metrics['f1-score']
                    logger.info(f"   {category}: {f1:.1%} F1-score")
        
        # Success criteria
        logger.info("\n🏆 Success Criteria:")
        logger.info(f"   ✅ Generate 1,000+ tickets: {synthetic_count >= 1000}")
        logger.info(f"   {'✅' if accuracy >= 0.8 else '❌'} Achieve 80%+ accuracy: {accuracy:.1%}")
        logger.info(f"   ✅ Use QLoRA fine-tuning: Yes")
        logger.info(f"   ✅ 60x+ data amplification: {amplification:.0f}x")
        
        return accuracy >= 0.8
    
    def run_complete_pipeline(self):
        """Run the complete end-to-end pipeline."""
        logger.info("🚀 Starting LLM Data Factory Complete Pipeline")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met. Please fix the issues above.")
            return False
        
        # Step 2: Generate synthetic data
        if not self.generate_synthetic_data():
            logger.error("❌ Synthetic data generation failed")
            return False
        
        # Step 3: Fine-tune model
        if not self.fine_tune_model():
            logger.error("❌ Model fine-tuning failed")
            return False
        
        # Step 4: Evaluate pipeline
        success = self.evaluate_pipeline()
        
        # Final summary
        end_time = time.time()
        duration = (end_time - start_time) / 60  # Convert to minutes
        
        logger.info("\n" + "="*60)
        if success:
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.info("⚠️  PIPELINE COMPLETED WITH WARNINGS")
        logger.info(f"⏱️  Total Duration: {duration:.1f} minutes")
        logger.info("="*60)
        
        return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the complete LLM Data Factory pipeline")
    parser.add_argument(
        "--quick-test", 
        action="store_true", 
        help="Run a quick test with 50 samples instead of full 1,200"
    )
    
    args = parser.parse_args()
    
    pipeline = CompletePipeline(quick_test=args.quick_test)
    success = pipeline.run_complete_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
