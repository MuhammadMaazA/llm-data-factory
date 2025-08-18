#!/usr/bin/env python3
"""
Complete Training Pipeline for LLM Data Factory

This script runs the complete pipeline:
1. Generate synthetic data (optional)
2. Fine-tune the student model
3. Run evaluation
4. Generate report

Usage:
    python run_complete_pipeline.py [--skip-data-generation] [--openai-key YOUR_KEY]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"⚠️ {description} completed with warnings")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if in correct directory
    if not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Check if virtual environment is activated
    if sys.prefix == sys.base_prefix:
        print("⚠️ Warning: Virtual environment not detected")
        print("   Consider activating your virtual environment first:")
        print("   source venv/bin/activate")
    
    # Check for seed examples
    if not Path("data/seed_examples.json").exists():
        print("❌ Seed examples not found: data/seed_examples.json")
        return False
    
    # Check for test data
    if not Path("data/test_data.json").exists():
        print("❌ Test data not found: data/test_data.json")
        return False
    
    print("✅ Prerequisites check passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete LLM Data Factory pipeline")
    parser.add_argument("--skip-data-generation", action="store_true", 
                       help="Skip synthetic data generation step")
    parser.add_argument("--openai-key", type=str, 
                       help="OpenAI API key (alternatively set OPENAI_API_KEY env var)")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (assumes model is already trained)")
    
    args = parser.parse_args()
    
    print("🚀 LLM Data Factory - Complete Pipeline")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Set OpenAI API key if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
        print("✅ OpenAI API key set")
    elif not os.getenv("OPENAI_API_KEY") and not args.skip_data_generation:
        print("⚠️ Warning: OPENAI_API_KEY not set")
        print("   Data generation will be skipped")
        args.skip_data_generation = True
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Generate synthetic data (optional)
    if not args.skip_data_generation and not args.eval_only:
        print(f"\n{'='*20} STEP 1: DATA GENERATION {'='*20}")
        
        if Path("data/synthetic_data.json").exists():
            print("✅ Synthetic data already exists")
            choice = input("Regenerate? (y/N): ").lower().strip()
            if choice != 'y':
                print("⏭️ Skipping data generation")
                success_steps.append("Data Generation (skipped)")
            else:
                if run_command("python scripts/01_generate_synthetic_data.py", 
                              "Generating synthetic data"):
                    success_steps.append("Data Generation")
                else:
                    failed_steps.append("Data Generation")
        else:
            if run_command("python scripts/01_generate_synthetic_data.py", 
                          "Generating synthetic data"):
                success_steps.append("Data Generation")
            else:
                failed_steps.append("Data Generation")
    else:
        print(f"\n⏭️ Skipping data generation")
        success_steps.append("Data Generation (skipped)")
    
    # Step 2: Fine-tune the model
    if not args.eval_only:
        print(f"\n{'='*20} STEP 2: MODEL TRAINING {'='*20}")
        
        if Path("final_student_model").exists():
            print("✅ Trained model already exists")
            choice = input("Retrain? (y/N): ").lower().strip()
            if choice != 'y':
                print("⏭️ Skipping model training")
                success_steps.append("Model Training (skipped)")
            else:
                if run_command("python scripts/02_finetune_student_model.py", 
                              "Fine-tuning student model"):
                    success_steps.append("Model Training")
                else:
                    failed_steps.append("Model Training")
        else:
            if run_command("python scripts/02_finetune_student_model.py", 
                          "Fine-tuning student model"):
                success_steps.append("Model Training")
            else:
                failed_steps.append("Model Training")
    else:
        print(f"\n⏭️ Skipping model training")
    
    # Step 3: Run evaluation
    print(f"\n{'='*20} STEP 3: EVALUATION {'='*20}")
    
    # Check if model exists
    if not Path("final_student_model").exists():
        print("❌ No trained model found. Cannot run evaluation.")
        failed_steps.append("Evaluation")
    else:
        print("🔄 Running evaluation notebook...")
        print("Note: This will execute the evaluation notebook")
        
        # Try to run evaluation notebook
        eval_command = """
cd notebooks && python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

try:
    with open('evaluation.ipynb', 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '../'}})
    
    with open('evaluation.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print('✅ Evaluation notebook executed successfully')
except Exception as e:
    print(f'❌ Error executing notebook: {e}')
    sys.exit(1)
"
"""
        
        if run_command(eval_command, "Running evaluation notebook", check=False):
            success_steps.append("Evaluation")
        else:
            # Fallback: try to run individual evaluation script
            print("⚠️ Notebook execution failed, trying alternative method...")
            
            eval_script = """
import sys
sys.path.append('..')
from app.inference import load_classifier, predict_ticket_category
import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

print("Running basic evaluation...")

# Load test data
with open('../data/test_data.json', 'r') as f:
    test_data = json.load(f)

test_df = pd.DataFrame(test_data)
print(f"Loaded {len(test_df)} test samples")

# Load model
classifier = load_classifier()
if classifier is None:
    print("Failed to load model")
    exit(1)

# Generate predictions
predictions = []
for _, row in test_df.iterrows():
    result = predict_ticket_category(classifier, row['customer_message'])
    predictions.append(result['predicted_category'])

# Calculate accuracy
accuracy = accuracy_score(test_df['category'], predictions)
print(f"Overall Accuracy: {accuracy:.3f}")

# Print classification report
report = classification_report(test_df['category'], predictions)
print("Classification Report:")
print(report)

print("✅ Basic evaluation completed")
"""
            
            eval_file = Path("temp_eval.py")
            eval_file.write_text(eval_script)
            
            if run_command("python temp_eval.py", "Running basic evaluation", check=False):
                success_steps.append("Evaluation (basic)")
                eval_file.unlink()  # Clean up
            else:
                failed_steps.append("Evaluation")
                if eval_file.exists():
                    eval_file.unlink()
    
    # Final report
    print(f"\n{'='*20} PIPELINE SUMMARY {'='*20}")
    
    print(f"\n✅ Successful Steps ({len(success_steps)}):")
    for step in success_steps:
        print(f"   • {step}")
    
    if failed_steps:
        print(f"\n❌ Failed Steps ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"   • {step}")
    
    print(f"\n📁 Generated Files:")
    files_to_check = [
        ("data/synthetic_data.json", "Synthetic training data"),
        ("final_student_model", "Fine-tuned model"),
        ("results/evaluation_summary.json", "Evaluation results"),
        ("results/detailed_predictions.csv", "Detailed predictions")
    ]
    
    for file_path, description in files_to_check:
        if Path(file_path).exists():
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} (not found)")
    
    # Next steps
    print(f"\n🎯 Next Steps:")
    if not failed_steps:
        print("   ✅ Pipeline completed successfully!")
        print("   • Start your demo app: ./start.sh")
        print("   • View results in notebooks/evaluation.ipynb")
        print("   • Check evaluation results in results/ folder")
    else:
        print("   ⚠️ Some steps failed. Please:")
        print("   • Check error messages above")
        print("   • Ensure all dependencies are installed")
        print("   • Verify your OpenAI API key (if using data generation)")
        print("   • Run individual scripts manually to debug")
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
