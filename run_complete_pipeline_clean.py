#!/usr/bin/env python3
"""
Complete Pipeline for LLM Data Factory
Runs the entire process from data generation to model evaluation
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"{description} completed successfully")
        else:
            print(f"Warning: {description} completed with warnings")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed: {e}")
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check if in correct directory
    if not Path("requirements.txt").exists():
        print("Error: Please run this script from the project root directory")
        return False
    
    # Check if virtual environment is active
    if not os.environ.get('VIRTUAL_ENV'):
        print("Warning: Virtual environment not detected")
        print("It's recommended to run this in a virtual environment")
    
    # Check if seed examples exist
    if not Path("data/seed_examples.json").exists():
        print("Error: Seed examples not found: data/seed_examples.json")
        print("Please create seed examples first")
        return False
    
    # Check if test data exists
    if not Path("data/test_data.json").exists():
        print("Error: Test data not found: data/test_data.json")
        print("Please create test data first")
        return False
    
    print("Prerequisites check passed")
    return True


def install_dependencies():
    """Install required dependencies."""
    if run_command("pip install -r requirements.txt", "Installing dependencies"):
        # Install additional dependencies that might be needed
        run_command("pip install python-dotenv", "Installing python-dotenv", check=False)
        run_command("pip install jupyter", "Installing Jupyter", check=False)
        return True
    return False


def main():
    """Main pipeline execution."""
    start_time = datetime.now()
    print("LLM Data Factory - Complete Pipeline")
    print("=" * 50)
    print(f"Started at: {start_time}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check API key
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key set")
    else:
        print("Warning: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        response = input("Continue without API key? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install dependencies
    print("\nStep 1: Installing Dependencies")
    if not install_dependencies():
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Step 2: Generate synthetic data
    print("\nStep 2: Data Generation")
    if Path("data/synthetic_data.json").exists():
        with open("data/synthetic_data.json", 'r') as f:
            try:
                data = json.load(f)
                if data and len(data) > 0:
                    print("Synthetic data already exists")
                    response = input("Regenerate data? (y/n): ")
                    if response.lower() != 'y':
                        print("Skipping data generation")
                        goto_training = True
                    else:
                        goto_training = False
                else:
                    goto_training = False
            except:
                goto_training = False
    else:
        goto_training = False
    
    if not goto_training:
        if not run_command("python scripts/01_generate_synthetic_data.py", 
                          "Generating synthetic data"):
            print("Data generation failed. Check your API key and try again.")
            sys.exit(1)
    else:
        print(f"\nSkipping data generation")
    
    # Step 3: Fine-tune model
    print("\nStep 3: Model Training")
    if Path("final_student_model").exists():
        print("Trained model already exists")
        response = input("Retrain model? (y/n): ")
        if response.lower() != 'y':
            print("Skipping model training")
            goto_evaluation = True
        else:
            goto_evaluation = False
    else:
        goto_evaluation = False
    
    if not goto_evaluation:
        if not run_command("python scripts/02_finetune_student_model.py", 
                          "Fine-tuning student model"):
            print("Model training failed. This requires significant computational resources.")
            print("Consider running on a machine with GPU support.")
            response = input("Continue to evaluation with base model? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        print(f"\nSkipping model training")
    
    # Step 4: Run evaluation
    print("\nStep 4: Model Evaluation")
    if not Path("final_student_model").exists() and not Path("notebooks/evaluation.ipynb").exists():
        print("Error: No trained model found. Cannot run evaluation.")
        sys.exit(1)
    
    print("Running evaluation notebook...")
    
    # Try to run the evaluation notebook
    try:
        # First, try with jupyter
        result = subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace",
            "notebooks/evaluation.ipynb"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            # Fallback: just inform user to run manually
            print("Automated notebook execution failed.")
            print("Please run the evaluation notebook manually:")
            print("jupyter notebook notebooks/evaluation.ipynb")
        else:
            print("Evaluation notebook executed successfully")
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Could not execute notebook automatically.")
        print("Please run the evaluation manually:")
        print("jupyter notebook notebooks/evaluation.ipynb")
    
    # Step 5: Start demo (optional)
    print("\nStep 5: Demo Application")
    response = input("Start the demo application? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting demo application...")
        print("Backend will start on http://localhost:8000")
        print("Frontend will start on http://localhost:5173")
        print("\nPress Ctrl+C to stop the servers when done.")
        
        try:
            # Start backend in background
            backend_process = subprocess.Popen([
                "python", "app/api_server.py"
            ], cwd=os.getcwd())
            
            # Wait a moment for backend to start
            time.sleep(3)
            
            # Start frontend
            frontend_process = subprocess.Popen([
                "npm", "run", "dev"
            ], cwd="frontend")
            
            # Wait for user to stop
            input("\nPress Enter to stop the demo servers...")
            
            # Clean up processes
            backend_process.terminate()
            frontend_process.terminate()
            
        except KeyboardInterrupt:
            print("\nStopping demo servers...")
        except Exception as e:
            print(f"Error running demo: {e}")
            print("You can start the servers manually:")
            print("Backend: cd app && python api_server.py")
            print("Frontend: cd frontend && npm run dev")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print(f"Total time: {duration}")
    print("\nWhat's been completed:")
    print("- Dependencies installed")
    print("- Synthetic data generated (if applicable)")
    print("- Model fine-tuned (if applicable)")
    print("- Evaluation ready to run")
    print("\nNext steps:")
    print("1. Review evaluation results in notebooks/evaluation.ipynb")
    print("2. Test the API: python test_api.py")
    print("3. Start the demo: cd app && python api_server.py")
    print("4. Start the frontend: cd frontend && npm run dev")


if __name__ == "__main__":
    main()
