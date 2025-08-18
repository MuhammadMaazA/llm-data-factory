#!/usr/bin/env python3
"""
Environment Setup Verification Script
Checks that all environment variables are properly configured
"""

import os
from dotenv import load_dotenv

def check_env_setup():
    """Verify that environment is properly set up"""
    
    # Load environment variables
    load_dotenv()
    
    print("LLM Data Factory - Environment Verification")
    print("=" * 50)
    
    # Critical variables
    critical_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'API_BASE_URL': 'API Base URL',
        'STUDENT_MODEL_NAME': 'Student Model Name',
        'TEACHER_MODEL_NAME': 'Teacher Model Name'
    }
    
    # Optional variables
    optional_vars = {
        'MODEL_CACHE_DIR': 'Model Cache Directory',
        'DATA_DIR': 'Data Directory',
        'OUTPUT_MODEL_DIR': 'Output Model Directory',
        'BATCH_SIZE': 'Training Batch Size',
        'LEARNING_RATE': 'Learning Rate'
    }
    
    all_good = True
    
    print("\nCritical Variables:")
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if value:
            if var == 'OPENAI_API_KEY':
                print(f"  {description}: {'*' * 20}...{value[-4:]} [OK]")
            else:
                print(f"  {description}: {value} [OK]")
        else:
            print(f"  {description}: [MISSING]")
            all_good = False
    
    print("\nOptional Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  {description}: {value} [OK]")
        else:
            print(f"  {description}: (using default)")
    
    print("\nSecurity Check:")
    
    # Check if .env files exist
    env_files = ['.env', 'frontend/.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"  {env_file}: Found [OK]")
        else:
            print(f"  {env_file}: Missing [ERROR]")
            all_good = False
    
    # Test git ignore
    try:
        import subprocess
        result = subprocess.run(['git', 'check-ignore', '.env'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  .env properly ignored by git: [OK]")
        else:
            print("  .env git ignore: [WARNING] - .env might be tracked!")
    except:
        print("  Git ignore check: Skipped (git not available)")
    
    print("\n" + "=" * 50)
    if all_good:
        print("Environment setup is COMPLETE!")
        print("\nNext steps:")
        print("  1. Run: python scripts/01_generate_synthetic_data.py")
        print("  2. Run: python scripts/02_finetune_student_model.py")
        print("  3. Run: python run_complete_pipeline.py")
    else:
        print("Some issues found. Please check the missing variables.")
        print("\nTo fix:")
        print("  1. Check your .env file")
        print("  2. Make sure your OpenAI API key is set")
        print("  3. Run this script again")

if __name__ == "__main__":
    check_env_setup()
