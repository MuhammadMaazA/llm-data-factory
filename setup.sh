#!/bin/bash

# LLM Data Factory - Quick Setup Script
# This script helps you get started quickly with the project

echo "🚀 LLM Data Factory - Quick Setup"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🔍 Checking prerequisites..."

# Check Python
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ OpenAI API key not set"
    echo "Please set your API key:"
    echo "export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or continue without data generation (will use existing test data)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SKIP_DATA_GEN="--skip-data-generation"
else
    echo "✅ OpenAI API key found"
    SKIP_DATA_GEN=""
fi

echo "✅ Prerequisites check passed"

echo ""
echo "🛠️ What would you like to do?"
echo "1. Complete setup and training"
echo "2. Just setup environment (no training)"
echo "3. Only start demo servers"
echo "4. Run evaluation only"

read -p "Choose an option (1-4): " -n 1 -r
echo

case $REPLY in
    1)
        echo "🔄 Running complete setup and training..."
        
        # Setup environment
        echo "📦 Setting up Python environment..."
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt
        
        # Setup frontend
        echo "📦 Setting up frontend..."
        cd frontend
        if [ ! -d "node_modules" ]; then
            npm install
        fi
        cd ..
        
        # Run complete pipeline
        echo "🚀 Running complete pipeline..."
        python run_complete_pipeline.py $SKIP_DATA_GEN
        
        echo ""
        echo "✅ Setup complete! To start the demo:"
        echo "./start.sh"
        ;;
        
    2)
        echo "🔄 Setting up environment only..."
        
        # Setup environment
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt
        
        # Setup frontend
        cd frontend
        if [ ! -d "node_modules" ]; then
            npm install
        fi
        cd ..
        
        echo "✅ Environment setup complete!"
        echo ""
        echo "Next steps:"
        echo "1. Activate environment: source venv/bin/activate"
        echo "2. Train model: python run_complete_pipeline.py"
        echo "3. Start demo: ./start.sh"
        ;;
        
    3)
        echo "🔄 Starting demo servers..."
        
        if [ ! -d "venv" ] || [ ! -d "frontend/node_modules" ]; then
            echo "❌ Environment not set up. Please run option 1 or 2 first."
            exit 1
        fi
        
        ./start.sh
        ;;
        
    4)
        echo "🔄 Running evaluation only..."
        
        if [ ! -d "venv" ]; then
            echo "❌ Python environment not set up."
            exit 1
        fi
        
        source venv/bin/activate
        python run_complete_pipeline.py --eval-only
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
