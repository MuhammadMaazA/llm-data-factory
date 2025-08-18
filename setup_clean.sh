#!/bin/bash

# LLM Data Factory - Interactive Setup Script
# This script provides an interactive menu for setting up and running the project

echo "LLM Data Factory - Quick Setup"
echo "================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Checking prerequisites..."

# Check Python
if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check Node.js  
if ! command_exists node; then
    echo "Error: Node.js is required but not installed"
    exit 1
fi

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ] && [ ! -f ".env" ]; then
    echo "Warning: OpenAI API key not set"
    echo "Please either:"
    echo "  1. Set OPENAI_API_KEY environment variable"
    echo "  2. Create .env file with OPENAI_API_KEY=your-key"
    echo ""
    echo "You can get an API key from: https://platform.openai.com/api-keys"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "OpenAI API key found"
fi

echo "Prerequisites check passed"

# Main menu
echo "What would you like to do?"
echo "1. Complete setup + train model (recommended for first time)"
echo "2. Setup environment only"  
echo "3. Start demo servers"
echo "4. Run evaluation only"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running complete setup and training..."
        
        # Setup environment
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt
        
        # Setup frontend
        cd frontend
        npm install
        cd ..
        
        # Copy env template if needed
        if [ ! -f ".env" ]; then
            cp .env.template .env
            echo "Created .env file. Please add your OpenAI API key!"
        fi
        
        echo "Running complete pipeline..."
        python run_complete_pipeline.py
        
        echo "Setup complete! To start the demo:"
        echo "  Backend: cd app && python api_server.py"
        echo "  Frontend: cd frontend && npm run dev"
        ;;
    2)
        echo "Setting up environment only..."
        
        # Setup Python environment
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt
        
        # Setup frontend
        cd frontend
        npm install
        cd ..
        
        # Copy env template if needed
        if [ ! -f ".env" ]; then
            cp .env.template .env
            echo "Created .env file. Please add your OpenAI API key!"
        fi
        
        echo "Environment setup complete!"
        echo "Next steps:"
        echo "  1. Add your OpenAI API key to .env file"
        echo "  2. Run: python run_complete_pipeline.py"
        echo "  3. Or use option 3 to start demo servers"
        ;;
    3)
        echo "Starting demo servers..."
        
        if [ ! -d "venv" ] || [ ! -d "frontend/node_modules" ]; then
            echo "Error: Environment not set up. Please run option 1 or 2 first."
            exit 1
        fi
        
        source venv/bin/activate
        ./start.sh
        ;;
    4)
        echo "Running evaluation only..."
        
        if [ ! -d "venv" ]; then
            echo "Error: Python environment not set up."
            echo "Please run option 1 or 2 first."
            exit 1
        fi
        
        source venv/bin/activate
        jupyter notebook notebooks/evaluation.ipynb
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Error: Invalid option"
        exit 1
        ;;
esac
