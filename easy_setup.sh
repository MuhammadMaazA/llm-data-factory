#!/bin/bash

# LLM Data Factory Setup Script
# This script helps you set up the entire project

echo "==========================================="
echo "  LLM Data Factory Setup Script"
echo "==========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

echo "Step 1: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Error: Could not activate virtual environment"
    exit 1
fi

echo ""
echo "Step 2: Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Step 3: Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "Created .env file from template"
    echo "Please edit .env and add your OpenAI API key"
else
    echo ".env file already exists"
fi

echo ""
echo "Step 4: Checking data files..."
if [ ! -f "data/seed_examples.json" ]; then
    echo "Warning: data/seed_examples.json not found"
else
    echo "Seed examples found"
fi

if [ ! -f "data/test_data.json" ]; then
    echo "Warning: data/test_data.json not found"
else
    echo "Test data found"
fi

echo ""
echo "Step 5: Installing frontend dependencies..."
if command_exists npm; then
    cd frontend
    npm install
    cd ..
    echo "Frontend dependencies installed"
else
    echo "Warning: npm not found. Frontend dependencies not installed."
    echo "Please install Node.js to use the frontend"
fi

echo ""
echo "==========================================="
echo "  Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run the complete pipeline:"
echo "   python run_complete_pipeline_clean.py"
echo ""
echo "Or run components individually:"
echo "- Generate data: python scripts/01_generate_synthetic_data.py"
echo "- Train model: python scripts/02_finetune_student_model.py" 
echo "- Start API: cd app && python api_server.py"
echo "- Start frontend: cd frontend && npm run dev"
echo ""
echo "For detailed instructions, see COMPLETE_SETUP_GUIDE.md"
