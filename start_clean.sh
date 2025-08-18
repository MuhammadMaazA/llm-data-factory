#!/bin/bash

# LLM Data Factory - Quick Start Script
# This script starts both the backend API and frontend development server

echo "Starting LLM Data Factory..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

if ! command_exists node; then
    echo "Error: Node.js is required but not installed"
    exit 1
fi

if ! command_exists npm; then
    echo "Error: npm is required but not installed"
    exit 1
fi

# Check for pip
if ! command_exists pip; then
    echo "Error: pip is required but not installed"
    exit 1
fi

# Check if Python environment is set up
echo "Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Python virtual environment found"
else
    echo "Error: Could not activate virtual environment"
    exit 1
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if python -c "import fastapi, uvicorn, transformers" 2>/dev/null; then
    # Check if dependencies are up to date
    if pip list --outdated | grep -E "(fastapi|uvicorn|transformers)" > /dev/null; then
        echo "Installing/updating Python dependencies..."
        pip install -r requirements.txt
    else
        echo "Python dependencies up to date"
    fi
else
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check if frontend dependencies exist
if [ -d "frontend/node_modules" ]; then
    echo "Frontend dependencies found"
    
    # Check if package.json has been updated
    cd frontend
    if [ package.json -nt node_modules ]; then
        echo "Installing updated frontend dependencies..."
        npm install
    else
        echo "Frontend dependencies up to date"
    fi
    cd ..
else
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Kill any existing processes on the ports we want to use
echo "Checking for existing processes..."
pkill -f "uvicorn.*8000" 2>/dev/null || true
pkill -f "npm.*dev" 2>/dev/null || true
sleep 2

# Start backend API server
echo "Starting backend API server..."
cd app
if python -c "import fastapi, uvicorn" 2>/dev/null; then
    python api_server.py &
    BACKEND_PID=$!
else
    echo "Error: Required Python packages not found. Installing dependencies..."
    cd ..
    pip install -r requirements.txt
    cd app
    python api_server.py &
    BACKEND_PID=$!
fi
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend development server  
echo "Starting frontend development server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Both servers started!"

echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:5173"

echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
trap 'echo "Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
