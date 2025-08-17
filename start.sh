#!/bin/bash

# Startup script for LLM Data Factory
# This script starts both the backend API and frontend development server

echo "🚀 Starting LLM Data Factory..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is required but not installed"
    exit 1
fi

# Check if we have pip
if ! command_exists pip; then
    echo "❌ pip is required but not installed"
    exit 1
fi

# Setup Python virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Python virtual environment found"
    source venv/bin/activate
    
    # Check if dependencies are installed by testing a few key imports
    echo "🔍 Checking Python dependencies..."
    if ! python -c "import fastapi, uvicorn, transformers" 2>/dev/null; then
        echo "📦 Installing/updating Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "✅ Python dependencies up to date"
    fi
fi

# Setup frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
else
    echo "✅ Frontend dependencies found"
    # Check if package.json is newer than node_modules
    if [ "frontend/package.json" -nt "frontend/node_modules" ]; then
        echo "📦 Updating frontend dependencies..."
        cd frontend
        npm install
        cd ..
    else
        echo "✅ Frontend dependencies up to date"
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Set up cleanup trap
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔧 Starting backend API server..."
cd app

# Verify that required Python modules are available
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ Required Python packages not found. Installing dependencies..."
    cd ..
    pip install -r requirements.txt
    cd app
fi

python api_server.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Start frontend development server
echo "🎨 Starting frontend development server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Both servers started!"
echo "🌐 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop the script
wait
