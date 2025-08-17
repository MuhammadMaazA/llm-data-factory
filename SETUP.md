# LLM Data Factory - Complete Setup Guide

This guide will walk you through setting up the complete LLM Data Factory system, including both the Python backend and React frontend.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Git** for version control
- **OpenAI API key** (for data generation)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/MuhammadMaazA/llm-data-factory.git
cd llm-data-factory

# Make the startup script executable
chmod +x start.sh

# Run the complete setup (this will install all dependencies)
./start.sh
```

The startup script will:
- Create a Python virtual environment
- Install Python dependencies
- Install frontend dependencies
- Start both backend and frontend servers

### 2. Set Environment Variables

```bash
# Set your OpenAI API key for data generation
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 3. Access the Application

Once the startup script completes:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## Manual Setup (Alternative)

If you prefer to set up components individually:

### Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
cd app
python api_server.py
```

### Frontend Setup

```bash
# Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev
```

## Project Structure Overview

```
llm-data-factory/
├── app/                    # Backend application
│   ├── api_server.py      # FastAPI REST API
│   ├── app.py            # Legacy Streamlit app
│   └── inference.py      # Model inference utilities
├── frontend/             # Modern React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── lib/         # API service and utilities
│   │   └── pages/       # Page components
│   └── package.json     # Frontend dependencies
├── data/                # Training and test data
├── scripts/             # Data generation and training scripts
├── notebooks/           # Jupyter notebooks for evaluation
├── start.sh            # Startup script
└── requirements.txt    # Python dependencies
```

## Usage Workflow

### 1. Generate Synthetic Data

```bash
# Ensure OpenAI API key is set
export OPENAI_API_KEY='your-key'

# Generate synthetic training data
python scripts/01_generate_synthetic_data.py
```

### 2. Fine-tune the Model

```bash
# Train the student model on synthetic data
python scripts/02_finetune_student_model.py
```

### 3. Evaluate Performance

```bash
# Open the evaluation notebook
jupyter notebook notebooks/evaluation.ipynb
```

### 4. Use the Web Interface

1. Start the servers with `./start.sh`
2. Open http://localhost:5173
3. Try the live classifier with example tickets
4. View model performance metrics

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /predict` - Classify a single support ticket
- `POST /batch-predict` - Classify multiple tickets
- `GET /model-info` - Get model information
- `GET /evaluation` - Get evaluation results
- `GET /health` - Health check

## Development

### Frontend Development

```bash
cd frontend

# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload for development
cd app
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure you've run the fine-tuning script first
2. **API connection failed**: Check that the backend server is running on port 8000
3. **Import errors**: Ensure all dependencies are installed in the virtual environment

### Port Conflicts

If you encounter port conflicts, you can modify the ports:

- Frontend: Edit `vite.config.ts` and change the port
- Backend: Modify the port in `api_server.py`

### Performance Issues

- The first prediction may be slow due to model loading
- Subsequent predictions should be much faster
- Consider using GPU acceleration for better performance

## Deployment

### Production Deployment

For production deployment, consider:

1. **Frontend**: Build and serve with a web server like Nginx
2. **Backend**: Use a production ASGI server like Gunicorn with Uvicorn workers
3. **Environment**: Set appropriate environment variables for production
4. **Security**: Configure CORS policies and API authentication

### Docker Deployment (Future Enhancement)

Docker configuration files can be added for containerized deployment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
