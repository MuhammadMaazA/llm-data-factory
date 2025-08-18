# Complete LLM Data Factory Setup Guide

Follow this guide to get your LLM Data Factory project running from scratch!

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for data generation)
- 8GB+ RAM (16GB+ recommended for model training)
- GPU recommended for faster training (optional)

## Step 1: Environment Setup

### 1.1 Clone and Navigate
```bash
git clone https://github.com/MuhammadMaazA/llm-data-factory.git
cd llm-data-factory
```

### 1.2 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Configure API Key

### 2.1 Create .env File
```bash
cp .env.template .env
```

### 2.2 Add Your OpenAI API Key
Edit the `.env` file and replace `your-openai-api-key-here` with your actual OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

**Important:** Never commit your `.env` file to version control!

## Step 3: Run the Complete Pipeline

### Option A: Automated Pipeline (Recommended)
```bash
python run_complete_pipeline_clean.py
```

This will:
- Install dependencies
- Generate synthetic data
- Fine-tune the model
- Run evaluation
- Optionally start the demo

### Option B: Manual Step-by-Step

#### 3.1 Generate Synthetic Data
```bash
python scripts/01_generate_synthetic_data.py
```
This creates `data/synthetic_data.json` using GPT-4 and your seed examples.

#### 3.2 Fine-tune the Student Model
```bash
python scripts/02_finetune_student_model.py
```
This creates the `final_student_model/` directory with your trained model.

#### 3.3 Run Evaluation
```bash
jupyter notebook notebooks/evaluation.ipynb
```
Open and run all cells to see model performance metrics.

## Step 4: Test the API

### 4.1 Start the Backend Server
```bash
cd app
python api_server.py
```
The API will be available at `http://localhost:8000`

### 4.2 Test API Endpoints
In a new terminal:
```bash
python test_api.py
```

## Step 5: Launch the Frontend Demo

### 5.1 Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 5.2 Start the Frontend
```bash
npm run dev
```
The demo will be available at `http://localhost:5173`

## Step 6: Validate Everything Works

1. **Check data files exist:**
   - `data/seed_examples.json` (provided)
   - `data/test_data.json` (provided)
   - `data/synthetic_data.json` (generated)

2. **Check model files exist:**
   - `final_student_model/` directory (after training)

3. **Test the complete flow:**
   - Backend API responds at `http://localhost:8000/health`
   - Frontend loads at `http://localhost:5173`
   - Can classify tickets through the web interface

## Troubleshooting

### Common Issues

**1. API Key Issues**
- Make sure your `.env` file exists and contains your real API key
- Verify the key starts with `sk-`
- Check you have OpenAI credits available

**2. Memory Issues During Training**
- Reduce batch size in `scripts/02_finetune_student_model.py`
- Use a machine with more RAM
- Consider using Google Colab for training

**3. Package Installation Issues**
- Make sure you're in a virtual environment
- Try upgrading pip: `pip install --upgrade pip`
- For PyTorch issues, visit: https://pytorch.org/get-started/locally/

**4. Frontend Issues**
- Make sure Node.js is installed
- Try clearing npm cache: `npm cache clean --force`
- Delete `node_modules` and run `npm install` again

### Performance Notes

- **Data Generation:** Takes 5-15 minutes depending on your API plan
- **Model Training:** Can take 30 minutes to several hours depending on hardware
- **Evaluation:** Takes 2-5 minutes
- **API/Frontend:** Should start in under 30 seconds

## What Each Component Does

### Data Generation (`scripts/01_generate_synthetic_data.py`)
- Uses GPT-4 to create realistic customer support tickets
- Generates diverse examples based on your seed data
- Creates balanced datasets across different categories

### Model Training (`scripts/02_finetune_student_model.py`)
- Fine-tunes Microsoft Phi-3-mini model using QLoRA
- Trains on your synthetic data for ticket classification
- Saves optimized model for fast inference

### Evaluation (`notebooks/evaluation.ipynb`)
- Tests model performance on held-out test data
- Generates classification reports and confusion matrices
- Compares performance against baseline models

### API Server (`app/api_server.py`)
- FastAPI server with REST endpoints
- Handles single and batch predictions
- Provides model information and health checks

### Frontend (`frontend/`)
- Modern React application with TypeScript
- Interactive demo for real-time ticket classification
- Beautiful UI with confidence visualization

## Next Steps

Once everything is working:

1. **Customize for your use case:**
   - Modify seed examples for your domain
   - Adjust categories in the training script
   - Customize the frontend branding

2. **Improve the model:**
   - Generate more training data
   - Experiment with different base models
   - Try different training parameters

3. **Deploy to production:**
   - Use Docker for containerization
   - Deploy to cloud platforms (AWS, GCP, Azure)
   - Set up monitoring and logging

4. **Scale the system:**
   - Add database for persistent storage
   - Implement user authentication
   - Add more sophisticated evaluation metrics

Congratulations! You now have a complete LLM data factory running!
