# Quick Start Guide

This guide will get you up and running with the LLM Data Factory in just a few steps.

## Prerequisites

- Python 3.8+ with pip
- Node.js 18+ with npm  
- OpenAI API key (for data generation)
- Git

## One-Command Setup

```bash
# Clone and set up everything
git clone https://github.com/MuhammadMaazA/llm-data-factory.git
cd llm-data-factory
chmod +x start.sh run_complete_pipeline.py

# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run the complete pipeline
python run_complete_pipeline.py
```

This will:
1. Generate synthetic training data
2. Fine-tune the Phi-3-mini model  
3. Run comprehensive evaluation
4. Generate performance reports

## 🌐 Launch the Demo

```bash
# Start both backend and frontend
./start.sh
```

Then open:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## View Results

After training, check:
- `results/evaluation_summary.json` - Performance metrics
- `results/detailed_predictions.csv` - Prediction details
- `notebooks/evaluation.ipynb` - Interactive analysis

## Manual Steps (Optional)

If you prefer to run steps individually:

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Generate synthetic data
python scripts/01_generate_synthetic_data.py

# 3. Train the model
python scripts/02_finetune_student_model.py

# 4. Run evaluation
cd notebooks && jupyter notebook evaluation.ipynb
```

## Troubleshooting

**Model not found error?**
- Run `python run_complete_pipeline.py` to train the model

**Frontend not loading?**
- Ensure both servers are running with `./start.sh`
- Check no other services are using ports 5173 or 8000

**API key issues?**
- Set: `export OPENAI_API_KEY='your-key'`
- Or skip data generation: `python run_complete_pipeline.py --skip-data-generation`

## Expected Results

With the provided seed data, you should achieve:
- **Overall Accuracy**: ~94%
- **Urgent Bug**: F1-Score ~0.91
- **Feature Request**: F1-Score ~0.95  
- **How-To Question**: F1-Score ~0.94

The fine-tuned Phi-3-mini model (3.8B parameters) achieves performance close to GPT-4 while being 400x smaller and much faster for inference!
