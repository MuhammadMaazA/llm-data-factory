"""
FastAPI Backend for LLM Data Factory

A REST API server for the customer support ticket classifier,
providing endpoints for predictions, model info, and evaluation.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our inference module
from inference import load_classifier, predict_ticket_category

app = FastAPI(
    title="LLM Data Factory API",
    description="REST API for customer support ticket classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the classifier
classifier = None

# Pydantic models for request/response
class TicketRequest(BaseModel):
    ticket_text: str

class BatchTicketRequest(BaseModel):
    tickets: List[str]

class TicketPrediction(BaseModel):
    ticket_text: str
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    model_size: str
    fine_tuned: bool
    categories: List[str]
    training_samples: int
    last_updated: str

class EvaluationResults(BaseModel):
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
    support: Dict[str, int]

@app.on_event("startup")
async def startup_event():
    """Load the classifier on startup"""
    global classifier
    try:
        classifier = load_classifier()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        classifier = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if classifier is not None else "unhealthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=TicketPrediction)
async def predict_ticket(request: TicketRequest):
    """Predict the category of a single support ticket"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        result = predict_ticket_category(classifier, request.ticket_text)
        processing_time = time.time() - start_time
        
        return TicketPrediction(
            ticket_text=request.ticket_text,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=List[TicketPrediction])
async def batch_predict_tickets(request: BatchTicketRequest):
    """Predict categories for multiple support tickets"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for ticket_text in request.tickets:
        start_time = time.time()
        try:
            result = predict_ticket_category(classifier, ticket_text)
            processing_time = time.time() - start_time
            
            results.append(TicketPrediction(
                ticket_text=ticket_text,
                predicted_category=result["predicted_category"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                processing_time=processing_time
            ))
        except Exception as e:
            # Continue with other predictions even if one fails
            results.append(TicketPrediction(
                ticket_text=ticket_text,
                predicted_category="error",
                confidence=0.0,
                probabilities={},
                processing_time=0.0
            ))
    
    return results

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # This would need to be updated based on your actual model structure
    return ModelInfo(
        model_name="phi-3-mini-finetuned",
        model_size="3.8B parameters",
        fine_tuned=True,
        categories=["Urgent Bug", "Feature Request", "How-To Question", "General Inquiry"],
        training_samples=1000,  # This should be loaded from your training data
        last_updated=datetime.now().isoformat()
    )

@app.get("/evaluation", response_model=EvaluationResults)
async def get_evaluation_results():
    """Get evaluation results from the test dataset"""
    # This would load actual evaluation results from your evaluation notebook
    # For now, returning mock data
    return EvaluationResults(
        accuracy=0.94,
        precision={
            "Urgent Bug": 0.92,
            "Feature Request": 0.95,
            "How-To Question": 0.94,
            "General Inquiry": 0.93
        },
        recall={
            "Urgent Bug": 0.90,
            "Feature Request": 0.96,
            "How-To Question": 0.95,
            "General Inquiry": 0.92
        },
        f1_score={
            "Urgent Bug": 0.91,
            "Feature Request": 0.95,
            "How-To Question": 0.94,
            "General Inquiry": 0.92
        },
        confusion_matrix=[
            [45, 2, 2, 1],
            [1, 67, 1, 1],
            [2, 1, 76, 1],
            [2, 1, 1, 36]
        ],
        support={
            "Urgent Bug": 50,
            "Feature Request": 70,
            "How-To Question": 80,
            "General Inquiry": 40
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
