#!/usr/bin/env python3
"""
FastAPI Backend for Customer Support Ticket Classifier

This serves the fine-tuned model via REST API for the frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
from inference import TicketClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LLM Data Factory - Ticket Classifier API",
    description="Fine-tuned Phi-3-mini model for customer support ticket classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

# Request/Response models
class TicketRequest(BaseModel):
    text: str
    return_probabilities: Optional[bool] = False

class BatchTicketRequest(BaseModel):
    texts: List[str]
    return_probabilities: Optional[bool] = False

class ClassificationResult(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    categories: List[str]

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global classifier
    try:
        logger.info("Loading classifier model...")
        classifier = TicketClassifier()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if classifier else "model_not_loaded",
        model_loaded=classifier is not None,
        categories=classifier.categories if classifier else []
    )

@app.post("/classify", response_model=ClassificationResult)
async def classify_ticket(request: TicketRequest):
    """Classify a single customer support ticket."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = classifier.predict(request.text, request.return_probabilities)
        return ClassificationResult(**result)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

@app.post("/classify-batch", response_model=List[ClassificationResult])
async def classify_tickets_batch(request: BatchTicketRequest):
    """Classify multiple customer support tickets."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    try:
        results = classifier.predict_batch(request.texts, request.return_probabilities)
        return [ClassificationResult(**result) for result in results]
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail="Batch classification failed")

@app.get("/categories")
async def get_categories():
    """Get available ticket categories."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "categories": classifier.categories,
        "total": len(classifier.categories)
    }

@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "base_model": "microsoft/phi-3-mini-4k-instruct",
        "fine_tuned": True,
        "categories": classifier.categories,
        "training_accuracy": "95%",  # From our training results
        "method": "QLoRA (Quantized Low-Rank Adaptation)",
        "parameters_trained": "8.9M out of 3.7B (0.24%)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
