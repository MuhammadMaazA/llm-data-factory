"""
FastAPI Backend for LLM Data Factory

A REST API server for the customer support ticket classifier,
providing endpoints for predictions, model info, evaluation, and pipeline management.
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# Global variables for pipeline status
generation_status = {
    "status": "idle",
    "progress": 0,
    "total_batches": 0,
    "current_batch": 0,
    "tickets_generated": 0,
    "message": "Ready to generate data"
}

training_status = {
    "status": "idle", 
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "message": "Ready to train model"
}

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

class GenerationRequest(BaseModel):
    num_samples: int = 1000

class TrainingRequest(BaseModel):
    epochs: int = 3
    
class GenerationStatus(BaseModel):
    status: str
    progress: float
    total_batches: int
    current_batch: int
    tickets_generated: int
    message: str

class TrainingStatus(BaseModel):
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: float
    message: str

class TaskResponse(BaseModel):
    message: str
    task_id: str

@app.on_event("startup")
async def startup_event():
    """Load the classifier on startup"""
    global classifier
    try:
        classifier = load_classifier()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
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

async def run_data_generation(num_samples: int):
    """Background task to run data generation"""
    global generation_status
    
    try:
        generation_status.update({
            "status": "running",
            "progress": 0,
            "total_batches": (num_samples + 9) // 10,  # Assuming batch size of 10
            "current_batch": 0,
            "tickets_generated": 0,
            "message": "Starting data generation..."
        })
        
        # Run the data generation script
        process = subprocess.Popen(
            ["python", "scripts/01_generate_synthetic_data.py"],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress (simplified - in real implementation, parse stdout)
        while process.poll() is None:
            await asyncio.sleep(2)
            # Update progress based on time elapsed (mock progress)
            generation_status["current_batch"] = min(
                generation_status["current_batch"] + 1,
                generation_status["total_batches"]
            )
            generation_status["progress"] = (
                generation_status["current_batch"] / generation_status["total_batches"] * 100
            )
            generation_status["tickets_generated"] = generation_status["current_batch"] * 10
            generation_status["message"] = f"Processing batch {generation_status['current_batch']}"
        
        if process.returncode == 0:
            generation_status.update({
                "status": "completed",
                "progress": 100,
                "message": f"Successfully generated {num_samples} tickets"
            })
        else:
            generation_status.update({
                "status": "error",
                "message": "Data generation failed"
            })
            
    except Exception as e:
        generation_status.update({
            "status": "error",
            "message": f"Error: {str(e)}"
        })

async def run_model_training():
    """Background task to run model training"""
    global training_status
    
    try:
        training_status.update({
            "status": "running",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": 3,
            "loss": 0.0,
            "message": "Starting model training..."
        })
        
        # Run the training script
        process = subprocess.Popen(
            ["python", "scripts/02_finetune_student_model.py"],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress (simplified)
        while process.poll() is None:
            await asyncio.sleep(5)
            training_status["current_epoch"] = min(
                training_status["current_epoch"] + 1,
                training_status["total_epochs"]
            )
            training_status["progress"] = (
                training_status["current_epoch"] / training_status["total_epochs"] * 100
            )
            training_status["message"] = f"Training epoch {training_status['current_epoch']}"
        
        if process.returncode == 0:
            training_status.update({
                "status": "completed",
                "progress": 100,
                "message": "Model training completed successfully"
            })
        else:
            training_status.update({
                "status": "error",
                "message": "Model training failed"
            })
            
    except Exception as e:
        training_status.update({
            "status": "error",
            "message": f"Error: {str(e)}"
        })

@app.post("/generate-data", response_model=TaskResponse)
async def start_data_generation(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Start synthetic data generation"""
    if generation_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Data generation already in progress")
    
    task_id = f"gen_{int(time.time())}"
    background_tasks.add_task(run_data_generation, request.num_samples)
    
    return TaskResponse(
        message="Data generation started",
        task_id=task_id
    )

@app.get("/generation-status", response_model=GenerationStatus)
async def get_generation_status():
    """Get current data generation status"""
    return GenerationStatus(**generation_status)

@app.post("/train-model", response_model=TaskResponse)
async def start_training(background_tasks: BackgroundTasks):
    """Start model training"""
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    task_id = f"train_{int(time.time())}"
    background_tasks.add_task(run_model_training)
    
    return TaskResponse(
        message="Model training started",
        task_id=task_id
    )

@app.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**training_status)

@app.post("/run-pipeline", response_model=TaskResponse)
async def run_complete_pipeline(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Run the complete pipeline: generate data then train model"""
    if generation_status["status"] == "running" or training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline already in progress")
    
    # Start data generation first, then training will start automatically
    task_id = f"pipeline_{int(time.time())}"
    
    async def run_pipeline():
        await run_data_generation(request.num_samples)
        if generation_status["status"] == "completed":
            await run_model_training()
    
    background_tasks.add_task(run_pipeline)
    
    return TaskResponse(
        message="Complete pipeline started",
        task_id=task_id
    )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
