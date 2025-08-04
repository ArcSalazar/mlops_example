"""
Prediction API endpoints.
Handles prediction requests and routes them to the inference service.
Supports concurrent requests by offloading CPU-bound operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from app.services.inference import route_prediction

# Create router for prediction endpoints
router = APIRouter()

# Thread pool for handling CPU-bound operations
# This allows the FastAPI server to handle multiple concurrent requests
# without blocking the event loop during model inference
thread_pool = ThreadPoolExecutor()

class PredictionRequest(BaseModel):
    features: List[float]

@router.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Predict customer churn probability based on input features.
    
    Routes requests to appropriate model and measures prediction latency.
    Uses a thread pool to handle CPU-bound model inference operations,
    allowing the API to handle multiple concurrent requests efficiently.
    
    Args:
        request: PredictionRequest containing features for prediction
        
    Returns:
        Dict containing churn probability, model used, and latency
    """
    # Run the CPU-bound prediction in a thread pool to avoid blocking the event loop
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, 
        route_prediction, 
        request.features
    )