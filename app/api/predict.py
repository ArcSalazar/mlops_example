"""
Prediction API endpoints.
Handles prediction requests and routes them to the inference service.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.inference import route_prediction

# Create router for prediction endpoints
router = APIRouter()

class PredictionRequest(BaseModel):
    features: List[float]

@router.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict customer churn probability based on input features.
    
    Routes requests to appropriate model and measures prediction latency.
    
    Args:
        request: PredictionRequest containing features for prediction
        
    Returns:
        Dict containing churn probability, model used, and latency
    """
    return route_prediction(request.features)