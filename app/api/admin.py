"""
Admin API endpoints.
Handles canary deployment, rollback, promotion, and monitoring.
"""

import os
import datetime
import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.core import state
from app.utils.stats import check_canary_health

# Create router for admin endpoints
router = APIRouter(prefix="/admin")

class CanaryDeployRequest(BaseModel):
    model_path: str

@router.post("/deploy-canary")
async def deploy_canary(request: CanaryDeployRequest):
    """
    Deploy a new canary model for A/B testing.
    
    Validates model file existence and loadability before deployment.
    Updates canary model and resets metrics upon successful deployment.
    
    Args:
        request: CanaryDeployRequest containing model path
        
    Returns:
        Dict containing deployment status and information
    """
    # Validate that the file exists
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
    
    # Try to load the model - this will be handled by the model manager now
    try:
        # Update canary model path in the model manager
        state.model_manager.canary_model_path = request.model_path
        # Test loading the model to ensure it works
        _ = state.model_manager.canary_model
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
    
    # Update canary-related variables with thread-safe access
    with state.state_lock:
        state.canary_start_time = datetime.datetime.now().isoformat()
        state.alert_status = {}
    
    # Reset metrics with thread-safe access
    with state.metrics_lock:
        state.canary_metrics = {
            "stable": {"latencies_ms": []},
            "canary": {"latencies_ms": []}
        }
    
    # Return success response
    return {
        "status": "success",
        "message": "Canary model deployed successfully",
        "model_path": request.model_path,
        "canary_start_time": state.canary_start_time
    }

@router.post("/rollback-canary")
async def rollback_canary():
    """
    Roll back an active canary deployment.
    
    Resets all canary-related state if a canary model is active.
    
    Returns:
        Dict containing rollback status and information
    """
    # Check if a canary model is active
    if state.canary_model() is None:
        return {
            "status": "error",
            "message": "No active canary to rollback"
        }
    
    # Reset all canary-related state with thread-safe access
    state.model_manager.canary_model_path = None
    
    with state.state_lock:
        state.canary_start_time = None
        state.alert_status = {}
        
    with state.metrics_lock:
        state.canary_metrics = {
            "stable": {"latencies_ms": []},
            "canary": {"latencies_ms": []}
        }
    
    # Return success response
    return {
        "status": "success",
        "message": "Canary rolled back successfully"
    }

@router.post("/promote-canary")
async def promote_canary():
    """
    Promote canary model to stable if it passed all checks.
    
    Verifies no alerts are triggered before promotion.
    Updates stable model and resets canary-related state upon successful promotion.
    
    Returns:
        Dict containing promotion status and information
    """
    # Check if a canary model is active
    if state.canary_model() is None:
        return {
            "status": "error",
            "message": "No active canary to promote"
        }
    
    # Check if any alerts were triggered
    if state.alert_status.get("alert_triggered", False):
        return {
            "status": "error",
            "message": "Cannot promote canary with active alerts"
        }
    
    # Store previous stable model path for response
    previous_stable_model = state.model_manager.stable_model_path
    
    # Promote canary to stable - update the model paths
    canary_path = state.model_manager.canary_model_path
    
    # Update the stable model path in the model manager
    # This requires modifying the internal attribute since we don't have a setter for stable_model_path
    state.model_manager._stable_model_path = canary_path
    
    # Reset canary-related state with thread-safe access
    state.model_manager.canary_model_path = None
    
    with state.state_lock:
        state.canary_start_time = None
        state.alert_status = {}
        
    with state.metrics_lock:
        state.canary_metrics = {
            "stable": {"latencies_ms": []},
            "canary": {"latencies_ms": []}
        }
    
    # Return success response
    return {
        "status": "success",
        "message": "Canary promoted to stable successfully",
        "previous_stable_model": os.path.basename(previous_stable_model),
        "new_stable_model": os.path.basename(canary_path)
    }

@router.post("/toggle-slowdown")
async def toggle_slowdown():
    """
    Toggle the simulate_slowdown boolean variable.
    
    This endpoint allows administrators to enable or disable the artificial slowdown
    of the canary model for testing purposes.
    
    Returns:
        Dict containing the updated slowdown status
    """
    # Toggle the boolean value with thread-safe access
    with state.state_lock:
        state.simulate_slowdown = not state.simulate_slowdown
        # Get the current value for the message
        current_slowdown = state.simulate_slowdown
    
    # Prepare the response message
    message = "Slowdown simulation enabled" if current_slowdown else "Slowdown simulation disabled"
    
    # Return the updated state
    return {
        "simulate_slowdown": bool(state.simulate_slowdown),  # Ensure it's a Python bool
        "message": message
    }

@router.get("/check-canary-health")
async def check_canary_health_endpoint():
    """
    Check the health of the canary deployment by comparing latency metrics.
    
    Delegates to the stats utility function to perform the analysis.
    
    Returns:
        Dict containing analysis results and alert status
    """
    return check_canary_health()