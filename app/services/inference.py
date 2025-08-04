"""
Inference service for prediction routing and latency measurement.
Handles model selection, prediction execution, and latency tracking.
"""

import time
import random
import logging
from typing import List, Dict, Any

from app.core import state

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference_service")

def route_prediction(features: List[float]) -> Dict[str, Any]:
    """
    Route prediction to appropriate model and measure latency.
    
    Args:
        features: List of float features for prediction
    
    Returns:
        Dict containing prediction result, model used, and latency
    """
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"[{request_id}] Processing prediction request with {len(features)} features")
    
    # Determine which model to use (exactly 10% to canary if available)
    # Use last digit of request_id timestamp for deterministic routing
    timestamp = int(request_id.split('_')[1])
    use_canary = state.canary_model() is not None and (timestamp % 10 == 0)
    model = state.canary_model() if use_canary else state.stable_model()
    model_name = "canary" if use_canary else "stable"
    
    logger.info(f"[{request_id}] Routing request to {model_name} model")
    
    # Prepare features for prediction
    features_array = [features]
    
    # Measure prediction latency
    start_time = time.perf_counter()
    
    # Add simulated slowdown for canary model if enabled
    if use_canary and state.simulate_slowdown:
        logger.debug(f"[{request_id}] Applying simulated slowdown for canary model")
        time.sleep(0.01)
    
    # Log model prediction start
    model_start_time = time.perf_counter()
    logger.debug(f"[{request_id}] Starting model prediction")
    
    # Get prediction probability for class 1 (churn)
    probabilities = model.predict_proba(features_array)
    churn_probability = float(probabilities[0][1])
    
    # Log model prediction completion
    model_end_time = time.perf_counter()
    model_time_ms = (model_end_time - model_start_time) * 1000
    logger.debug(f"[{request_id}] Model prediction completed in {model_time_ms:.2f}ms")
    
    # Calculate total latency
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # Store latency in metrics with thread-safe access
    with state.metrics_lock:
        state.canary_metrics[model_name]["latencies_ms"].append(latency_ms)
    
    logger.info(f"[{request_id}] Request completed: model={model_name}, latency={latency_ms}ms")
    
    # Return prediction result
    return {
        "churn_probability": churn_probability,
        "model_used": model_name,
        "latency_ms": latency_ms,
        "request_id": request_id
    }