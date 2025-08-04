"""
Global state variables for model management.
Centralizes all state variables used across the application.
"""

import joblib
import datetime
import logging
import os
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app_state")

# Model cache
_model_cache = {}

class ModelManager:
    """Class to manage model loading and caching"""
    
    def __init__(self):
        self._stable_model_path = 'models/model_v1.joblib'
        self._canary_model_path = None
        self._model_cache = {}
    
    def get_model(self, model_path):
        """Lazy load models to improve performance"""
        if not model_path:
            return None
            
        if model_path not in self._model_cache:
            logger.info(f"Loading model from {model_path}")
            start_time = datetime.datetime.now()
            self._model_cache[model_path] = joblib.load(model_path)
            end_time = datetime.datetime.now()
            load_time = (end_time - start_time).total_seconds() * 1000
            logger.info(f"Model loaded in {load_time:.2f}ms")
        return self._model_cache[model_path]
    
    @property
    def stable_model_path(self):
        return self._stable_model_path
        
    @property
    def canary_model_path(self):
        return self._canary_model_path
        
    @canary_model_path.setter
    def canary_model_path(self, path):
        self._canary_model_path = path
    
    @property
    def stable_model(self):
        return self.get_model(self._stable_model_path)
        
    @property
    def canary_model(self):
        return self.get_model(self._canary_model_path)
        
    def clear_cache(self):
        """Clear the model cache"""
        self._model_cache.clear()
        logger.info("Model cache cleared")

# Create model manager instance
model_manager = ModelManager()

# For backwards compatibility, expose properties at module level
stable_model_path = model_manager.stable_model_path
canary_model_path = model_manager.canary_model_path

# These are now properties that use the model manager
def stable_model():
    return model_manager.stable_model
    
def canary_model():
    return model_manager.canary_model

# Canary deployment tracking
canary_start_time: Optional[str] = None
canary_metrics: Dict[str, Dict[str, List[float]]] = {
    "stable": {"latencies_ms": []},
    "canary": {"latencies_ms": []}
}

# Alert management
alert_status: Dict[str, Any] = {}

# Feature flags
simulate_slowdown: bool = False