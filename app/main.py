"""
FastAPI application for customer churn prediction with canary deployment support.
Manages both stable and canary models with performance monitoring.
"""

from fastapi import FastAPI

from app.api import predict, admin

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API", 
    description="API for customer churn prediction with canary deployment support"
)

# Include routers from api modules
app.include_router(predict.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    from app.core import state
    
    return {
        "message": "Churn Prediction API",
        "stable_model": state.stable_model_path,
        "canary_model": state.canary_model_path,
        "canary_active": state.canary_model is not None
    }