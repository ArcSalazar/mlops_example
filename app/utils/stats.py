"""
Statistical utilities for canary deployment health monitoring.
Provides functions for analyzing latency metrics and determining canary health.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any

from app.core import state

def check_canary_health() -> Dict[str, Any]:
    """
    Check the health of the canary deployment by comparing latency metrics.
    
    Performs Welch's t-test to determine if canary latency is significantly higher.
    Requires at least 20 latency samples for both stable and canary models.
    Triggers an alert if p-value < 0.05 and canary average latency > stable average.
    
    Returns:
        Dict containing analysis results and alert status
    """
    # Check if a canary model is active
    if state.canary_model is None:
        return {
            "alert_triggered": False,
            "message": "No active canary deployment to monitor."
        }
    
    # Get latency samples
    stable_latencies = state.canary_metrics["stable"]["latencies_ms"]
    canary_latencies = state.canary_metrics["canary"]["latencies_ms"]
    
    # Check if we have enough samples
    stable_count = len(stable_latencies)
    canary_count = len(canary_latencies)
    
    if stable_count < 20 or canary_count < 20:
        return {
            "alert_triggered": False,
            "message": "Insufficient data for statistical analysis. Need at least 20 samples for both models.",
            "stable_sample_count": stable_count,
            "canary_sample_count": canary_count
        }
    
    # Calculate average latencies
    stable_avg = np.mean(stable_latencies)
    canary_avg = np.mean(canary_latencies)
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(canary_latencies, stable_latencies, equal_var=False)
    
    # Determine if alert should be triggered
    alert_triggered = p_value < 0.05 and canary_avg > stable_avg
    
    # Update alert status in global state
    state.alert_status["alert_triggered"] = alert_triggered
    
    # Prepare response message
    if alert_triggered:
        message = "ALERT: Canary latency is significantly higher than stable."
    else:
        message = "Canary performance is acceptable."
    
    # Return appropriate response
    return {
        "alert_triggered": alert_triggered,
        "p_value": round(float(p_value), 3),
        "message": message,
        "stable_avg_latency_ms": round(float(stable_avg), 1),
        "canary_avg_latency_ms": round(float(canary_avg), 1),
        "stable_sample_count": stable_count,
        "canary_sample_count": canary_count
    }