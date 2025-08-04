# MLOps Take-Home Project: Canary Deployment with Statistical Performance Monitoring

This project implements a canary deployment system for machine learning models with statistical performance monitoring. It allows for safely deploying a new model version by first routing a small percentage of traffic to it and monitoring its performance.

## Project Overview

The system serves a customer churn prediction model via a FastAPI service. When a new model version is developed, it can be deployed as a "canary" that receives 10% of the traffic. The system actively monitors the prediction response time (latency) of both the stable and canary models and performs statistical tests to detect if the canary model is significantly slower.

### Key Features

- **Traffic Splitting**: 10% of requests are routed to the canary model when active
- **Latency Monitoring**: Tracks response times for both stable and canary models
- **Statistical Testing**: Uses Welch's t-test to compare performance
- **Admin Controls**: Endpoints for deploying, monitoring, promoting, and rolling back canary deployments

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Scikit-learn
- SciPy
- Joblib

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the server:
   ```
   python main.py
   ```

## Demonstration Workflow

Below is a step-by-step demonstration of the canary deployment process using curl commands.

### 1. Deploy Canary

Deploy the new model (model_v2.joblib) as a canary:

```bash
curl -X POST "http://localhost:8000/admin/deploy-canary" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/model_v2.joblib"}'
```

Expected response:
```json
{
  "status": "success",
  "message": "Canary model deployed successfully",
  "model_path": "models/model_v2.joblib",
  "canary_start_time": "2025-08-03T20:16:00.123456"
}
```

### 2. Generate Traffic

Generate prediction requests to populate the latency metrics:

```bash
for i in {1..200}; do 
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'; 
done
```

Alternatively, you can use this Python script:

```python
import requests
import time

url = "http://localhost:8000/predict"
data = {"features": [0.5, 1.2, 0.8, 0.3, 1.1]}

for _ in range(200):
    response = requests.post(url, json=data)
    print(response.json())
    time.sleep(0.01)  # Small delay to avoid overwhelming the server
```

### 3. First Health Check

Check the health of the canary deployment:

```bash
curl -X GET "http://localhost:8000/admin/check-canary-health"
```

Expected response (no alert):
```json
{
  "alert_triggered": false,
  "p_value": 0.234,
  "message": "Canary performance is acceptable.",
  "stable_avg_latency_ms": 15.2,
  "canary_avg_latency_ms": 16.1,
  "stable_sample_count": 180,
  "canary_sample_count": 20
}
```

### 4. Introduce Slowness

Enable the simulated slowdown for the canary model:

```bash
curl -X POST "http://localhost:8000/admin/toggle-slowdown"
```

Expected response:
```json
{
  "simulate_slowdown": true,
  "message": "Slowdown simulation enabled"
}
```

### 5. Generate More Traffic

Generate more prediction requests with the slowdown enabled:

```bash
for i in {1..200}; do 
  curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'; 
done
```

### 6. Trigger the Alert

Check the health again to see the alert:

```bash
curl -X GET "http://localhost:8000/admin/check-canary-health"
```

Expected response (alert triggered):
```json
{
  "alert_triggered": true,
  "p_value": 0.031,
  "message": "ALERT: Canary latency is significantly higher than stable.",
  "stable_avg_latency_ms": 15.2,
  "canary_avg_latency_ms": 25.8,
  "stable_sample_count": 180,
  "canary_sample_count": 20
}
```

### 7. Rollback

Roll back the canary deployment:

```bash
curl -X POST "http://localhost:8000/admin/rollback-canary"
```

Expected response:
```json
{
  "status": "success",
  "message": "Canary rolled back successfully"
}
```

### 8. Confirm Rollback

Make a prediction request to confirm the server is using only the stable model:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'
```

Expected response:
```json
{
  "churn_probability": 0.23,
  "model_used": "stable",
  "latency_ms": 15.2
}
```

## Alternative Scenario: Promoting Canary

If the canary model performs well (no alerts), you can promote it to stable:

```bash
curl -X POST "http://localhost:8000/admin/promote-canary"
```

Expected response:
```json
{
  "status": "success",
  "message": "Canary promoted to stable successfully",
  "previous_stable_model": "model_v1.joblib",
  "new_stable_model": "model_v2.joblib"
}
```

## API Documentation

### Prediction Endpoint

- **POST /predict**
  - Request: `{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}`
  - Response: `{"churn_probability": 0.23, "model_used": "stable", "latency_ms": 15.2, "request_id": "req_1722889260123"}`

#### Curl Example for Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'
```

### Admin Endpoints

- **POST /admin/deploy-canary**
  - Request: `{"model_path": "models/model_v2.joblib"}`
  - Response: Success or error message

- **POST /admin/rollback-canary**
  - Request: No body required
  - Response: Success or error message

- **GET /admin/check-canary-health**
  - Request: No body required
  - Response: Health check results with statistical analysis

- **POST /admin/promote-canary**
  - Request: No body required
  - Response: Success or error message

- **POST /admin/toggle-slowdown**
  - Request: No body required
  - Response: Updated slowdown status

## Automated Demonstration Script

The project includes a Python script (`demo.py`) that automates the entire canary deployment workflow. This script is useful for quickly demonstrating the system's functionality without having to manually execute each step.

### Usage

```bash
python demo.py [--url URL] [--requests NUM_REQUESTS] [--workflow {full,deploy,promote}]
```

### Options

- `--url URL`: Base URL of the API (default: http://localhost:8000)
- `--requests NUM_REQUESTS`: Number of requests to generate for each traffic batch (default: 200)
- `--workflow WORKFLOW`: Workflow to run:
  - `full`: Run both failure and success workflows (default)
  - `deploy`: Run only the deployment failure workflow
  - `promote`: Run only the successful promotion workflow

### Workflows

The script can demonstrate two main workflows:

1. **Deployment Failure Workflow**: Shows how the system detects performance issues in a canary model and rolls back
2. **Successful Promotion Workflow**: Shows how a well-performing canary model can be promoted to stable

Each workflow follows the same steps described in the manual demonstration section, but automates the process and provides detailed output at each step.

### Example

To run the complete demonstration:

```bash
python demo.py
```

This will execute both workflows sequentially, showing how the system handles both failure and success scenarios.

## Power Analysis

The minimum number of samples needed to detect a 10ms latency increase with 80% power can be calculated using a power analysis. For a t-test with alpha=0.05 and 80% power to detect a 10ms difference, assuming a standard deviation of approximately 15ms, we would need approximately 36 samples for each model.

This is why our health check requires at least 36 samples for both the stable and canary models before performing the statistical test.