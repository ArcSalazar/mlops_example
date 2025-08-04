# MLOps Project: Canary Deployment of a New Model with Statistical Performance Monitoring

## Objective

This project implements an advanced MLOps pattern: a canary release of a new model version with automated, statistically-driven performance checks. The system is a FastAPI service that can safely deploy an updated logistic regression model.

A new model version will first enter a one-hour "canary" period where it receives 10% of traffic. During this time, the system actively monitors the **prediction response time (latency)** of both the stable and canary models. The system provides an endpoint that performs a **statistical test** to detect if the canary model is significantly slower. If an alert is triggered, an operator can roll the deployment back.

## Scenario

The API serves a stable customer churn model (model_v1.joblib). A data scientist has provided a new, potentially more accurate version (model_v2.joblib). There is a concern that the new model might be slower. The system deploys model_v2 as a canary, compares its latency against model_v1 in real-time, and provides a clear, statistically-backed signal if performance has degraded.

## Core Components

### 1. Installation & Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

The project requires the following packages:
- numpy
- scipy
- scikit-learn
- joblib
- fastapi
- uvicorn
- requests
- statsmodels
- testcontainers
- pytest

This project was tested using Python 3.8 on Ubuntu Linux.

### 2. Model Training & Artifacts

The `train.py` script creates and saves two distinct models:
1. `model_v1.joblib`: The initial, stable model.
2. `model_v2.joblib`: The new canary candidate.

Both models are trained on slightly different data with different random states to simulate model evolution.

To run the training script:

```bash
python utils/train.py
```

### 3. Application State Management

The FastAPI application manages the following state:

- `stable_model`: The loaded scikit-learn object for the v1 model.
- `stable_model_path`: The file path for the stable model (initially "model_v1.joblib").
- `canary_model`: The loaded scikit-learn object for the v2 model (is None when no canary is active).
- `canary_model_path`: The file path for the canary model (is None when no canary is active).
- `canary_start_time`: A datetime object marking the start of the canary release.
- `canary_metrics`: A dictionary to store lists of observed latencies.
  ```json
  {
    "stable": {"latencies_ms": []},
    "canary": {"latencies_ms": []}
  }
  ```
- `alert_status`: A dictionary holding the latest health check result.
- `simulate_slowdown`: A boolean flag to enable/disable artificial latency for testing (default: False).

### 3. FastAPI Application Endpoints

#### Prediction Endpoint

- **POST /predict**
  - Accepts JSON with customer features for prediction
  - Routes 10% of traffic to the canary model if active
  - Measures prediction latency
  - Returns prediction probability and which model was used

#### Admin Endpoints

- **POST /admin/deploy-canary**
  - Deploys a new canary model for A/B testing
  - Validates model file existence and loadability
  - Sets up canary state and resets metrics

- **POST /admin/rollback-canary**
  - Rolls back an active canary deployment
  - Resets all canary-related state

- **GET /admin/check-canary-health**
  - Performs statistical analysis on latency metrics
  - Uses Welch's T-test to compare stable and canary performance
  - Triggers an alert if canary is significantly slower

- **POST /admin/promote-canary**
  - Promotes canary model to stable if it passed all checks
  - Updates stable model and resets canary-related state

- **POST /admin/toggle-slowdown**
  - Toggles the simulate_slowdown flag for testing
  - Allows deterministic testing of the statistical check

## Statistical Implementation

The system uses Welch's T-test (a variant of the independent two-sample t-test that doesn't assume equal variance) to compare the latency distributions of the stable and canary models. An alert is triggered if:

1. The p-value is less than 0.05 (statistically significant difference)
2. The canary's average latency is higher than the stable's

The system requires at least 20 samples for both models to perform the statistical test.

## Power Analysis

A power analysis was performed to determine the minimum number of samples needed to detect a 10ms increase in latency with 80% power and 5% significance level. The analysis assumes a standard deviation of 15ms.

Result: **Minimum sample size required per group: 36**

## Demonstration

Here's how to demonstrate the canary deployment system:

### Scenario 1: Canary Deployment with Performance Degradation (Rollback)

#### 1. Start the FastAPI Application

```bash
uvicorn main:app --reload
```

#### 2. Deploy Canary

```bash
curl -X POST "http://localhost:8000/admin/deploy-canary" -H "Content-Type: application/json" -d '{"model_path": "models/model_v2.joblib"}'
```

#### 3. Generate Traffic

```bash
for i in {1..200}; do 
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'; 
  echo ""; 
done
```

#### 4. First Health Check

```bash
curl -X GET "http://localhost:8000/admin/check-canary-health"
```

This should show no alert, as we haven't introduced the artificial slowdown yet.

#### 5. Introduce Slowness

```bash
curl -X POST "http://localhost:8000/admin/toggle-slowdown"
```

#### 6. Generate More Traffic

```bash
for i in {1..200}; do 
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'; 
  echo ""; 
done
```

#### 7. Trigger the Alert

```bash
curl -X GET "http://localhost:8000/admin/check-canary-health"
```

The response should now show `"alert_triggered": true` and provide the statistical evidence (p-value, average latencies).

#### 8. Rollback

```bash
curl -X POST "http://localhost:8000/admin/rollback-canary"
```

#### 9. Confirm Rollback

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'
```

The response should show that the server is back to using only the stable model.

### Scenario 2: Successful Canary Deployment (Promotion)

#### 1. Deploy Canary

```bash
curl -X POST "http://localhost:8000/admin/deploy-canary" -H "Content-Type: application/json" -d '{"model_path": "models/model_v2.joblib"}'
```

#### 2. Generate Traffic

```bash
for i in {1..200}; do 
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'; 
  echo ""; 
done
```

#### 3. Check Canary Health

```bash
curl -X GET "http://localhost:8000/admin/check-canary-health"
```

The response should show `"alert_triggered": false` indicating that the canary model is performing well.

#### 4. Promote Canary to Stable

```bash
curl -X POST "http://localhost:8000/admin/promote-canary"
```

#### 5. Confirm Promotion

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 1.2, 0.8, 0.3, 1.1]}'
```

The response should show that the server is now using the new model (previously canary, now stable) for all predictions.

## Concurrency Support

The system is designed to handle concurrent requests efficiently:

- **Thread-Safe Model Management**: The ModelManager class uses locks to ensure thread-safe access to the model cache and model paths, preventing race conditions during model loading and access.
- **Asynchronous API Endpoints**: The prediction endpoint is implemented as an async function that offloads CPU-bound model inference to a thread pool, allowing the FastAPI server to handle multiple concurrent requests without blocking the event loop.
- **Thread-Safe State Management**: All shared state variables are protected by appropriate locks:
  - `metrics_lock`: Protects access to latency metrics collection
  - `state_lock`: Protects access to other state variables like alert_status, canary_start_time, and simulate_slowdown
- **Thread-Safe Admin Operations**: All admin operations that modify state (deploy, rollback, promote, toggle slowdown) use appropriate locks to ensure thread safety.

## Bonus: Power Analysis for the Health Check

The power analysis script (`utils/power_analysis.py`) calculates the minimum number of samples needed to detect a 10ms latency increase with 80% power. This ensures that the statistical test has sufficient power to detect meaningful performance degradation.

```bash
python  utils/power_analysis.py
```

output:

Minimum sample size required per group: 36
