import os
import pickle
import logging
import pandas as pd
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from clearml import Task, Model

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InflationAPI")

# ML Resources container for hot-swapping
ml_resources = {}

# --- CORE LOGIC: SMART MODEL LOADER ---
def load_production_model():
    """
    Fetch the latest model from the task tagged as 'winner-ARIMA' or 'production'.
    """
    logger.info("Connecting to ClearML Registry via Smart Search...")
    
    #  Find the latest completed task with the winner tag
    tasks = Task.get_tasks(
        project_name="Inflation_Forecast_2026",
        tags=["winner-ARIMA"], 
        task_filter={'status': ['completed']}
    )
    
    if not tasks:
        logger.warning("No task found with 'winner-ARIMA'. Trying 'production' tag...")
        tasks = Task.get_tasks(
            project_name="Inflation_Forecast_2026",
            tags=["production"],
            task_filter={'status': ['completed']}
        )

    if not tasks:
        raise Exception("No winning tasks found with 'winner-ARIMA' or 'production' tags.")

    # Task.get_tasks returns results sorted by newest first
    latest_task = tasks[0]
    logger.info(f"Found winning task: {latest_task.name} (ID: {latest_task.id})")

    # Extract the model associated with this task
    models_dict = latest_task.get_models_sdks()
    output_models = models_dict.get('output', [])

    if not output_models:
        raise Exception(f"Task {latest_task.id} found, but it has no output models.")

    # Download and Load
    registered_model = output_models[0]
    local_path = registered_model.get_local_copy()
    
    with open(local_path, "rb") as f:
        model_data = pickle.load(f)
    
    model_type = type(model_data).__name__
    return model_data, registered_model.id, model_type

# LIFESPAN HANDLER
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Starting Up...")
    try:
        model, m_id, m_type = load_production_model()
        ml_resources["model"] = model
        ml_resources["model_id"] = m_id
        ml_resources["model_type"] = m_type
        logger.info(f"Startup: Loaded {m_type} (ID: {m_id})")
    except Exception as e:
        logger.error(f"Startup Fetch Failed: {e}")
        ml_resources["model"] = None
        ml_resources["model_id"] = "none"
        ml_resources["model_type"] = "none"

    yield
    logger.info("Shutting Down...")
    ml_resources.clear()

# APP INITIALIZATION 
app = FastAPI(
    title="Nigeria Inflation Forecast API",
    description="Production API with automated task-based model fetching.",
    version="1.6.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENDPOINTS 

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "current_model_id": ml_resources.get("model_id"),
        "model_type": ml_resources.get("model_type"),
        "timestamp": pd.Timestamp.now(tz='Africa/Lagos').isoformat()
    }

@app.post("/refresh", tags=["Admin"])
async def refresh_model():
    try:
        model, m_id, m_type = load_production_model()
        ml_resources["model"] = model
        ml_resources["model_id"] = m_id
        ml_resources["model_type"] = m_type
        logger.info(f"Hot-Swap Successful: {m_type} active.")
        return {"status": "success", "new_model_id": m_id, "type": m_type}
    except Exception as e:
        logger.error(f"Refresh Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict", tags=["Forecasting"])
async def predict(months: int = Query(12, ge=1, le=24)):
    model = ml_resources.get("model")
    m_type = ml_resources.get("model_type")
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Check ClearML tags.")

    try:
        # Generate forecast dates starting from Feb 2026
        dates = pd.date_range(start="2026-02-01", periods=months, freq='ME')
        
        # Logic Router
        if "SARIMAX" in m_type or "ARIMAResults" in m_type:
            forecast = model.get_forecast(steps=months)
            values = forecast.predicted_mean.values
            conf = forecast.conf_int()
            low_bounds = conf.iloc[:, 0].values
            high_bounds = conf.iloc[:, 1].values
        elif "Prophet" in m_type:
            future = model.make_future_dataframe(periods=months, freq='ME')
            forecast = model.predict(future).tail(months)
            values = forecast['yhat'].values
            low_bounds = forecast['yhat_lower'].values
            high_bounds = forecast['yhat_upper'].values
        else:
            values = model.predict(np.array(range(months)).reshape(-1, 1))
            low_bounds = values * 0.95
            high_bounds = values * 1.05

        results = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "rate": round(float(v), 2),
                "low": round(float(l), 2),
                "high": round(float(h), 2)
            }
            for d, v, l, h in zip(dates, values, low_bounds, high_bounds)
        ]
        
        return {
            "model_id": ml_resources["model_id"], 
            "model_type": m_type,
            "data": results
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)