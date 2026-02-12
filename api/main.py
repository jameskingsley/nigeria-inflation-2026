import os
import pickle
import logging
import pandas as pd
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from clearml import Model

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InflationAPI")

# ML Resources container for hot-swapping
ml_resources = {}

# --- CORE LOGIC: ROBUST MODEL LOADER ---
def load_production_model():
    """
    Directly fetches the latest MODEL registered with the 'production' tag.
    This bypasses Task lookups and goes straight to the weights.
    """
    logger.info("Connecting to ClearML Registry...")
    
    try:
        # Search specifically for the Model object tagged 'production'
        models = Model.query_models(
            project_name="Inflation_Forecast_2026",
            tags=["production"],
            only_published=False  # Set to True if you always call .publish() in train.py
        )
        
        if not models:
            raise Exception("No model found with 'production' tag in ClearML Registry.")

        # query_models returns newest first by default
        latest_model_entry = models[0]
        logger.info(f"Found Production Model: {latest_model_entry.name} (ID: {latest_model_entry.id})")

        # Download to local temp storage
        local_path = latest_model_entry.get_local_copy()
        
        if not local_path or not os.path.exists(local_path):
            raise Exception(f"Failed to download model weights for ID: {latest_model_entry.id}")

        with open(local_path, "rb") as f:
            model_data = pickle.load(f)
        
        model_type = type(model_data).__name__
        return model_data, latest_model_entry.id, model_type

    except Exception as e:
        logger.error(f"Error in load_production_model: {str(e)}")
        raise e

# LIFESPAN HANDLER
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Starting Up...")
    try:
        model, m_id, m_type = load_production_model()
        ml_resources["model"] = model
        ml_resources["model_id"] = m_id
        ml_resources["model_type"] = m_type
        logger.info(f"Startup Successful: Loaded {m_type} (ID: {m_id})")
    except Exception as e:
        logger.critical(f"CRITICAL STARTUP FAILURE: {e}")
        # Initialize with None so the app stays alive but returns 503 errors
        ml_resources["model"] = None
        ml_resources["model_id"] = "none"
        ml_resources["model_type"] = "none"

    yield
    logger.info("Shutting Down...")
    ml_resources.clear()

# APP INITIALIZATION 
app = FastAPI(
    title="Nigeria Inflation Forecast API",
    description="Production API with automated Model Registry fetching.",
    version="1.7.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "model_loaded": ml_resources.get("model") is not None,
        "current_model_id": ml_resources.get("model_id"),
        "model_type": ml_resources.get("model_type"),
        "timestamp": pd.Timestamp.now(tz='Africa/Lagos').isoformat()
    }

@app.post("/refresh", tags=["Admin"])
async def refresh_model():
    """Triggered by the training pipeline to swap the model in-memory."""
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
        raise HTTPException(
            status_code=503, 
            detail="Model is currently unavailable. Ensure ClearML keys are set in Render."
        )

    try:
        # Start predictions from February 2026
        dates = pd.date_range(start="2026-02-01", periods=months, freq='ME')
        
        # Router for different model types
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
            # Fallback for XGBoost or simpler models
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