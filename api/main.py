import os
import pickle
import logging
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from clearml import Model

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InflationAPI")

# ML Resources container for hot-swapping
ml_resources = {}

# --- CORE LOGIC: MODEL LOADER ---
def load_production_model():
    """
    Fetch the latest 'production' tagged model from ClearML.
    """
    logger.info("Connecting to ClearML Registry...")
    # Matches the name we used in OutputModel in train.py
    models = Model.query_models(
        project_name="Inflation_Forecast_2026",
        model_name="Nigeria_Inflation_Forecast_Model", 
        tags=["production"],
        only_published=True
    )
    
    if not models:
        raise Exception("No model found with the 'production' tag in ClearML.")

    # Get the newest model from the list
    registered_model = models[0]
    local_path = registered_model.get_local_copy()
    
    with open(local_path, "rb") as f:
        model_data = pickle.load(f)
    
    # We store the class name to know how to predict later
    model_type = type(model_data).__name__
    return model_data, registered_model.id, model_type

# --- LIFESPAN HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown logic."""
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

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Nigeria Inflation Forecast API",
    description="Production API with automated model hot-swapping.",
    version="1.5.0",
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
        "current_model_id": ml_resources.get("model_id"),
        "model_type": ml_resources.get("model_type"),
        "timestamp": pd.Timestamp.now(tz='Africa/Lagos').isoformat()
    }

@app.post("/refresh", tags=["Admin"])
async def refresh_model():
    """Triggered by the Prefect pipeline to swap to the new winner."""
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
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Start dates from the current month in 2026
        dates = pd.date_range(start="2026-02-01", periods=months, freq='ME')
        
        # 2. Logic Router: Handle different model prediction APIs
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

        else: # Fallback for XGBoost/Scikit-Learn
            # Simple simulation of confidence intervals for non-probabilistic models
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
    uvicorn.run(app, host="0.0.0.0", port=8000)