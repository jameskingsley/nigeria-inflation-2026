import pandas as pd
import numpy as np
import pickle
import os
import requests
import logging
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from clearml import Task, Dataset, OutputModel, Model
from prefect import task, flow

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingPipeline")

# --- PREFECT TASKS ---

@task(retries=2, retry_delay_seconds=60)
def prepare_data():
    """Fetches the latest dataset from ClearML."""
    logger.info("Fetching dataset from ClearML...")
    dataset_path = Dataset.get(
        dataset_project="Inflation_Forecast_2026", 
        dataset_name="Nigeria_Inflation_Data"
    ).get_local_copy()
    
    df = pd.read_csv(os.path.join(dataset_path, "nigeria_inflation.csv"))
    df['ds'] = pd.to_datetime(df['ds'])
    return df

@task
def evaluate_models(df):
    """Trains models, selects winner, and handles cloud promotion."""
    
    # Force cloud upload by setting output_uri to True
    task_tracking = Task.init(
        project_name="Inflation_Forecast_2026", 
        task_name="Monthly_Model_Competition",
        reuse_last_task_id=False,
        output_uri=True 
    )
    
    test_len = 3
    train_df = df[:-test_len]
    test_actuals = df[-test_len:]['y'].values
    
    # --- MODEL TRAINING ---
    def train_arima(train, t_len):
        model = SARIMAX(train['y'], order=(1,1,1))
        res = model.fit(disp=False)
        return res.forecast(steps=t_len), res

    def train_prophet(train, t_len):
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=t_len, freq='ME')
        forecast = m.predict(future)
        return forecast.tail(t_len)['yhat'].values, m

    def train_xgboost(data, t_len):
        df_xgb = data.copy()
        df_xgb['lag_1'] = df_xgb['y'].shift(1)
        df_xgb = df_xgb.dropna()
        train_xgb = df_xgb[:-t_len]
        test_xgb = df_xgb[-t_len:]
        model = XGBRegressor(n_estimators=100, learning_rate=0.05)
        model.fit(train_xgb[['lag_1']], train_xgb['y'])
        preds = model.predict(test_xgb[['lag_1']])
        return preds, model

    # Run Competition
    logger.info("Starting model competition...")
    p_preds, p_mod = train_prophet(train_df, test_len)
    a_preds, a_mod = train_arima(train_df, test_len)
    x_preds, x_mod = train_xgboost(df, test_len)
    
    scores = {
        "Prophet": mean_absolute_error(test_actuals, p_preds),
        "ARIMA": mean_absolute_error(test_actuals, a_preds),
        "XGBoost": mean_absolute_error(test_actuals, x_preds)
    }
    
    for name, score in scores.items():
        task_tracking.get_logger().report_single_value(f"{name}_MAE", score)
    
    best_model_name = min(scores, key=scores.get)
    logger.info(f"Winner: {best_model_name} (MAE: {scores[best_model_name]:.4f})")
    
    # --- SAVE WINNER ---
    os.makedirs("models", exist_ok=True)
    winner_obj = {"Prophet": p_mod, "ARIMA": a_mod, "XGBoost": x_mod}[best_model_name]
    model_path = "models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner_obj, f)
    
    # --- FIXED PROMOTION LOGIC ---
    
    # 1. Clear old 'production' tags using list assignment (universal for all Model types)
    old_production_models = Model.query_models(
        project_name="Inflation_Forecast_2026",
        tags=["production"]
    )
    for old_m in old_production_models:
        new_tags = [t for t in old_m.tags if t != "production"]
        old_m.tags = new_tags 
        logger.info(f"Cleaned up old production model: {old_m.id}")

    # 2. Register new winner
    output_model = OutputModel(
        task=task_tracking, 
        name="Nigeria_Inflation_Forecast_Model", 
        framework="Scikit-Learn" 
    )
    
    # This automatically uploads the file because task_tracking has output_uri=True
    output_model.update_weights(weights_filename=model_path)
    
    # Add tags via property (Safe & Consistent)
    output_model.tags = ["production", f"winner-{best_model_name}"]
    output_model.publish()
    
    logger.info("✅ SUCCESS: Model uploaded to cloud and tagged 'production'.")
    task_tracking.close()
    return best_model_name

@task
def notify_api_refresh():
    """Tells the Render FastAPI server to hot-swap to the new model."""
    render_url = "https://nigeria-inflation-2026.onrender.com/refresh"
    try:
        response = requests.post(render_url, timeout=30)
        logger.info(f"API Refresh Response: {response.status_code}")
    except Exception as e:
        logger.warning(f"API Refresh skipped (Server might be sleeping): {e}")

# --- PREFECT FLOW ---

@flow(name="Nigeria Inflation Training Pipeline")
def training_flow():
    data = prepare_data()
    evaluate_models(data)
    notify_api_refresh()

if __name__ == "__main__":
    training_flow()