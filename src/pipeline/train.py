import pandas as pd
import numpy as np
import pickle
import os
import requests
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from clearml import Task, Dataset, OutputModel, Model
from prefect import task, flow

# --- CORE LOGIC: REPLACING YOUR HELPER ---

def get_production_model_path():
    """
    Finds the latest model in the Registry with the 'production' tag.
    """
    try:
        # We query for the model directly
        models = Model.query_models(
            project_name="Inflation_Forecast_2026",
            tags=["production"],
            only_published=True
        )
        
        if not models:
            # Fallback if the user forgot to "Publish" the model
            models = Model.query_models(
                project_name="Inflation_Forecast_2026",
                tags=["production"],
                only_published=False
            )

        if not models:
            raise Exception("No model found with tag 'production'")
            
        return models[0].get_local_copy()
    except Exception as e:
        print(f"Error fetching model: {e}")
        return None

# --- PREFECT TASKS ---

@task(retries=2, retry_delay_seconds=60)
def prepare_data():
    """Fetches the latest dataset from ClearML."""
    dataset_path = Dataset.get(
        dataset_project="Inflation_Forecast_2026", 
        dataset_name="Nigeria_Inflation_Data"
    ).get_local_copy()
    
    df = pd.read_csv(os.path.join(dataset_path, "nigeria_inflation.csv"))
    df['ds'] = pd.to_datetime(df['ds'])
    return df

@task
def evaluate_models(df):
    """Trains multiple models, selects winner, and promotes it via direct tagging."""
    task_tracking = Task.init(
        project_name="Inflation_Forecast_2026", 
        task_name="Monthly_Model_Competition",
        reuse_last_task_id=False 
    )
    
    test_len = 3
    train_df = df[:-test_len]
    test_actuals = df[-test_len:]['y'].values
    
    # --- MODEL TRAINING FUNCTIONS ---
    def train_arima(train, t_len):
        model = SARIMAX(train['y'], order=(1,1,1))
        res = model.fit(disp=False)
        return res.forecast(steps=t_len), res

    def train_prophet(train, t_len):
        m = Prophet(yearly_seasonality=True)
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
    print(f"Winner: {best_model_name} (MAE: {scores[best_model_name]:.4f})")
    
    # --- SAVE WINNER ---
    os.makedirs("models", exist_ok=True)
    winner_obj = {"Prophet": p_mod, "ARIMA": a_mod, "XGBoost": x_mod}[best_model_name]
    model_path = f"models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner_obj, f)
    
    # --- BULLETPROOF PRODUCTION PROMOTION ---
    
    # 1. Clear old 'production' tags using property assignment
    old_production_models = Model.query_models(
        project_name="Inflation_Forecast_2026",
        tags=["production"]
    )
    for old_m in old_production_models:
        current_tags = old_m.tags
        if "production" in current_tags:
            # Overwrite tags list without 'production'
            old_m.tags = [t for t in current_tags if t != "production"]
            print(f"Removed production tag from old model: {old_m.id}")

    # 2. Register and Tag the new winner
    output_model = OutputModel(
        task=task_tracking, 
        name="Nigeria_Inflation_Forecast_Model", 
        framework="Scikit-Learn" 
    )
    output_model.update_weights(weights_filename=model_path)
    
    # Direct assignment for the new model
    output_model.tags = ["production", f"winner-{best_model_name}"]
    output_model.publish()
    
    print(f"✅ Success: {best_model_name} promoted to Production.")
    task_tracking.close()
    return best_model_name

@task
def notify_api_refresh():
    """Tells the Render FastAPI server to hot-swap to the new model."""
    try:
        render_url = "https://nigeria-inflation-2026.onrender.com/refresh"
        response = requests.post(render_url, timeout=15)
        if response.status_code == 200:
            print("🚀 Render API Refresh Triggered!")
        else:
            print(f"⚠️ API Refresh status: {response.status_code}")
    except Exception as e:
        print(f"❌ API Refresh not reachable: {e}")

# --- PREFECT FLOW ---

@flow(name="Nigeria Inflation Training Pipeline")
def training_flow():
    data = prepare_data()
    evaluate_models(data)
    notify_api_refresh()

if __name__ == "__main__":
    training_flow()