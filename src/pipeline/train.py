import pandas as pd
import numpy as np
import pickle
import os
import requests
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from clearml import Task, Dataset, OutputModel
from prefect import task, flow

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
    """Trains multiple models, selects winner, and auto-tags in Registry."""
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
        future = m.make_future_dataframe(periods=t_len, freq='YE')
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

    # Run Shootout
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
    
    # --- SAVE & AUTO-TAG (FIXED VERSION) ---
    os.makedirs("models", exist_ok=True)
    winner_obj = {"Prophet": p_mod, "ARIMA": a_mod, "XGBoost": x_mod}[best_model_name]
    model_path = "models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner_obj, f)
    
    # 1. Initialize OutputModel
    output_model = OutputModel(
        task=task_tracking, 
        name="Nigeria_Inflation_ARIMA_2026", 
        framework="Scikit-Learn"
    )
    output_model.update_weights(weights_filename=model_path)
    
    # 2. Correct way to add tags in ClearML: Tag the task itself
    task_tracking.add_tags(["production", f"winner-{best_model_name}"])
    
    # 3. Finalize
    output_model.publish()
    
    print(f"Model {best_model_name} registered and Task tagged as 'production'.")
    task_tracking.close()
    return best_model_name

@task
def notify_api_refresh():
    """Tells the FastAPI server to hot-swap to the new model."""
    try:
        # Update with your specific API address if different
        response = requests.post("http://127.0.0.1:8000/refresh", timeout=10)
        if response.status_code == 200:
            print("API Refresh Triggered Successfully!")
        else:
            print(f"API Refresh failed with status: {response.status_code}")
    except Exception as e:
        print(f"API not reachable for refresh: {e}")

# --- PREFECT FLOW ---

@flow(name="Nigeria Inflation Training Pipeline")
def training_flow():
    """Main orchestration flow."""
    data = prepare_data()
    best_name = evaluate_models(data)
    notify_api_refresh()

if __name__ == "__main__":
    training_flow()