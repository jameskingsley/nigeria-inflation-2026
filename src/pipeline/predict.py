import pandas as pd
import pickle
import os
from clearml import Task, Dataset

def generate_forecast():
    # 1. Initialize ClearML Task for tracking predictions
    task = Task.init(project_name="Inflation_Forecast_2026", task_name="Final_2026_Forecast")
    
    # 2. Load the best model
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run train.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 3. Generate Forecast for 2026
    # Assuming your data ended in late 2025, we forecast 12 steps (months)
    forecast_steps = 12
    forecast_results = model.get_forecast(steps=forecast_steps)
    forecast_values = forecast_results.predicted_mean
    
    # Create a simple date range for the forecast (Monthly)
    dates = pd.date_range(start="2026-01-01", periods=forecast_steps, freq='ME')
    
    forecast_df = pd.DataFrame({
        'Month': dates,
        'Predicted_Inflation': forecast_values.values
    })

    # 4. Print and Log Results
    december_2026_val = forecast_df.iloc[-1]['Predicted_Inflation']
    print("-" * 30)
    print(f"PROJECTED INFLATION (DEC 2026): {december_2026_val:.2f}%")
    print("-" * 30)
    print(forecast_df)

    # Upload results to ClearML
    task.upload_artifact("forecast_2026_csv", forecast_df)
    
    return december_2026_val

if __name__ == "__main__":
    generate_forecast()