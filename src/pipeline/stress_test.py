import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task

def run_stress_test():
    task = Task.init(project_name="Inflation_Forecast_2026", task_name="Scenario_Stress_Test")
    
    # Load the winning ARIMA model
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Baseline Forecast
    forecast_steps = 12
    baseline = model.get_forecast(steps=forecast_steps).predicted_mean.values
    
    # Apply the "June Shock"
    # I assume an external event adds 5% to the inflation rate from June onwards
    shock_value = 5.0 
    stressed_forecast = baseline.copy()
    stressed_forecast[5:] = stressed_forecast[5:] + shock_value # Apply shock from month 6 (June)

    # Create Comparison DataFrame
    dates = pd.date_range(start="2026-01-01", periods=12, freq='ME')
    df = pd.DataFrame({
        'Month': dates,
        'Baseline': baseline,
        'Stressed': stressed_forecast
    })

    # Visualize the Gap
    plt.figure(figsize=(10, 5))
    plt.plot(df['Month'], df['Baseline'], label='Original Forecast (12.33%)', color='green', marker='o')
    plt.plot(df['Month'], df['Stressed'], label='Shock Scenario (Fuel/FX Hike)', color='red', linestyle='--', marker='x')
    plt.fill_between(df['Month'], df['Baseline'], df['Stressed'], color='red', alpha=0.1)
    
    plt.title('2026 Inflation Stress Test: Baseline vs. Shock Scenario')
    plt.ylabel('Inflation Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Final Stats
    print(f"Original Dec 2026: {baseline[-1]:.2f}%")
    print(f"Stressed Dec 2026: {stressed_forecast[-1]:.2f}%")
    
    plt.show()
    task.get_logger().report_matplotlib_figure(title="Stress Test", series="Scenario A", figure=plt)

if __name__ == "__main__":
    run_stress_test()