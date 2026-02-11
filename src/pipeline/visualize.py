import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from clearml import Dataset, Task

def create_forecast_plot():
    # 1. Initialize ClearML for the Plot
    # We use reuse_last_task_id=False to ensure a fresh upload
    task = Task.init(project_name="Inflation_Forecast_2026", task_name="Visual_Analysis")
    
    # 2. Load Historical Data
    dataset_path = Dataset.get(dataset_project="Inflation_Forecast_2026", dataset_name="Nigeria_Inflation_Data").get_local_copy()
    hist_df = pd.read_csv(os.path.join(dataset_path, "nigeria_inflation.csv"))
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])

    # 3. Load Model and Generate Forecast
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print("Error: models/best_model.pkl not found!")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    forecast_steps = 12
    forecast_results = model.get_forecast(steps=forecast_steps)
    forecast_values = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int() 

    # Create Forecast DataFrame
    dates = pd.date_range(start="2026-01-01", periods=forecast_steps, freq='ME')
    forecast_df = pd.DataFrame({'ds': dates, 'y': forecast_values.values})

   # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Historical
    plt.plot(hist_df['ds'], hist_df['y'], label='HistoricalInflation', color='#1f77b4', linewidth=2)
    
    # Plot Forecast
    plt.plot(forecast_df['ds'], forecast_df['y'], label='2026 Forecast (ARIMA)', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # Add Uncertainty Shading
    plt.fill_between(dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2, label='Confidence Interval')

    # Formatting
    plt.title('Nigeria Inflation Forecast: Road to December 2026', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save Locally
    plot_path = "plots/inflation_forecast_2026.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Local plot saved to {plot_path}")
    
    # --- CLEARML FORCE UPLOAD ---
    # 1. Report as a Debug Sample (Found in "Debug Samples" -> "Plots")
    task.get_logger().report_matplotlib_figure(
        title="Inflation Forecast", 
        series="2026 Projection", 
        figure=plt,
        report_image=True
    )
    
    # 2. Report as an Artifact (Found in "Artifacts" tab)
    task.upload_artifact("final_forecast_plot", artifact_object=plot_path)
    
    print("Plot successfully pushed to ClearML (Check Debug Samples and Artifacts tabs)!")
    
    plt.show()

if __name__ == "__main__":
    create_forecast_plot()