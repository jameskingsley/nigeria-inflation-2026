import wbgapi as wb
import pandas as pd
from prefect import task, flow
from clearml import Dataset
import os
from datetime import datetime

# Define the indicator for Nigerian Inflation (Consumer Prices %)
INDICATOR = 'FP.CPI.TOTL.ZG'
COUNTRY = 'NGA'

@task(retries=3, retry_delay_seconds=10)
def fetch_world_bank_data():
    """Fetches historical inflation data from World Bank API."""
    print(f"Fetching data for {COUNTRY}...")
    # Fetching up to the most recent available data in 2026
    df = wb.data.DataFrame(INDICATOR, COUNTRY, time=range(2000, 2027))
    return df

@task
def transform_data(df):
    """Cleans and formats data for time-series forecasting."""
    df = df.T.reset_index()
    df.columns = ['ds', 'y']
    # Clean year strings (e.g., 'YR2024' -> '2024')
    df['ds'] = pd.to_datetime(df['ds'].str.replace('YR', ''), format='%Y')
    # Drop rows with missing values to ensure model stability
    df = df.dropna().sort_values('ds')
    return df

@task
def save_and_version_dataset(df):
    """Saves data locally and registers it in ClearML as a versioned dataset."""
    local_path = "data/raw/nigeria_inflation.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(local_path, index=False)
    
    # ClearML Dataset Versioning
    dataset = Dataset.create(
        dataset_name="Nigeria_Inflation_Data",
        dataset_project="Inflation_Forecast_2026"
    )
    dataset.add_files(local_path)
    dataset.upload()
    dataset.finalize()
    print(f"Dataset versioned and finalized in ClearML: {dataset.id}")
    return local_path

@flow(name="Data Ingestion Flow")
def ingestion_flow():
    """Main Orchestration Flow for Data Ingestion."""
    raw_data = fetch_world_bank_data()
    clean_data = transform_data(raw_data)
    path = save_and_version_dataset(clean_data)
    return path

if __name__ == "__main__":
    ingestion_flow()