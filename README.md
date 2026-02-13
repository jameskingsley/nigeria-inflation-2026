# Nigeria (NG) Inflation Forecasting System (MLOps 2026)

An end-to-end MLOps ecosystem designed to forecast Nigerian inflation rates. This project goes beyond simple modeling by implementing an automated "Model Shootout" pipeline that retrains, evaluates, and hot-swaps the best-performing model into production every month.

## System Architecture
The project is built on a distributed microservices architecture:

* Data Layer: Managed via ClearML Datasets, versioning historical inflation data from the National Bureau of Statistics (NBS).

* Orchestration Layer: Prefect 3.0 schedules a monthly training flow. It handles retries, caching, and infrastructure-as-code deployments.

* The "Shootout" Logic: Every month, three architectures compete on the latest data:

* ARIMA (Statistical): Captures linear auto-regressive trends.

* Prophet (Additive): Handles yearly seasonality and economic shocks.

* XGBoost (ML): Captures complex non-linear relationships.

* Model Registry: ClearML tracks every experiment. The winner (lowest MAE) is auto-tagged as production.

* Inference Layer: FastAPI (hosted on Render) pulls the production model and provides a /predict endpoint.

* Presentation Layer: A Streamlit dashboard provides real-time visualization for stakeholders.

## Project Structure
####  Deployment & Usage
###### Research & EDA
View the notebook/eda.ipynb to see the Augmented Dickey-Fuller (ADF) tests, stationarity transformations, and initial model benchmarking.

###### Running the Pipeline
To start the automated worker that polls for retraining tasks:

###### API Inference
The API supports hot-swapping. When the pipeline finds a new winner, it calls the /refresh endpoint to update the API without downtime.

* Endpoint: GET /predict?months=12

* Endpoint: POST /refresh (Internal use)

###### Model Performance
The current production model is ARIMA, achieving a Mean Absolute Error (MAE) of 12.85. The system is designed to pivot to Prophet or XGBoost automatically if economic volatility shifts the data distribution.

###### Tech Stack
* Core: Python 3.10

* Time Series: Statsmodels (SARIMAX), Prophet
* XGB
* MLOps: Prefect 3.0, ClearML

* DevOps:  Render, GitHub Actions

* UI: Plotly, Streamlit

**Live Dashboard:** [View Live App](https://nigeria-inflation-2026-nt3fwyrct2rs9j9dlxglta.streamlit.app/)


Maintained by **James Kingsley**  Looking to leverage data to solve economic challenges in emerging markets.