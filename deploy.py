from src.pipeline.train import training_flow
from prefect.client.schemas.schedules import CronSchedule
import logging

# Setup professional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrefectDeployment")

def run_deployment():
    logger.info("Initializing Prefect 3.0 Deployment...")

    # .serve() registers the flow and starts a local worker 
    # that listens for the schedule.
    training_flow.serve(
        name="monthly-inflation-retraining",
        schedule=CronSchedule(cron="0 0 1 * *"), # 1st of every month at midnight
        description="Automated retraining of ARIMA, Prophet, and XGBoost models for Nigeria Inflation 2026.",
        tags=["production", "mlops-nigeria"],
        # No need for Deployment.apply() or build_from_flow anymore!
    )

if __name__ == "__main__":
    try:
        run_deployment()
    except KeyboardInterrupt:
        logger.info("Deployment worker stopped manually.")