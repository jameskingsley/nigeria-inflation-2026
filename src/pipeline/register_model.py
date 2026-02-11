from clearml import Task, OutputModel
import os

def register_existing_model():
    # Task ID from your dashboard: f885ff1a7b7e45f186aed391fb0f5341
    task_id = "f885ff1a7b7e45f186aed391fb0f5341"
    task = Task.get_task(task_id=task_id)
    
    print(f"Found task: {task.name} (ID: {task_id})")

    # 2. Get the artifact path
    # This downloads the .pkl from your ClearML storage to a local temp folder
    model_artifact_path = task.artifacts['best_model'].get_local_copy()
    
    # Register it in the Model Registry
    # This creates a standalone entry in the "MODELS" tab of your project
    output_model = OutputModel(
        task=task,
        name="Nigeria_Inflation_ARIMA_2026",
        framework="Scikit-Learn", # ARIMA is statsmodels, but scikit-learn is a good generic tag
        tags=["production", "2026-forecast"],
        comment="Official 12.33% target model for Dec 2026"
    )
    
    # Upload the weights to the registry
    output_model.update_weights(weights_filename=model_artifact_path)
    
    # Mark it as 'Published' so it's locked and ready for the API
    output_model.publish()
    
    print(f"Model registered successfully!")
    print(f"New Model ID: {output_model.id}")

if __name__ == "__main__":
    register_existing_model()