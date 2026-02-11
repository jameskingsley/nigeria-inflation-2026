from clearml import Model

def get_production_model_path():
    """
    Finds the latest model in the Registry with the 'production' tag.
    """
    try:
        # This searches for the model auto-tagged in train.py
        model = Model.query_models(
            project_name="Inflation_Forecast_2026",
            model_name="Nigeria_Inflation_Forecast_Model",
            tags=["production"],
            only_published=True
        )
        
        if not model:
            raise Exception("No model found with tag 'production'")
            
        # Download the .pkl to the Render instance's local temp storage
        return model[0].get_local_copy()
    except Exception as e:
        print(f"Error fetching model: {e}")
        return None