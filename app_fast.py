"""
Concurrency:
- multiple tasks are managed at the same time, but not necessarily executing simultaneously
- useful for I/O-bound tasks where the program is waiting for external resources.

Parallelism:
- multiple tasks execute at the exact same time, 
- beneficial for CPU-bound tasks that require a lot of computation.
"""
import pandas as pd
import json
import os
import re
import joblib
from typing import List
import typer

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel

# Import your existing functions
from scripts.utils import find_latest_file, get_latest_artifacts
from scripts.data_preprocess import preprocess_leads

# --- Pydantic Model for Input Validation ---
# --- Pydantic Model for Input Validation ---
class Lead(BaseModel):
    user_email: str
    optin_time: str
    country_code: str
    member_rating: int
    tag_count: int
    tag_aws_webinar: int
    tag_learning_lab: int
    tag_learning_lab_05: int
    tag_learning_lab_09: int
    tag_learning_lab_11: int
    tag_learning_lab_12: int
    tag_learning_lab_13: int
    tag_learning_lab_14: int
    tag_learning_lab_15: int
    tag_learning_lab_16: int
    tag_learning_lab_17: int
    tag_learning_lab_18: int
    tag_learning_lab_19: int
    tag_learning_lab_20: int
    tag_learning_lab_21: int
    tag_learning_lab_22: int
    tag_learning_lab_23: int
    tag_learning_lab_24: int
    tag_learning_lab_25: int
    tag_learning_lab_26: int
    tag_learning_lab_27: int
    tag_learning_lab_28: int
    tag_learning_lab_29: int
    tag_learning_lab_30: int
    tag_learning_lab_31: int
    tag_learning_lab_32: int
    tag_learning_lab_33: int
    tag_learning_lab_34: int
    tag_learning_lab_35: int
    tag_learning_lab_36: int
    tag_learning_lab_37: int
    tag_learning_lab_38: int
    tag_learning_lab_39: int
    tag_learning_lab_40: int
    tag_learning_lab_41: int
    tag_learning_lab_42: int
    tag_learning_lab_43: int
    tag_learning_lab_44: int
    tag_learning_lab_45: int
    tag_learning_lab_46: int
    tag_learning_lab_47: int
    tag_learning_lab_48: int
    tag_learning_lab_49: int
    tag_learning_lab_50: int
    tag_learning_lab_51: int
    tag_learning_lab_52: int
    tag_learning_lab_53: int
    tag_learning_lab_54: int
    tag_learning_lab_55: int
    tag_learning_lab_56: int
    tag_learning_lab_57: int
    tag_learning_lab_58: int
    tag_learning_lab_59: int
    tag_learning_lab_60: int
    tag_learning_lab_61: int
    tag_learning_lab_62: int
    tag_learning_lab_63: int
    tag_learning_lab_64: int
    tag_learning_lab_65: int
    tag_learning_lab_66: int
    tag_learning_lab_67: int
    tag_learning_lab_68: int
    tag_learning_lab_69: int
    tag_learning_lab_70: int
    tag_learning_lab_71: int
    tag_learning_lab_72: int
    tag_learning_lab_73: int
    tag_learning_lab_74: int
    tag_learning_lab_75: int
    tag_learning_lab_76: int
    tag_learning_lab_77: int
    tag_learning_lab_78: int
    tag_learning_lab_79: int
    tag_learning_lab_80: int
    tag_learning_lab_81: int
    tag_learning_lab_82: int
    tag_learning_lab_83: int
    tag_learning_lab_84: int
    tag_learning_lab_85: int
    tag_learning_lab_86: int
    tag_learning_lab_87: int
    tag_learning_lab_88: int
    tag_learning_lab_89: int
    tag_learning_lab_90: int
    tag_learning_lab_91: int
    tag_learning_lab_92: int
    tag_learning_lab_93: int
    tag_learning_lab_94: int
    tag_learning_lab_95: int
    tag_learning_lab_96: int
    tag_learning_lab_97: int
    tag_learning_lab_98: int
    tag_learning_lab_99: int
    tag_learning_lab_100: int
    tag_time_series_webinar: int
    tag_webinar_no_degree: int
    tag_webinar_no_degree_02: int
    tag_webinar: int
    tag_webinar_01: int

# --- API Instance ---
# Note: 'lifespan' replaces 'on_event'
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, label_encoders, feature_list, model_metadata
    try:
        # --- Startup Event ---
        typer.echo(typer.style("🚀 ---- Starting up API ----", fg=typer.colors.BRIGHT_CYAN))
        model, label_encoders, feature_list = get_latest_artifacts()

        # Load metadata
        model_path = find_latest_file("models", "xgboost_ray_best_", ".pkl")
        if model_path:
            timestamp_match = re.search(r'(\d{8}_\d{6})', model_path)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                metadata_path = f"models/json/xgboost_ray_best_metadata_{timestamp}.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        model_metadata = json.load(f)

        typer.echo(typer.style("✅ Model, encoders, and feature list loaded successfully.", fg=typer.colors.BRIGHT_GREEN))
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        # Setting artifacts to None to indicate a failure
        model = None
        label_encoders = None
        feature_list = None
        model_metadata = None

    # --- Yield control to the application to handle requests ---
    yield

    # --- Shutdown Event ---
    print("👋 Shutting down API and releasing resources...")
    # You could add any cleanup logic here if needed, e.g., closing connections.
    print("✅ Resources released successfully.")

app = FastAPI(title="Lead Scoring API", lifespan=lifespan)

# --- API Endpoints ---
@app.get("/health")
def health_check():
    """Performs a health check on the API."""
    if model:
        return {"status": "ok", "message": "API is running and model is loaded."}
    else:
        return {"status": "error", "message": "API is running but model failed to load."}

@app.get("/features")
def get_model_features():
    """Returns the list of features expected by the model."""
    if feature_list:
        return {"features": feature_list}
    else:
        raise HTTPException(status_code=404, detail="Feature list not loaded.")

@app.get("/metadata")
def get_model_metadata():
    """Returns the full metadata of the currently loaded model."""
    if model_metadata:
        return {"metadata": model_metadata}
    else:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

@app.post("/predict")
def predict(leads: List[Lead]):
    """
    Receives a list of lead data, preprocesses it, and returns purchase predictions.
    """
    if not all([model, label_encoders, feature_list]):
        raise HTTPException(status_code=503, detail="Model or artifacts not loaded. Service unavailable.")

    # Convert Pydantic models to a list of dicts, then to a DataFrame
    leads_dict = [lead.model_dump() for lead in leads]
    df_raw = pd.DataFrame(leads_dict)
    print(f"\nReceived {len(df_raw)} record(s) for prediction.")

    # Preprocess the data (same as before)
    try:
        df_processed, _ = preprocess_leads(df_raw, label_encoders=label_encoders, fit_encoders=False)
        print("✅ Preprocessing successful.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
        
    # Align columns with the feature list (same as before)
    try:
        df_for_prediction = df_processed[feature_list]
        print("✅ Columns aligned with training features.")
    except KeyError as e:
        missing_cols = set(feature_list) - set(df_processed.columns)
        raise HTTPException(status_code=400, detail=f"Input data is missing expected columns: {missing_cols}")

    # Make prediction (same as before)
    try:
        prediction_proba = model.predict_proba(df_for_prediction)[:, 1]
        print("✅ Prediction successful.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Return predictions with a simple structure
    predictions = [
        {"user_email": lead.user_email, "prediction_proba": float(proba)}
        for lead, proba in zip(leads, prediction_proba)
    ]
    
    return {"predictions": predictions}

# --- Practice with FastAPI Parameters ---
@app.get("/predict_by_email")
def predict_by_email(email: str = Query(
    ..., # ... means this parameter is required
    title="User Email",
    description="The email of the user to get a prediction for.",
    min_length=5,
    max_length=100,
    regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)):
    """
    Retrieves a single prediction for a specific user email.
    """
    
    # You would need to retrieve lead data for this email from a database or file here.
    # For this example, we'll return a placeholder.
    
    # In a real-world scenario, you would:
    # 1. Fetch the lead's data using the `email` from your data source.
    # 2. Convert the data to a format your `preprocess_leads` function can use.
    # 3. Call `preprocess_leads` and then `model.predict_proba`.
    
    return {
        "user_email": email,
        "prediction_proba": 0.50 # Placeholder value
    }


@app.get("/predict_by_id/{lead_id}")
def predict_by_id(
    lead_id: int = Path(
        ...,
        title="Lead ID",
        description="The unique Mailchimp ID for a lead.",
        gt=0  # Ensure the ID is a positive integer
    )
):
    """
    Retrieves a single prediction for a specific user ID.
    """

    # In a real-world scenario, you would:
    # 1. Fetch the lead's data from your database using the `lead_id`.
    # 2. Preprocess the data into a DataFrame.
    # 3. Use your loaded model to make a prediction.
    
    # Placeholder for demonstration
    return {
        "lead_id": lead_id,
        "prediction_proba": 0.65
    }
