"""
Concurrency:
- multiple tasks are managed at the same time, but not necessarily executing simultaneously
- useful for I/O-bound tasks where the program is waiting for external resources.

Parallelism:
- multiple tasks execute at the exact same time, 
- beneficial for CPU-bound tasks that require a lot of computation.
"""

import pandas as pd
from typing import List
import typer

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI
from pydantic import BaseModel

# Import your existing functions
from scripts.utils import find_latest_file, get_latest_artifacts
from scripts.data_preprocess import preprocess_leads

# --- Pydantic Model for Input Validation ---
class Lead(BaseModel):
    user_email: str
    optin_time: str # FastAPI will handle string-to-datetime conversion if needed, but string is safer here
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
    tag_time_series_webinar: int
    tag_webinar: int
    tag_webinar_01: int
    tag_webinar_no_degree: int
    tag_webinar_no_degree_02: int

# --- FastAPI Application ---
app = FastAPI(title="Lead Conversion API", version="1.0")

# Load artifacts at startup
try:
    model, label_encoders, feature_list = get_latest_artifacts()
    typer.echo(typer.style(f"\n✅ Model, encoders, and {len(feature_list)} features loaded! API is ready.", fg=typer.colors.BRIGHT_GREEN))
except (FileNotFoundError, ValueError) as e:
    print(f"\n❌ CRITICAL ERROR: Could not load artifacts. {e}")
    model, label_encoders, feature_list = None, None, None

# Root endpoint for health check
@app.get("/")
def health_check():
    return {"status": "Lead Conversion Prediction API is up and running!"}

# Prediction endpoint
@app.post("/predict")
def predict(leads: List[Lead]):
    """
    Receives a list of lead data, preprocesses it, and returns purchase predictions.
    """
    if not all([model, label_encoders, feature_list]):
        return {"error": "Model or artifacts not loaded. Check server logs."}

    # Convert Pydantic models to a list of dicts, then to a DataFrame
    leads_dict = [lead.model_dump() for lead in leads]
    df_raw = pd.DataFrame(leads_dict)
    print(f"\nReceived {len(df_raw)} record(s) for prediction.")

    # Preprocess the data (same as before)
    try:
        df_processed, _ = preprocess_leads(df_raw, label_encoders=label_encoders, fit_encoders=False)
        print("✅ Preprocessing successful.")
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}
        
    # Align columns with the feature list (same as before)
    try:
        df_for_prediction = df_processed[feature_list]
        print("✅ Columns aligned with training features.")
    except KeyError as e:
        missing_cols = set(feature_list) - set(df_processed.columns)
        return {"error": f"Input data is missing expected columns: {missing_cols}"}

    # Make prediction (same as before)
    try:
        prediction_proba = model.predict_proba(df_for_prediction)[:, 1]
        print("✅ Prediction successful.")
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Return the result
    return {"prediction_probability": prediction_proba.tolist()}

