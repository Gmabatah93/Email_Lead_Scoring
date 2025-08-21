from ray import serve
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize Ray Serve
serve.start()

# Load the trained model
model_path = "models/ray/xgboost_ray_best_20230820_091847.pkl"  # Update with the correct path
model = joblib.load(model_path)

# Define request schema
class LeadScoringRequest(BaseModel):
    features: list

# Define the model deployment
@serve.deployment(route_prefix="/predict")
@serve.ingress(FastAPI)
class LeadScoringModel:
    def __init__(self, model):
        self.model = model

    async def predict(self, request: LeadScoringRequest):
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        # Make predictions
        prediction = self.model.predict(features_df)
        probability = self.model.predict_proba(features_df)[:, 1]
        return {"prediction": int(prediction[0]), "probability": float(probability[0])}

# Deploy the model
LeadScoringModel.deploy(model)