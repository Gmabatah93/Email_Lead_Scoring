from ray import serve
import joblib
import pandas as pd
import numpy as np

@serve.deployment(num_replicas=2)
class XGBoostModel:
    def __init__(self, model_path):
        # Load the trained model
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")

    @serve.batch(max_batch_size=32)  # Batch up to 32 requests
    async def predict(self, inputs: list):
        # Convert inputs to a DataFrame
        input_df = pd.DataFrame(inputs)
        # Perform batch inference
        predictions = self.model.predict_proba(input_df)[:, 1]  # Probability of positive class
        return predictions.tolist()

# Deploy the model
model_path = "models/ray/xgboost_ray_best_20250820_112607.pkl"
serve.run(XGBoostModel.bind(model_path), name="XGBoostModel", route_prefix="/predict")