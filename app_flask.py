import flask
import joblib
import pandas as pd
import os
import re
import json
from typing import Tuple, List

# Import the specific preprocessing function from your script
from scripts.utils import find_latest_file, get_latest_artifacts
from scripts.data_preprocess import preprocess_leads

# --- Helper Functions to Load Latest Artifacts ---

# def find_latest_file(directory: str, prefix: str, extension: str) -> str:
#     """Finds the most recent file in a directory based on a timestamp."""
#     files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
#     if not files:
#         return None
#     latest_file = sorted(files, reverse=True)[0]
#     return os.path.join(directory, latest_file)

# def get_latest_artifacts() -> Tuple[any, dict, list]:
#     """
#     Loads the latest model, its corresponding label encoders, and the feature list.
#     """
#     print("🔎 Searching for the latest model and artifacts...")

#     # Find the latest model file
#     model_dir = "models"
#     model_path = find_latest_file(model_dir, prefix="xgboost_ray_best_", extension=".pkl")
#     if not model_path:
#         raise FileNotFoundError(f"No model files found in '{model_dir}'.")

#     # Extract timestamp from the model filename to find matching artifacts
#     timestamp_match = re.search(r'(\d{8}_\d{6})', model_path)
#     if not timestamp_match:
#         raise ValueError(f"Could not extract timestamp from model file: {model_path}")
#     timestamp = timestamp_match.group(1)
    
#     # Find corresponding encoder and feature files
#     encoder_path = f"models/labels/xgboost_label_encoders_{timestamp}.pkl"
#     features_path = f"models/features/xgboost_features_{timestamp}.json"

#     if not os.path.exists(encoder_path):
#         raise FileNotFoundError(f"Missing encoder file for timestamp {timestamp}: {encoder_path}")
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"Missing feature list for timestamp {timestamp}: {features_path}")

#     # Load all artifacts
#     print(f"✅ Loading model: {model_path}")
#     model = joblib.load(model_path)
    
#     print(f"✅ Loading encoders: {encoder_path}")
#     encoders = joblib.load(encoder_path)

#     print(f"✅ Loading feature list: {features_path}")
#     with open(features_path, 'r') as f:
#         feature_list = json.load(f)
    
#     return model, encoders, feature_list


# --- Flask Application ---

app = flask.Flask(__name__)

# Load all artifacts when the application starts
try:
    model, label_encoders, feature_list = get_latest_artifacts()
    print(f"\n🎉 Model, encoders, and {len(feature_list)} features loaded successfully! Ready for predictions.")
except (FileNotFoundError, ValueError) as e:
    print(f"\n❌ CRITICAL ERROR: Could not load artifacts. {e}")
    model, label_encoders, feature_list = None, None, None

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, label_encoders, feature_list]):
        return flask.jsonify({'error': 'Model or artifacts not loaded. Check server logs.'}), 500

    json_data = flask.request.get_json()
    if not json_data:
        return flask.jsonify({'error': 'No input data provided'}), 400

    df_raw = pd.DataFrame(json_data)
    print(f"\nReceived {len(df_raw)} record(s) for prediction.")

    # Preprocess the new data
    try:
        df_processed, _ = preprocess_leads(df_raw, label_encoders=label_encoders, fit_encoders=False)
        print("✅ Preprocessing successful.")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return flask.jsonify({'error': f"Preprocessing failed: {str(e)}"}), 400
        
    # --- THIS IS THE CRUCIAL FIX ---
    # Align columns of the processed data with the feature list from training
    try:
        df_for_prediction = df_processed[feature_list]
        print("✅ Columns aligned with training features.")
    except KeyError as e:
        missing_cols = set(feature_list) - set(df_processed.columns)
        print(f"❌ Column mismatch. Missing: {missing_cols}. Error: {e}")
        return flask.jsonify({'error': f"Input data is missing expected columns: {missing_cols}"}), 400

    # Make a prediction
    try:
        prediction_proba = model.predict_proba(df_for_prediction)[:, 1]
        print("✅ Prediction successful.")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return flask.jsonify({'error': f"Prediction failed: {str(e)}"}), 500

    # Return the result
    return flask.jsonify({'prediction_probability': prediction_proba.tolist()})

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return "Lead Conversion Prediction API is up and running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
