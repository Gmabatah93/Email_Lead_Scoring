import joblib
import os
import re
import json
from typing import Tuple, List, Dict, Any
import typer

def find_latest_file(directory: str, prefix: str, extension: str) -> str:
    """Finds the most recent file in a directory based on a timestamp."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files: return None
    return os.path.join(directory, sorted(files, reverse=True)[0])

def get_latest_artifacts() -> Tuple[Any, Dict, List]:
    """Loads the latest model, encoders, and feature list."""
    typer.echo(typer.style("🔎 Searching for the latest model and artifacts...", fg=typer.colors.BRIGHT_YELLOW))
    model_dir = "models"
    model_path = find_latest_file(model_dir, "xgboost_ray_best_", ".pkl")
    if not model_path: raise FileNotFoundError(f"No model files in '{model_dir}'.")

    timestamp_match = re.search(r'(\d{8}_\d{6})', model_path)
    if not timestamp_match: raise ValueError(f"No timestamp in model file: {model_path}")
    timestamp = timestamp_match.group(1)
    
    encoder_path = f"models/labels/xgboost_label_encoders_{timestamp}.pkl"
    features_path = f"models/features/xgboost_features_{timestamp}.json"

    if not os.path.exists(encoder_path): raise FileNotFoundError(f"Missing encoder: {encoder_path}")
    if not os.path.exists(features_path): raise FileNotFoundError(f"Missing features: {features_path}")

    typer.echo(typer.style(f"⚙️ Loading model: {model_path}...", fg=typer.colors.BRIGHT_YELLOW))
    model = joblib.load(model_path)
    typer.echo(typer.style(f"⚙️ Loading encoders: {encoder_path}...", fg=typer.colors.BRIGHT_YELLOW))
    encoders = joblib.load(encoder_path)
    typer.echo(typer.style(f"⚙️ Loading feature list: {features_path}...", fg=typer.colors.BRIGHT_YELLOW))
    with open(features_path, 'r') as f:
        feature_list = json.load(f)
    
    return model, encoders, feature_list