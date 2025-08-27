"""
FUTURE ENHANCEMENT:
- turn into a cli app
- making hardcoded values configurable via command line arguments
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import time
from datetime import datetime
import mlflow
import joblib
import os
import json
from pathlib import Path
import typer
from typing_extensions import Annotated
from typing import Tuple, Dict, Any
from sklearn.metrics import f1_score, recall_score, roc_auc_score

import ray
from ray import tune
from ray.tune import CheckpointConfig
from ray.tune.search import BasicVariantGenerator
from ray.util.metrics import Counter, Histogram, Gauge
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air import session

# Import preprocessing functions from your preprocess file
from data_preprocess import prepare_xgboost_data, preprocess_leads

# Initialize Typer app
app = typer.Typer()

def trainable_xgboost(config: Dict[str, Any]) -> None:
    """
    Train an XGBoost model using Ray Tune for hyperparameter optimization.

    Args:
        config (Dict[str, Any]): Hyperparameter configuration for the XGBoost model.

    Returns:
        None. Reports metrics to Ray Tune and saves model checkpoints.
    """
    start_time = time.time()
    
    # Retrieve data from Ray object store
    X_train = ray.get(X_train_ref)
    X_val = ray.get(X_val_ref)
    y_train = ray.get(y_train_ref)
    y_val = ray.get(y_val_ref)

    # Initialize XGB
    model = xgb.XGBClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1
    )

    # Train model
    model.fit(X_train, y_train)
    
    # Predictions on validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Metrics on validation set
    f1 = f1_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)

    # Report to Ray
    tune.report({"f1": f1, "recall": recall, "roc_auc": roc_auc})

def save_best_model(
        best_result: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series, 
        timestamp: str) -> Tuple[xgb.XGBClassifier, str, str]:
    """
    Recreate and save the best XGBoost model using the best hyperparameters found by Ray Tune.

    Args:
        best_result (Any): Ray Tune result object containing best config and metrics.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        timestamp (str): Timestamp string for file naming.

    Returns:
        Tuple containing:
            - best_model (xgb.XGBClassifier): The retrained best model.
            - model_path (str): Path to the saved model file.
            - metadata_path (str): Path to the saved metadata JSON file.
    """
    
    # Convert NumPy arrays to pandas DataFrames/Series if necessary
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(y_val, np.ndarray):
        y_val = pd.Series(y_val)
    
    # Combine training and validation sets
    X_full_train = pd.concat([X_train, X_val])
    y_full_train = pd.concat([y_train, y_val])
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Recreate best model
    best_model = xgb.XGBClassifier(
        **best_result.config,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    
    # Retrain on full training set (training + validation)
    best_model.fit(X_full_train, y_full_train)
    
    # Save model
    model_path = f"models/xgboost_ray_best_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": "XGBoost_Ray_Tune",
        "best_config": best_result.config,
        "best_metrics": best_result.metrics,
        "training_samples": len(X_full_train),
        "validation_samples": len(X_val),  # For reference
        "search_algorithm": "Random Search"
    }
    
    metadata_path = f"models/json/xgboost_ray_best_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(typer.style(f"✅ Best model saved: {model_path}", fg=typer.colors.BRIGHT_GREEN))
    typer.echo(typer.style(f"✅ Metadata saved: {metadata_path}", fg=typer.colors.BRIGHT_GREEN))

    return best_model, model_path, metadata_path

@app.command()
def main(
    input_path: Annotated[Path, typer.Option(help="Path to the cleaned leads CSV file.")] = "data/leads_cleaned.csv",
    mlflow_uri: Annotated[str, typer.Option(help="MLflow tracking URI.")] = "file:./mlruns",
    experiment_name: Annotated[str, typer.Option(help="Name of the MLflow experiment.")] = "xgboost_ray_experiment",
    metric: Annotated[str, typer.Option(help="Metric to optimize for Ray Tune.")] = "roc_auc",
    num_trials: Annotated[int, typer.Option(help="Number of Ray Tune trials.")] = 12,
    num_workers: Annotated[int, typer.Option(help="Number of concurrent workers for Ray Tune.")] = 3
):
    """
    Run the complete XGBoost model training and hyperparameter optimization pipeline with Ray Tune.
    """
    global trial_counter, training_time_histogram, roc_auc_gauge
    global X_train_ref, X_val_ref, y_train_ref, y_val_ref

    # SETUP ============================================================
    typer.echo("=" * 50)
    typer.echo(typer.style("SETUP", fg=typer.colors.CYAN))
    typer.echo("=" * 50)

    # MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    typer.echo(f"🖥️ MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Create Unique Experiment Name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name_with_timestamp = f"{experiment_name}_{timestamp}"
    typer.echo(f"🖥️ Experiment name: {experiment_name_with_timestamp}\n")
    
    # Initialize Ray
    typer.echo(typer.style("⚙️ Initializing Ray...", fg=typer.colors.BRIGHT_YELLOW))
    ray.init(
        num_cpus=num_workers,
        object_store_memory=2_000_000_000,
        log_to_driver=False,
        include_dashboard=True
    )
    typer.echo(typer.style(f"✅ Ray initialized with resources: {ray.cluster_resources()}\n", fg=typer.colors.GREEN))

    # PREPROCESS ============================================================
    typer.echo("=" * 50)
    typer.echo(typer.style("PREPROCESS", fg=typer.colors.CYAN))
    typer.echo("=" * 50)

    # Use the preprocessing function from preprocess.py
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = prepare_xgboost_data(
        data_path=input_path,
        test_size=0.2,
        val_size=0.2,
        random_state=123
    )
    
    typer.echo(typer.style(f"✅ Ready for XGBoost training with {X_train.shape[0]} training samples.", fg=typer.colors.GREEN))
    
    # Save the test set to CSV files
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    typer.echo(typer.style("✅ Test set saved to 'data/X_test.csv' and 'data/y_test.csv'", fg=typer.colors.GREEN))

    # Save label encoders
    label_encoder_path = f"models/labels/xgboost_label_encoders_{timestamp}.pkl"
    joblib.dump(label_encoders, label_encoder_path)
    typer.echo(typer.style(f"✅ Label encoders saved to '{label_encoder_path}'\n", fg=typer.colors.GREEN))

    # Convert validation and test sets to numpy arrays
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    # Store large objects in the Ray object store (for efficient distributed access)
    X_train_ref = ray.put(X_train)
    X_val_ref = ray.put(X_val)
    y_train_ref = ray.put(y_train)
    y_val_ref = ray.put(y_val)

    typer.echo(typer.style("💾 Stored all data references in Ray.\n", fg=typer.colors.BRIGHT_YELLOW))
 
    # RAY: Tune ===========================================================
    typer.echo("=" * 50)
    typer.echo(typer.style(f" RAY TUNE: {metric}", fg=typer.colors.BRIGHT_MAGENTA))
    typer.echo("=" * 50)

    # Metric Selection
    typer.echo(typer.style(f"🎯 Using metric: {metric} for optimization\n", fg=typer.colors.RED))

    # RAY: Search Space
    search_space = {
        'max_depth': tune.randint(3, 8),
        'learning_rate': tune.uniform(0.01, 0.3),
        'n_estimators': tune.randint(50, 200),  # Smaller range for faster trials
        'subsample': tune.uniform(0.7, 1.0),
        'colsample_bytree': tune.uniform(0.7, 1.0)
    }
    # typer.echo(f"🔥 Search Space: {search_space}")
    typer.echo(f"🔥 Search Space:\n{json.dumps(search_space, indent=2, default=str)}")

    # RAY: Tune Configuration
    tune_config = tune.TuneConfig(
        metric=metric,
        mode="max",
        num_samples=num_trials,
        max_concurrent_trials=num_workers,
        search_alg=BasicVariantGenerator(random_state=42) 
    )
    typer.echo(f"🔥 Tune Config: Random Search w/ {num_trials} trials, {num_workers} workers.")

    # RAY: Run Configuration
    run_config=tune.RunConfig(
            name=f"RAY_xgboost_els_{metric}_{timestamp}",
            storage_path='/Users/isiomamabatah/Desktop/Python/Projects/Email_Lead_Scoring/results/training',
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri="file:./mlruns",
                    experiment_name=experiment_name_with_timestamp,
                    save_artifact=True,
                    tags={"algorithm": "random_search", "model": "xgboost"}
                )
            ]
    )
    typer.echo(f"🔥 Run Config: set with MLflow logging at {experiment_name_with_timestamp}.")

    # Run Ray Tune: 
    # - Random Search Algorithm
    tuner_Random = tune.Tuner(
        trainable=trainable_xgboost,
        tune_config=tune_config,
        param_space=search_space,
        run_config=run_config
    )
    typer.echo("🔥 Ray Tuner:  Initialized (Random Search) \n")

    # XGBoost: RAY FIT ===========================================================
    typer.echo("=" * 50)
    typer.echo(typer.style(f"👾 XGBoost: RAY Fit {metric} 👾", fg=typer.colors.BRIGHT_MAGENTA))
    typer.echo("=" * 50)

    results = tuner_Random.fit()
    typer.echo(typer.style("✅ Ray Tune hyperparameter optimization completed!\n", fg=typer.colors.BRIGHT_GREEN))

    # RESULTS ===========================================================
    typer.echo("=" * 50)
    typer.echo(typer.style("RESULTS", fg=typer.colors.CYAN))
    typer.echo("=" * 50)

    best_result = results.get_best_result(metric=metric, mode="max")
    typer.echo(f"📝 Best trial config: {best_result.config}")
    # print("📝 Best trial metrics:", best_result.metrics)

    # EXPAND YOUR ANALYSIS:
    # Get all results for deeper analysis
    df_results = results.get_dataframe()
    sorted_df = df_results.sort_values(by="roc_auc", ascending=False)
    typer.echo(f"\n📝 Total trials completed: {len(df_results)}")
    typer.echo(f"📝 Best ROC AUC achieved: {df_results['roc_auc'].max():.4f}")
    typer.echo(f"📝 Average ROC AUC: {df_results['roc_auc'].mean():.4f}")

    # Mlflow Logging 
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(experiment_name_with_timestamp).experiment_id):
        mlflow.log_params(best_result.config)
        numeric_metrics = {k: v for k, v in best_result.metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric_metrics)

    typer.echo(typer.style("✅ Best trial logged to MLflow!\n", fg=typer.colors.BRIGHT_GREEN))

    # SAVE BEST MODEL ===========================================================
    best_model, model_path, metadata_path = save_best_model(
        best_result, 
        X_train, 
        y_train, 
        X_val, 
        y_val,
        timestamp
    )

    # SHUTDOWN ===========================================================
    typer.echo(typer.style("Dashboard is running. Press Ctrl+C to shut down...", fg=typer.colors.BRIGHT_YELLOW))
    try:
        while True:
            time.sleep(1) # Keeps the process alive without consuming CPU
    except KeyboardInterrupt:
        typer.echo("Shutting down Ray...")
    finally:
        ray.shutdown()
    
if __name__ == "__main__":
    app()
