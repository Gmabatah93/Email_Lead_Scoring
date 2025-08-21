"""
FUTURE ENHANCEMENT:
- turn into a cli app
- making hardcoded values configurable via command line arguments
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import xgboost as xgb
import time
from datetime import datetime
import mlflow
import joblib
import os
import json

import ray
from ray import tune
from ray.tune import CheckpointConfig
from ray.tune.search import BasicVariantGenerator
from ray.util.metrics import Counter, Histogram, Gauge
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air import session

# Import preprocessing functions from your preprocess file
from preprocess import prepare_xgboost_data, preprocess_leads

def trainable_xgboost(config):
    """Train XGBoost model with Ray Tune and Prometheus metrics"""
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
    
    # Save checkpoint AFTER training and metrics calculation
    checkpoint_dir = os.path.join("checkpoints", f"trial_{time.time()}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Record Prometheus metrics
    trial_counter.inc()
    training_time_histogram.observe(time.time() - start_time)
    roc_auc_gauge.set(roc_auc)

    # Report to Ray
    tune.report({"f1": f1, "recall": recall, "roc_auc": roc_auc})

def save_best_model(best_result, X_train, y_train, X_val, y_val, timestamp):
    """Recreate and save the best model with metadata"""
    
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
    model_path = f"models/ray/xgboost_ray_best_{timestamp}.pkl"
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
    
    metadata_path = f"models/ray/metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Best model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
    return best_model, model_path, metadata_path

if __name__ == "__main__":
    # SETUP ============================================================
    print("=" * 50)
    print("SETUP")
    print("=" * 50)

    # Get raw data
    df_leads_raw = pd.read_csv("data/leads_raw.csv")
    print(f"✅ Loaded raw data with {df_leads_raw.shape[0]} records and {df_leads_raw.shape[1]} features.")

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    print(f"🖥️ MLflow tracking URI: {mlflow.get_tracking_uri()}")

    experiment_name = f"xgboost_ray_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"🖥️ Experiment name: {experiment_name}")

    # Define the metric to optimize
    METRIC = "roc_auc"
    print(f"🖥️ Using metric: {METRIC} for optimization")

    # Create a unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize Ray
    ray.init(
        num_cpus=6,
        object_store_memory=2_000_000_000,
        log_to_driver=False,
        include_dashboard=True,
        _metrics_export_port=8080  # Enable Prometheus metrics
    )
    print(f"✅ Ray initialized with resources: {ray.cluster_resources()}\n\n")

    # PREPROCESS ============================================================
    print("=" * 50)
    print("PREPROCESS")
    print("=" * 50)

    # Apply general preprocessing
    df_leads_processed = preprocess_leads(df_leads_raw)
    print(f"✅ Preprocessed data with {df_leads_processed.shape[0]} records and {df_leads_processed.shape[1]} features.")
    print(f"✅ Preprocessing completed! Saved {df_leads_processed.shape[0]} records with {df_leads_processed.shape[1]} features.")
    print("📝 Data saved to: data/leads_cleaned.csv\n")

    # Use the preprocessing function from preprocess.py
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = prepare_xgboost_data(
        data_path="data/leads_cleaned.csv",
        test_size=0.2,
        val_size=0.2,
        random_state=123
    )

    # Save the test set to CSV files
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("✅ Test set saved to 'data/X_test.csv' and 'data/y_test.csv'")

    # Convert validation and test sets to numpy arrays
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    # Store large objects in the Ray object store (for efficient distributed access)
    global X_train_ref, X_val_ref, X_test_ref, y_train_ref, y_val_ref, y_test_ref
    X_train_ref = ray.put(X_train)
    X_val_ref = ray.put(X_val)
    X_test_ref = ray.put(X_test)
    y_train_ref = ray.put(y_train)
    y_val_ref = ray.put(y_val)
    y_test_ref = ray.put(y_test)

    print("✅ Preprocessing completed using imported function!")
    print(f"✅ Ready for XGBoost training with {X_train.shape[0]} training samples.\n\n")

    # XGBoost: Function ===================================================
    print("=" * 50)
    print("XGBoost: Trainable Function")
    print("=" * 50)

    # Custom metrics
    global trial_counter, training_time_histogram, roc_auc_gauge
    trial_counter = Counter("xgboost_trials_total", description="Total XGBoost trials completed")
    training_time_histogram = Histogram(
        "xgboost_training_duration_seconds", 
        description="XGBoost training time",
        boundaries=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]  # Time buckets in seconds
    )
    roc_auc_gauge = Gauge("xgboost_best_roc_auc", description="Best ROC AUC achieved")

    print("🔥 XGBoost trainable function defined with Ray Tune and Prometheus metrics!\n\n")

    # RAY: Tune ===========================================================
    print("=" * 50)
    print(f" RAY TUNE: {METRIC}")
    print("=" * 50)

    NUM_OF_TRIALS = 24
    WORKERS = 6

    # RAY: Search Space
    search_space = {
        'max_depth': tune.randint(3, 8),
        'learning_rate': tune.uniform(0.01, 0.3),
        'n_estimators': tune.randint(50, 200),  # Smaller range for faster trials
        'subsample': tune.uniform(0.7, 1.0),
        'colsample_bytree': tune.uniform(0.7, 1.0)
    }
    print(f"🔥 Search Space: {search_space}")

    # RAY: Tune Configuration
    tune_config = tune.TuneConfig(
        metric=METRIC,
        mode="max",
        num_samples=NUM_OF_TRIALS,
        max_concurrent_trials=WORKERS,
        search_alg=BasicVariantGenerator(random_state=42) 
    )
    print(f"🔥 Tune Config: Random Search w/ {NUM_OF_TRIALS} trials, {WORKERS} workers.")

    # RAY: Run Configuration
    run_config=tune.RunConfig(
            name=f"RAY_xgboost_els_{METRIC}_{timestamp}",
            storage_path='/Users/isiomamabatah/Desktop/Python/Projects/Email_Lead_Scoring/results/training',
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri="file:./mlruns",
                    experiment_name=experiment_name,
                    save_artifact=True,
                    tags={"algorithm": "random_search", "model": "xgboost"}
                )
            ],
            checkpoint_config = CheckpointConfig(
                num_to_keep=1,  # Keep only the best checkpoint
                checkpoint_score_attribute=METRIC,  # Metric to monitor
                checkpoint_score_order="max"  # Higher roc_auc is better
            )

    )
    print("🔥 Run Config: set with MLflow logging and checkpointing.")

    # Run Ray Tune: 
    # - Random Search Algorithm
    tuner_Random = tune.Tuner(
        trainable=trainable_xgboost,
        tune_config=tune_config,
        param_space=search_space,
        run_config=run_config
    )
    print("🔥🔥 Ray Tuner:  Initialized! 🔥🔥\n\n")

    # XGBoost: RAY FIT ===========================================================
    print("=" * 50)
    print(f"👾 XGBoost: RAY Fit {METRIC} 👾")
    print("=" * 50)

    results_Random = tuner_Random.fit()
    print("✅ Ray Tune hyperparameter optimization completed!\n\n")

    # RESULTS ===========================================================
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)

    best_result = results_Random.get_best_result(metric=METRIC, mode="max")
    print("📝 Best trial config:", best_result.config)
    # print("📝 Best trial metrics:", best_result.metrics)

    # Checkpoint
    best_result.best_checkpoints

    # EXPAND YOUR ANALYSIS:
    # Get all results for deeper analysis
    df_results = results_Random.get_dataframe()
    sorted_df = df_results.sort_values(by="roc_auc", ascending=False)
    print(f"\n📝 Total trials completed: {len(df_results)}")
    print(f"📝 Best ROC AUC achieved: {df_results['roc_auc'].max():.4f}")
    print(f"📝 Average ROC AUC: {df_results['roc_auc'].mean():.4f}")

    # Mlflow Logging 
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{METRIC} DESC"]
    )
    mlflow.log_params(best_result.config)
    # Filter out non-numeric values from the metrics dictionary
    numeric_metrics = {k: v for k, v in best_result.metrics.items() if isinstance(v, (int, float))}
    # Log only numeric metrics to MLflow
    mlflow.log_metrics(numeric_metrics)

    print("📝 Best trial logged to MLflow!")

    # SAVE BEST MODEL ===========================================================
    best_model, model_path, metadata_path = save_best_model(
        best_result, 
        X_train, 
        y_train, 
        X_val, 
        y_val,
        timestamp
    )