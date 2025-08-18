import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import xgboost as xgb
import time

import ray
from ray import tune
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from ray.util.metrics import Counter, Histogram, Gauge

# Import preprocessing functions from your preprocess file
from preprocess import prepare_xgboost_data

# PREPROCESS ============================================================
print(10 * "=" + " PREPROCESS " + 10 * "=")

# Use the preprocessing function from preprocess.py
X_train, X_test, y_train, y_test, label_encoders = prepare_xgboost_data()

print("Preprocessing completed using imported function!")
print(f"Ready for XGBoost training with {X_train.shape[0]} training samples.")

# XGBoost: Function ===================================================

# Custom metrics
trial_counter = Counter("xgboost_trials_total", description="Total XGBoost trials completed")
training_time_histogram = Histogram(
    "xgboost_training_duration_seconds", 
    description="XGBoost training time",
    boundaries=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]  # Time buckets in seconds
)
roc_auc_gauge = Gauge("xgboost_best_roc_auc", description="Best ROC AUC achieved")

def trainable_xgboost(config):
    """Train XGBoost model with Ray Tune and Prometheus metrics"""
    start_time = time.time()
    
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
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Record Prometheus metrics
    trial_counter.inc()
    training_time_histogram.observe(time.time() - start_time)
    roc_auc_gauge.set(roc_auc)

    # Report to Ray
    tune.report({"f1": f1, "recall": recall, "roc_auc": roc_auc})

# RAY: Tune ===========================================================
print(10 * "=" + " RAY TUNE " + 10 * "=")

# Initialize Ray
ray.init(
    num_cpus=6,
    object_store_memory=2_000_000_000,
    log_to_driver=False,
    include_dashboard=True,
    _metrics_export_port=8080  # Enable Prometheus metrics
)
print(f"Ray initialized with resources: {ray.cluster_resources()}")

# RAY: Search Space
search_space = {
    'max_depth': tune.randint(3, 8),
    'learning_rate': tune.uniform(0.01, 0.3),
    'n_estimators': tune.randint(50, 200),  # Smaller range for faster trials
    'subsample': tune.uniform(0.7, 1.0),
    'colsample_bytree': tune.uniform(0.7, 1.0)
}
print("Ray Tune setup complete!")
print(f"Search space: {search_space}\n")

# Run Ray Tune: 

# - Random Search Algorithm
print("Starting Ray Tune hyperparameter optimization...")
tuner_Random = tune.Tuner(
    trainable=trainable_xgboost,
    tune_config=tune.TuneConfig(
        metric="roc_auc",  # Primary metric to optimize
        mode="max",        # Maximize ROC AUC
        num_samples=24,     # Number of trials to run
        max_concurrent_trials=6,
        search_alg=BasicVariantGenerator(random_state=42)
    ),
    param_space=search_space,
    run_config=tune.RunConfig(
        name='RAY_xgboost_els_auc_random',
        storage_path='/Users/isiomamabatah/Desktop/Python/Projects/Email_Lead_Scoring/results'
    )
)

# - Bayesian Optimization with Optuna
tuner_Optuna = tune.Tuner(
    trainable=trainable_xgboost,
    tune_config=tune.TuneConfig(
        metric="roc_auc",
        mode="max",
        num_samples=24,
        max_concurrent_trials=6,
        search_alg=OptunaSearch(seed=42)  # Bayesian optimization
    ),
    param_space=search_space,
    run_config=tune.RunConfig(
        name='RAY_xgboost_auc_els_optuna',
        storage_path='/Users/isiomamabatah/Desktop/Python/Projects/Email_Lead_Scoring/results'
    )
)

# Execute the tuning
results_Random = tuner_Random.fit()
results_Optuna = tuner_Optuna.fit()

# RESULTS ===========================================================
print(10 * "=" + " RESULTS " + 10 * "=")

best_result = results_Optuna.get_best_result(metric="roc_auc", mode="max")
print("Best trial config:", best_result.config)
print("Best trial metrics:", best_result.metrics)

# EXPAND YOUR ANALYSIS:
# Get all results for deeper analysis
df_results = results_Optuna.get_dataframe()
print(f"\nTotal trials completed: {len(df_results)}")
print(f"Best ROC AUC achieved: {df_results['roc_auc'].max():.4f}")
print(f"Average ROC AUC: {df_results['roc_auc'].mean():.4f}")


# SAVE BEST MODEL ===========================================================
import os
import joblib
import json
from datetime import datetime

def save_best_model(best_result, X_train, y_train, X_test, y_test):
    """Recreate and save the best model with metadata"""
    
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
    
    # Retrain on full training set
    best_model.fit(X_train, y_train)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f"models/xgboost_ray_best_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": "XGBoost_Ray_Tune",
        "best_config": best_result.config,
        "best_metrics": best_result.metrics,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "search_algorithm": "Optuna"
    }
    
    metadata_path = f"models/metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Best model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    
    return best_model, model_path, metadata_path

best_model, model_path, metadata_path = save_best_model(
    best_result, X_train, y_train, X_test, y_test
)