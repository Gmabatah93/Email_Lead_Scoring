# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 1: PYCARET INTEGRATION
# ----

# Core
import pandas as pd
import numpy as np
# Machine Learning
import pycaret.classification as clf
# MLFlow Models
import mlflow
# Lead Scoring
import email_lead_scoring as els

# RECAP 

leads_tags_df = els.db_read_and_process_els_data()

# 1.0 PYCARET'S MLFLOW INTEGRATION ----

# When we setup pycaret, we setup logging experiments with MLFlow

# Job & Experiment Logging: clf.setup()
    # n_jobs = -1,
    # session_id = 123,
    # log_experiment=True,
    # experiment_name = 'email_lead_scoring_1'


# 2.0 MLFLOW UI ----

# !mlflow ui
# http://localhost:5000/

# 1. GUI OVERVIEW: HOW TO FIND THE MODELS YOU'RE LOOKING FOR
# 2. SORTING: AUC 
# 3. SEARCHING: tags.Source = 'finalize_model'
# 4. DRILLING INTO MODELS


# 3.0 MLFLOW TRACKING & EXPERIMENT INTERFACE ----


# 3.1 TRACKING URI (FOLDER WHERE YOUR EXPERIMENTS & RUNS ARE STORED) ----

mlflow.get_tracking_uri()

# 3.2 WORKING WITH EXPERIMENTS (GROUP OF RUNS) ----

# Listing Experiments

mlflow.list_experiments()

mlflow.get_experiment("1")

mlflow.get_experiment_by_name('email_lead_scoring_0')

# Programmatically Working With Experiments

mlflow_experiments = mlflow.list_experiments()

experiment_1 = mlflow_experiments[1]

experiment_1.experiment_id
experiment_1.artifact_location
experiment_1.name

experiment_1.tags
experiment_1.lifecycle_stage

# 3.3 SEARCHING WITH THE EXPERIMENT NAME ----

logs_df = mlflow.search_runs(
    experiment_ids=experiment_1.experiment_id
)

best_run_id = logs_df \
    .query("`tags.Source` in ['finalize_model']") \
    .sort_values('metrics.AUC', ascending = False) \
    ['run_id'] \
    .values \
    [0]



# pycaret interface to get experiments

clf.get_logs(experiment_name='email_lead_scoring_0')



# 4.0 WORKING WITH RUNS ----

# Finding Runs in Experiments

best_run_id

mlflow.list_run_infos("1")[0]

# Finding Runs from Run ID

best_run_details = mlflow.get_run('26c14ff733ec4034a17b7448d52a14d3')

# 5.0 MAKING PREDICTIONS ----

# Using the mlflow model 

import mlflow
logged_model = 'runs:/21bdcd19c79e4b0b9049c75c63b57001/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(leads_tags_df)

# Issue - Mlflow does not give probability. We need probabilities for lead scoring.

# Solution 1 - Extract out the sklearn model and use the 
#   sklearn .predict_proba() method

loaded_model._model_impl

loaded_model._model_impl.predict_proba(leads_tags_df)[:,1]

# Solution 2 - Predict with Pycaret's prediction function in production

clf.predict_model(
    estimator = loaded_model._model_impl,
    data      = leads_tags_df,
    raw_score = True
)

clf.load_model(
    model_name=f'mlruns/1/{best_run_id}/artifacts/model/model'
)


# CONCLUSIONS ----
# 1. Pycaret simplifies the logging process for mlflow
# 2. Pycaret includes a lot of detail (saved metrics) by default
# 3. We haven't actually logged anything of our own 
# 4. Let's explore MLFlow deeper with H2O (We can actually use this process for Scikit Learn too!)
