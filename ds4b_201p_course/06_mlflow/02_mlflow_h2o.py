# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 2: H2O AUTOML + MLFLOW
# ----

import pandas as pd
import numpy as np

import email_lead_scoring as els

import h2o
from h2o.automl import H2OAutoML

import mlflow
import mlflow.h2o as mlflow_h2o
import mlflow.sklearn as mlflow_sklearn

# Collect Data

leads_df = els.db_read_and_process_els_data()

# Initialize H2O

h2o.init()

# Convert to H2O Frame
leads_h2o = h2o.H2OFrame(leads_df)

leads_h2o['made_purchase'] = leads_h2o['made_purchase'].asfactor()

# Prep for ML
x_cols = [
    # 'mailchimp_id', 'user_full_name', 'user_email', 
    'member_rating', 
    # 'optin_time', 
    'country_code', 
    # 'made_purchase', 
    'tag_count', 'optin_days', 'email_provider', 'tag_count_by_optin_day', 'tag_aws_webinar', 'tag_learning_lab', 'tag_learning_lab_05', 'tag_learning_lab_09', 'tag_learning_lab_11', 'tag_learning_lab_12', 'tag_learning_lab_13', 'tag_learning_lab_14', 'tag_learning_lab_15', 'tag_learning_lab_16', 'tag_learning_lab_17', 'tag_learning_lab_18', 'tag_learning_lab_19', 'tag_learning_lab_20', 'tag_learning_lab_21', 'tag_learning_lab_22', 'tag_learning_lab_23', 'tag_learning_lab_24', 'tag_learning_lab_25',
    'tag_learning_lab_26', 'tag_learning_lab_27', 'tag_learning_lab_28', 'tag_learning_lab_29', 'tag_learning_lab_30', 'tag_learning_lab_31', 'tag_learning_lab_32', 'tag_learning_lab_33', 'tag_learning_lab_34', 'tag_learning_lab_35', 'tag_learning_lab_36', 'tag_learning_lab_37', 'tag_learning_lab_38', 'tag_learning_lab_39', 'tag_learning_lab_40', 'tag_learning_lab_41', 'tag_learning_lab_42', 'tag_learning_lab_43', 'tag_learning_lab_44', 'tag_learning_lab_45', 'tag_learning_lab_46', 'tag_learning_lab_47', 'tag_time_series_webinar', 'tag_webinar', 'tag_webinar_01', 'tag_webinar_no_degree', 'tag_webinar_no_degree_02'
]

y_col = 'made_purchase'


# 1.0 MLFLOW + H2O SETUP

# Initialize H2O

# Create an mlflow client

EXPERIMENT_NAME = 'automl_lead_scoring_1'

client = mlflow.MlflowClient()

# Create an mlflow experiment (if not already present)
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except:
    print(f"Experiment Name: {EXPERIMENT_NAME} already exists.")
    
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

experiment

experiment.experiment_id

mlflow.set_experiment(experiment_name=experiment.name)

# 2.0 MLFLOW UI
# !mlflow ui



# 3.0 MLFLOW + H2O AUTOML INTEGRATION ----

# Start an mlflow run (an experiment must be created):

mlflow.start_run()
    
# Run H2O AutoML
aml = H2OAutoML(
    max_runtime_secs = 45,
    exclude_algos    = ['DeepLearning'],
    nfolds           = 5,
    seed             = 123
)

aml.train(
    x = x_cols,
    y = y_col,
    training_frame=leads_h2o
)

aml.leaderboard

# Mlflow H2O Integration: Used mlflow_h2o.log_model()

mlflow_h2o.log_model(
    h2o_model=aml.leader,
    artifact_path="model"
)

# Scikit Learn Integration: Use mlflow_sklearn.log_model()


# Log Metrics
aml.leader.logloss()
aml.leader.auc()

mlflow.log_metric("log_loss", aml.leader.logloss())
mlflow.log_metric("AUC", aml.leader.auc())

# Set a tag

mlflow.set_tag("Source", "h2o_automl_model")

active_run_id = mlflow.active_run().info.run_id

mlflow.set_tag("Run ID", active_run_id)

# Print Model URI (location)
model_uri = mlflow.get_artifact_uri("model")
print(model_uri)


# Print and view AutoML Leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
print(lb.head(rows = lb.nrows))

# Save leaderboard as CSV Artifact
experiment_id = experiment.experiment_id
run_id = mlflow.active_run().info.run_id
try:
    lb_path = f'mlruns/{experiment_id}/{run_id}/artifacts/model/leaderboard.csv'
    lb.as_data_frame().to_csv(lb_path, index = False)
    print(f'Leaderboard saved in location: {lb_path}')
except:
    print("Could not save leaderboard as a CSV File.")
    
# END THE MLFLOW RUN

mlflow.end_run()


# PREDICTIONS ---- 
# - Copy from MLFlow UI Artifacts 

import mlflow
logged_model = 'runs:/556c111fb66146f0870dad0581c8664e/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(leads_df)['p1']

loaded_model._model_impl

# CONCLUSIONS ----
# 1. We have learned a framework for creating our own MLFlow Experiments, Runs, and storing models like H2O and Scikit Learn
# 2. Takes more setup than the Pycaret Integration, but can be used in the same way to track experiments and runs

