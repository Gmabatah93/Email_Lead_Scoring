# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | PYCARET
# ----

# Core
import os
import pandas as pd
import numpy as np
import pycaret.classification as clf

# Lead Scoring
import email_lead_scoring as els

# RECAP ----

leads_df = els.db_read_and_process_els_data() 



# 1.0 PREPROCESSING (SETUP) ---- 
# - Infers data types (requires user to say yes)
# - Returns a preprocessing pipeline

# ?clf.setup

leads_df.info()

# Removing Unnecessary Columns

df = leads_df \
    .drop(
        ["mailchimp_id", "user_full_name", "user_email", "optin_time", "email_provider"], 
        axis = 1
    )
    
df.info()

# Numeric Features
_tag_mask = df.columns.str.match('^tag_')

numeric_features = df.columns[_tag_mask].to_list()

numeric_features.append('optin_days')

numeric_features

# Categorical Features 

categorical_features = ['country_code']

# Ordinal Features

df['member_rating'].unique()

ordinal_features = {
    'member_rating': ["1", "2", "3", "4", "5"]
}

# Classifier Setup

clf_1 = clf.setup(
    data       = df,
    target     = "made_purchase",
    train_size = 0.8,
    preprocess  = True,
    
    # Imputation
    imputation_type='simple',
    
    # Categorical 
    categorical_features = categorical_features,
    handle_unknown_categorical=True,
    combine_rare_levels = True,
    rare_level_threshold= 0.005,
    
    # Ordinal Features
    ordinal_features = ordinal_features,
    
    # Numeric Features
    numeric_features = numeric_features,
    
    # K-Fold
    fold_strategy='stratifiedkfold',
    fold = 5,
    
    # Job Experiment Logging
    n_jobs = -1,
    session_id = 123,
    log_experiment=True,
    experiment_name = 'email_lead_scoring_0',
    
    # Silent: Turns off asking for data types infered correctly
    silent = True,
    
)

clf_1

# 2.0 GET CONFIGURATION ----
# - Understanding what Pycaret is doing underneath
# - Can extract pre/post transformed data
# - Get the Scikit Learn Pipeline

# ?clf.get_config

# Transformed Dataset

clf.get_config("data_before_preprocess")

clf.get_config("X").head(3)

# Extract Scikit learn Pipeline

pipeline = clf.get_config("prep_pipe")


# Check difference in columns

pipeline.fit_transform(df)


# 3.0 MACHINE LEARNING (COMPARE MODELS) ----

# Available Models

clf.models()


# Running All Available Models

best_models = clf.compare_models(
    sort = "AUC",
    n_select = 3,
    budget_time = 2,
)

# Get the grid

clf.pull()


# Top 3 Models

best_models

len(best_models)

best_models[0]

best_models[1]

best_models[2]

# Make predictions

clf.predict_model(best_models[0])

clf.predict_model(
    estimator=best_models[1],
    data = df.iloc[[0]]
)


# Refits on Full Dataset

best_model_0_finalized = clf.finalize_model(best_models[0])

# Save / load model

os.mkdir("models")

clf.save_model(
    model = best_model_0_finalized,
    model_name="models/best_model_0"
)

clf.load_model("models/best_model_0")

# 4.0 PLOTTING MODEL PERFORMANCE -----

# Get all plots 
# - Note that this can take a long time for certain plots
# - May want to just plot individual (see that next)

clf.evaluate_model(best_model_0_finalized)

# - ROC Curves & PR Curves
clf.plot_model(best_models[0], plot="auc")

clf.plot_model(best_models[0], plot = "pr")

# Confusion Matrix / Error

clf.plot_model(
    best_models[1], 
    plot = "confusion_matrix",
    plot_kwargs={'percent': True}
)


# Gain/Lift Plots

clf.plot_model(best_models[1], plot = "gain")

clf.plot_model(best_models[1], plot = "lift")


# Feature Importance

clf.plot_model(best_models[1], plot = "feature")

clf.plot_model(best_models[0], plot = "feature_all")


# Shows the Precision/Recall/F1

clf.plot_model(best_models[1], plot = "class_report")


# Get model parameters used

clf.plot_model(best_models[1], plot = "parameter")



# 5.0 CREATING & TUNING INDIVIDUAL MODELS ----

clf.models()

# Create more models

xgb_model = clf.create_model(
    estimator="xgboost"
)


# Tuning Models

xgb_model_tuned = clf.tune_model(
    estimator=xgb_model, 
    n_iter=5,
    optimize="AUC"
)

# Save xgb tuned

xgb_model_tuned_finalized = clf.finalize_model(xgb_model_tuned)

clf.save_model(
    model = xgb_model_tuned_finalized,
    model_name="models/xgb_model_tuned"
)

clf.load_model("models/xgb_model_tuned")


# 6.0 INTERPRETING MODELS ----
# - SHAP Package Integration

# ?clf.interpret_model


# 1. Summary Plot: Overall top features

clf.interpret_model(best_models[1], plot="summary")

# 2. Analyze Specific Features ----

# Our Exploratory Function
els.explore_sales_by_category(
    leads_df, 
    'member_rating', 
    sort_by='prop_in_group'
)

# Correlation Plot

clf.interpret_model(
    best_models[1],
    plot = "correlation",
    feature = "member_rating"
)

clf.interpret_model(
    best_models[1],
    plot = "correlation",
    feature = "optin_days"
)



# Partial Dependence Plot

clf.interpret_model(
    best_models[1],
    plot = "pdp",
    feature="member_rating",
    ice = True
)


# 3. Analyze Individual Observations
leads_df.iloc[[0]]

# Shap Force Plot

clf.interpret_model(
    best_models[1],
    plot= "reason",
    X_new_sample=leads_df.iloc[[1]]
)

clf.predict_model(
    best_models[1],
    leads_df.iloc[[1]]
)


# 7.0 BLENDING MODELS (ENSEMBLES) -----

blended_models = clf.blend_models(
    best_models,
    optimize="AUC"
)

# 8.0 CALIBRATION ----
# - Improves the probability scoring (makes the probability realistic)

blended_models_calibrated = clf.calibrate_model(blended_models)


clf.plot_model(blended_models, plot = "calibration")

clf.plot_model(blended_models_calibrated, plot = "calibration")

# 9.0 FINALIZE MODEL ----
# - Equivalent of refitting on full dataset

blended_models_final = clf.finalize_model(blended_models_calibrated)



# 10.0 MAKING PREDICTIONS & RANKING LEADS ----

# Prediction

predictions_df = clf.predict_model(
    estimator = blended_models_final,
    data = leads_df
)

predictions_df.query("Label == 1")

# Scoring

leads_scored_df = pd.concat([1-predictions_df['Score'], leads_df], axis=1)

leads_scored_df.sort_values('Score', ascending=False)


# SAVING / LOADING PRODUCTION MODELS -----

clf.save_model(
    model = blended_models_final,
    model_name="models/blended_models_final"
)

clf.load_model("models/blended_models_final")

# CONCLUSIONS ----

# - We now have an email lead scoring model
# - Pycaret simplifies the process of building, selecting, improving machine learning models
# - Scikit Learn would take 1000's of lines of code to do all of this


