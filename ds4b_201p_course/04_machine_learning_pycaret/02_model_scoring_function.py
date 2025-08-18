
# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | MODEL LEAD SCORING FUNCTION
# ----

import numpy as np
import pandas as pd
import pycaret.classification as clf
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

# MODEL LOAD FUNCTION ----

def model_score_leads(
    data, 
    model_path = "models/blended_models_final"
):
    
    mod = clf.load_model(model_path)
    
    predictions_df = clf.predict_model(
        estimator=mod,
        data = data
    )
    
    # FIX ----
    
    # leads_scored_df = pd.concat(
    #     [1-predictions_df['Score'], data], 
    #     axis=1
    # )
    
    df = predictions_df
    
    predictions_df['Score'] = np.where(df['Label'] == 0, 1 - df['Score'], df['Score'])
    
    predictions_df['Score']
    
    leads_scored_df = pd.concat(
        [predictions_df['Score'], data], 
        axis=1
    )
    
    # END FIX ----
    
    return leads_scored_df

model_score_leads(leads_df)

# TEST OUT

import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()
leads_df

els.model_score_leads(
    data = leads_df,
    model_path="models/xgb_model_tuned"
)

