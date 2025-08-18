# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 5: ADVANCED MACHINE LEARNING 
# PART 2: H2O AUTOML
# ----

import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

import email_lead_scoring as els

# Collect Data

leads_df = els.db_read_and_process_els_data()


# 1.0 H2O PREPARATION

# Initialize H2O

h2o.init(
    max_mem_size = 4
)

# Convert to H2O Frame

leads_h2o = h2o.H2OFrame(leads_df)

leads_h2o.describe()

leads_h2o['made_purchase'] = leads_h2o['made_purchase'].asfactor()

leads_h2o.describe()

# Prep for AutoML

print(leads_h2o.columns)

x_cols = [
    # 'mailchimp_id', 'user_full_name', 'user_email', 
    'member_rating', 
    # 'optin_time', 
    'country_code', 'tag_count', 
    # 'made_purchase', 
    'optin_days', 'email_provider', 'tag_count_by_optin_day', 'tag_aws_webinar', 'tag_learning_lab', 'tag_learning_lab_05', 'tag_learning_lab_09', 'tag_learning_lab_11', 'tag_learning_lab_12', 'tag_learning_lab_13', 'tag_learning_lab_14', 'tag_learning_lab_15', 'tag_learning_lab_16', 'tag_learning_lab_17', 'tag_learning_lab_18', 'tag_learning_lab_19', 'tag_learning_lab_20', 'tag_learning_lab_21', 'tag_learning_lab_22', 'tag_learning_lab_23', 'tag_learning_lab_24', 'tag_learning_lab_25', 'tag_learning_lab_26', 'tag_learning_lab_27', 'tag_learning_lab_28', 'tag_learning_lab_29', 'tag_learning_lab_30', 'tag_learning_lab_31', 'tag_learning_lab_32', 'tag_learning_lab_33', 'tag_learning_lab_34', 'tag_learning_lab_35', 'tag_learning_lab_36', 'tag_learning_lab_37', 'tag_learning_lab_38', 'tag_learning_lab_39', 'tag_learning_lab_40', 'tag_learning_lab_41', 'tag_learning_lab_42', 'tag_learning_lab_43', 'tag_learning_lab_44', 'tag_learning_lab_45', 'tag_learning_lab_46', 'tag_learning_lab_47', 'tag_time_series_webinar', 'tag_webinar', 'tag_webinar_01', 'tag_webinar_no_degree', 'tag_webinar_no_degree_02'
]

y_col = 'made_purchase'


# 2.0 RUN H2O AUTOML ----

# H2OAutoML

aml = H2OAutoML(
    nfolds           = 5,
    exclude_algos    = ['DeepLearning'],
    max_runtime_secs = 3*60,
    seed             = 123
)

aml.train(
    x = x_cols,
    y = y_col,
    training_frame=leads_h2o
)

aml.leaderboard

# Save / load the model

model_h2o_stacked_ensemble = h2o.get_model(
    model_id='StackedEnsemble_AllModels_3_AutoML_1_20230601_164916'
)

h2o.save_model(
    model_h2o_stacked_ensemble,
    path = "models",
    filename="h2o_stacked_ensemble",
    force = True
)

h2o.load_model("models/h2o_stacked_ensemble")

h2o.__version__

# Predictions

predictions_h2o = model_h2o_stacked_ensemble.predict(leads_h2o)

predictions_df = predictions_h2o.as_data_frame()

pd.concat([predictions_df, leads_df], axis=1) \
    .sort_values('p1', ascending=False)

# CONCLUSIONS ----
# 1. H2O AutoML handles A LOT of stuff for you (preprocessing)
# 2. H2O is highly scalable
# 3. (CON) H2O depends on Java, which adds another complexity when you take your model into production
