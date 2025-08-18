import pandas as pd
from pycaret.classification import ClassificationExperiment

df_leads = pd.read_csv("data/leads_cleaned.csv")
df_leads.info()

# 1. PREPROCESSING (Model) ========================================================

# Removing Unnecessary Columns
REMOVE_COLUMNS = ["mailchimp_id","user_full_name","user_email","optin_time","email_provider"]

df_prec = df_leads.drop(columns=REMOVE_COLUMNS, axis=1)
df_prec.info()

# Numeric Features
OPTIN = ['optin_days', 'optin_month', 'optin_day_of_week', 'optin_day_of_year', 'optin_quarter', 'optin_is_weekend']

_tag_mask = df_prec.columns.str.match('^tag_')
numeric_features = df_prec.columns[_tag_mask].to_list()
numeric_features.extend(OPTIN)

# Categorical Features
categorical_features = ['country_code']

# Ordinal Features
df_prec['member_rating'].unique()
ordinal_features = {
    'member_rating': ["1", "2", "3", "4", "5"]
}

# 2. CLASSIFIER SETUP ========================================================
s = ClassificationExperiment()
# - basic setup
help(s.setup)
s.setup(
    data=df_prec,
    target='made_purchase',
    ordinal_features=ordinal_features,
    log_experiment='mlflow',
    experiment_name='test_experiment',
    profile=True,
    session_id=123
)

s.models()
# Data Transformation
s.get_config("X_train")
s.get_config("X_train_transformed")

# 3. COMPARE MODELS ========================================================
best_f1 = s.compare_models(sort="F1", n_select=5)
best_recall = s.compare_models(sort="Recall", n_select=5)
best_auc = s.compare_models(sort="AUC", n_select=1)

# NOTE: Class Imbalance
# Recall: Sesitivity / True Positive Rate
# - The biggest business cost is a missed opportunity—failing to identify a customer who was ready to buy (a "false negative"). 
# - A high recall ensures you are finding the maximum number of potential buyers. 
# - You'd rather send a marketing email to a few uninterested people than miss out on a sale.
s.pull().sort_values(by='Recall', ascending=False)

# F1 Score:
# - While you want to find as many buyers as possible (high Recall), you also don't want to send emails to a massive list of people who will never buy (low Precision). 
# - The F1-Score is arguably the best single metric to use for imbalanced problems because it finds a sweet spot between identifying a high percentage of true buyers and not being too wasteful with your marketing efforts
s.pull().sort_values(by='F1', ascending=False)

# AUC: measures the model's overall ability to distinguish between the two classes (buyers vs. non-buyers)
# - A great "big picture" metric. 
# - It tells you how good the model is at separating your classes, independent of any specific probability threshold. 
# - It's excellent for comparing the general discriminative power of different algorithms.
s.pull().sort_values(by='AUC', ascending=False)

# Just want to compare tree models
TREE_MODELS = ['dt', 'rf', 'et', 'gbc', 'xgboost', 'lightgbm', 'catboost']
tree_models = s.compare_models(include=TREE_MODELS)

# Just GPU algorithms
GPU_MODELS = ['xgboost', 'lightgbm', 'catboost']
gpu_models = s.compare_models(include=GPU_MODELS)

# 4. TUNE MODEL ========================================================


# 5. ANALYZE MODEL ==================================================== 
s.evaluate_model(best_auc)
help(s.evaluate_model)

help(s.plot_model)
s.plot_model(best_f1, plot = 'auc')
s.plot_model(best_f1, plot = 'confusion_matrix')
s.plot_model(best_f1, plot = 'feature')

s.evaluate_model(tree_models)

help(s.interpret_model)
s.interpret_model(tree_models)
s.interpret_model(tree_models, plot = 'reason', observation = 1)
# 6. PREDICTIONS ========================================================
s.predict_model(best_f1, raw_score=True)
s.predict_model(best_recall, raw_score=True)
s.predict_model(best_auc, raw_score=True)

# 7. SAVE MODEL ================================
s.save_model(best_f1, 'models/basic_model_lda_f1')
s.save_model(best_recall, 'models/basic_model_nb_recall')
s.save_model(best_auc, 'models/basic_model_gbc_auc')