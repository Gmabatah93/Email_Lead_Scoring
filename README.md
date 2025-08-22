# File Structure

```
data/
├── crm_database.sqlite
├── data.py
docs/
├── els/
├──── preprocess.md
├──── RAY_model_evaluation.md
├──── RAY_model_xgboost.md
├── index.md
results/
├── evaluation/
├── training/
scripts/
├── data_ingestion.py
├── data_testing.py
├── preprocess.py
├── RAY_model_xgboost.py
├── RAY_model_evaluation.py
├── RAY_model_serve.py
eda.py
mkdocs.yml
README.md
SystemDesign.md
```

# 1. Get the Data 'data_ingestion.py' 
- Source SQL DB
- Pull Tables [Subscribers | Tags | Transaction]
- Clean (datatypes) & Merge
- Save 'subscribers_joined.csv' to local drive

