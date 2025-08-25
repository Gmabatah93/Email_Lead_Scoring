# Introduction
This project develops a machine learning model to score email leads based on their engagement with a Customer Relationship Management (CRM) system. The goal is to identify high-potential leads by leveraging data on subscriber profiles, tags, and transaction history. The entire workflow is built as a data pipeline, with each step handled by a dedicated Python script and a user-friendly Command-Line Interface (CLI).


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
├── data_quality/
├── evaluation/
├── training/
scripts/
├── data_ingestion.py
├── data_testing.py
├── data_preprocess.py
├── RAY_model_xgboost.py
├── RAY_model_evaluation.py
├── RAY_model_serve.py
eda.py
mkdocs.yml
README.md
SystemDesign.md
```

# 1. Get the Data `data_ingestion.py`
- Source CRM (SQL DB)
- Pull Tables [Subscribers | Tags | Transaction]
- Clean (datatypes) & Merge

✅ Output Saved to `data/subscribers_joined.csv` on local drive

# 2. Test the Data `data_testing.py`

## CRM Source
Runs a suite of data quality tests on the raw, source data tables from the CRM database (Subscribers, Tags, and Transactions).

Checks Include: 
- _Row Count_
- _Column Count_
- _Column Order_
- _Primary Keys_
- _Formating_

✅ Ouput Saved to `results/data_quality/crm_validation_results.json` on local drive

## Processed Data
Validates the integrity of the processed `subscribers_joined.csv` file, checking for schema compliance and data types.

Checks Include:
- _Row Count_
- _Column Count_
- _Column Order_
- _Formating_

✅ Ouput Saved to `results/data_quality/subscribers_joined_validation_results.json` on local drive

# 3. Preprocess Data `data_preprocess.py`
- Takes the `subscribers_joined.csv` & Tags from the database to create a new feature -> Output: `leads_raw.csv`
- Additional feature engineering and cleaning is done 
    - [ Date | Email | Activity ]
- Also some final cleaning 
- Their is also some functions to prepare the data for xgboost

✅ Output Saved to `data/leads_raw.csv` & `data/leads_cleaned.csv` on local drive

# 4. 👾 Model Training: `RAY_model_xgboost.py`
We use Ray Tune to perform hyperparameter optimization for an XGBoost model. It automates the process of finding the best-performing model configuration and logs all results to MLflow for tracking and reproducibility.

## Key Features
- **Hyperparameter Search:** Uses Ray Tune's BasicVariantGenerator to perform a random search over a defined hyperparameter space for the XGBoost classifier. This is more efficient than a manual grid search.  
- **Distributed Training:** The script parallelizes the training of different hyperparameter configurations across multiple CPU cores using Ray. This significantly reduces the time required for optimization.
- **Experiment Tracking:** Integrates with MLflow via the MLflowLoggerCallback to automatically log the parameters, metrics (F1-score, Recall, ROC AUC), and artifacts for each trial.
- **Final Model Saving:** After the search is complete, the script retrieves the best-performing model and retrains it on the combined training and validation datasets. The final model and its metadata are then saved for future use.

# 🖥️ Running from the CLI

## 1. Data Ingestion (`data_ingestion.py`)

### Basic Usage
```bash
python scripts/data_ingestion.py

# Get help and see all available options
python scripts/data_ingestion.py --help

# Enable verbose output (shows detailed statistics)
python scripts/data_ingestion.py --verbose

# Custom output path
python scripts/data_ingestion.py --output-path results/my_data.csv

# Combine multiple options
python scripts/data_ingestion.py --verbose --output-path results/detailed_data.csv
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-path` | Path | `data/subscribers_joined.csv` | Path to save the processed CSV file |
| `--verbose` | Flag | `False` | Enable detailed output and statistics |

### Output
- Creates `subscribers_joined.csv` with merged subscriber, tag, and purchase data
- Shows purchase rate statistics
- With `--verbose`: displays detailed data shapes, columns, and previews

## 2. Data Testing (`data_testing.py')

### Usage
```bash
# Run only raw CRM data quality tests
python scripts/data_testing.py crm

# Run only Processed "JOINED" data quality tests
python scripts/data_testing.py joined

# Run all data tests (raw CRM data, processed joined data, and business rules)
python scripts/data_testing.py run-all-tests
```

### Parameters
The both **crm** & **joined** command has the options:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| --results-path| Path | results/data_quality/{NAME}.json | Path to save the validation results as JSON |

## 3. Preprocess (`data_preprocess.py`)

### Usage
```bash
# Get help and see all available options
python scripts/preprocess.py --help

# Run the complete preprocessing pipeline
python scripts/preprocess.py

# Custom input and output paths
python scripts/preprocess.py --input-path data/subscribers_joined.csv --raw-output-path data/leads_raw.csv --cleaned-output-path data/leads_cleaned.csv
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input-path` | Path | `data/subscribers_joined.csv` | Path to save the processed CSV file |
| `--raw-output-path` | Path | `data/leads_raw.csv` | Path to save the merged raw data CSV file |
| `--clean-output-path` | Path | `data/leads_cleaned.csv` | Path to save the final cleaned data CSV file |

## 4. RAY XGBoost (`RAY_model_xgboost.py`)

### Usage
```bash
# Get help and see all available options
python RAY_model_xgboost.py --help

# Run the complete model training pipeline with default parameters
python RAY_model_xgboost.py main

# Customize the number of trials and workers
python RAY_model_xgboost.py main --num-trials 40 --num-workers 8

# Specify a different MLflow experiment name and metric
python RAY_model_xgboost.py main --experiment-name my_project_ray --metric recall
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input-path` | Path | `data/leads_raw.csv` | Path to the raw leads CSV file. |
| `--mlflow-uri` | str | `file:./mlruns` | MLflow tracking URI. |
| `--experiment-name` | str | `xgboost_ray_experiment` | Name of the MLflow experiment. |
| `--metric` | str | `roc_auc` | Metric to optimize for Ray Tune. |
| `--num-trials` | int | `24` | Number of Ray Tune trials. |
| `--num-workers` | int | `6` | Number of concurrent workers for Ray Tune. |