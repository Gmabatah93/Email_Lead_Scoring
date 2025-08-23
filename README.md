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

✅ Output Saved to `data/subscribers_joined.csv` on local drive

# 2. Test the Data 'data_testing.py'

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

✅ Ouput Saved to `results/data_quality/processed_validation_results.json` on local drive

# 🖥️ Running from the CLI

## 1. Data Ingestion (`data_ingestion.py`)

### Basic Usage
```bash
python scripts/data_ingestion.py
```

### Advanced Options
```bash
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
