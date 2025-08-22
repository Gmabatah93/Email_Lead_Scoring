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