# Email Lead Scoring System Design

## Data Ingest
- **Source Systems**: SQLite CRM database (`data/crm_database.sqlite`)
  - Subscribers table: User profiles, ratings, opt-in data
  - Tags table: User engagement/behavior tags
  - Transactions table: Purchase history
- **Data Pipeline**: `1_data.py` - SQLAlchemy connections, data extraction
- **Output**: `data/subscribers_joined.csv` - Consolidated dataset
- **Real-world equivalent**: Mailchimp API, CRM systems (Salesforce, HubSpot)

## Data Testing
- **Testing Framework**: Great Expectations + custom business rule validation
- **Test Script**: `1.2_data_testing.py` - Validates data quality post-ingestion
- **Schema Validation**:
  - Column existence and order verification
  - Data type consistency (int64, string formats)
  - Primary key uniqueness (mailchimp_id, user_email)
- **Data Quality Checks**:
  - Null value validation for critical fields
  - Email format validation (regex pattern matching)
  - Member rating range validation (1-5)
  - Tag count non-negative validation
- **Business Rule Testing**:
  - Conversion rate sanity checks (1-50% range)
  - Country code format validation (2-character ISO codes)
  - Email domain distribution analysis
  - Target variable binary validation (0/1)
- **Output**: `data/validation_results.json` - Detailed test results
- **Quality Gate**: Prevents invalid data from entering ML pipeline

## Data Analysis
- **EDA Script**: `2_eda.py` - Exploratory data analysis to understand data patterns
- **High Cardinality Analysis**:
  - Country code performance analysis (sales & conversion rates by region)
  - Pareto analysis to identify top-performing segments
  - Cumulative proportion analysis for feature selection
- **Ordinal Feature Analysis**:
  - Member rating impact on conversion rates
  - Rating distribution and purchase correlation
- **Interaction Effects**:
  - Tag count distribution by purchase behavior
  - Quantile analysis (10th, 50th, 90th percentiles)
  - Feature interaction exploration
- **Key Insights Discovery**:
  - Data sufficiency validation for ML task
  - Distribution shift detection capabilities
  - Anomaly identification and outlier analysis
- **Iterative Process**: Revisited throughout development for distribution monitoring
- **Output**: Statistical summaries and feature relationship insights


## Data Preprocessing
- **Feature Engineering Scripts**: `3_preprocessing.py`, `4_pycaret.py`, `5_ray_xgboost.py`
- **Data Cleaning**:
  - Remove unnecessary columns (mailchimp_id, user_full_name, user_email, optin_time, email_provider)
  - Missing value handling (fillna with -999 for XGBoost compatibility)
  - Data type validation and conversion
- **Feature Engineering**:
  - **Temporal Features**: optin_days, optin_month, optin_day_of_week, optin_day_of_year, optin_quarter, optin_is_weekend
  - **Engagement Features**: tag_count, individual tag indicators (tag_learning_lab_*, tag_webinar_*)
  - **Categorical Encoding**: Label encoding for country_code (XGBoost), ordinal encoding for member_rating
- **Feature Categories**:
  - **Numeric Features**: Tag engagement metrics and temporal opt-in features
  - **Categorical Features**: country_code (high cardinality geographic data)
  - **Ordinal Features**: member_rating (1-5 scale with meaningful order)
- **Framework-Specific Processing**:
  - **PyCaret**: Automated preprocessing with ordinal feature specification
  - **XGBoost**: Manual label encoding and missing value imputation
  - **Ray Tune**: Streamlined preprocessing for hyperparameter optimization
- **Train/Test Split**: 80/20 stratified split maintaining target class distribution
- **Output**: Clean feature matrix ready for model training across multiple frameworks

## Model Training 
- **Hyperparameter Optimization:** Random Search & Bayesian


## Model Tuning

## Model Evaluation
### Offline
- **Holdout**

### Online

## Model Testing 