import pandas as pd
import sqlalchemy as sql
import janitor as jn
import re 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# LOAD AND PREPARE TAG DATA ===================================================
def merge_tags_with_leads(df_raw, db_path="data/crm_database.sqlite", output_path="data/leads_raw.csv"):
    """
    Merge raw leads data with tag data from the database and save to CSV.
    
    Args:
        df_raw (pd.DataFrame): Raw leads dataframe.
        db_path (str): Path to the SQLite database.
        output_path (str): Path to save the merged CSV.
        
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    with sql.create_engine(f"sqlite:///{db_path}").connect() as conn:
        tags_df = pd.read_sql("SELECT * FROM Tags", conn)
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
        print(f"✅ Tags loaded from database: {db_path}")

    tags_wide_leads_df = tags_df \
        .assign(value=1) \
        .pivot(
            index='mailchimp_id',
            columns='tag',
            values='value'
        ) \
        .fillna(0) \
        .pipe(jn.clean_names)

    tags_wide_leads_df.columns = tags_wide_leads_df.columns \
        .to_series() \
        .apply(lambda x: f"tag_{x}") \
        .to_list()

    tags_wide_leads_df = tags_wide_leads_df.reset_index()
    print(f"📝 Tags data transformed to wide format with {tags_wide_leads_df.shape[1]} columns.")

    # Merge raw data with tag data
    df_leads_raw = df_raw.merge(tags_wide_leads_df, how='left')
    df_leads_raw.to_csv(output_path, index=False)
    print(f"📝 Merged leads and tags saved to {output_path}\n")
    return df_leads_raw

# PREPROCESSING FUNCTIONS =====================================================
def preprocess_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for email leads data.
    
    Args:
        df: Raw dataframe with subscriber data
        
    Returns:
        Preprocessed dataframe ready for ML
    """
    print("⚙️ Starting preprocessing...")
    df_processed = df.copy()
    
    # 1. DATE FEATURES
    if 'optin_time' in df_processed.columns:
        df_processed['optin_time'] = pd.to_datetime(df_processed['optin_time'])
        date_max = df_processed['optin_time'].max()
        
        # Temporal features
        df_processed['optin_days'] = (df_processed['optin_time'] - date_max).dt.days
        df_processed['optin_month'] = df_processed['optin_time'].dt.month
        df_processed['optin_day_of_week'] = df_processed['optin_time'].dt.dayofweek
        df_processed['optin_day_of_year'] = df_processed['optin_time'].dt.dayofyear
        df_processed['optin_quarter'] = df_processed['optin_time'].dt.quarter
        df_processed['optin_is_weekend'] = df_processed['optin_time'].dt.dayofweek.isin([5, 6]).astype(int)
    print("📝 Date features Added.")

    # 2. EMAIL FEATURES
    if 'user_email' in df_processed.columns:
        df_processed['email_provider'] = df_processed['user_email'].str.split("@").str[1]
        
        # Email domain categories
        free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        df_processed['is_free_email'] = df_processed['email_provider'].isin(free_providers).astype(int)
    print("📝 Email features Added.")

    # 3. ACTIVITY FEATURES
    if 'tag_count' in df_processed.columns and 'optin_days' in df_processed.columns:
        df_processed['tag_count_by_optin_day'] = df_processed['tag_count'] / (abs(df_processed['optin_days']) + 1)
    print("📝 Activity features Added.")

    # 4. COUNTRY STANDARDIZATION
    if 'country_code' in df_processed.columns:
        countries_to_keep = [
            'us', 'in', 'au', 'uk', 'br', 'ca', 'de', 'fr', 'es', 'mx',
            'nl', 'sg', 'dk', 'pl', 'my', 'ae', 'co', 'id', 'ng', 'jp', 'be'
        ]
        df_processed['country_code'] = df_processed['country_code'].apply(
            lambda x: x if x in countries_to_keep else 'other'
        )
    print("📝 Country codes standardized.")

    # 5. CLEAN TAG COLUMNS
    tag_columns = [col for col in df_processed.columns if col.startswith('tag_')]
    for col in tag_columns:
        df_processed[col] = df_processed[col].fillna(0)
    print(f"📝 Tag columns cleaned: {len(tag_columns)} columns.")

    return df_processed

def preprocess_for_xgboost(df):
    """XGBoost-specific preprocessing"""
    
    df_processed = df.copy()
    
    # Remove unnecessary columns for XGBoost
    REMOVE_COLUMNS = ["mailchimp_id","user_full_name","user_email","optin_time","email_provider"]
    columns_to_remove = [col for col in REMOVE_COLUMNS if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_remove, axis=1)
    
    # Categorical Features
    print("Encoding categorical features...")
    categorical_features = ['country_code']
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder() 
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"{col} encoded: {le.classes_}\n")

    # Check for missing values
    print("Checking for missing values...")
    missing_values = df_processed.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()

    if len(missing_cols) > 0:
        print(f"Missing values found in columns: {missing_cols}\n")
    else:
        print("No missing values found.\n")

    df_processed = df_processed.fillna(-999)

    # Separate features and target
    print("Separating features and target variable...")
    X = df_processed.drop('made_purchase', axis=1)
    y = df_processed['made_purchase']

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}\n")
    
    return X, y, label_encoders

def prepare_xgboost_data(data_path="data/leads_cleaned.csv", test_size=0.2, val_size=0.2, random_state=123):
    """Complete data preparation pipeline for XGBoost with separate test set"""
    
    print(10 * "=" + " PREPROCESS: XGBoost " + 10 * "=")
    
    # Load data
    df_leads = pd.read_csv(data_path)
    
    # Apply XGBoost preprocessing
    X, y, label_encoders = preprocess_for_xgboost(df_leads)
    
    print(f"Ready for XGBoost training with {X.shape[0]} samples and {X.shape[1]} features.\n")
    
    # Split data into training + temp (validation + test)
    print("Splitting the data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=test_size + val_size,  # Combine validation and test sizes
        random_state=random_state,
        stratify=y
    )
    
    # Further split temp into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=test_size / (test_size + val_size),  # Adjust proportion
        random_state=random_state,
        stratify=y_temp
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoders

if __name__ == "__main__":
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    # Load raw data
    df = pd.read_csv("data/subscribers_joined.csv")
    print("✅ Raw data loaded from: data/subscribers_joined.csv")

    # Merge tags with leads
    df = merge_tags_with_leads(df, db_path="data/crm_database.sqlite", output_path="data/leads_raw.csv")
    
    # Preprocess leads
    df_processed = preprocess_leads(df)
    df_processed.to_csv("data/leads_cleaned.csv", index=False)
    print("✅ Preprocessing complete. Saved to data/leads_cleaned.csv.")
