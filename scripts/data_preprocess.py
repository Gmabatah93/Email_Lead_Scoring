import pandas as pd
import sqlalchemy as sql
import janitor as jn
import re
import typer
from typing import Tuple, Dict, Any
from typing_extensions import Annotated
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = typer.Typer()

# LOAD AND PREPARE TAG DATA ===================================================
def merge_tags_with_leads(
    df_raw: pd.DataFrame,
    db_path: str = "data/crm_database.sqlite",
    output_path: str = "data/leads_raw.csv"
) -> pd.DataFrame:
    """
    Merge raw leads DataFrame with tag data from a SQLite database, transform tags to wide format,
    and save the merged DataFrame to a CSV file.

    Args:
        df_raw (pd.DataFrame): Raw leads DataFrame.
        db_path (str, optional): Path to the SQLite database. Defaults to "data/crm_database.sqlite".
        output_path (str, optional): Path to save the merged CSV. Defaults to "data/leads_raw.csv".

    Returns:
        pd.DataFrame: Merged DataFrame with tag columns.
    """
    typer.echo(typer.style("⚙️ Merging tags with leads...", fg=typer.colors.BRIGHT_YELLOW))
    with sql.create_engine(f"sqlite:///{db_path}").connect() as conn:
        tags_df = pd.read_sql("SELECT * FROM Tags", conn)
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
        typer.echo(typer.style(f"✅ Tags loaded from database: {db_path}", fg=typer.colors.GREEN))

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
    typer.echo(typer.style(f"✅ Merged leads and tags saved to {output_path}\n", fg=typer.colors.GREEN))
    return df_leads_raw

# PREPROCESSING FUNCTIONS =====================================================
def preprocess_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and cleaning steps to the leads DataFrame.

    Args:
        df (pd.DataFrame): Raw DataFrame with subscriber data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for machine learning.
    """
    typer.echo(typer.style("⚙️ Starting preprocessing...", fg=typer.colors.BRIGHT_YELLOW))
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
    typer.echo("📝 Date features Added.")

    # 2. EMAIL FEATURES
    if 'user_email' in df_processed.columns:
        df_processed['email_provider'] = df_processed['user_email'].str.split("@").str[1]

        # Email domain categories
        free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        df_processed['is_free_email'] = df_processed['email_provider'].isin(free_providers).astype(int)
    typer.echo("📝 Email features Added.")

    # 3. ACTIVITY FEATURES
    if 'tag_count' in df_processed.columns and 'optin_days' in df_processed.columns:
        df_processed['tag_count_by_optin_day'] = df_processed['tag_count'] / (abs(df_processed['optin_days']) + 1)
    typer.echo("📝 Activity features Added.")

    # 4. COUNTRY STANDARDIZATION
    if 'country_code' in df_processed.columns:
        countries_to_keep = [
            'us', 'in', 'au', 'uk', 'br', 'ca', 'de', 'fr', 'es', 'mx',
            'nl', 'sg', 'dk', 'pl', 'my', 'ae', 'co', 'id', 'ng', 'jp', 'be'
        ]
        df_processed['country_code'] = df_processed['country_code'].apply(
            lambda x: x if x in countries_to_keep else 'other'
        )
    typer.echo("📝 Country codes standardized.")

    # 5. CLEAN TAG COLUMNS
    tag_columns = [col for col in df_processed.columns if col.startswith('tag_')]
    for col in tag_columns:
        df_processed[col] = df_processed[col].fillna(0)
    typer.echo(f"📝 Tag columns cleaned: {len(tag_columns)} columns.")

    return df_processed

def preprocess_for_xgboost(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Prepare features and target for XGBoost, including label encoding and column removal.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
            - X: Features DataFrame.
            - y: Target Series.
            - label_encoders: Dictionary of fitted LabelEncoders for categorical columns.
    """

    df_processed = df.copy()

    # Remove unnecessary columns for XGBoost
    REMOVE_COLUMNS = ["mailchimp_id","user_full_name","user_email","optin_time","email_provider"]
    columns_to_remove = [col for col in REMOVE_COLUMNS if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_remove, axis=1)
    typer.echo(typer.style(f"❌ Removed columns: {columns_to_remove}", fg=typer.colors.BRIGHT_RED))

    # Categorical Features
    categorical_features = ['country_code']
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        typer.echo(f"📝 {col} encoded: {le.classes_}")

    # Check for missing values
    missing_values = df_processed.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()

    if len(missing_cols) > 0:
        typer.echo(f"Missing values found in columns: {missing_cols}\n")
    else:
        typer.echo("📝 No missing values found.\n")

    df_processed = df_processed.fillna(-999)

    # Separate features and target
    X = df_processed.drop('made_purchase', axis=1)
    y = df_processed['made_purchase']
    typer.echo(typer.style("🎯 Separated features and target variable", fg=typer.colors.BRIGHT_RED))

    typer.echo(f"    Features shape: {X.shape}")
    positive_pct = y.value_counts(normalize=True).get(1, 0) * 100
    typer.echo(f"    Target distribution (1) %: {positive_pct:.2f}\n")
    
    return X, y, label_encoders

def prepare_xgboost_data(
    data_path: str = "data/leads_cleaned.csv",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, LabelEncoder]]:
    """
    Complete data preparation pipeline for XGBoost, including loading, preprocessing,
    and splitting into train, validation, and test sets.

    Args:
        data_path (str, optional): Path to the cleaned leads CSV. Defaults to "data/leads_cleaned.csv".
        test_size (float, optional): Proportion of data for test set. Defaults to 0.2.
        val_size (float, optional): Proportion of data for validation set. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 123.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_val (pd.DataFrame): Validation features.
            - X_test (pd.DataFrame): Test features.
            - y_train (pd.Series): Training target.
            - y_val (pd.Series): Validation target.
            - y_test (pd.Series): Test target.
            - label_encoders (Dict[str, LabelEncoder]): Label encoders for categorical columns.
    """
    typer.echo(typer.style("⚙️ Preparing data for XGBoost...", fg=typer.colors.BRIGHT_YELLOW))

    # Load data
    df_leads = pd.read_csv(data_path)
    typer.echo(typer.style(f"✅ Data loaded from: {data_path}", fg=typer.colors.GREEN))
    typer.echo(f"📊 Data shape: {df_leads.shape}\n")

    # Apply XGBoost preprocessing
    X, y, label_encoders = preprocess_for_xgboost(df_leads)

    # Split data into training + temp (validation + test)
    typer.echo(typer.style("✂️ Splitting the data into training, validation, and test sets...", fg=typer.colors.BRIGHT_RED))
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

    typer.echo(f"📝 Training set: {X_train.shape}")
    typer.echo(f"📝 Validation set: {X_val.shape}")
    typer.echo(f"📝 Test set: {X_test.shape}")

    positive_pct = y_train.value_counts(normalize=True).get(1, 0) * 100
    typer.echo(f"🎯 Training target distribution (1) %: {positive_pct:.2f}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoders

@app.command()
def main(
    input_path: Annotated[Path, typer.Option(help="Path to the raw subscribers CSV file.")] = "data/subscribers_joined.csv",
    db_path: Annotated[str, typer.Option(help="Path to the CRM database.")] = "data/crm_database.sqlite",
    raw_output_path: Annotated[Path, typer.Option(help="Path to save the merged raw data CSV file.")] = "data/leads_raw.csv",
    cleaned_output_path: Annotated[Path, typer.Option(help="Path to save the final cleaned data CSV file.")] = "data/leads_cleaned.csv"
):
    """
    Complete data preparation pipeline from ingestion to a cleaned CSV.
    """
    typer.echo("=" * 50)
    typer.echo(typer.style("PREPROCESSING", fg=typer.colors.CYAN))
    typer.echo("=" * 50)

    # Load raw data
    df = pd.read_csv(input_path)
    typer.echo(typer.style(f"✅ Raw data loaded from: {input_path}", fg=typer.colors.GREEN))
    typer.echo(f"📊 Raw data shape: {df.shape}\n")

    # Merge tags with leads
    df = merge_tags_with_leads(df, db_path=db_path, output_path=raw_output_path)

    # Preprocess leads
    df_processed = preprocess_leads(df)
    df_processed.to_csv(cleaned_output_path, index=False)
    typer.echo(typer.style(f"\n✅ Preprocessing complete. Saved to {cleaned_output_path}", fg=typer.colors.GREEN))
    typer.echo(f"📊 Final processed data shape: {df_processed.shape}")


if __name__ == "__main__":
    app()