
import pandas as pd
import sqlalchemy as sql
import re
import janitor as jn

# Read & Combine Raw Data

def db_read_els_data(conn_string = "sqlite:///00_database/crm_database.sqlite"):
    """Function to read in the Subscribers, Tags, and Transactions tables and 
    combine them into a DataFrame with tag_count and made_purchase columns

    Args:
        conn_string (str, optional): Defaults to "sqlite:///00_database/crm_database.sqlite".

    Returns:
        _type_: Pandas DataFrame
    """
    
    # Connect to engine
    engine = sql.create_engine(conn_string)
    
    # Raw Data Collect
    with engine.connect() as conn:
        
        # Subscribers        
        subscribers_df = pd.read_sql("SELECT * FROM Subscribers", conn)
        
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')

        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')

        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64')

        # Tags
        tags_df = pd.read_sql("SELECT * FROM Tags", conn)
        
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
        
        # Transactions
        transactions_df = pd.read_sql("SELECT * FROM Transactions", conn)
        
        transactions_df['purchased_at'] = transactions_df['purchased_at'].astype('datetime64')

        transactions_df['product_id'] = transactions_df['product_id'].astype('int')
        
    # MERGE TAG COUNTS
    
    user_events_df = tags_df \
        .groupby('mailchimp_id') \
        .agg(dict(tag = 'count')) \
        .set_axis(['tag_count'], axis=1) \
        .reset_index()
    
    subscribers_joined_df = subscribers_df \
        .merge(user_events_df, how='left') \
        .fillna(dict(tag_count = 0))
        
    subscribers_joined_df['tag_count'] = subscribers_joined_df['tag_count'].astype('int')
    
    # MERGE TARGET VARIABLE
    emails_made_purchase = transactions_df['user_email'].unique()
    
    subscribers_joined_df['made_purchase'] = subscribers_joined_df['user_email'] \
        .isin(emails_made_purchase) \
        .astype('int')
    
        
    return subscribers_joined_df

# Read Table Names
def db_read_els_table_names(conn_string = "sqlite:///00_database/crm_database.sqlite"):
    """Reads the Table Names for each table in the crm database.

    Args:
        conn_string (str, optional): Defaults to "sqlite:///00_database/crm_database.sqlite".

    Returns:
        _type_: List with table names
    """
    
    engine = sql.create_engine(conn_string)
    
    inspect = sql.inspect(engine)
    
    table_names = inspect.get_table_names()
    
    return table_names

# Get Raw Table
def db_read_raw_els_table(table = "Products", conn_string = "sqlite:///00_database/crm_database.sqlite"):
    """Reads a single raw table from the CRM database.

    Args:
        table (str, optional): Table Name. Defaults to "Products". See
        db_read_els_table_names() to get the full list of table names. 
        conn_string (str, optional): Defaults to "sqlite:///00_database/crm_database.sqlite".

    Returns:
        _type_: Pandas DataFrame
    """
    
    engine = sql.create_engine(conn_string)
    
    with engine.connect() as conn:
        
        df = pd.read_sql(
            sql=f"SELECT * FROM {table}",
            con=conn
        )
    
    return df

# PROCESSING ---- 

def process_lead_tags(leads_df, tags_df):
    """Processing Pipeline that combines leads and tags data frames and prepares for machine learning. 

    Args:
        leads_df (DataFrame): els.db_read_els_data()
        tags_df (DataFrame): els.db_read_raw_els_table("Tags")

    Returns:
        DataFrame: Leads and Tags combined and prepared for machine learning analysis
    """
    
    # Date Features
    
    date_max = leads_df['optin_time'].max()
    
    leads_df['optin_days'] = (leads_df['optin_time'] - date_max).dt.days
    
    # Email Features
    
    leads_df['email_provider'] = leads_df['user_email'] \
        .map(lambda x: x.split("@")[1])
    
    # Activity Features (Rate Features)
    
    leads_df['tag_count_by_optin_day'] = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)
    
    # Specific Tag Features (Actions)
    
    tags_wide_leads_df = tags_df \
        .assign(value = lambda x: 1) \
        .pivot(
            index = 'mailchimp_id',
            columns = 'tag',
            values = 'value'
        ) \
        .fillna(value = 0) \
        .pipe(
            func=jn.clean_names
        )
    
    # Merge Tags
    
    tags_wide_leads_df.columns = tags_wide_leads_df.columns \
        .to_series() \
        .apply(func = lambda x: f"tag_{x}") \
        .to_list()
        
    tags_wide_leads_df = tags_wide_leads_df.reset_index()
    
    leads_tags_df = leads_df \
        .merge(tags_wide_leads_df, how='left') 
    
    # Fill NA selectively
    
    def fillna_regex(data, regex, value = 0, **kwargs):
        for col in data.columns:
            if re.match(pattern=regex, string = col):
                # print(col)
                data[col] = data[col].fillna(value=value, **kwargs)
        return data

    leads_tags_df = fillna_regex(leads_tags_df, regex="^tag_", value = 0)
        
    # High Cardinality Features: Country Code
    
    countries_to_keep = [
        'us',
        'in',
        'au',
        'uk',
        'br',
        'ca',
        'de',
        'fr',
        'es',
        'mx',
        'nl',
        'sg',
        'dk',
        'pl',
        'my',
        'ae',
        'co',
        'id',
        'ng',
        'jp',
        'be'
    ]
    
    leads_tags_df['country_code'] = leads_tags_df['country_code'] \
        .apply(lambda x: x if x in countries_to_keep else 'other')

    
    return leads_tags_df

# FINAL PIPELINE ----

def db_read_and_process_els_data(
    conn_string='sqlite:///00_database/crm_database.sqlite'
):
    leads_df = db_read_els_data(conn_string=conn_string)

    tags_df = db_read_raw_els_table(
        table = "Tags",
        conn_string=conn_string
    )
    
    df = process_lead_tags(leads_df, tags_df)
    
    return df
