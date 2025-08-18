# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 2: DATA IMPORT FUNCTIONS
# ----

# LIBRARIES ----

import pandas as pd
import numpy as np
import sqlalchemy as sql


# IMPORT RAW DATA ----

# Read & Combine Raw Data

def db_read_els_data(conn_string = "sqlite:///00_database/crm_database.sqlite"):
    
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
        

db_read_els_data().info()    


# Read Table Names
def db_read_els_table_names(conn_string = "sqlite:///00_database/crm_database.sqlite"):
    
    engine = sql.create_engine(conn_string)
    
    inspect = sql.inspect(engine)
    
    table_names = inspect.get_table_names()
    
    return table_names
    
db_read_els_table_names() 
    

# Get Raw Table
def db_read_raw_els_table(table = "Products", conn_string = "sqlite:///00_database/crm_database.sqlite"):
    
    engine = sql.create_engine(conn_string)
    
    with engine.connect() as conn:
        
        df = pd.read_sql(
            sql=f"SELECT * FROM {table}",
            con=conn
        )
    
    return df

db_read_els_table_names()
db_read_raw_els_table("Website")


# TEST IT OUT -----

import email_lead_scoring as els

els.db_read_els_data()

els.db_read_els_table_names()

els.db_read_raw_els_table("Website")
