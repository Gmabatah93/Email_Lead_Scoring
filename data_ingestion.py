import pandas as pd
import sqlalchemy as sql


# 1 Connect to the database ----
url = "sqlite:///data/crm_database.sqlite"


engine = sql.create_engine(url)

with engine.connect() as conn:
    # Subscribers        
    subscribers_df = pd.read_sql("SELECT * FROM Subscribers", conn)
    # - clean
    subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
    subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
    subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')

    # Tags
    tags_df = pd.read_sql("SELECT * FROM Tags", conn)
    tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
    # Transactions
    transactions_df = pd.read_sql("SELECT * FROM Transactions", conn)
    transactions_df['purchased_at'] = transactions_df['purchased_at'].astype('datetime64[ns]')
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


subscribers_joined_df.to_csv("data/subscribers_joined.csv", index=False)