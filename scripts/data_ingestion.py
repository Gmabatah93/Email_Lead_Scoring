import pandas as pd
import sqlalchemy as sql
import argparse

def main(output_path="data/subscribers_joined.csv", verbose=False):
    print("=" * 50)
    print("DATA INGESTTION")
    print("=" * 50)

    # 1 Connect to the database ----
    url = "sqlite:///data/crm_database.sqlite"
    engine = sql.create_engine(url)
    print(f"✅ Connected to database: {url}")

    with engine.connect() as conn:
        # Subscribers        
        subscribers_df = pd.read_sql("SELECT * FROM Subscribers", conn)

        if verbose:
            print(f"📊 Raw subscribers shape: {subscribers_df.shape}")
            print(f"📊 Subscribers columns: {list(subscribers_df.columns)}")

        # - clean
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')

        # Tags
        tags_df = pd.read_sql("SELECT * FROM Tags", conn)

        if verbose:
            print(f"📊 Raw tags shape: {tags_df.shape}")
            print(f"📊 Tags columns: {list(tags_df.columns)}")
        # - clean
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
        
        # Transactions
        transactions_df = pd.read_sql("SELECT * FROM Transactions", conn)
        
        if verbose:
            print(f"📊 Raw transactions shape: {transactions_df.shape}")
            print(f"📊 Transactions columns: {list(transactions_df.columns)}")
        # - clean
        transactions_df['purchased_at'] = transactions_df['purchased_at'].astype('datetime64[ns]')
        transactions_df['product_id'] = transactions_df['product_id'].astype('int')

    print("✅ Data loaded from database")
    print(f"Subscribers: {subscribers_df.shape[0]}")
    print(f"Tags: {tags_df.shape[0]}")
    print(f"Transactions: {transactions_df.shape[0]}")
    
    # MERGE TAG COUNTS
    user_events_df = tags_df \
        .groupby('mailchimp_id') \
        .agg(dict(tag = 'count')) \
        .set_axis(['tag_count'], axis=1) \
        .reset_index()

    if verbose:
        print(f"📊 Tag count distribution: {user_events_df['tag_count'].describe()}")

    subscribers_joined_df = subscribers_df \
        .merge(user_events_df, how='left') \
        .fillna(dict(tag_count = 0))
        
    subscribers_joined_df['tag_count'] = subscribers_joined_df['tag_count'].astype('int')

    # MERGE TARGET VARIABLE
    emails_made_purchase = transactions_df['user_email'].unique()

    subscribers_joined_df['made_purchase'] = subscribers_joined_df['user_email'] \
        .isin(emails_made_purchase) \
        .astype('int')

    if verbose:
        print(f"📊 Purchase rate: {subscribers_joined_df['made_purchase'].mean():.2%}")
        print(f"📊 Final dataset shape: {subscribers_joined_df.shape}")
        
    print(f"✅ Subscribers joined with tags and purchase info: {subscribers_joined_df.shape[0]}")

    subscribers_joined_df.to_csv(output_path, index=False)
    print(f"✅ Subscribers joined data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and process CRM data")
    parser.add_argument("--output_path", type=str, default="data/subscribers_joined.csv", help="Path to save the processed CSV file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    main(output_path=args.output_path, verbose=args.verbose)