import pandas as pd
import sqlalchemy as sql
import typer
from typing_extensions import Annotated
from pathlib import Path

from rich.console import Console
from rich.table import Table
console = Console()

app = typer.Typer()

@app.command()
def main(
    output_path: Annotated[Path, typer.Option(help="Path to save the processed CSV file")] = "data/subscribers_joined.csv",
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False
):
    """Ingest and process CRM data from a SQLite database, then save the cleaned and merged data to a CSV file."""
    
    typer.echo("=" * 50)
    typer.echo(typer.style("DATA INGESTION", fg=typer.colors.CYAN))
    typer.echo("=" * 50)

    # 1 Connect to the database ----
    url = f"sqlite:///data/crm_database.sqlite"
    engine = sql.create_engine(url)
    typer.echo(typer.style(f"✅ Connected to database: data/crm_database.sqlite\n", fg=typer.colors.GREEN))
    
    with engine.connect() as conn:
        # Subscribers        
        subscribers_df = pd.read_sql("SELECT * FROM Subscribers", conn)

        if verbose:
            typer.echo('--' * 6 + " 📊 Subscribers " + 6 * "--")
            typer.echo(typer.style(f"📊 Raw Subscribers shape: {subscribers_df.shape}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Subscribers columns: {list(subscribers_df.columns)}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Raw Subscribers preview:\n{subscribers_df.head().to_string()}", fg=typer.colors.BRIGHT_YELLOW))
        # - clean
        subscribers_df['mailchimp_id'] = subscribers_df['mailchimp_id'].astype('int')
        subscribers_df['member_rating'] = subscribers_df['member_rating'].astype('int')
        subscribers_df['optin_time'] = subscribers_df['optin_time'].astype('datetime64[ns]')

        # Tags
        tags_df = pd.read_sql("SELECT * FROM Tags", conn)

        if verbose:
            typer.echo('--' * 8 + " 📊 Tags " + 8 * "--")
            typer.echo(typer.style(f"📊 Raw Tags shape: {tags_df.shape}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Tags columns: {list(tags_df.columns)}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Raw Tags preview:\n{tags_df.head().to_string()}", fg=typer.colors.BRIGHT_YELLOW))
        # - clean
        tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype("int")
        
        # Transactions
        transactions_df = pd.read_sql("SELECT * FROM Transactions", conn)
        
        if verbose:
            typer.echo('--' * 6 + " 📊 Transactions " + 6 * "--")
            typer.echo(typer.style(f"📊 Raw Transactions shape: {transactions_df.shape}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Transactions columns: {list(transactions_df.columns)}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo(typer.style(f"📊 Raw Transactions preview:\n{transactions_df.head().to_string()}", fg=typer.colors.BRIGHT_YELLOW))
            typer.echo('--' * 20)
        # - clean
        transactions_df['purchased_at'] = transactions_df['purchased_at'].astype('datetime64[ns]')
        transactions_df['product_id'] = transactions_df['product_id'].astype('int')

    typer.echo(typer.style("✅ Data loaded from database", fg=typer.colors.GREEN))
    typer.echo(f"     - Subscribers: {subscribers_df.shape[0]}")
    typer.echo(f"     - Tags: {tags_df.shape[0]}")
    typer.echo(f"     - Transactions: {transactions_df.shape[0]}")

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

    if verbose:
        typer.echo('--' * 6 + " 📊 Subscribers_Joined " + 6 * "--")
        typer.echo(typer.style(f"📊 Final dataset shape: {subscribers_joined_df.shape}", fg=typer.colors.BRIGHT_YELLOW))
        typer.echo(typer.style(f"📊 Final dataset columns: {list(subscribers_joined_df.columns)}", fg=typer.colors.BRIGHT_YELLOW))
        typer.echo(typer.style(f"📊 Final dataset preview:\n{subscribers_joined_df.head().to_string()}", fg=typer.colors.BRIGHT_YELLOW))
        typer.echo('--' * 20)

     # Target Variable Statistics
    typer.echo(typer.style(f"🎯 Target Column 'made_purchase': {subscribers_joined_df['made_purchase'].mean():.2%}", fg=typer.colors.BRIGHT_RED))

    typer.echo(typer.style(f"✅ Subscribers joined with tags and purchase info: {subscribers_joined_df.shape[0]}", fg=typer.colors.GREEN))

    # print(subscribers_joined_df.head())
    console.print(subscribers_joined_df.head().to_string())
    subscribers_joined_df.to_csv(output_path, index=False)
    typer.echo(typer.style(f"✅ Subscribers joined data saved to: {output_path}", fg=typer.colors.GREEN))
    
# python scripts/data_ingestion.py --help 
if __name__ == "__main__":
    app()

