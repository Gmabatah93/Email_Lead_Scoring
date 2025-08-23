import pandas as pd
import great_expectations as gx
import sqlalchemy as sql
import logging
import warnings
import json
import typer
from typing_extensions import Annotated

app = typer.Typer()

# Suppress most logging and warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@app.command()
def crm(
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
    results_path: Annotated[str, typer.Option(help="Path to save validation results as JSON")] = "results/data_quality/crm_validation_results.json"
):
    """Test raw data quality from CRM database tables using Great Expectations."""
    typer.echo("=" * 70)
    typer.echo(typer.style("🔎 CRM DATA QUALITY TESTS", fg=typer.colors.CYAN))
    typer.echo("=" * 70)

    context = gx.get_context()
    crm_source = context.data_sources.add_sqlite(
        name="crm_database",
        connection_string="sqlite:///data/crm_database.sqlite"
    )

    # ========== SUBSCRIBERS ==========
    typer.echo(typer.style("---" * 1 + " TABLE: Subscribers " + 15 * "---", fg=typer.colors.BRIGHT_GREEN))
    
    # Setup
    subscribers_asset = crm_source.add_table_asset(
        name="subscribers_raw",
        table_name="Subscribers"
    )
    subscribers_batch = subscribers_asset.build_batch_request()
    subscribers_suite = context.suites.add(gx.ExpectationSuite(name="subscribers_quality_tests"))
    subscribers_validator = context.get_validator(
        batch_request=subscribers_batch,
        expectation_suite=subscribers_suite
    )

    # Checks
    typer.echo(typer.style("🔎 - Checking row count...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=50000)
    
    typer.echo(typer.style("\n🔎 - Checking column count...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_table_column_count_to_equal(6)

    typer.echo(typer.style("\n🔎 - Checking column order...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_table_columns_to_match_ordered_list([
        'mailchimp_id', 'user_full_name', 'user_email',
        'member_rating', 'optin_time', 'country_code'
    ])

    typer.echo(typer.style("\n🔎 - Checking mailchimp_id uniqueness and not null...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_be_unique("mailchimp_id")
    subscribers_validator.expect_column_values_to_not_be_null("mailchimp_id")

    typer.echo(typer.style("\n🔎 - Checking user_email, not null, and format...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_not_be_null("user_email")
    subscribers_validator.expect_column_values_to_match_regex(
        "user_email", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    typer.echo(typer.style("\n🔎 - Checking user_full_name not null...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_not_be_null("user_full_name")

    typer.echo(typer.style("\n🔎 - Checking country_code format and allowed values...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_match_regex(
        "country_code", r'^[a-zA-Z]{2}$'
    )
    valid_countries = ['us', 'ca', 'gb', 'au', 'de', 'fr', 'in', 'it', 'co', None]
    subscribers_validator.expect_column_values_to_be_in_set("country_code", valid_countries)

    typer.echo(typer.style("\n🔎 - Checking optin_time format...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_match_regex(
        "optin_time", r'^\d{4}-\d{2}-\d{2}$'
    )

    typer.echo(typer.style("\n🔎 - Checking member_rating range...", fg=typer.colors.BRIGHT_RED))
    subscribers_validator.expect_column_values_to_be_between("member_rating", 1, 5)

    # ========== TAGS ==========
    print("\n")
    typer.echo(typer.style("---" * 1 + " TABLE: Tags " + 18 * "---", fg=typer.colors.BRIGHT_GREEN))
    
    # Setup
    tags_asset = crm_source.add_table_asset(
        name="tags_raw",
        table_name="Tags"
    )
    tags_batch = tags_asset.build_batch_request()
    tags_suite = context.suites.add(gx.ExpectationSuite(name="tags_quality_tests"))
    tags_validator = context.get_validator(
        batch_request=tags_batch,
        expectation_suite=tags_suite
    )

    # Checks
    typer.echo(typer.style("🔎 - Checking row count...", fg=typer.colors.BRIGHT_RED))
    tags_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)

    typer.echo(typer.style("\n🔎 - Checking column count...", fg=typer.colors.BRIGHT_RED))
    tags_validator.expect_table_column_count_to_equal(2)
    
    typer.echo(typer.style("\n🔎 - Checking column order...", fg=typer.colors.BRIGHT_RED))
    tags_validator.expect_table_columns_to_match_ordered_list(['mailchimp_id', 'tag'])
    
    typer.echo(typer.style("\n🔎 - Checking mailchimp_id not null...", fg=typer.colors.BRIGHT_RED))
    tags_validator.expect_column_values_to_not_be_null("mailchimp_id")
    
    typer.echo(typer.style("  • Checking tag not null and not empty...", fg=typer.colors.BRIGHT_RED))
    tags_validator.expect_column_values_to_not_be_null("tag")
    tags_validator.expect_column_values_to_match_regex("tag", r'^.+$')

    # ========== TRANSACTIONS ==========
    print("\n")
    typer.echo(typer.style("---" * 1 + " TABLE: Transactions " + 16 * "---", fg=typer.colors.BRIGHT_GREEN))
    
    # Setup
    transactions_asset = crm_source.add_table_asset(
        name="transactions_raw",
        table_name="Transactions"
    )
    transactions_batch = transactions_asset.build_batch_request()
    transactions_suite = context.suites.add(gx.ExpectationSuite(name="transactions_quality_tests"))
    transactions_validator = context.get_validator(
        batch_request=transactions_batch,
        expectation_suite=transactions_suite
    )

    # Checks
    typer.echo(typer.style("🔎 - Checking row count...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=20000)
    
    typer.echo(typer.style("\n🔎 - Checking column count...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_table_column_count_to_equal(6)
    
    typer.echo(typer.style("\n🔎 - Checking column order...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_table_columns_to_match_ordered_list([
        'transaction_id', 'purchased_at', 'user_full_name',
        'user_email', 'charge_country', 'product_id'
    ])
    
    typer.echo(typer.style("\n🔎 - Checking transaction_id uniqueness and not null...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_column_values_to_be_unique("transaction_id")
    transactions_validator.expect_column_values_to_not_be_null("transaction_id")

    typer.echo(typer.style("\n🔎 - Checking user_email and product_id not null and email format...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_column_values_to_not_be_null("user_email")
    transactions_validator.expect_column_values_to_not_be_null("product_id")
    transactions_validator.expect_column_values_to_match_regex(
        "user_email", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    typer.echo(typer.style("\n🔎 - Checking purchased_at not null and format...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_column_values_to_not_be_null("purchased_at")
    transactions_validator.expect_column_values_to_match_regex(
        "purchased_at", r'^\d{4}-\d{2}-\d{2}$'
    )

    typer.echo(typer.style("\n🔎 - Checking product_id range...", fg=typer.colors.BRIGHT_RED))
    transactions_validator.expect_column_values_to_be_between("product_id", 1, 100)

    typer.echo(typer.style("\n🔎 - Checking charge_country allowed values...", fg=typer.colors.BRIGHT_RED))
    valid_charge_countries = ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'NZ', None]
    transactions_validator.expect_column_values_to_be_in_set("charge_country", valid_charge_countries)

    # ========== RUN VALIDATIONS ==========
    print("\n")
    typer.echo("-" * 70)
    typer.echo(typer.style("🧾 SUMMARY", fg=typer.colors.BRIGHT_YELLOW))
    typer.echo("-" * 70)

    subscribers_result = subscribers_validator.validate(result_format="BASIC")
    typer.echo(typer.style(f"Subscribers: {'✅ PASSED' if subscribers_result.success else '❌ FAILED'}", fg=typer.colors.GREEN if subscribers_result.success else typer.colors.BRIGHT_RED))

    tags_result = tags_validator.validate(result_format="SUMMARY")
    typer.echo(typer.style(f"Tags: {'✅ PASSED' if tags_result.success else '❌ FAILED'}", fg=typer.colors.GREEN if tags_result.success else typer.colors.BRIGHT_RED))

    transactions_result = transactions_validator.validate(result_format="COMPLETE")
    typer.echo(typer.style(f"Transactions: {'✅ PASSED' if transactions_result.success else '❌ FAILED'}", fg=typer.colors.GREEN if transactions_result.success else typer.colors.BRIGHT_RED))

    typer.echo(typer.style(f"\n🎉 Overall: {'✅ ALL PASSED' if all([subscribers_result.success, tags_result.success, transactions_result.success]) else '❌ SOME FAILED'}", fg=typer.colors.BRIGHT_GREEN))

    # Save results to JSON
    results = {
        "subscribers": subscribers_result.to_json_dict(),
        "tags": tags_result.to_json_dict(),
        "transactions": transactions_result.to_json_dict()
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(typer.style(f"✅ Results saved to {results_path}", fg=typer.colors.BRIGHT_GREEN))

def test_processed_data():
    """Test processed subscribers_joined.csv data quality"""
    print("🔍 Testing processed data...")
    # Placeholder for future update
    pass

def test_business_rules():
    """Test business logic and domain-specific rules"""
    print("🔍 Testing business rules...")
    # Placeholder for future update
    pass

def run_all_tests():
    """Run complete data testing pipeline"""
    print("🚀 Starting comprehensive data testing...")

    # Test source data
    crm_results = crm()

    # Test processed data (placeholder)
    test_processed_data()

    # Test business rules (placeholder)
    test_business_rules()

    print("🎉 All data testing completed!")
    return crm_results

# CLI interface for running specific tests
if __name__ == "__main__":
    app()