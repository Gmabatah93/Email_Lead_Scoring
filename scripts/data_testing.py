import pandas as pd
import great_expectations as gx
import sqlalchemy as sql
import logging
import warnings
import json
from pathlib import Path

# Suppress most logging and warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# BROKEN:
def print_failed_expectations(result, table_name):
    """Prints failed expectations for a given validation result."""
    if not result.success:
        print(f"  ❌ {table_name} table failures:")
        for res in result.results:
            if not res.success:
                exp_type = res.expectation_config.expectation_type
                column = res.expectation_config.kwargs.get("column", "")
                print(f"    - {exp_type} on column '{column}':")
                print(f"      Details: {res.result}")
    else:
        print(f"  ✅ All expectations passed for {table_name}.")

def test_crm_source_data():
    """Test raw data quality from CRM database tables using Great Expectations."""
    print("🔍 Testing CRM source data...")

    context = gx.get_context()
    crm_source = context.data_sources.add_sqlite(
        name="crm_database",
        connection_string="sqlite:///data/crm_database.sqlite"
    )

    # ========== SUBSCRIBERS ==========
    print("\n--- Subscribers Table Checks ---")
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

    print("  • Checking row count...")
    subscribers_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=50000)
    print("  • Checking column count...")
    subscribers_validator.expect_table_column_count_to_equal(6)
    print("  • Checking column order...")
    subscribers_validator.expect_table_columns_to_match_ordered_list([
        'mailchimp_id', 'user_full_name', 'user_email',
        'member_rating', 'optin_time', 'country_code'
    ])
    print("  • Checking mailchimp_id uniqueness and not null...")
    subscribers_validator.expect_column_values_to_be_unique("mailchimp_id")
    subscribers_validator.expect_column_values_to_not_be_null("mailchimp_id")
    print("  • Checking user_email uniqueness, not null, and format...")
    subscribers_validator.expect_column_values_to_be_unique("user_email")
    subscribers_validator.expect_column_values_to_not_be_null("user_email")
    subscribers_validator.expect_column_values_to_match_regex(
        "user_email", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    print("  • Checking user_full_name not null...")
    subscribers_validator.expect_column_values_to_not_be_null("user_full_name")
    print("  • Checking country_code format and allowed values...")
    subscribers_validator.expect_column_values_to_match_regex(
        "country_code", r'^[a-z]{2}$'
    )
    valid_countries = ['us', 'ca', 'gb', 'au', 'de', 'fr', 'in', 'it', 'co', None]
    subscribers_validator.expect_column_values_to_be_in_set("country_code", valid_countries)
    print("  • Checking optin_time format...")
    subscribers_validator.expect_column_values_to_match_regex(
        "optin_time", r'^\d{4}-\d{2}-\d{2}$'
    )
    print("  • Checking member_rating range...")
    subscribers_validator.expect_column_values_to_be_between("member_rating", 1, 5)

    # ========== TAGS ==========
    print("\n--- Tags Table Checks ---")
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
    print("  • Checking row count...")
    tags_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)
    print("  • Checking column count...")
    tags_validator.expect_table_column_count_to_equal(2)
    print("  • Checking column order...")
    tags_validator.expect_table_columns_to_match_ordered_list(['mailchimp_id', 'tag'])
    print("  • Checking mailchimp_id not null...")
    tags_validator.expect_column_values_to_not_be_null("mailchimp_id")
    print("  • Checking tag not null and not empty...")
    tags_validator.expect_column_values_to_not_be_null("tag")
    tags_validator.expect_column_values_to_match_regex("tag", r'^.+$')

    # ========== TRANSACTIONS ==========
    print("\n--- Transactions Table Checks ---")
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
    print("  • Checking row count...")
    transactions_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=20000)
    print("  • Checking column count...")
    transactions_validator.expect_table_column_count_to_equal(6)
    print("  • Checking column order...")
    transactions_validator.expect_table_columns_to_match_ordered_list([
        'transaction_id', 'purchased_at', 'user_full_name',
        'user_email', 'charge_country', 'product_id'
    ])
    print("  • Checking transaction_id uniqueness and not null...")
    transactions_validator.expect_column_values_to_be_unique("transaction_id")
    transactions_validator.expect_column_values_to_not_be_null("transaction_id")
    print("  • Checking user_email and product_id not null and email format...")
    transactions_validator.expect_column_values_to_not_be_null("user_email")
    transactions_validator.expect_column_values_to_not_be_null("product_id")
    transactions_validator.expect_column_values_to_match_regex(
        "user_email", r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    print("  • Checking purchased_at not null and format...")
    transactions_validator.expect_column_values_to_not_be_null("purchased_at")
    transactions_validator.expect_column_values_to_match_regex(
        "purchased_at", r'^\d{4}-\d{2}-\d{2}$'
    )
    print("  • Checking product_id range...")
    transactions_validator.expect_column_values_to_be_between("product_id", 1, 100)
    print("  • Checking charge_country allowed values...")
    valid_charge_countries = ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'NZ', None]
    transactions_validator.expect_column_values_to_be_in_set("charge_country", valid_charge_countries)

    # ========== RUN VALIDATIONS ==========
    print("\n🔍 Running Subscribers validation...")
    subscribers_result = subscribers_validator.validate()
    print(f"Subscribers: {'✅ PASSED' if subscribers_result.success else '❌ FAILED'}")
    print_failed_expectations(subscribers_result, "Subscribers")

    print("🔍 Running Tags validation...")
    tags_result = tags_validator.validate()
    print(f"Tags: {'✅ PASSED' if tags_result.success else '❌ FAILED'}")
    print_failed_expectations(tags_result, "Tags")

    print("🔍 Running Transactions validation...")
    transactions_result = transactions_validator.validate()
    print(f"Transactions: {'✅ PASSED' if transactions_result.success else '❌ FAILED'}")
    print_failed_expectations(transactions_result, "Transactions")

    print(f"\n🎉 Overall: {'✅ ALL PASSED' if all([subscribers_result.success, tags_result.success, transactions_result.success]) else '❌ SOME FAILED'}")

    return {
        "subscribers": subscribers_result,
        "tags": tags_result,
        "transactions": transactions_result
    }

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
    crm_results = test_crm_source_data()

    # Test processed data (placeholder)
    test_processed_data()

    # Test business rules (placeholder)
    test_business_rules()

    print("🎉 All data testing completed!")
    return crm_results

# CLI interface for running specific tests
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "crm":
            test_crm_source_data()
        elif test_type == "processed":
            test_processed_data()
        elif test_type == "business":
            test_business_rules()
        else:
            print("Usage: python data_testing.py [crm|processed|business]")
    else:
        # Run all tests by default
        run_all_tests()