import great_expectations as gx

context = gx.get_context()
context.variables.analytics_enabled = False

assert type(context).__name__ == "EphemeralDataContext"

# Create Data Source
crm_source = context.data_sources.add_sqlite(
    name="crm_database",
    connection_string="sqlite:///data/crm_database.sqlite"
)

# =================== SUBSCRIBERS TABLE ===================
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

# Schema & Structure Validations
subscribers_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=50000)
subscribers_validator.expect_table_column_count_to_equal(6)
subscribers_validator.expect_table_columns_to_match_ordered_list([
    'mailchimp_id', 'user_full_name', 'user_email', 
    'member_rating', 'optin_time', 'country_code'
])

# Primary Key & Uniqueness Checks
subscribers_validator.expect_column_values_to_be_unique("mailchimp_id")
subscribers_validator.expect_column_values_to_not_be_null("mailchimp_id")
# subscribers_validator.expect_column_values_to_be_unique("user_email")
subscribers_validator.expect_column_values_to_not_be_null("user_email")
subscribers_validator.expect_column_values_to_not_be_null("user_full_name")

# Data Format & Pattern Validation
subscribers_validator.expect_column_values_to_match_regex(
    "user_email", 
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
subscribers_validator.expect_column_values_to_match_regex(
    "country_code", 
    r'^[a-zA-Z]{2}$'
)
subscribers_validator.expect_column_values_to_match_regex(
    "optin_time", 
    r'^\d{4}-\d{2}-\d{2}$'  # Matches YYYY-MM-DD format
)

# Range & Domain Validation
subscribers_validator.expect_column_values_to_be_between("member_rating", 1, 5)
subscribers_validator.expect_column_values_to_match_regex(
    "optin_time", 
    r'^\d{4}-\d{2}-\d{2}$'
)

# Valid country codes (allow None for missing values)
valid_countries = ['us', 'ca', 'gb', 'au', 'de', 'fr', 'in', 'it', 'co', None]
subscribers_validator.expect_column_values_to_be_in_set("country_code", valid_countries)

# =================== TAGS TABLE ===================
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

# Schema & Structure
tags_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)
tags_validator.expect_table_column_count_to_equal(2)
tags_validator.expect_table_columns_to_match_ordered_list(['mailchimp_id', 'tag'])

# Data Quality
tags_validator.expect_column_values_to_not_be_null("mailchimp_id")
tags_validator.expect_column_values_to_not_be_null("tag")

# Tag format validation (should not be empty strings)
tags_validator.expect_column_values_to_match_regex("tag", r'^.+$')  # At least 1 character

# =================== TRANSACTIONS TABLE ===================
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

# Schema & Structure
transactions_validator.expect_table_row_count_to_be_between(min_value=1000, max_value=20000)
transactions_validator.expect_table_column_count_to_equal(6)
transactions_validator.expect_table_columns_to_match_ordered_list([
    'transaction_id', 'purchased_at', 'user_full_name', 
    'user_email', 'charge_country', 'product_id'
])

# Primary Key & Required Fields
transactions_validator.expect_column_values_to_be_unique("transaction_id")
transactions_validator.expect_column_values_to_not_be_null("transaction_id")
transactions_validator.expect_column_values_to_not_be_null("user_email")
transactions_validator.expect_column_values_to_not_be_null("product_id")
transactions_validator.expect_column_values_to_not_be_null("purchased_at")

# Data Format Validation
transactions_validator.expect_column_values_to_match_regex(
    "user_email", 
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)
transactions_validator.expect_column_values_to_match_regex(
    "purchased_at",
    r'^\d{4}-\d{2}-\d{2}$'  # Matches YYYY-MM-DD format
)

# Range & Domain Validation
transactions_validator.expect_column_values_to_be_between("product_id", 1, 100)
transactions_validator.expect_column_values_to_match_regex(
    "purchased_at",
    r'^\d{4}-\d{2}-\d{2}$'  # Matches YYYY-MM-DD format
)

# Valid country codes for transactions
valid_charge_countries = ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'NZ', None]
transactions_validator.expect_column_values_to_be_in_set("charge_country", valid_charge_countries)

# =================== RUN ALL VALIDATIONS ===================
print("🔍 Running Subscribers validation...")
subscribers_result = subscribers_validator.validate(only_return_failures=True)
print(f"Subscribers: {'✅ PASSED' if subscribers_result.success else '❌ FAILED'}")

print("🔍 Running Tags validation...")
tags_result = tags_validator.validate()
print(f"Tags: {'✅ PASSED' if tags_result.success else '❌ FAILED'}")

print("🔍 Running Transactions validation...")
transactions_result = transactions_validator.validate()
print(f"Transactions: {'✅ PASSED' if transactions_result.success else '❌ FAILED'}")

print(f"\n🎉 Overall: {'✅ ALL PASSED' if all([subscribers_result.success, tags_result.success, transactions_result.success]) else '❌ SOME FAILED'}")