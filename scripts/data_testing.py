import pandas as pd
import great_expectations as ge
from great_expectations.core import ExpectationSuite
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_subscribers_data():
    """Test data quality of subscribers_joined.csv using Great Expectations"""
    
    # Load the saved CSV data
    df = pd.read_csv("data/subscribers_joined.csv")
    logger.info(f"Loaded {len(df)} records from subscribers_joined.csv")
    
    # Convert to Great Expectations DataFrame
    ge_df = ge.from_pandas(df)
    
    # Basic schema validation
    logger.info("Running schema validation tests...")
    ge_df.expect_table_columns_to_match_ordered_list([
        'mailchimp_id', 'user_full_name', 'user_email', 'member_rating',
        'optin_time', 'country_code', 'tag_count', 'made_purchase'
    ])
    
    # Data quality tests
    logger.info("Running data quality tests...")
    
    # Primary key validation
    ge_df.expect_column_values_to_be_unique('mailchimp_id')
    ge_df.expect_column_values_to_not_be_null('mailchimp_id')
    
    # Email validation
    ge_df.expect_column_values_to_not_be_null('user_email')
    ge_df.expect_column_values_to_match_regex('user_email', 
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ge_df.expect_column_values_to_be_unique('user_email')
    
    # Business rule validation
    ge_df.expect_column_values_to_be_between('member_rating', 1, 5)
    ge_df.expect_column_values_to_be_of_type('member_rating', 'int64')
    
    # Tag count validation
    ge_df.expect_column_values_to_be_between('tag_count', 0, None)
    ge_df.expect_column_values_to_be_of_type('tag_count', 'int64')
    
    # Target variable validation
    ge_df.expect_column_values_to_be_in_set('made_purchase', [0, 1])
    ge_df.expect_column_values_to_be_of_type('made_purchase', 'int64')
    
    # Date validation (if optin_time is loaded as string)
    ge_df.expect_column_values_to_not_be_null('optin_time')
    
    # Run all validations
    logger.info("Executing all validation tests...")
    validation_result = ge_df.validate()
    
    # Print results
    if validation_result.success:
        logger.info("✅ All data quality tests PASSED!")
    else:
        logger.error("❌ Some data quality tests FAILED!")
        
        # Print failed expectations
        for result in validation_result.results:
            if not result.success:
                logger.error(f"FAILED: {result.expectation_config.expectation_type}")
                logger.error(f"Details: {result.result}")
    
    return validation_result

def test_business_rules():
    """Additional business rule testing"""
    df = pd.read_csv("data/subscribers_joined.csv")
    
    logger.info("Running business rule tests...")
    
    # Test: Users with purchases should have tag_count >= 0
    purchase_users = df[df['made_purchase'] == 1]
    assert len(purchase_users) > 0, "No users with purchases found"
    logger.info(f"✅ Found {len(purchase_users)} users with purchases")
    
    # Test: Email providers distribution
    df['email_domain'] = df['user_email'].str.split('@').str[1]
    common_domains = df['email_domain'].value_counts().head()
    logger.info(f"✅ Email domain distribution: {common_domains.to_dict()}")
    
    # Test: Country code format (should be 2 characters or null)
    invalid_countries = df[df['country_code'].notna() & 
                          (df['country_code'].str.len() != 2)]
    assert len(invalid_countries) == 0, f"Found {len(invalid_countries)} invalid country codes"
    logger.info("✅ All country codes are valid format")
    
    # Test: Conversion rate is reasonable (between 1-50%)
    conversion_rate = df['made_purchase'].mean()
    assert 0.01 <= conversion_rate <= 0.5, f"Conversion rate {conversion_rate:.2%} seems unrealistic"
    logger.info(f"✅ Conversion rate is {conversion_rate:.2%}")

if __name__ == "__main__":
    logger.info("Starting data quality testing...")
    
    # Run Great Expectations tests
    ge_results = test_subscribers_data()
    
    # Run business rule tests
    test_business_rules()
    
    logger.info("Data testing completed!")
    
    # Save validation results
    if hasattr(ge_results, 'to_json_dict'):
        import json
        with open('data/validation_results.json', 'w') as f:
            json.dump(ge_results.to_json_dict(), f, indent=2)
        logger.info("Validation results saved to data/validation_results.json")