import great_expectations as gx
import pandas as pd

df = pd.read_csv("data/subscribers_joined.csv")
context = gx.get_context()

suite = context.suites.add(gx.ExpectationSuite(name="subscribers_joined_quality_tests"))

# This is the most version-agnostic way for v1.x:
validator = context.get_validator(
    batch_data=df,
    expectation_suite=suite,
)