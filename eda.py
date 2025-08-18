"""
Exploratory data analysis (EDA) to understand the signals and nuances of our dataset. 
It's a cyclical process that can be done at various points of our development process (before/after labeling, preprocessing, etc. depending on how well the problem is defined. 
For example, if we're unsure how to label or preprocess our data, we can use EDA to figure it out.

We're going to start our project with EDA, a vital (and fun) process that's often misconstrued. 
Here's how to think about EDA:

- not just to visualize a prescribed set of plots (correlation matrix, etc.).
- goal is to convince yourself that the data you have is sufficient for the task.
- use EDA to answer important questions and to make it easier to extract insight
- not a one time process; as your data grows, you want to revisit EDA to catch distribution shifts, anomalies, etc.
"""

import pandas as pd

df_raw = pd.read_csv("data/subscribers_joined.csv")


# 1. EXPLORATION ========================================================

# High Cardinality: Lots of unique values 
df_raw \
    .groupby('country_code') \
    .agg(
        dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])
    ) \
    .set_axis(['sales', 'prop_in_group'], axis=1) \
    .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
    .sort_values(by = 'sales', ascending=False) \
    .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum()) \
    .query("sales > 5") 


# Ordinal Features: Categories have meaningful order
df_raw \
    .groupby('member_rating') \
    .agg(
        dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])
    ) \
    .set_axis(['sales', 'prop_in_group'], axis=1) \
    .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
    .sort_values(by = 'sales', ascending=False) \
    .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum())

# Interaction: Essentially, variables don't act in isolation; their combined effect is more complex than the sum of their individual effects
feature_list = ['made_purchase', 'tag_count']
df_raw[feature_list] \
    .groupby('made_purchase') \
    .quantile(q = [0.10, 0.50, 0.90])

# Target variable distribution
conversion_rate = df_raw['made_purchase'].mean()
print(f"Overall conversion rate: {conversion_rate:.2%}")

# Temporal analysis
df_raw['optin_month'] = pd.to_datetime(df_raw['optin_time']).dt.month
monthly_conversions = df_raw.groupby('optin_month')['made_purchase'].agg(['count', 'sum', 'mean'])

# Email provider analysis
df_raw['email_domain'] = df_raw['user_email'].str.split('@').str[1]
email_performance = df_raw.groupby('email_domain')['made_purchase'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)




