
import pandas as pd
import numpy as np

# Function: Explore Sales By Category  
def explore_sales_by_category(data, category = 'country_code', sort_by = ['sales', 'prop_in_group']):
    """This function explores the 'made_purchase' column by a categorical column

    Args:
        data (DataFrame): The subscriber data with a column 'made_purchase'
        category (str, optional): A categorical column. Defaults to 'country_code'.
        sort_by (list, optional): A column to sort by. One of ['sales', 'prop_in_group'].

    Returns:
        DataFrame
    """
    
    # Handle sort_by
    if (type(sort_by) is list):
        sort_by = sort_by[0]
    
    # Data Manipulation
    ret = data \
        .groupby(category) \
        .agg(
            dict(made_purchase = ['sum', lambda x: sum(x) / len(x)])
        ) \
        .set_axis(['sales', 'prop_in_group'], axis=1) \
        .assign(prop_overall = lambda x: x['sales'] / sum(x['sales'])) \
        .sort_values(by = sort_by, ascending=False) \
        .assign(prop_cumsum = lambda x: x['prop_overall'].cumsum())
    
    return ret

# Function: Explore Sales by Numeric Feature
def explore_sales_by_numeric(data, numeric = 'tag_count', q = [0.10, 0.50, 0.90]):
    """Exploring the subscriber data using the column 'made_purchase' and any numeric column(s).

    Args:
        data (DataFrame): The subscriber data with a column added 'made_purchase'
        numeric (str, list, optional): One or more numeric columns. Defaults to 'tag_count'.
        q (list, optional): The values to apply the quantile function. Defaults to [0.10, 0.50, 0.90].

    Returns:
        DataFrame
    """
    
    if (type(numeric) is list):
        feature_list = ['made_purchase', *numeric]
    else: 
        feature_list = ['made_purchase', numeric]
    
    
    ret = data[feature_list] \
        .groupby('made_purchase') \
        .quantile(q = q)
    
    
    return ret
