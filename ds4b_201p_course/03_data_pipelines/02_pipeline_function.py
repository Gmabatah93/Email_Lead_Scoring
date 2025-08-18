# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA PROCESSING PIPELINE FUNCTION
# ----

# LIBRARIES ----
# Core
import pandas as pd
import numpy as np
# EDA 
import re
import janitor as jn

import email_lead_scoring as els

leads_df = els.db_read_els_data()

tags_df = els.db_read_raw_els_table("Tags")

# 1.0 CREATE PROCESSING FUNCTION ----

def process_lead_tags(leads_df, tags_df):
    
    # Date Features
    
    date_max = leads_df['optin_time'].max()
    
    leads_df['optin_days'] = (leads_df['optin_time'] - date_max).dt.days
    
    # Email Features
    
    leads_df['email_provider'] = leads_df['user_email'] \
        .map(lambda x: x.split("@")[1])
    
    # Activity Features (Rate Features)
    
    leads_df['tag_count_by_optin_day'] = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)
    
    # Specific Tag Features (Actions)
    
    tags_wide_leads_df = tags_df \
        .assign(value = lambda x: 1) \
        .pivot(
            index = 'mailchimp_id',
            columns = 'tag',
            values = 'value'
        ) \
        .fillna(value = 0) \
        .pipe(
            func=jn.clean_names
        )
    
    # Merge Tags
    
    tags_wide_leads_df.columns = tags_wide_leads_df.columns \
        .to_series() \
        .apply(func = lambda x: f"tag_{x}") \
        .to_list()
        
    tags_wide_leads_df = tags_wide_leads_df.reset_index()
    
    leads_tags_df = leads_df \
        .merge(tags_wide_leads_df, how='left') 
    
    # Fill NA selectively
    
    def fillna_regex(data, regex, value = 0, **kwargs):
        for col in data.columns:
            if re.match(pattern=regex, string = col):
                # print(col)
                data[col] = data[col].fillna(value=value, **kwargs)
        return data

    leads_tags_df = fillna_regex(leads_tags_df, regex="^tag_", value = 0)
        
    # High Cardinality Features: Country Code
    
    countries_to_keep = [
        'us',
        'in',
        'au',
        'uk',
        'br',
        'ca',
        'de',
        'fr',
        'es',
        'mx',
        'nl',
        'sg',
        'dk',
        'pl',
        'my',
        'ae',
        'co',
        'id',
        'ng',
        'jp',
        'be'
    ]
    
    leads_tags_df['country_code'] = leads_tags_df['country_code'] \
        .apply(lambda x: x if x in countries_to_keep else 'other')

    
    return leads_tags_df
    

# 2.0 TEST OUT

process_lead_tags(leads_df, tags_df) \
    .head(3)


# 3.0 IMPROVE OUR DATA PIPELINE

def db_read_and_process_els_data(
    conn_string='sqlite:///00_database/crm_database.sqlite'
):
    leads_df = els.db_read_els_data(conn_string=conn_string)

    tags_df = els.db_read_raw_els_table(
        table = "Tags",
        conn_string=conn_string
    )
    
    df = process_lead_tags(leads_df, tags_df)
    
    return df

db_read_and_process_els_data()

# 4.0 TRY PACKAGE 

import email_lead_scoring as els

els.db_read_and_process_els_data()

