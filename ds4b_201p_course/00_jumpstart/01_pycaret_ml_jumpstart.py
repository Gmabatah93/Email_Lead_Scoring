# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 0: MACHINE LEARNING & API'S JUMPSTART 
# PART 1: PYCARET
# ----

# GOAL: Make an introductory ML lead scoring model
#  that can be used in an API

# LIBRARIES

import os
import pandas as pd
import sqlalchemy as sql
import pycaret.classification as clf


# 1.0 READ DATA ----

# Connect to SQL Database ----

engine = sql.create_engine("sqlite:///00_database/crm_database.sqlite")

conn = engine.connect()

sql.inspect(engine).get_table_names()

# * Subscribers ---

subscribers_df = pd.read_sql(
    sql = "SELECT * FROM Subscribers", 
    con = conn
)

# * Transactions

transactions_df = pd.read_sql(
    sql = "SELECT * FROM Transactions",
    con = conn
)


# *Close Connection ----

conn.close()

# 2.0 SIMPLIFIED DATA PREP ----

subscribers_joined_df = subscribers_df

emails_made_purchase = transactions_df['user_email'] \
    .unique()
    
subscribers_joined_df['made_purchase'] =subscribers_joined_df['user_email'] \
    .isin(emails_made_purchase) \
    .astype('int')
    
subscribers_joined_df

# 3.0 QUICKSTART MACHINE LEARNING WITH PYCARET ----

# * Subset the data ----

df = subscribers_joined_df[['member_rating', 'country_code', 'made_purchase']]
df


# * Setup the Classifier ----

clf_1 = clf.setup(
    data       = df,
    target     = "made_purchase",
    train_size = 0.8,
    session_id = 123
)

clf_1

# * Make A Machine Learning Model ----

xgb_model = clf.create_model(
    estimator='xgboost'
)

# * Finalize the model ----

xgb_model_finalized = clf.finalize_model(xgb_model)

# * Predict -----

new_df = pd.DataFrame(
    dict(
        member_rating = [5],
        country_code  = ["us"]
    )
)

clf.predict_model(
    estimator = xgb_model_finalized,
    data      = new_df,
    raw_score = True
)


# * Save the Model ----

os.mkdir("00_jumpstart/models")

clf.save_model(
    model = xgb_model_finalized,
    model_name = "00_jumpstart/models/xgb_model_finalized"
)

# * Load the model -----

clf.load_model(
    model_name = "00_jumpstart/models/xgb_model_finalized"
)



# CONCLUSIONS:
# * Insane that we did all of this in 90 lines of code
# * And the model was better than random guessing...
# * But, there are questions that come to mind...

# KEY QUESTIONS:
# * SHOULD WE EVEN TAKE ON THIS PROJECT? (COST/BENEFIT)
# * MACHINE LEARNING MODEL - IS IT GOOD?
# * WHAT CAN WE DO TO IMPROVE THE MODEL?
# * WHAT ARE THE KEY FEATURES IN THE MODEL?
# * CAN WE EXPLAIN WHY CUSTOMERS ARE BUYING / NOT BUYING?
# * CAN THE COMPANY MAKE A RETURN ON INVESTMENT FROM THIS MODEL?


