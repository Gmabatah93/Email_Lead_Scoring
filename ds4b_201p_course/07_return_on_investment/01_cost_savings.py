# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI
# PART 1: COST VS SAVINGS TRADEOFF & ESTIMATE
# ----

# LIBRARIES -----

import pandas as pd
import numpy as np
import email_lead_scoring as els

# RECAP ----

leads_df = els.db_read_and_process_els_data()

leads_scored_df = els.model_score_leads(
    data = leads_df,
    model_path = "models/xgb_model_tuned"
)

# Optional (MLFlow)
leads_scored_df_2 = els.mlflow_score_leads(
    data=leads_df, 
    run_id = els.mlflow_get_best_run('automl_lead_scoring_1')
)

# REVIEW COSTS ----

els.cost_calc_monthly_cost_table()

els.cost_simulate_unsub_costs()

# 1.0 LEAD TARGETING STRATEGY ----

# 1.1 Make Lead Strategy

leads_scored_small_df = leads_scored_df[['user_email', 'Score', 'made_purchase']]

leads_ranked_df = leads_scored_small_df \
    .sort_values('Score', ascending = False) \
    .assign(rank = lambda x: np.arange(0, len(x['made_purchase'])) + 1 ) \
    .assign(gain = lambda x: np.cumsum(x['made_purchase']) / np.sum(x['made_purchase']))
    
leads_ranked_df

# Threshold selection

thresh = 0.95

strategy_df = leads_ranked_df \
    .assign(category = lambda x: np.where(x['gain'] <= thresh, "Hot-Lead", "Cold-Lead"))
    
strategy_for_markeketing_df = leads_scored_df \
    .merge(
        right       = strategy_df[['category']],
        how         = 'left',
        left_index  = True,
        right_index = True
    )
    
strategy_for_markeketing_df

# 1.2 Aggregate Results

results_df = strategy_df \
    .groupby('category') \
    .agg(
        count = ('made_purchase', 'count'),
        sum_made_purchase = ('made_purchase', 'sum')
    )

results_df

# 2.0 CONFUSION MATRIX ANALYSIS ----

email_list_size = 1e5
unsub_rate_per_sales_email = 0.005
sales_emails_per_month = 5

avg_sales_per_month = 250000
avg_sales_emails_per_month = 5

customer_conversion_rate = 0.05
avg_customer_value = 2000

sample_factor = 5

# 2.1 Confusion Matrix Calculations ----

results_df

try:
    cold_lead_count = results_df['count']['Cold-Lead']
except:
    cold_lead_count = 0

try:
    hot_lead_count = results_df['count']['Hot-Lead']
except:
    hot_lead_count = 0

try:
    missed_purchases = results_df['sum_made_purchase']['Cold-Lead']
except:
    missed_purchases = 0
    
try:
    made_purchases = results_df['sum_made_purchase']['Hot-Lead']
except:
    made_purchases = 0
    


# 2.2 Confusion Matrix Summaries ----

total_count = (cold_lead_count + hot_lead_count)

total_purchases = (missed_purchases + made_purchases)

sample_factor = email_list_size / total_count

sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month

# 3.0 PRELIMINARY EXPECTED VALUE CALCULATIONS

results_df

# 3.1 [Savings] Cold That Are Not Targeted

savings_cold_no_target = cold_lead_count * \
    sales_emails_per_month * unsub_rate_per_sales_email * \
    customer_conversion_rate * avg_customer_value * \
    sample_factor

savings_cold_no_target

# 3.2 [Cost] Missed Sales That Are Not Targeted

missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)

cost_missed_purchases = sales_per_email_sent * sales_emails_per_month * missed_purchase_ratio

cost_missed_purchases


# 3.3 [Cost] Hot Leads Targeted That Unsubscribe

cost_hot_target_but_unsub = hot_lead_count * \
    sales_emails_per_month * unsub_rate_per_sales_email * \
    customer_conversion_rate * avg_customer_value * \
    sample_factor
    
cost_hot_target_but_unsub
    
# 3.4 [Savings] Sales Achieved

made_purchase_ratio = made_purchases / (missed_purchases + made_purchases)

savings_made_purchases = sales_per_email_sent * sales_emails_per_month * made_purchase_ratio

savings_made_purchases

# 4.0 FINAL EXPECTED VALUE TO REPORT TO MANAGEMENT

# 4.1 Expected Monthly Sales (Realized)

savings_made_purchases

# 4.2 Expected Monthly Value (Unrealized because of delayed nuture effect)

ev = savings_made_purchases + \
    savings_cold_no_target - cost_missed_purchases

# 4.3 Expected Monthly Savings (Unrealized until nurture takes effect)

es = savings_cold_no_target - cost_missed_purchases


# 4.4 Expected Saved Customers (Unrealized until nuture takes effect)

esc = savings_cold_no_target / avg_customer_value


# 4.5 EXPECTED VALUE SUMMARY OUTPUT

'${:,.0f}'.format(ev)

print(f"Expected Value: {'${:,.0f}'.format(ev)}")
print(f"Expected Savings: {'${:,.0f}'.format(es)}")
print(f"Monthly Sales: {'${:,.0f}'.format(savings_made_purchases)}")
print(f"Saved Customers: {'{:,.0f}'.format(esc)}")



# CONCLUSIONS -----
# - Can save a lot of money with this strategy
# - But there is a tradeoff with the threshold selected
# - Need to somehow optimize for threshold



