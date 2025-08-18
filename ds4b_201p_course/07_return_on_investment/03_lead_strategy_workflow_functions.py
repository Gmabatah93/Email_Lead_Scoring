# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI 
# PART 3: LEAD STRATEGY FUNCTIONAL WORKFLOW
# ----

# IMPORTS
import email_lead_scoring as els

# WORKFLOW ----

leads_df = els.db_read_and_process_els_data()
leads_df

leads_scored_df = els.model_score_leads(leads_df)
leads_scored_df

# CREATE FUNCTIONS ----
#  els > lead_strategy.py

# ?els.lead_score_strategy_optimization

optimization_results = els.lead_score_strategy_optimization(
    leads_scored_df
)

optimization_results.keys()

keys_list = list(optimization_results.keys())

optimization_results[keys_list[0]]

optimization_results[keys_list[1]]

optimization_results[keys_list[2]]

optimization_results[keys_list[3]]
