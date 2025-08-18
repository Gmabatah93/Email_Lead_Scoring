# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----


# TEST CALCULATIONS ----

import email_lead_scoring as els

# DON'T NEED TO DO THIS IF INIT.PY FILE IMPORTS 
# import email_lead_scoring.cost_calculations as cost
# cost.cost_calc_monthly_cost_table()

# ?cost.cost_calc_monthly_cost_table

# Once __init__.py file imports the submodule function
els.cost_calc_monthly_cost_table()

# ?els.cost_total_unsub_cost

els.cost_calc_monthly_cost_table() \
    .cost_total_unsub_cost()
    
    
# ?els.cost_simulate_unsub_costs
els.cost_simulate_unsub_costs()

# ?els.cost_plot_simulated_unsub_costs

els.cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=[0, 0.015, 0.025, 0.035]
) \
    .cost_plot_simulated_unsub_costs()


