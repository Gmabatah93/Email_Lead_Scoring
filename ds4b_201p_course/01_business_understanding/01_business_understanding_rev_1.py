# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----

# LIBRARIES ----

import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px


# BUSINESS SCIENCE PROBLEM FRAMEWORK ----

# View Business as a Machine ----


# Business Units: 
#   - Marketing Department
#   - Responsible for sales emails  
# Project Objectives:
#   - Target Subscribers Likely To Purchase
#   - Nurture Subscribers to take actions that are known to increase probability of purchase
# Define Machine:
#   - Marketing sends out email blasts to everyone
#   - Generates Sales
#   - Also ticks some members off
#   - Members unsubscribe, this reduces email growth and profitability
# Collect Outcomes:
#   - Revenue has slowed, Email growth has slowed


# Understand the Drivers ----

#   - Key Insights:
#     - Company has Large Email List: 100,000 
#     - Email list is growing at 6,000/month less 2500 unsub for total of 3500
#     - High unsubscribe rates: 500 people per sales email
#   - Revenue:
#     - Company sales cycle is generating about $250,000 per month
#     - Average Customer Lifetime Value: Estimate $2000/customer
#   - Costs: 
#     - Marketing sends 5 Sales Emails Per Month
#     - 5% of lost customers likely to convert if nutured



# COLLECT OUTCOMES ----

email_list_size_1 = 100000

unsub_count_per_sales_email_1 = 500

unsub_rate_1 = unsub_count_per_sales_email_1 / email_list_size_1
unsub_rate_1

sales_emails_per_month_1 = 5

conversion_rate_1 = 0.05

lost_customer_1 = email_list_size_1 * unsub_rate_1 * sales_emails_per_month_1 * conversion_rate_1

average_customer_value_1 = 2000

lost_revenue_per_month_1 = lost_customer_1 * average_customer_value_1



# No-growth scenario $3M cost

cost_no_growth_1 = lost_revenue_per_month_1 * 12
cost_no_growth_1

# Growth scenario: 
#   amount = principle * ((1+rate)**time)

growth_rate = 3500 / 100000

100000 * ((1+growth_rate) ** 0)

100000 * ((1+growth_rate) ** 1)

100000 * ((1+growth_rate) ** 6)

100000 * ((1+growth_rate) ** 11)


# Cost Table 

time = 12

# Period

period_series = pd.Series(np.arange(0,12), name="period")

len(period_series)

cost_table_df = period_series.to_frame()

# Email Size - No Growth

cost_table_df['email_size_no_growth'] = np.repeat(email_list_size_1, time)

cost_table_df

# Lost Customers - No Growth

# *** FIX ***
#  Missed the conversion_rate_1

# WRONG:
# cost_table_df['lost_customers_no_growth'] = cost_table_df['email_size_no_growth'] * unsub_rate_1 * sales_emails_per_month_1

# CORRECT:
cost_table_df['lost_customers_no_growth'] = cost_table_df['email_size_no_growth'] * unsub_rate_1 * sales_emails_per_month_1 * conversion_rate_1

cost_table_df

# Cost - No Growth

# WRONG:
# cost_table_df['cost_no_growth'] = cost_table_df['lost_customers_no_growth'] * conversion_rate_1 * average_customer_value_1

# CORRECT: 
cost_table_df['cost_no_growth'] = cost_table_df['lost_customers_no_growth'] * average_customer_value_1

cost_table_df

# Email Size - With Growth

cost_table_df['email_size_with_growth'] = cost_table_df['email_size_no_growth'] * ((1+growth_rate) ** cost_table_df['period'])

cost_table_df

px.line(
    data_frame=cost_table_df,
    y = ['email_size_no_growth', 'email_size_with_growth']
) \
    .add_hline(y = 0)
    
# Lost Customers - With Growth

# WRONG: 
# cost_table_df['lost_customers_with_growth'] = cost_table_df['email_size_with_growth'] * unsub_rate_1 * sales_emails_per_month_1

# CORRECT:
cost_table_df['lost_customers_with_growth'] = cost_table_df['email_size_with_growth'] * unsub_rate_1 * sales_emails_per_month_1 * conversion_rate_1

cost_table_df

# Cost - With Growth

# WRONG:
# cost_table_df['cost_with_growth'] = cost_table_df['lost_customers_with_growth'] * conversion_rate_1 * average_customer_value_1

# CORRECT:
cost_table_df['cost_with_growth'] = cost_table_df['lost_customers_with_growth'] * average_customer_value_1

cost_table_df

px.line(
    cost_table_df,
    y = ['cost_no_growth', 'cost_with_growth']
) \
    .add_hline(y = 0)

# Compare Cost - With / No Growth

cost_table_df[ ['cost_no_growth', 'cost_with_growth'] ] \
    .sum()

3.65 / 3


# If reduce unsubscribe rate by 30%

cost_table_df['cost_no_growth'].sum() * 0.30

cost_table_df['cost_with_growth'].sum() * 0.30



# COST CALCULATION FUNCTIONS ----

# Function: Calculate Monthly Unsubscriber Cost Table ----

def cost_calc_monthly_cost_table(
    email_list_size = 1e5,
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
):
    
    # Period
    period_series = pd.Series(np.arange(0, n_periods), name="period")
    
    cost_table_df = period_series.to_frame()
    
    # Email Size - No Growth

    cost_table_df['email_size_no_growth'] = np.repeat(email_list_size, n_periods)
    
    # Lost Customers - No Growth

    cost_table_df['lost_customers_no_growth'] = cost_table_df['email_size_no_growth'] * unsub_rate_per_sales_email * sales_emails_per_month  
    
    # Cost - No Growth

    cost_table_df['cost_no_growth'] = cost_table_df['lost_customers_no_growth'] * customer_conversion_rate * average_customer_value
    
    # Email Size - With Growth

    cost_table_df['email_size_with_growth'] = cost_table_df['email_size_no_growth'] * ((1+email_list_growth_rate) ** cost_table_df['period'])
    
    # Lost Customers - With Growth

    cost_table_df['lost_customers_with_growth'] = cost_table_df['email_size_with_growth'] * unsub_rate_per_sales_email * sales_emails_per_month
    
    # Cost - With Growth

    cost_table_df['cost_with_growth'] = cost_table_df['lost_customers_with_growth'] * customer_conversion_rate * average_customer_value
        
    
    return cost_table_df
    
cost_calc_monthly_cost_table(
    email_list_size= 50000,
    sales_emails_per_month=1,
    unsub_rate_per_sales_email=0.001,
    n_periods=24
)

# Function: Sumarize Cost ----

cost_table_df[ ['cost_no_growth', 'cost_with_growth'] ] \
    .sum() \
    .to_frame() \
    .transpose()
    
def cost_total_unsub_cost(cost_table):
    
    summary_df = cost_table[ ['cost_no_growth', 'cost_with_growth'] ] \
        .sum() \
        .to_frame() \
        .transpose()
    
    return summary_df


cost_total_unsub_cost(cost_table_df)

# ARE OBJECTIVES BEING MET?
# - We can see a large cost due to unsubscription
# - However, some attributes may vary causing costs to change


# SYNTHESIZE OUTCOMES (COST SIMULATION) ----
# - Make a cartesian product of inputs that can vary
# - Use list comprehension to perform simulation
# - Visualize results

# Cartesian Product (Expand Grid)

# ?jn.expand_grid

data_dict = dict(
    email_list_monthly_growth_rate = np.linspace(0, 0.05, num = 10),
    customer_conversion_rate = np.linspace(0.04, 0.06, num = 3)
)

parameter_grid_df = jn.expand_grid(others=data_dict)

# List Comprehension

def temporary_function(x, y):
    
    cost_table_df = cost_calc_monthly_cost_table(
        email_list_growth_rate=x,
        customer_conversion_rate=y
    )
    
    summary_df = cost_total_unsub_cost(cost_table_df)
    
    return summary_df

temporary_function(0.10, y=0.10)


summary_list = [temporary_function(x, y) for x, y in zip(
    parameter_grid_df['email_list_monthly_growth_rate'], 
    parameter_grid_df['customer_conversion_rate']
)]

simulation_results_df = pd.concat(summary_list, axis = 0) \
    .reset_index() \
    .drop('index', axis = 1) \
    .merge(parameter_grid_df, left_index=True, right_index=True)

# Function

def cost_simulate_unsub_costs(
    email_list_monthly_growth_rate = [0, 0.035],
    customer_conversion_rate = [0.04, 0.05, 0.06],
    **kwargs
):
    
    # Parameter Grid
    data_dict = dict(
        email_list_monthly_growth_rate = email_list_monthly_growth_rate,
        customer_conversion_rate = customer_conversion_rate
    )

    parameter_grid_df = jn.expand_grid(others=data_dict)
    
    # Temporary Function
    
    # List Comprehension

    def temporary_function(x, y):
        
        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate=x,
            customer_conversion_rate=y,
            **kwargs
        )
        
        summary_df = cost_total_unsub_cost(cost_table_df)
        
        return summary_df
    
    # List Comprehension
    summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df['email_list_monthly_growth_rate'], parameter_grid_df['customer_conversion_rate'])]

    simulation_results_df = pd.concat(summary_list, axis = 0) \
        .reset_index() \
        .drop('index', axis = 1) \
        .merge(parameter_grid_df, left_index=True, right_index=True)
    
    
        
    return simulation_results_df

cost_simulate_unsub_costs()

# VISUALIZE COSTS

simulation_results_wide_df = cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=[0.01, 0.02],
    customer_conversion_rate=[0.04, 0.06],
    email_list_size = 100000
) \
    .drop('cost_no_growth', axis=1) \
    .pivot(
        index   = 'email_list_monthly_growth_rate',
        columns = 'customer_conversion_rate',
        values  ='cost_with_growth'
    )

# ?px.imshow

px.imshow(
    simulation_results_wide_df,
    origin='lower',
    aspect = 'auto',
    title = "Lead Cost Simulation",
    labels = dict(
        x = 'Customer Conversion Rate',
        y = 'Monthly Email Growth Rate',
        color = 'Cost of Unsubscription'
    )
)

# Function: Plot Simulated Unsubscriber Costs

def cost_plot_simulated_unsub_costs(simulation_results):
    
    simulation_results_wide_df = simulation_results \
        .drop('cost_no_growth', axis=1) \
        .pivot(
            index   = 'email_list_monthly_growth_rate',
            columns = 'customer_conversion_rate',
            values  ='cost_with_growth'
        )
        
    fig = px.imshow(
        simulation_results_wide_df,
        origin='lower',
        aspect = 'auto',
        title = "Lead Cost Simulation",
        labels = dict(
            x = 'Customer Conversion Rate',
            y = 'Monthly Email Growth Rate',
            color = 'Cost of Unsubscription'
        )
    )
    
    return fig


cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=[0.01, 0.02, 0.03],
    customer_conversion_rate=[0.04, 0.05, 0.06],
    email_list_size = 100000
) \
    .pipe(cost_plot_simulated_unsub_costs)




# ARE OBJECTIVES BEING MET?
# - Even with simulation, we see high costs
# - What if we could reduce by 30% through better targeting?



# - What if we could reduce unsubscribe rate from 0.5% to 0.17% (marketing average)?
# - Source: https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-unsubscribe-rate/



# HYPOTHESIZE DRIVERS

# - What causes a customer to convert of drop off?
# - If we know what makes them likely to convert, we can target the ones that are unlikely to nurture them (instead of sending sales emails)
# - Meet with Marketing Team
# - Notice increases in sales after webinars (called Learning Labs)
# - Next: Begin Data Collection and Understanding



