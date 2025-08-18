# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 7: ROI  
# PART 2: THRESHOLD OPTIMIZATION & PROFIT MAXIMIZATION
# ----

# LIBRARIES -----

import pandas as pd
import numpy as np
import plotly.express as px
import email_lead_scoring as els

# RECAP ----

leads_df = els.db_read_and_process_els_data()

leads_scored_df = els.model_score_leads(leads_df)


# 1.0 MAKE THE LEAD STRATEGY FROM THE SCORED SUBSCRIBERS:
#   lead_make_strategy()

def lead_make_strategy(leads_scored_df, thresh = 0.99, for_marketing_team = False, verbose = False):
    
    # Ranking the leads
    leads_scored_small_df = leads_scored_df[['user_email', 'Score', 'made_purchase']]

    leads_ranked_df = leads_scored_small_df \
        .sort_values('Score', ascending = False) \
        .assign(rank = lambda x: np.arange(0, len(x['made_purchase'])) + 1 ) \
        .assign(gain = lambda x: np.cumsum(x['made_purchase']) / np.sum(x['made_purchase']))
        
    # Make the Strategy
    strategy_df = leads_ranked_df \
        .assign(category = lambda x: np.where(x['gain'] <= thresh, "Hot-Lead", "Cold-Lead"))
    
    if for_marketing_team:
        strategy_df = leads_scored_df \
            .merge(
                right       = strategy_df[['category']],
                how         = 'left',
                left_index  = True,
                right_index = True
            )
    
    if verbose:
        print("Strategy Created.")
        
    return strategy_df
    

# Workflow

lead_make_strategy(
    leads_scored_df,
    thresh = 0.90,
    for_marketing_team = False   
)


# 2.0 AGGREGATE THE LEAD STRATEGY RESULTS
#  lead_aggregate_strategy_results()

def lead_aggregate_strategy_results(strategy_df):
    
    results_df = strategy_df \
        .groupby('category') \
        .agg(
            count = ('made_purchase', 'count'),
            sum_made_purchase = ('made_purchase', 'sum')
        )
    
    return results_df

# Workflow

lead_make_strategy(leads_scored_df, thresh=0.90) \
    .pipe(lead_aggregate_strategy_results)

    
# 3.0 CALCULATE EXPECTED VALUE (FOR ONE SET OF INPUTS) ----
#  lead_strategy_calc_expected_value()

def lead_strategy_calc_expected_value(
    results_df,
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.001,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    verbose = False,
):
    
    # Confusion matrix calculations ----
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
        
    # Confusion matrix summaries
    
    total_count = (cold_lead_count + hot_lead_count)

    total_purchases = (missed_purchases + made_purchases)

    sample_factor = email_list_size / total_count

    sales_per_email_sent = avg_sales_per_month / avg_sales_emails_per_month
    
    # Preliminary Expected Value Calcuations
    
    # 3.1 [Savings] Cold That Are Not Targeted

    savings_cold_no_target = cold_lead_count * \
        sales_emails_per_month * unsub_rate_per_sales_email * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
        
    # 3.2 [Cost] Missed Sales That Are Not Targeted

    missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)

    cost_missed_purchases = sales_per_email_sent * sales_emails_per_month * missed_purchase_ratio
    
    # 3.3 [Cost] Hot Leads Targeted That Unsubscribe

    cost_hot_target_but_unsub = hot_lead_count * \
        sales_emails_per_month * unsub_rate_per_sales_email * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
        
    # 3.4 [Savings] Sales Achieved

    made_purchase_ratio = made_purchases / (missed_purchases + made_purchases)

    savings_made_purchases = sales_per_email_sent * sales_emails_per_month * made_purchase_ratio
    
    # 4.2 Expected Monthly Value (Unrealized because of delayed nuture effect)

    ev = savings_made_purchases + \
        savings_cold_no_target - cost_missed_purchases

    # 4.3 Expected Monthly Savings (Unrealized until nurture takes effect)

    es = savings_cold_no_target - cost_missed_purchases


    # 4.4 Expected Saved Customers (Unrealized until nuture takes effect)

    esc = savings_cold_no_target / avg_customer_value
    
    if verbose:
        print(f"Expected Value: {'${:,.0f}'.format(ev)}")
        print(f"Expected Savings: {'${:,.0f}'.format(es)}")
        print(f"Monthly Sales: {'${:,.0f}'.format(savings_made_purchases)}")
        print(f"Saved Customers: {'{:,.0f}'.format(esc)}")
        
    return {
        'expected_value': ev,
        'expected_savings': es,
        'monthly_sales': savings_made_purchases,
        'expected_customers_saved': esc
    }
        
    

# Workflow:

lead_make_strategy(
    leads_scored_df,
    thresh = 0.95,
    verbose = True
) \
    .pipe(
        lead_aggregate_strategy_results
    ) \
    .pipe(
        lead_strategy_calc_expected_value,
        email_list_size = 1e5,
        unsub_rate_per_sales_email = 0.005,
        sales_emails_per_month = 5,
        avg_sales_per_month = 250000,
        avg_sales_emails_per_month = 5,
        customer_conversion_rate = 0.05,
        avg_customer_value = 2000,
        verbose = False,
    )

# 4.0 OPTIMIZE THE THRESHOLD AND GENERATE A TABLE
#  lead_strategy_create_thresh_table()

def lead_strategy_create_thresh_table(
    leads_scored_df,
    thresh = np.linspace(0, 1, num=100),
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.005,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    highlight_max = True,
    highlight_max_color = "yellow",
    verbose = False,
):
    
    
    thresh_df = pd.Series(thresh, name = "thresh").to_frame()
    
    # List comprehension
    # [tup[0] for tup in zip(thresh_df['thresh'])]
    sim_results_list = [
        lead_make_strategy(
            leads_scored_df,
            thresh = tup[0],
            verbose = verbose
        ) \
            .pipe(
                lead_aggregate_strategy_results
            ) \
            .pipe(
                lead_strategy_calc_expected_value,
                email_list_size = email_list_size,
                unsub_rate_per_sales_email = unsub_rate_per_sales_email,
                sales_emails_per_month = sales_emails_per_month,
                avg_sales_per_month = avg_sales_per_month,
                avg_sales_emails_per_month = avg_sales_emails_per_month,
                customer_conversion_rate = customer_conversion_rate,
                avg_customer_value = avg_customer_value,
                verbose = verbose,
            )
        for tup in zip(thresh_df['thresh'])
    ]
    
    sim_results_df = pd.Series(sim_results_list, name = "sim_results").to_frame()
    
    sim_results_df = sim_results_df['sim_results'].apply(pd.Series)
    
    thresh_optim_df = pd.concat([thresh_df, sim_results_df], axis=1)
    
    if highlight_max:
        thresh_optim_df = thresh_optim_df.style.highlight_max(
            color = highlight_max_color,
            axis  = 0
        )
    
    return thresh_optim_df
    


# Workflow:

thresh_optim_df = lead_strategy_create_thresh_table(
    leads_scored_df     = leads_scored_df,
    thresh              = np.linspace(0,1, num = 100),
    highlight_max_color = "green",
    verbose             = True
)


# 5.0 SELECT THE BEST THRESHOLD
#  def lead_select_optimum_thresh()

def lead_select_optimum_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    monthly_sales_reduction_safe_gaurd = 0.90,
    verbose = False
):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
        
    # Find optim
    _filter_1 = thresh_optim_df[optim_col] == thresh_optim_df[optim_col].max()
    
    # Find safegaurd
    _filter_2 = thresh_optim_df['monthly_sales'] >= monthly_sales_reduction_safe_gaurd * thresh_optim_df['monthly_sales'].max()
    
    # Test i optim is in the safegaurd
    if (all(_filter_1 + _filter_2 == _filter_2)):
        _filter = _filter_1
    else:
        _filter = _filter_2
        
    # Apply filter
    thresh_selected = thresh_optim_df[_filter].head(1)
    
    # Values
    ret = thresh_selected['thresh'].values[0]
    
    if verbose:
        print(f"Optimal threshold: {ret}")
        
    return ret 

# Workflow


thresh_optim_df = lead_strategy_create_thresh_table(
    leads_scored_df     = leads_scored_df,
    thresh              = np.linspace(0,1, num = 100),
    highlight_max_color = "green",
    verbose             = True
)

thresh_optim = lead_select_optimum_thresh(
    thresh_optim_df,
    monthly_sales_reduction_safe_gaurd = 0.90
)

# 6.0 GET EXPECTED VALUE RESULTS ----
#  def lead_get_expected_value()

def lead_get_expected_value(thresh_optim_df, threshold = 0.85, verbose = False):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
        
    df = thresh_optim_df[thresh_optim_df.thresh >= threshold].head(1)
    
    if verbose:
        print("Expected Value Table:")
        print(df)
        
    return df
    
# Workflow 

lead_get_expected_value(
    thresh_optim_df=thresh_optim_df,
    threshold = lead_select_optimum_thresh(
        thresh_optim_df=thresh_optim_df,
        monthly_sales_reduction_safe_gaurd= 0.90
    )
)   


# 7.0 PLOT THE OPTIMAL THRESHOLD ----
#  def lead_plot_optim_thresh()

thresh_optim_df

fig = px.line(
    thresh_optim_df, 
    x = 'thresh',
    y = 'expected_value'
)

fig.add_hline(y = 0, line_color = 'black')

fig.add_vline(
    x = lead_select_optimum_thresh(thresh_optim_df),
    line_color = "red",
    line_dash  = "dash"
)

fig

def lead_plot_optim_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    monthly_sales_reduction_safegaurd = 0.90,
    verbose = False
):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
        
    # Make the plog
    fig = px.line(
        thresh_optim_df, 
        x = 'thresh',
        y = optim_col
    )

    fig.add_hline(y = 0, line_color = 'black')

    fig.add_vline(
        x = lead_select_optimum_thresh(
            thresh_optim_df, 
            optim_col=optim_col,    
            monthly_sales_reduction_safe_gaurd=monthly_sales_reduction_safegaurd
        ),
        line_color = "red",
        line_dash  = "dash"
    )
    
    if verbose:
        print("Plot created.")
        
    return fig
    


# Workflow

lead_plot_optim_thresh(
    thresh_optim_df,
    optim_col='expected_value',
    monthly_sales_reduction_safegaurd=0.70,
    verbose = True
)


# 8.0 MAKE THE OPTIMAL STRATEGY ----



# FINAL OPTIMIZATION RESULTS ----
# - Use one function to perform the automation and collect the results
#   def lead_score_strategy_optimization()


def lead_score_strategy_optimization(
    leads_scored_df,
    
    thresh = np.linspace(0, 1, num=100),
    optim_col = "expected_value",
    monthly_sales_reduction_safe_gaurd = 0.90,
    for_marketing_team = True,
    
    email_list_size = 1e5,
    unsub_rate_per_sales_email = 0.005,
    sales_emails_per_month = 5,
    avg_sales_per_month = 250000,
    avg_sales_emails_per_month = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    highlight_max = True,
    highlight_max_color = "yellow",
    
    verbose = False,
):
    
    # Lead strategy create thresh table
    thresh_optim_df = lead_strategy_create_thresh_table(
        leads_scored_df=leads_scored_df,
        thresh=thresh,
        email_list_size=email_list_size,
        unsub_rate_per_sales_email = unsub_rate_per_sales_email,
        sales_emails_per_month = sales_emails_per_month,
        avg_sales_per_month = avg_sales_per_month,
        avg_sales_emails_per_month = avg_sales_emails_per_month,
        customer_conversion_rate = customer_conversion_rate,
        avg_customer_value = avg_customer_value,
        highlight_max = highlight_max,
        highlight_max_color = highlight_max_color,
        verbose=verbose
    )
    
    # Lead select optimum thresh
    thresh_optim = lead_select_optimum_thresh(
        thresh_optim_df,
        optim_col = optim_col,
        monthly_sales_reduction_safe_gaurd = monthly_sales_reduction_safe_gaurd,
        verbose = verbose
    )
    
    # Expected value
    expected_value = lead_get_expected_value(
        thresh_optim_df, 
        threshold = thresh_optim, 
        verbose = verbose
    )
    
    # Lead plot
    
    thresh_plot = lead_plot_optim_thresh(
        thresh_optim_df,
        optim_col = optim_col,
        monthly_sales_reduction_safegaurd = monthly_sales_reduction_safe_gaurd,
        verbose = verbose
    )
    
    # Recalculate Lead Strategy
    lead_strategy_df = lead_make_strategy(
        leads_scored_df, 
        thresh             = thresh_optim, 
        for_marketing_team = for_marketing_team, 
        verbose            = verbose
    )    
    
    # Dictionary for return
    
    ret = dict(
        lead_strategy_df = lead_strategy_df,
        expected_value   = expected_value,
        thresh_optim_df  = thresh_optim_df,
        thresh_plot      = thresh_plot
    )
    
    return(ret)


# Workflow

optimization_results = lead_score_strategy_optimization(
    leads_scored_df=leads_scored_df,
    monthly_sales_reduction_safe_gaurd=0.90,
    verbose = True
)

optimization_results.keys()

optimization_results['lead_strategy_df']

optimization_results['expected_value']

optimization_results['thresh_optim_df']

optimization_results['thresh_plot']

# CONCLUSIONS ----

# Business Leaders may freak out if they see a big hit in sales
#  even though it could be beneficial in long term
# Recommend to start with a smaller strategy of focusing on 
#  top 95% 
# We can set this up with the Threshold Override

