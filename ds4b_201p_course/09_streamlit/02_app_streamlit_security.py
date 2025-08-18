# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT 
# FRONTEND USER STREAMLIT APP FOR API - SECURITY ENABLED
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/02_app_streamlit_security.py

import streamlit as st
import requests
import pandas as pd

import sys
import pathlib

# NEEDED FOR EMAIL LEAD SCORING TO BE DETECTED
# APPEND PROJECT DIRECTORY TO PYTHONPATH
working_dir = str(pathlib.Path().absolute())
print(working_dir)
sys.path.append(working_dir)

import email_lead_scoring as els

ENDPOINT = 'http://localhost:8000'

AUTHORIZED_API_KEYS = {"X-API-KEY": "my-secret-api-key"}

# Initialize or access the session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Only show authentication UI if the user is not authenticated
if not st.session_state.authenticated:
    st.title('API Key Authentication')
    user_name = st.text_input("Enter User Name")
    user_input_key = st.text_input("Enter API Key", type="password")
    
    # Store the user name and password in the session state
    st.session_state.user_name = user_name
    st.session_state.user_input_key = user_input_key

    if st.button('Authenticate'):
        # Step 3: Verification
        if user_name in AUTHORIZED_API_KEYS.keys() and user_input_key == AUTHORIZED_API_KEYS[user_name]:
            st.session_state.authenticated = True
            st.balloons()
            st.write("Redirecting to the main application...")
            st.experimental_rerun()  # This refreshes the app to ensure the authentication UI elements aren't displayed.
        else:
            st.error("Invalid API Key. Please try again or contact support.")
else:

    HEADERS = {st.session_state.user_name : st.session_state.user_input_key}
    
    # 1.0 TITLE

    st.title("Email Lead Scoring Streamlit Frontend App")

    st.text(f"User Name: {st.session_state.user_name}")

    # 2.0 DATA INPUT
    #   CACHING DATA - NEEDED TO PREVENT REQUIRING DATA TO BE RE-INPUT

    uploaded_file = st.file_uploader(
        "Upload Email Subscribers File",
        type = ['csv'],
        accept_multiple_files= False
    )

    @st.cache_data()
    def load_data(filename):
        leads_df = pd.read_csv(uploaded_file)
        return leads_df


    # 3.0 APP BODY
    if uploaded_file:

        leads_df = load_data(uploaded_file)
        full_data_json = leads_df.to_json()
        
        # Checkbox - Show Table 
        if st.checkbox("Show raw data"):
            st.subheader("Sample of Raw Data (Fisrt 10 Rows)")
            st.write(leads_df.head(10))

        
        st.write("---")
        st.markdown("# Lead Scoring Analysis")
        
        # User Inputs - Add Sliders / Buttons
        estimated_monthly_sales = st.number_input("How much in email sales per month ($ on average)", 0, value = 250000, step = 1000)
        
        monthly_sales_reduction_safe_guard = st.slider(
            "How much of the monthly sales should be maintained (%)?",
            0., 1., 0.9, step = 0.01
        )
        # print(monthly_sales_reduction_safe_guard)
        
        sales_limit = "${:,.0f}".format(monthly_sales_reduction_safe_guard*estimated_monthly_sales)
        
        st.subheader(f"Month sales will not go below: {sales_limit}")
        
        
        # Run Analysis 
        
        if st.button("Run Analysis"):
        
            # Spinner
            with st.spinner("Lead scoring in progress. Almost done..."):
            
                # Make Request
                res = requests.post(
                    url = f'{ENDPOINT}/calculate_lead_strategy',
                    json = full_data_json,
                    params = dict(
                        monthly_sales_reduction_safe_gaurd=float(monthly_sales_reduction_safe_guard),
                        email_list_size=100000,
                        unsub_rate_per_sales_email=0.005,
                        sales_emails_per_month=5,
                        avg_sales_per_month=float(estimated_monthly_sales),
                        avg_sales_emails_per_month=5,
                        customer_conversion_rate=0.05,
                        avg_customer_value=2000.0,
                    ),
                    headers=HEADERS
                )
                
                # Collect JSON / Convert Data\
                
                print(res.json().keys())
                
                lead_strategy_df = pd.read_json(res.json()['lead_strategy'])
                
                expected_value_df = pd.read_json(res.json()['expected_value'])
                
                thresh_optim_table_df = pd.read_json(res.json()['thresh_optim_table'])
                
                print(thresh_optim_table_df.head(10))
                
                
                # Display Results
                st.success("Success! Lead Scoring is complete. Download the results below.")
                
                # Display Strategy Summary
                st.subheader("Lead Strategy Summary:")
                st.write(expected_value_df)
                
                
                # Display Expected Value Plot
                st.subheader("Expected Value Plot")
                st.plotly_chart(
                    els.lead_plot_optim_thresh(
                        thresh_optim_table_df,
                        monthly_sales_reduction_safegaurd=monthly_sales_reduction_safe_guard
                    )
                )
                
                
                # Display Sample Lead Strategy
                st.subheader("Sample of Lead Strategy (First 10 Rows)")
                st.write(lead_strategy_df.head(10))
                
                
                # Download button - Get lead scoring results
                st.download_button(
                    label = "Download Lead Scoring Strategy",
                    data = lead_strategy_df.to_csv(index = False),
                    file_name='lead_strategy.csv',
                    mime="text/csv",
                    key = "download-csv"
                )
            


        
    
