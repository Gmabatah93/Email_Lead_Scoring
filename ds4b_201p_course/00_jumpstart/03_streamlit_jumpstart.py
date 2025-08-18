# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT 
# STREAMLIT JUMPSTART
# ----

# To run app (put this in Terminal):
#   streamlit run 00_jumpstart/03_streamlit_jumpstart.py


# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px


# 1.0 Title and Introduction
st.title("Business Dashboard")
st.write("""
This dashboard provides insights into sales, customer demographics, and product performance. Upload your data to get started.
""")

# 2.0 Data Input
st.header("Upload Business Data")

uploaded_file = st.file_uploader("Choose a CSV File", type = "csv", accept_multiple_files=False)


# 3.0 App Body 
#  What Happens Once Data Is Loaded?

# data = pd.read_csv("00_jumpstart/data/sales.csv")

if uploaded_file:
    
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the Uploaded Data:")
    st.write(data.head())

    # * Sales insights
    st.header("Sales Insights")
    if 'sales_date' in data.columns and 'sales_amount' in data.columns: 
        st.write("Sales Over Time")
        fig = px.line(data, x = 'sales_date', y='sales_amount', title = 'Sales Over Time')
        st.plotly_chart(fig)
    else:
        st.warning("Please ensure your data has 'sales_date' and 'sales_amount' columns for sales visualization.")


    # * Customer Segmentation by Region
    st.header("Customer Segmentation")
    if 'region' in data.columns and 'sales_amount' in data.columns:
        st.write("Customer Segmentation")
        fig = px.pie(data, names = "region", values = 'sales_amount')
        st.plotly_chart(fig)
    else:
        st.warning("Please snsure your data has a 'region' column for customer segmentation.")

    # * Product Analysis
    st.header("Product Analysis")
    if 'product' in data.columns and 'sales_amount' in data.columns:
        st.write("Top Products by Sales")
        
        top_products_df = data.groupby('product').sum('sales_amount').nlargest(10, 'sales_amount')
        
        fig = px.bar(top_products_df, x = top_products_df.index, y='sales_amount', title = "Top Products By Sales")
        
        st.plotly_chart(fig)
        
    else:
        st.warning("Please ensure your data has 'product' and 'sales_amount' columns for product analysis.")
    


    # * Feedback Form
    st.header("Feedback (Your Opinion Counts)")
    feedback = st.text_area("Please provide any feedback or suggestions.")
    if st.button("Submit Feedback"):
        st.success('Thank you for your feedback.')


# 4.0 Footer
st.write("---")
st.write("This business dashboard template is flexible. Expand upon it based on your specific business needs.")


if __name__ == "__main__":
    pass