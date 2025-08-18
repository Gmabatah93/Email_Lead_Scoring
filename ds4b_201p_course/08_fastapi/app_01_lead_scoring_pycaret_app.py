# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 8: FASTAPI 
# PART 1: BUILDING THE API
# ----

# To Run this App:
# - Open Terminal
# - uvicorn 08_fastapi.app_01_lead_scoring_pycaret_app:app --reload --port 8000
# - Navigate to localhost:8000
# - Navigate to localhost:8000/docs
# - Shutdown App: Ctrl/Cmd + C

# PACKAGES

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import json

import pandas as pd
import email_lead_scoring as els


app = FastAPI()

# DATA
leads_df = els.db_read_and_process_els_data()

# 0.0 INTRODUCE YOUR API
@app.get("/")
async def main():
    
    content = """
    <body>
    <h1>Welcome to the Email Lead Scoring Project</h1>
    <p>This API helps users score leads using our proprietary lead scoring models.</p>
    <p>Navigate to the <code>/docs</code> to see the API documentation.</p>
    </body>
    """
    
    return HTMLResponse(content = content)


# 1.0 GET: EXPOSE THE EMAIL SUBSCRIBER DATA AS AN ENDPOINT
@app.get("/get_email_subscribers")
async def get_email_subscribers():
    
    json = leads_df.to_json()
    
    return JSONResponse(json)
    
    
    
# 2.0 POST: PASSING DATA TO AN API
@app.post("/data")
async def data(request: Request):
    
    request_body = await request.body()
    
    # print(request_body)
    
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json)
    
    # print(leads_df)
    
    leads_json = leads_df.to_json()
    
    return JSONResponse(leads_json)
    
    

# 3.0 MAKING PREDICTIONS FROM AN API
@app.post("/predict")
async def predict(request: Request):
    
    # Handle incoming JSON request
    request_body = await request.body()
    
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json) 
    
    # Load model
    leads_scored_df = els.model_score_leads(
        data = leads_df,
        model_path="models/xgb_model_tuned"
    )
    
    # print(leads_scored_df)
    
    # Convert to JSON
    scores = leads_scored_df[['Score']].to_dict()
    
    # print(scores)
    
    return JSONResponse(scores)


# 4.0 POST: MAKE LEAD SCORING STRATEGY
@app.post("/calculate_lead_strategy")
async def calculate_lead_strategy(
    request: Request,
    monthly_sales_reduction_safe_gaurd: float=0.9,
    email_list_size:int=100000,
    unsub_rate_per_sales_email:float=0.005,
    sales_emails_per_month:int=5,
    avg_sales_per_month: float=250000.0,
    avg_sales_emails_per_month: int=5,
    customer_conversion_rate: float=0.05,
    avg_customer_value: float=2000.0,
):
    
    # Handle incoming JSON request
    request_body = await request.body()
    
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json) 
    
    # Load model
    leads_scored_df = els.model_score_leads(
        data = leads_df,
        model_path="models/xgb_model_tuned"
    )
    
    # Optimization Results
    optimization_results = els.lead_score_strategy_optimization(
        leads_scored_df=leads_scored_df,
        
        monthly_sales_reduction_safe_gaurd=monthly_sales_reduction_safe_gaurd,
        
        email_list_size=email_list_size,
        unsub_rate_per_sales_email=unsub_rate_per_sales_email,
        sales_emails_per_month=sales_emails_per_month,
        avg_sales_per_month=avg_sales_per_month,
        avg_sales_emails_per_month=avg_sales_emails_per_month,
        customer_conversion_rate= customer_conversion_rate,
        avg_customer_value=avg_customer_value,
        
    )
    
    # print(optimization_results)
    
    results = {
        'lead_strategy': optimization_results['lead_strategy_df'].to_json(),
        'expected_value': optimization_results['expected_value'].to_json(),
        'thresh_optim_table': optimization_results['thresh_optim_df'].data.to_json()
    }
    
    return JSONResponse(results)



# DEFINE THE API PORT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
