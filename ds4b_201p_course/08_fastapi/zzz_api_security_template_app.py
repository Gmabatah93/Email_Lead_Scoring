# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 8: FASTAPI 
# TEMPLATE: HOW TO ADD SECURITY TO AN API

# To Run this App:
# - Open Terminal
# - uvicorn 08_fastapi.zzz_api_security_template_app:app --reload
# - Navigate to localhost:8000
# - Navigate to localhost:8000/docs
# - Shutdown App: Ctrl/Cmd + C

# 1.0 APP ----

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# Define the name of the header that will contain the API key
API_KEY_NAME = "X-API-KEY"

# Define the expected API key value (this should be kept secret)
API_KEY = "my-secret-api-key"

# Create an instance of the APIKeyHeader class
api_key_header = APIKeyHeader(
    name=API_KEY_NAME, 
    auto_error=True
)

# Define a dependency that will check the API key in the request header
async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    return api_key

# Create an instance of the FastAPI application
app = FastAPI()

# Define a route that requires the API key
@app.get("/secure-endpoint", dependencies=[Depends(get_api_key)])
async def secure_endpoint():
    return {"message": "This is a secure endpoint"}

# Define a route that does not require the API key
@app.get("/public-endpoint")
async def public_endpoint():
    return {"message": "This is a public endpoint"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)