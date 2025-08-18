# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 8: FASTAPI 
# TEMPLATE: HOW TO TEST SECURITY WITH AN API KEY


import requests

url = "http://localhost:8000/secure-endpoint"
headers = {"X-API-KEY": "my-secret-api-key"}

# Not secure: No API Key
response = requests.get(url)
print(response.json())

# Secure: Pass API Key
response = requests.get(url, headers=headers)
print(response.json())