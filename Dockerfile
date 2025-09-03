# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main application file
COPY app_fast.py .

# Create the scripts directory inside the container and copy only the necessary scripts
RUN mkdir scripts
COPY scripts/utils.py scripts/
COPY scripts/data_preprocess.py scripts/

# Copy models
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with uvicorn from the root directory
CMD ["uvicorn", "app_fast:app", "--host", "0.0.0.0", "--port", "8000"]