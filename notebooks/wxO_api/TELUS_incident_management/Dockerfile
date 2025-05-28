# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the rest of the application
COPY . .

# Expose port 8080 for Uvicorn
EXPOSE 8080

# Launch the FastAPI app
CMD ["uvicorn", "FastAPI_Analysis_and_XML:app", "--host", "0.0.0.0", "--port", "8080"]