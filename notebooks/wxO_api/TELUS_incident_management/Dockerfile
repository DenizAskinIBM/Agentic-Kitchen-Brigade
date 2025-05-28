ARG REGISTRY
ARG VERSION
FROM ${REGISTRY}/agentic-asset-base:${VERSION}

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Python can import from /app
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Uvicorn
EXPOSE 8080

# Launch the FastAPI application
CMD ["uvicorn", "FastAPI_Analysis_and_XML:app", "--host", "0.0.0.0", "--port", "8080"]