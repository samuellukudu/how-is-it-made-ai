FROM python:3.10-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and start the server
EXPOSE 8002
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]