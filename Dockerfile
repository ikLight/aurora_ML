FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# IMPORTANT: Do NOT hardcode EXPOSE 8000; Render sets the PORT variable
# EXPOSE 8000  <- you can remove or leave, doesn't matter for Render

# Correct startup command
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
