FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create writable dirs
RUN mkdir -p data/raw data/processed data/processed/training_sets logs reports

# Initialize SQLite database
RUN sqlite3 data/processed/churn_data.db < database/init.sql

# Default command
CMD ["python", "main_pipeline.py"]
