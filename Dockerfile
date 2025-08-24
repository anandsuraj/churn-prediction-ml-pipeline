FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System dependencies including git for DVC and AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI for S3 access
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration files
COPY config/ ./config/

# Copy source code and other files
COPY src/ ./src/
COPY database/ ./database/
COPY airflow/ ./airflow/
COPY main_pipeline.py problem_formulation.md ./

# Create symbolic links for DVC files (maintaining compatibility)
RUN ln -sf config/dvc/dvc.yaml dvc.yaml \
    && ln -sf config/dvc/.dvcignore .dvcignore \
    && ln -sf config/env/.env.example .env.example

# Create necessary directories
RUN mkdir -p data/raw data/cleaned data/processed data/processed/training_sets \
             data/feature_store data/eda/raw data/eda/cleaned \
             logs reports .dvc/cache airflow/logs airflow/plugins

# Initialize SQLite database
RUN sqlite3 data/processed/churn_data.db < database/init.sql

# Initialize DVC
RUN dvc init --no-scm

# Make DVC setup script executable
RUN chmod +x src/setup_dvc.sh

# Set up environment
COPY config/env/.env.example .env

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import dvc; print('DVC OK')" || exit 1

# Default command
CMD ["python", "main_pipeline.py"]
