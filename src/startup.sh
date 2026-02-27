#!/bin/bash
set -e

echo "🚀 Starting Churn Prediction Pipeline Services..."

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_command" >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Initialize DVC if needed
if [ "$1" = "dvc-setup" ]; then
    echo "🔧 Setting up DVC..."
    bash /app/src/setup_dvc.sh
    exit 0
fi

# Initialize Airflow if needed
if [ "$1" = "airflow-init" ]; then
    echo "🔧 Initializing Airflow..."
    python /app/airflow/setup_airflow.py
    exit 0
fi

# Start Airflow webserver
if [ "$1" = "airflow-webserver" ]; then
    echo "🌐 Starting Airflow webserver..."
    wait_for_service "Airflow DB" "test -f /app/airflow/airflow.db"
    exec airflow webserver --port 8080 --hostname 0.0.0.0
fi

# Start Airflow scheduler
if [ "$1" = "airflow-scheduler" ]; then
    echo "📅 Starting Airflow scheduler..."
    wait_for_service "Airflow DB" "test -f /app/airflow/airflow.db"
    exec airflow scheduler
fi

# Default: run main pipeline
echo "🏃 Running main pipeline..."
exec python main_pipeline.py