#!/bin/bash

echo "🚀 Starting Airflow in Docker..."
echo ""

# Navigate to project root (two levels up from src/airflow_docker/)
cd "$(dirname "$0")/../.."

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down

# Build the image
echo "Building Docker image..."
docker-compose build

# Start Airflow services
echo "Starting Airflow services..."
docker-compose up airflow-init airflow-webserver airflow-scheduler

echo "✅ Airflow is starting up!"
echo "📊 Web UI will be available at: http://localhost:8080"
echo "👤 Username: admin"
echo "🔑 Password: admin"
echo ""
echo "To stop Airflow, press Ctrl+C or run: docker-compose down"