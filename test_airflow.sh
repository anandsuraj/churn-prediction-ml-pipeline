#!/bin/bash

# Test Airflow Setup Script
echo "Testing Airflow Setup..."

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow
export PYTHONFAULTHANDLER=true
export MPLBACKEND=Agg

echo "1. Checking Airflow installation..."
airflow version

echo "2. Checking DAG import..."
python -c "
import sys
sys.path.append('airflow/dags')
try:
    from churn_prediction_pipeline import dag
    print('✅ DAG imported successfully')
    print(f'DAG ID: {dag.dag_id}')
    print(f'Tasks: {[task.task_id for task in dag.tasks]}')
except Exception as e:
    print(f'❌ DAG import failed: {e}')
    exit(1)
"

echo "3. Testing DAG syntax..."
airflow dags list | grep churn_prediction_pipeline
if [ $? -eq 0 ]; then
    echo "✅ DAG found in Airflow"
else
    echo "❌ DAG not found in Airflow"
    exit(1)
fi

echo "4. Testing complete pipeline task..."
airflow tasks test churn_prediction_pipeline complete_pipeline 2025-08-24

echo "5. All tests passed! ✅"
echo ""
echo "To start Airflow:"
echo "Terminal 1: airflow webserver --port 8080"
echo "Terminal 2: airflow scheduler"
echo "Then visit: http://localhost:8080"