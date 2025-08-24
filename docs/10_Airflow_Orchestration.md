# Airflow Orchestration Guide

## Overview

Apache Airflow orchestrates the complete 9-step churn prediction pipeline:
1. Data Ingestion → 2. Raw Data Storage → 3. Data Validation → 4. Data Preparation → 
5. Data Transformation → 6. Feature Store → 7. Data Versioning → 8. Model Building → 9. Pipeline Success

## Prerequisites
- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt`
- Data file available: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Quick Start Commands

### 1. Start Airflow (Standalone Mode - Recommended)
```bash
# Set environment variables
export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags

# Initialize database (first time only)
airflow db migrate

# Start Airflow standalone (includes webserver + scheduler)
airflow standalone
```

### 2. Start Airflow (Production Mode)
```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler  
airflow scheduler

# Terminal 3: Start worker (if using Celery)
airflow celery worker
```

### 3. Access Airflow Web UI
```
URL: http://localhost:8080
Default Login: admin/admin (created automatically in standalone mode)
```

## Pipeline Commands

### Run Pipeline Manually
```bash
# Trigger the DAG immediately
airflow dags trigger churn_prediction_pipeline

# Run specific task
airflow tasks run churn_prediction_pipeline data_ingestion 2024-01-01

# Test single task (dry run)
airflow tasks test churn_prediction_pipeline data_ingestion 2024-01-01
```

### Monitor Pipeline Status
```bash
# List all DAGs
airflow dags list

# Show DAG details
airflow dags show churn_prediction_pipeline

# List DAG runs
airflow dags list-runs -d churn_prediction_pipeline

# Show task instances for a run
airflow tasks list churn_prediction_pipeline --tree
```

### View Logs
```bash
# View task logs
airflow tasks logs churn_prediction_pipeline data_ingestion 2024-01-01 1

# View all logs for a DAG run
airflow dags logs churn_prediction_pipeline
```

## Monitoring & Management

### 1. Web UI Monitoring
- **DAGs View**: See all pipelines and their status
- **Graph View**: Visualize task dependencies and execution status
- **Gantt Chart**: View task execution timeline
- **Task Duration**: Monitor performance over time
- **Logs**: Access detailed task logs

### 2. Command Line Monitoring
```bash
# Check DAG state
airflow dags state churn_prediction_pipeline 2024-01-01

# List failed tasks
airflow tasks failed-deps churn_prediction_pipeline 2024-01-01

# Clear task state (to rerun)
airflow tasks clear churn_prediction_pipeline

# Pause/Unpause DAG
airflow dags pause churn_prediction_pipeline
airflow dags unpause churn_prediction_pipeline
```

### 3. Pipeline Health Checks
```bash
# Validate DAG structure
airflow dags check churn_prediction_pipeline

# Test DAG import
python -c "from airflow.models import DagBag; print(DagBag().get_dag('churn_prediction_pipeline'))"

# Check for import errors
airflow dags list-import-errors
```

## Production Configuration

### Environment Variables
```bash
# Essential settings
export AIRFLOW_HOME=/path/to/airflow
export AIRFLOW__CORE__DAGS_FOLDER=/path/to/dags
export AIRFLOW__CORE__EXECUTOR=LocalExecutor  # or CeleryExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://user:pass@localhost/airflow
```

### Key Configuration Files
- `airflow/airflow.cfg` - Main configuration
- `airflow/dags/` - DAG files location
- `logs/` - Task execution logs

## Troubleshooting

### Common Issues
```bash
# DAG not appearing
airflow dags list-import-errors

# Task stuck in running state
airflow tasks clear churn_prediction_pipeline -t task_name

# Database issues
airflow db reset  # WARNING: Deletes all data
airflow db upgrade

# Permission issues
chmod +x airflow/dags/churn_prediction_pipeline.py
```

### Log Locations
- **Task Logs**: `$AIRFLOW_HOME/logs/dag_id/task_id/`
- **Scheduler Logs**: `$AIRFLOW_HOME/logs/scheduler/`
- **Webserver Logs**: `$AIRFLOW_HOME/logs/webserver/`

## Pipeline Schedule

The pipeline is configured to run **every 6 hours**:
- Schedule: `timedelta(hours=6)`
- Start Date: `2024-01-01`
- Catchup: Disabled

To modify the schedule, edit `airflow/dags/churn_prediction_pipeline.py`:
```python
schedule=timedelta(hours=6),  # Change this value
```

## Monitoring Outputs

### Generated Files
```
data/
├── raw/telco_churn_YYYYMMDD_HHMMSS.csv
├── processed/training_sets/churn_prediction_v1_YYYYMMDD_HHMMSS.csv
├── feature_store/ (SQLite database)
└── models/logreg_model_YYYYMMDD_HHMMSS.joblib

logs/
└── data_validation.log
```

### MLflow Tracking
```bash
# Start MLflow UI to view model experiments
mlflow ui --backend-store-uri file:///tmp/mlflow-runs
# Access at: http://localhost:5000
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Set environment variables
export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags

# 3. Initialize database
airflow db migrate

# 4. Start Airflow
airflow standalone
```

## Web UI Features
- **Graph View**: Visual pipeline representation
- **Tree View**: Historical runs and task status  
- **Gantt Chart**: Task duration and timing
- **Task Logs**: Detailed execution logs
- **Code View**: DAG source code

## Quick Health Check
```bash
# One-command pipeline health check
airflow dags trigger churn_prediction_pipeline && \
airflow dags list-runs -d churn_prediction_pipeline --limit 1
```

## Complete Pipeline Testing

### 1. Quick Test (Recommended)
```bash
# Start Airflow
export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
airflow standalone

# In another terminal, trigger pipeline
airflow dags trigger churn_prediction_pipeline

# Monitor progress
airflow dags list-runs -d churn_prediction_pipeline
```

### 2. Step-by-Step Testing
```bash
# Test individual tasks
airflow tasks test churn_prediction_pipeline data_ingestion 2024-01-01
airflow tasks test churn_prediction_pipeline raw_data_storage 2024-01-01
airflow tasks test churn_prediction_pipeline data_validation 2024-01-01
airflow tasks test churn_prediction_pipeline data_preparation 2024-01-01
airflow tasks test churn_prediction_pipeline data_transformation 2024-01-01
airflow tasks test churn_prediction_pipeline feature_store 2024-01-01
airflow tasks test churn_prediction_pipeline data_versioning 2024-01-01
airflow tasks test churn_prediction_pipeline model_building 2024-01-01
```

### 3. Validation Commands
```bash
# Validate DAG structure
airflow dags check churn_prediction_pipeline

# List all tasks
airflow tasks list churn_prediction_pipeline --tree

# Check for import errors
airflow dags list-import-errors
```

## Expected Results
✅ **Pipeline Duration**: 60-90 seconds  
✅ **Data Processed**: 7,043 customer records  
✅ **Features Created**: 45 engineered features  
✅ **Model Accuracy**: >74%  
✅ **Files Generated**: Raw data, processed data, model artifacts  

## Troubleshooting

### Common Issues & Solutions
```bash
# Issue: DAG not appearing
airflow dags list-import-errors

# Issue: Task stuck
airflow tasks clear churn_prediction_pipeline -t task_name

# Issue: Permission denied
chmod +x airflow/dags/churn_prediction_pipeline.py

# Issue: Module not found
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Log Locations
- **Task Logs**: Web UI → Graph View → Click Task → View Log
- **Scheduler Logs**: `$AIRFLOW_HOME/logs/scheduler/`
- **Pipeline Logs**: `logs/data_validation.log`

## Success Indicators
When the pipeline runs successfully, you should see:
1. All 9 tasks marked as SUCCESS (green) in Web UI
2. Generated files in `data/` directories
3. Model saved in `data/models/`
4. Feature store populated
5. DVC version created
6. MLflow experiment logged

## Best Practices
1. **Monitor Web UI**: Use Graph View for real-time status
2. **Check Logs**: Review task logs for detailed information
3. **Validate Output**: Verify all expected files are created
4. **Resource Monitoring**: Watch CPU/memory usage during execution
5. **Error Handling**: Review failed tasks and retry if needed