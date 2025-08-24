"""
Churn Prediction Pipeline DAG
====================================================
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# DAG configuration
dag = DAG(
    'churn_prediction_pipeline',
    default_args={
        'owner': 'data-team',
        'depends_on_past': False,
        'start_date': datetime(2025, 8, 24),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'catchup': False,
    },
    description='Complete churn prediction pipeline',
    schedule=timedelta(hours=6),
    max_active_runs=1,
    tags=['churn', 'ml', 'pipeline', 'sequential'],
)


def run_python_script(script_code, task_name, **context):
    """Helper function to run Python scripts safely"""
    print(f"Starting {task_name}...")

    # Set environment variables
    env = os.environ.copy()
    env['PYTHONFAULTHANDLER'] = 'true'
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src"

    try:
        result = subprocess.run(
            [sys.executable, '-c', script_code],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes per task
            check=False
        )

        if result.returncode == 0:
            print(f"{task_name} completed successfully!")
            print("Output:", result.stdout[-500:])  # Last 500 chars
            return {"status": "success", "output": result.stdout}
        else:
            print(f"{task_name} failed!")
            print("Error:", result.stderr[-500:])
            raise RuntimeError(f"{task_name} failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print(f"{task_name} timed out!")
        raise RuntimeError(f"{task_name} timed out after 5 minutes")
    except Exception as e:
        print(f"{task_name} crashed: {str(e)}")
        raise


# Data Ingestion
def data_ingestion(**context):
    """Data Ingestion"""
    script = """
import sys
sys.path.append('src')
from data_ingestion import DataIngestionPipeline

print("Running data ingestion...")
pipeline = DataIngestionPipeline()
result = pipeline.run_ingestion()
print(f"Data Ingestion Result: {result}")
"""
    return run_python_script(script, "Data Ingestion", **context)


# Raw Data Storage
def raw_data_storage(**context):
    """Raw Data Storage"""
    script = """
import sys
sys.path.append('src')
from raw_data_storage import RawDataStorage

print("Running raw data storage...")
storage = RawDataStorage()
result = storage.create_data_catalog()
print(f"Raw Data Storage Result: {result}")
"""
    return run_python_script(script, "Raw Data Storage", **context)


# Data Validation
def data_validation(**context):
    """Data Validation"""
    script = """
import sys
sys.path.append('src')
from data_validation import DataValidator

print("Running data validation...")
validator = DataValidator()
result = validator.run_validation()
print(f"Data Validation Result: {result}")
"""
    return run_python_script(script, "Data Validation", **context)


# Data Preparation
def data_preparation(**context):
    """Data Preparation"""
    script = """
import sys
import os
sys.path.append('src')

print("Running data preparation...")
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONFAULTHANDLER'] = 'true'

try:
    from data_preparation_safe import SafeDataPreparationPipeline
    pipeline = SafeDataPreparationPipeline()
    result = pipeline.run_preparation_auto()
    print(f"Data Preparation Result: {result}")
except Exception as e:
    print(f"Data Preparation Error: {str(e)}")
    # Fallback to regular data preparation
    from data_preparation import DataPreparationPipeline
    pipeline = DataPreparationPipeline()
    result = pipeline.run_preparation_auto()
    print(f"Data Preparation Fallback Result: {result}")
"""
    return run_python_script(script, "Data Preparation", **context)


# Data Transformation
def data_transformation(**context):
    """Data Transformation and Storage"""
    script = """
import sys
sys.path.append('src')
from data_transformation_storage import DataTransformationStorage

print("Running data transformation and storage...")
transformation = DataTransformationStorage()
result = transformation.run_transformation_pipeline_auto()
print(f"Data Transformation Result: {result}")
"""
    return run_python_script(script, "Data Transformation", **context)


# Feature Store
def feature_store(**context):
    """Feature Store"""
    script = """
import sys
sys.path.append('src')
from feature_store import SimpleChurnFeatureStore

print("Setting up feature store...")
feature_store = SimpleChurnFeatureStore()
result = feature_store.auto_populate_from_latest_data()
feature_store.close()
print(f"Feature Store Result: {result}")
"""
    return run_python_script(script, "Feature Store", **context)


# Data Versioning
def data_versioning(**context):
    """Data Versioning"""
    script = """
import sys
sys.path.append('src')
from data_versioning import version_pipeline_step
from datetime import datetime

print("Final data versioning...")
tag = version_pipeline_step(
    "Airflow Pipeline Complete",
    f"Complete pipeline run at {datetime.now().isoformat()}"
)
print(f"Data Versioning Result: {tag}")
"""
    return run_python_script(script, "Data Versioning", **context)


# Model Building
def model_building(**context):
    """Model Building"""
    script = """
import sys
sys.path.append('src')
from build_model import TrainCustomModel

print("Starting model training pipeline...")
model_builder = TrainCustomModel()
model_builder.train_model(model_type="logistic_regression")
print("Model Building completed successfully!")
"""
    return run_python_script(script, "Model Building", **context)


# Pipeline Success Callback
def pipeline_success(**context):
    """Final success callback"""
    print("Complete Churn Prediction Pipeline finished!")
    print("All tasks executed successfully!")
    return "Pipeline Success"


# Define all pipeline tasks
task_data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag,
    doc_md="Fetch data from multiple sources"
)

task_raw_data_storage = PythonOperator(
    task_id='raw_data_storage',
    python_callable=raw_data_storage,
    dag=dag,
    doc_md="Organize and catalog raw data"
)

task_data_validation = PythonOperator(
    task_id='data_validation',
    python_callable=data_validation,
    dag=dag,
    doc_md="Validate data quality and generate reports"
)

task_data_preparation = PythonOperator(
    task_id='data_preparation',
    python_callable=data_preparation,
    dag=dag,
    doc_md="Clean and preprocess data"
)

task_data_transformation = PythonOperator(
    task_id='data_transformation',
    python_callable=data_transformation,
    dag=dag,
    doc_md="Feature engineering and transformation"
)

task_feature_store = PythonOperator(
    task_id='feature_store',
    python_callable=feature_store,
    dag=dag,
    doc_md="Manage engineered features in feature store"
)

task_data_versioning = PythonOperator(
    task_id='data_versioning',
    python_callable=data_versioning,
    dag=dag,
    doc_md="Version control for datasets with DVC"
)

task_model_building = PythonOperator(
    task_id='model_building',
    python_callable=model_building,
    dag=dag,
    doc_md="Train machine learning model"
)

task_pipeline_success = PythonOperator(
    task_id='pipeline_success',
    python_callable=pipeline_success,
    dag=dag,
    doc_md="Final success notification"
)

# Define the complete pipeline sequence
task_data_ingestion >> task_raw_data_storage >> task_data_validation >> task_data_preparation >> task_data_transformation >> task_feature_store >> task_data_versioning >> task_model_building >> task_pipeline_success