#!/usr/bin/env python3
"""
Main Pipeline Runner
===================

Script to run the complete data management pipeline for customer churn prediction.

Pipeline Steps:
1. Problem Formulation (Step 1) - Business problem definition
2. Data Ingestion (Step 2) - Fetch data from multiple sources
3. Raw Data Storage (Step 3) - Store data in organized structure
4. Data Validation (Step 4) - Validate data quality
5. Data Preparation (Step 5) - Clean and preprocess data
6. Data Transformation (Step 6) - Feature engineering and storage
7. Feature Store (Step 7) - Manage engineered features

Usage:
    python main_pipeline.py
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add src directory to path
sys.path.append('src')

from data_ingestion import DataIngestionPipeline
from raw_data_storage import RawDataStorage
from data_validation import DataValidator
from data_transformation_storage import DataTransformationStorage
from data_preparation import DataPreparationPipeline
from feature_store import ChurnFeatureStore

def main():
    """Run the complete data management pipeline"""

    print("Customer Churn Data Management Pipeline")
    print("=" * 50)

    try:
        # Run data ingestion
        print("Step 2: Running data ingestion...")
        pipeline = DataIngestionPipeline()
        ingestion_result = pipeline.run_ingestion()

        print("Step 3: Running raw data storage...")
        storage = RawDataStorage()
        storage_result = storage.create_data_catalog()

        print("Step 4: Running data validation...")
        validator = DataValidator()
        validation_result = validator.run_validation()

        print("Step 5: Running data preparation...")
        preparation = DataPreparationPipeline()
        preparation_result = preparation.run_preparation_auto()

        print("Step 6: Running data transformation and storage...")
        transformation = DataTransformationStorage()
        transformation_result = transformation.run_transformation_pipeline_auto()

        print("Step 7: Setting up feature store...")
        feature_store = ChurnFeatureStore()
        # Auto-populate feature store from latest training data
        populate_result = feature_store.auto_populate_from_latest_data()
        
        print(f"\nPipeline completed successfully!")
        print(f"CSV File: {ingestion_result['csv_file']}")
        print(f"Hugging Face File: {ingestion_result['huggingface_file']}")
        print(f"Data Catalog: {storage_result}")
        print(f"Validation Report: {validation_result['report_path']}")
        print(f"Data Preparation: {preparation_result}")
        print(f"Data Transformation: {transformation_result}")
        print(f"Feature store: {populate_result}")
        print(f"Feature Store DB Path: {feature_store.db_path}")
        print(f"Check logs: logs/data_ingestion.log, logs/data_validation.log, logs/data_preparation.log, logs/data_transformation_storage.log, logs/feature_store.log")

        # Close feature store connection
        feature_store.close()

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
