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
5. Data Validation (Step 56) - Validate data quality

Usage:
    python main_pipeline.py
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_ingestion import DataIngestionPipeline
from raw_data_storage import RawDataStorage
from data_validation import DataValidator
from data_transformation_storage import DataTransformationStorage
from data_preparation import DataPreparationPipeline

def main():
    """Run the data ingestion pipeline"""

    print("Customer Churn Data Ingestion Pipeline")
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

        print(f"\nPipeline completed successfully!")
        print(f"CSV File: {ingestion_result['csv_file']}")
        print(f"Hugging Face File: {ingestion_result['huggingface_file']}")
        print(f"Data Catalog: {storage_result}")
        print(f"Validation Report: {validation_result['report_path']}")
        print(f"Data Preparation: {preparation_result}")
        print(f"Data Transformation: {transformation_result}")
        print(f"Check logs: logs/data_ingestion.log, logs/data_validation.log, logs/data_preparation.log, logs/data_transformation_storage.log")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
