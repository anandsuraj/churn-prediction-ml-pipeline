#!/usr/bin/env python3
"""
Main Pipeline Runner
===================

Simple script to run the data ingestion pipeline.

Usage:
    python main_pipeline.py
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_ingestion import DataIngestionPipeline
from data_validation import DataValidator


def main():
    """Run the data ingestion pipeline"""

    print("Customer Churn Data Ingestion Pipeline")
    print("=" * 50)

    try:
        # Run data ingestion
        print("Step 1: Running data ingestion...")
        pipeline = DataIngestionPipeline()
        ingestion_result = pipeline.run_ingestion()

        # Run data validation
        print("Step 2: Running data validation...")
        validator = DataValidator()
        validation_result = validator.run_validation()

        print(f"\nPipeline completed successfully!")
        print(f"CSV File: {ingestion_result['csv_file']}")
        print(f"Hugging Face File: {ingestion_result['huggingface_file']}")
        print(f"Validation Report: {validation_result['report_path']}")
        print(f"Check logs: logs/data_ingestion.log, logs/data_validation.log")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
