#!/usr/bin/env python3
"""
Main Pipeline Runner
===================

Simple script to run the data ingestion pipeline.

Usage:
    python main_pipeline.py
"""

from data_ingestion import DataIngestionPipeline
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append('src')


def main():
    """Run the data ingestion pipeline"""

    print("Customer Churn Data Ingestion Pipeline")
    print("=" * 50)

    try:
        # Run data ingestion
        pipeline = DataIngestionPipeline()
        result = pipeline.run_ingestion()

        print(f"\nPipeline completed successfully!")
        print(f"CSV File: {result['csv_file']}")
        print(f"Hugging Face File: {result['huggingface_file']}")
        print(f"Check logs: logs/data_ingestion.log")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
