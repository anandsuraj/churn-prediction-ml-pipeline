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
8. Data Versioning (Step 8) - Version control for datasets
9. Model Training (Step 9) - Train a custom model

Usage:
    python main_pipeline.py
"""

import sys

# Add src directory to path
sys.path.append('src')

# Import pipeline components
from src.data_ingestion import DataIngestionPipeline
from src.data_validation import DataValidator
from src.raw_data_storage import RawDataStorage
from src.data_preparation import DataPreparationPipeline
from src.data_transformation_storage import DataTransformationStorage
from src.feature_store import SimpleChurnFeatureStore
from src.data_versioning import version_pipeline_step
from src.build_model import TrainCustomModel


def run_data_ingestion_steps():
    """Run data ingestion and storage steps"""
    print("Step 2: Running data ingestion...")
    pipeline = DataIngestionPipeline()
    ingestion_result = pipeline.run_ingestion()

    print("Step 2.1: Versioning raw data...")
    raw_version_tag = version_pipeline_step(
        "Data Ingestion",
        "Raw data from ingestion pipeline"
    )

    print("Step 3: Running raw data storage...")
    storage = RawDataStorage()
    storage_result = storage.create_data_catalog()

    return ingestion_result, raw_version_tag, storage_result


def run_data_processing_steps():
    """Run data validation and preparation steps"""
    print("Step 4: Running data validation...")
    validator = DataValidator()
    validation_result = validator.run_validation()

    print("Step 5: Running data preparation...")
    preparation = DataPreparationPipeline()
    preparation_result = preparation.run_preparation_auto()

    print("Step 5.1: Versioning cleaned data...")
    cleaned_version_tag = version_pipeline_step(
        "Data Preparation",
        "Cleaned and preprocessed data"
    )

    return validation_result, preparation_result, cleaned_version_tag


def run_transformation_steps():
    """Run data transformation and feature store steps"""
    print("Step 6: Running data transformation and storage...")
    transformation = DataTransformationStorage()
    transformation_result = transformation.run_transformation_pipeline_auto()

    print("Step 6.1: Versioning transformed data...")
    transformed_version_tag = version_pipeline_step(
        "Data Transformation",
        "Transformed features for ML training"
    )

    print("Step 7: Setting up feature store...")
    feature_store = SimpleChurnFeatureStore()
    populate_result = feature_store.auto_populate_from_latest_data()

    return transformation_result, transformed_version_tag, populate_result, feature_store


def run_model_training():
    """Run model training step"""
    print("Step 9: Starting model training pipeline....")
    try:
        model_builder = TrainCustomModel()
        # Use the latest processed data instead of raw data
        model_builder.train_model(model_type="logistic_regression")
        return True
    except FileNotFoundError as e:
        print(f"Model training error: {str(e)}")
        print("Continuing pipeline without model training...")
        return False


def main():
    """Run the complete data management pipeline"""
    print("Customer Churn Data Management Pipeline")
    print("=" * 50)

    try:
        # Run pipeline steps
        ingestion_result, raw_version_tag, storage_result = run_data_ingestion_steps()
        validation_result, preparation_result, cleaned_version_tag = run_data_processing_steps()
        transformation_result, transformed_version_tag, populate_result, feature_store = run_transformation_steps()

        print("Step 8: Final data versioning...")
        final_version_tag = version_pipeline_step(
            "Pipeline Complete",
            "Complete pipeline with all processed data"
        )

        # Run model training (optional)
        model_success = run_model_training()

        # Print results
        print_pipeline_results(
            ingestion_result, storage_result, validation_result,
            preparation_result, transformation_result, populate_result,
            feature_store, raw_version_tag, cleaned_version_tag,
            transformed_version_tag, final_version_tag, model_success
        )

        # Close feature store connection
        feature_store.close()

    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("Make sure all required modules are installed and accessible.")
        return False
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        print("Make sure all required data files and directories exist.")
        return False
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        print("Check the logs for more detailed error information.")
        return False

    return True


def print_pipeline_results(ingestion_result, storage_result, validation_result,
                          preparation_result, transformation_result, populate_result,
                          feature_store, raw_version_tag, cleaned_version_tag,
                          transformed_version_tag, final_version_tag, model_success):
    """Print pipeline results summary"""
    print("\nPipeline completed successfully!")
    print(f"CSV File: {ingestion_result['csv_file']}")
    print(f"Hugging Face File: {ingestion_result['huggingface_file']}")
    print(f"Data Catalog: {storage_result}")
    print(f"Validation Report: {validation_result['report_path']}")
    print(f"Feature store: {populate_result}")
    print(f"Feature Store Path: {feature_store.store_path}")
    print(f"Raw Data Version: {raw_version_tag}")
    print(f"Cleaned Data Version: {cleaned_version_tag}")
    print(f"Transformed Data Version: {transformed_version_tag}")
    print(f"Final Version: {final_version_tag}")
    
    if model_success:
        print("Find Trained models at: data/models")
    else:
        print("Model training skipped due to errors")

    # Log file locations
    log_files = [
        "logs/data_ingestion.log",
        "logs/data_validation.log",
        "logs/data_preparation.log",
        "logs/data_transformation_storage.log",
        "logs/feature_store.log",
        "logs/data_versioning.log"
    ]
    print(f"Check logs: {', '.join(log_files)}")


if __name__ == "__main__":
    SUCCESS = main()
    if not SUCCESS:
        sys.exit(1)
