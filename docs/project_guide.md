# Project Guide

## Overview 

This codebase implements a data ingestion and analysis pipeline for churn prediction. It includes modules for data extraction, transformation, and modeling.

## Installation

### Running on Terminal

```bash
# create new virtual environment
python -m venv .venv

# switch to the newly created venv
# on MacOS/Linux
source .venv/bin/activate

# on Windows
.venv/Scripts/activate

# install dependencies
pip install -r requirements.txt

# run main pipeline 
python main_pipeline.py

```

### Running on a Docker

```bash
# build and run the docker container
docker-compose up --build
```


## Quick File Descriptions

* `src/data_preparation.py`: Data ingestion and cleaning.
* `src/data_transformation_storage.py`: Feature engineering and data storage.
* `src/build_model.py`: Model training and evaluation.
* `src/data_validation.py`: Data validation and quality checks.
* `src/feature_store.py`: Feature store implementation.
* `src/raw_data_storage.py`: Raw data storage.
* `src/utils/logger.py`: Logging utilities.
* `scheduler.py`: Pipeline scheduler.
* `main_pipeline.py`: Entry point to the pipeline.

Find detailed descriptions for each script in `/docs` directory

## Components Overview

### 1. Data Ingestion (`data_ingestion.py`)
- Fetches data from two sources (CSV, APIs)
  1) [Telco CSV (public GitHub)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
  2) [Hugging Face dataset API (JSON)](https://datasets-server.huggingface.co/rows)
- Handles errors and logging
- Stores raw data with timestamps

### 2. Raw Data Storage (`raw_data_storage.py`)
- Organizes raw files into a partitioned local structure
- Creates a simple JSON catalog of stored files for discoverability.
- Supports local and cloud storage (use `STORAGE_TYPE=cloud` for S3 support)

### 3. Data Validation (`data_validation.py`)
- Performs basic quality checks on ingested CSV and JSON datasets:
    - Missing values
    - Duplicate records
    - Data types and negative values in numeric columns
- Generates an Excel report summarizing findings.

### 4. Data Preparation (`data_preparation.py`)
- Cleans and preprocesses dataset for modeling:
- Handles missing values (numeric: median, categorical: mode)
- One-hot encodes categorical features; maps 'Churn' to 0/1
- Creates derived features and caps outliers (IQR method)
- Scales numerical features using StandardScaler
- Saves EDA outputs to `data/eda/raw` and `data/eda/cleaned`
- Saves cleaned and scaled datasets to `data/processed`

### 5. Data Transformation (`data_transformation_storage.py`)
- Transforms cleaned data into richer feature sets, scales features, and stores them in SQLite database for downstream querying and training set management.
- Tracks feature metadata and training set summaries.

### 6. Feature Store (`feature_store.py`)
- Custom lightweight feature store implementation that works with CSV data
- Automatically finds the latest training data CSV file from the `training_sets` directory
- Creates sample features for demonstration when no data is available

### 7. Model Building (`build_model.py`)
- Supports two models: Random Forest and Logistic Regression.
- Loads and cleans customer churn data from a CSV file.
- Splits data into training and testing sets.
- Trains the selected model and evaluates its performance.
- Logs model parameters, metrics, and artifacts to MLflow.
- Saves the trained model locally at `src/models`

