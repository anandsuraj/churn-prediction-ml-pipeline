"""
Data Ingestion
--------------
Fetches raw churn data from two sources:
  1) Telco CSV (public GitHub)
  2) Hugging Face dataset API (JSON)

Adds basic logger, retry with backoff for the API, and returns file paths
for downstream stages (storage, validation, preparation).
"""
import pandas as pd
import requests
import os
from datetime import datetime
import json
import glob
import time

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_INGESTION'])

# Class: orchestrates ingestion from CSV and Hugging Face API
class DataIngestionPipeline:
    """Ingestion pipeline for fetching CSV and Hugging Face JSON data."""
    # Initialize with base directory for saving raw files
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        os.makedirs(raw_data_path, exist_ok=True)

    # Download CSV dataset and save to raw folder
    def ingest_csv_data(self):
        """Ingest customer churn data from CSV source"""
        try:
            logger.info("Starting CSV data ingestion...")
            
            # IBM Telco Customer Churn dataset
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"customer_churn_{timestamp}.csv"
                filepath = os.path.join(self.raw_data_path, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Validate data
                df = pd.read_csv(filepath)
                logger.info(f"CSV data successfully ingested: {filepath}")
                logger.info(f"Records: {len(df)}, Columns: {len(df.columns)}")
                
                return filepath
            else:
                raise Exception(f"Failed to fetch CSV data: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"CSV ingestion failed: {str(e)}")
            raise

    # Fetch JSON rows from Hugging Face dataset server (with retry)
    def ingest_huggingface_data(self):
        """Ingest customer data from Hugging Face API"""
        try:
            logger.info("Starting Hugging Face data ingestion...")
            
            # Hugging Face Datasets API endpoint
            api_url = "https://datasets-server.huggingface.co/rows"
            params = {
                'dataset': 'scikit-learn/churn-prediction',
                'config': 'default',
                'split': 'train',
                'offset': 0,
                'length': 100
            }
            # Retry with exponential backoff on transient errors (e.g., 5xx)
            max_retries = 3
            backoff = 2
            last_status = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(api_url, params=params, timeout=30)
                    last_status = response.status_code
                    if response.status_code == 200:
                        break
                    logger.warning(f"HF API HTTP {response.status_code} (attempt {attempt}/{max_retries})")
                except Exception as e:
                    logger.warning(f"HF API request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)

            if last_status == 200:
                api_data = response.json()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"huggingface_churn_{timestamp}.json"
                filepath = os.path.join(self.raw_data_path, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(api_data, f, indent=2)
                
                records_count = len(api_data.get('rows', []))
                
                logger.info(f"Hugging Face data successfully ingested: {filepath}")
                logger.info(f"Records: {records_count}")
                
                return filepath
            else:
                raise Exception(f"Failed to fetch Hugging Face data: HTTP {last_status}")
            
        except Exception as e:
            logger.error(f"Hugging Face ingestion failed: {str(e)}")
            # Fallback to latest cached HF file if available
            try:
                hf_files = glob.glob(os.path.join(self.raw_data_path, "huggingface_churn_*.json"))
                if hf_files:
                    latest_hf = max(hf_files, key=os.path.getctime)
                    logger.warning(f"Using cached Hugging Face file: {latest_hf}")
                    return latest_hf
            except Exception:
                pass
            return None

    # Run full ingestion (CSV + HF) and return file paths
    def run_ingestion(self):
        """Run complete data ingestion from both sources"""
        try:
            logger.info("Starting data ingestion pipeline...")
            
            # Ingest from both sources
            csv_file = self.ingest_csv_data()
            hf_file = self.ingest_huggingface_data()
            
            logger.info("Data ingestion completed successfully")
            logger.info(f"CSV file: {csv_file}")
            if hf_file:
                logger.info(f"Hugging Face file: {hf_file}")
            else:
                logger.warning("Hugging Face file not available; proceeding with CSV only")
            
            return {
                'status': 'success',
                'csv_file': csv_file,
                'huggingface_file': hf_file,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    result = pipeline.run_ingestion()
    print(f"Ingestion completed: {result}")