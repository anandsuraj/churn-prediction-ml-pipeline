"""
Data Ingestion
--------------
Fetches raw churn data from two sources:
  1) Telco CSV (public GitHub)
  2) Hugging Face dataset API (JSON)

Adds basic logging, retry with backoff for the API, and returns file paths
for downstream stages (storage, validation, preparation).
"""
import pandas as pd
import requests
import os
import logging
from datetime import datetime
import json
import glob
import time

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_ingestion.log'),
        logging.StreamHandler()
    ]
)

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
            logging.info("Starting CSV data ingestion...")
            
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
                logging.info(f"CSV data successfully ingested: {filepath}")
                logging.info(f"Records: {len(df)}, Columns: {len(df.columns)}")
                
                return filepath
            else:
                raise Exception(f"Failed to fetch CSV data: HTTP {response.status_code}")
                
        except Exception as e:
            logging.error(f"CSV ingestion failed: {str(e)}")
            raise

    # Fetch JSON rows from Hugging Face dataset server (with retry)
    def ingest_huggingface_data(self):
        """Ingest customer data from Hugging Face API"""
        try:
            logging.info("Starting Hugging Face data ingestion...")
            
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
                    logging.warning(f"HF API HTTP {response.status_code} (attempt {attempt}/{max_retries})")
                except Exception as e:
                    logging.warning(f"HF API request failed (attempt {attempt}/{max_retries}): {e}")
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
                
                logging.info(f"Hugging Face data successfully ingested: {filepath}")
                logging.info(f"Records: {records_count}")
                
                return filepath
            else:
                raise Exception(f"Failed to fetch Hugging Face data: HTTP {last_status}")
            
        except Exception as e:
            logging.error(f"Hugging Face ingestion failed: {str(e)}")
            # Fallback to latest cached HF file if available
            try:
                hf_files = glob.glob(os.path.join(self.raw_data_path, "huggingface_churn_*.json"))
                if hf_files:
                    latest_hf = max(hf_files, key=os.path.getctime)
                    logging.warning(f"Using cached Hugging Face file: {latest_hf}")
                    return latest_hf
            except Exception:
                pass
            return None

    # Run full ingestion (CSV + HF) and return file paths
    def run_ingestion(self):
        """Run complete data ingestion from both sources"""
        try:
            logging.info("Starting data ingestion pipeline...")
            
            # Ingest from both sources
            csv_file = self.ingest_csv_data()
            hf_file = self.ingest_huggingface_data()
            
            logging.info("Data ingestion completed successfully")
            logging.info(f"CSV file: {csv_file}")
            if hf_file:
                logging.info(f"Hugging Face file: {hf_file}")
            else:
                logging.warning("Hugging Face file not available; proceeding with CSV only")
            
            return {
                'status': 'success',
                'csv_file': csv_file,
                'huggingface_file': hf_file,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Data ingestion pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    result = pipeline.run_ingestion()
    print(f"Ingestion completed: {result}")