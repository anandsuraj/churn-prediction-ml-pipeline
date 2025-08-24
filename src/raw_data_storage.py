"""
Raw Data Storage
----------------
Organizes raw files into a partitioned local structure and optionally mirrors
the same layout in S3 when STORAGE_TYPE=cloud. Also creates a simple JSON
catalog of stored files for discoverability.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from utils.logger import get_logger, PIPELINE_NAMES

# Load environment variables from .env file
load_dotenv()

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['RAW_DATA_STORAGE'])

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Cloud storage will be disabled.")

# Class: manages raw file layout locally and in S3
class RawDataStorage:
    """Manage local/S3 storage of raw ingested files and metadata catalog."""
    def __init__(self, storage_type=None, base_path="data/raw"):
        # Prefer explicit param, otherwise env var, fallback to "local"
        env_storage = os.environ.get('STORAGE_TYPE')
        self.storage_type = storage_type if storage_type is not None else (env_storage or "local")
        self.base_path = Path(base_path)
        self.s3_client = None
        self.bucket_name = os.environ.get('S3_BUCKET_NAME', 'churn-data-lake')

        if self.storage_type == "cloud" and BOTO3_AVAILABLE:
            self._init_s3_client()

        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage initialized: type={self.storage_type}, base_path={self.base_path}")

    # Initialize S3 client and ensure bucket exists (if permissions allow)
    def _init_s3_client(self):
        """Initialize S3 client and create bucket if missing."""
        try:
            aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            aws_region = os.environ.get('AWS_REGION', 'ap-south-1')
            
            if not aws_access_key or not aws_secret_key:
                raise ValueError("AWS credentials not found in environment variables")
            
            self.s3_client = boto3.client(
                's3',
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            
            # Check if bucket exists, create if not
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': aws_region}
                    )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                else:
                    raise
            
            logger.info(f"S3 connected to bucket: {self.bucket_name}")
        
        except Exception as e:
            logger.error(f"S3 initialization failed: {str(e)}")
            self.s3_client = None

    # Copy file into partitioned layout and optionally upload to S3
    def store_file(self, source_path, source, data_type="churn"):
        """Store a single file locally and optionally to S3.

        Layout: data/raw/sources/{source}/{data_type}/YYYY/MM/DD/filename
        """
        timestamp = datetime.now()
        date_partition = timestamp.strftime("%Y/%m/%d")

        # Create destination path: data/raw/sources/{source}/{data_type}/{date_partition}/filename
        destination_dir = self.base_path / "sources" / source / data_type / date_partition
        destination_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{source}_{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}{Path(source_path).suffix}"
        destination_path = destination_dir / filename

        # Copy to local
        shutil.copy2(source_path, destination_path)
        logger.info(f"File stored locally: {destination_path}")

        # Upload to cloud if enabled
        s3_url = None
        if self.storage_type == "cloud" and self.s3_client:
            # S3 key mirrors local relative path
            relative_path = destination_path.relative_to(self.base_path)
            s3_key = str(relative_path).replace("\\", "/")
            try:
                self.s3_client.upload_file(str(destination_path), self.bucket_name, s3_key)
                s3_url = f"s3://{self.bucket_name}/{s3_key}"
                logger.info(f"Uploaded to S3: {s3_url}")
            except Exception as e:
                logger.error(f"S3 upload failed: {str(e)}")

        return {"local_path": str(destination_path), "s3_url": s3_url}

    # Store outputs from the ingestion step (CSV and/or HF JSON)
    def store_ingested_files(self, ingestion_result):
        """Store files from DataIngestionPipeline.run_ingestion"""
        try:
            results = []
            # Store CSV file
            if 'csv_file' in ingestion_result and ingestion_result['csv_file']:
                result = self.store_file(
                    source_path=ingestion_result['csv_file'],
                    source='telco_csv',
                    data_type='churn'
                )
                results.append(result)
                logger.info(f"Stored CSV file from ingestion: {result['local_path']}")

            # Store Hugging Face JSON file
            if 'huggingface_file' in ingestion_result and ingestion_result['huggingface_file']:
                result = self.store_file(
                    source_path=ingestion_result['huggingface_file'],
                    source='huggingface',
                    data_type='churn'
                )
                results.append(result)
                logger.info(f"Stored Hugging Face file from ingestion: {result['local_path']}")

            return results

        except Exception as e:
            logger.error(f"Failed to store ingested files: {str(e)}")
            raise

    # Walk storage and emit a JSON catalog of files
    def create_data_catalog(self):
        """Create metadata catalog for stored data"""
        catalog = {
            'datasets': [],
            'last_updated': datetime.now().isoformat()
        }

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_info = {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'size_bytes': file_path.stat().st_size,
                    'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                }
                catalog['datasets'].append(file_info)

        catalog_path = self.base_path / 'data_catalog.json'
        import json
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Data catalog created: {catalog_path}")
        return str(catalog_path)

if __name__ == "__main__":
    from data_ingestion import DataIngestionPipeline
    storage = RawDataStorage(storage_type=os.environ.get('STORAGE_TYPE'))  # Change to "cloud" for S3 if setup

    pipeline = DataIngestionPipeline()
    ingestion_result = pipeline.run_ingestion()
    stored_files = storage.store_ingested_files(ingestion_result)
    catalog = storage.create_data_catalog()

    print(f"Stored files: {stored_files}")
    print(f"Catalog created: {catalog}")
    print("""
    Folder/Bucket Structure:
    - data/raw/sources/{source}/{data_type}/{YYYY}/{MM}/{DD}/{filename}
      - source: e.g., 'telco_csv' or 'huggingface'
      - data_type: e.g., 'customer' or 'churn'
      - Timestamp partition: YYYY/MM/DD
      - S3 mirrors this under the bucket root.
    """)