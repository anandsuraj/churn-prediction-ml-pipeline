import os
import shutil
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. Cloud storage will be simulated.")

logging.basicConfig(level=logging.INFO)

class RawDataStorage:
    def __init__(self, storage_type="local", base_path="data/raw"):
        self.storage_type = storage_type
        self.base_path = base_path
        self.s3_client = None
        
        if storage_type == "cloud" and BOTO3_AVAILABLE:
            self._init_s3_client()
        
        self.setup_storage_structure()

    def _init_s3_client(self):
        """Initialize S3 client"""
        try:
            # Get AWS credentials from environment variables
            aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            aws_region = os.environ.get('AWS_REGION', 'us-east-1')
            bucket_name = os.environ.get('S3_BUCKET_NAME', 'churn-data-lake')
            
            if not aws_access_key or not aws_secret_key:
                raise ValueError("AWS credentials not found in environment variables")
            
            self.s3_client = boto3.client(
                's3',
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"S3 connected to bucket: {bucket_name}")
            
        except Exception as e:
            logging.error(f"S3 initialization failed: {str(e)}")
            self.s3_client = None

    def setup_storage_structure(self):
        """Create organized folder structure for raw data"""
        folders = [
            "customer_data",
            "usage_data", 
            "billing_data",
            "support_data"
        ]

        for folder in folders:
            for subfolder in ["daily", "weekly", "monthly"]:
                path = os.path.join(self.base_path, folder, subfolder)
                os.makedirs(path, exist_ok=True)

        logging.info("Storage structure created")

    def store_file(self, source_path, data_type, frequency="daily"):
        """Store file with organized naming and partitioning"""
        timestamp = datetime.now()
        date_partition = timestamp.strftime("%Y/%m/%d")

        destination_dir = os.path.join(
            self.base_path, 
            f"{data_type}_data", 
            frequency, 
            date_partition
        )
        os.makedirs(destination_dir, exist_ok=True)

        filename = f"{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}{Path(source_path).suffix}"
        destination_path = os.path.join(destination_dir, filename)

        shutil.copy2(source_path, destination_path)
        logging.info(f"File stored: {destination_path}")

        return destination_path

    def upload_to_cloud(self, local_path, s3_key=None):
        """Upload file to S3"""
        if self.storage_type != "cloud":
            logging.info("Cloud storage not enabled")
            return local_path
        
        if not self.s3_client:
            logging.warning("S3 client not available")
            return local_path
        
        try:
            # Get bucket name from environment
            bucket_name = os.environ.get('S3_BUCKET_NAME', 'churn-data-lake')
            
            # Generate S3 key if not provided
            if not s3_key:
                timestamp = datetime.now().strftime("%Y/%m/%d")
                filename = Path(local_path).name
                s3_key = f"raw_data/{timestamp}/{filename}"
            
            # Upload to S3
            self.s3_client.upload_file(local_path, bucket_name, s3_key)
            s3_url = f"s3://{bucket_name}/{s3_key}"
            logging.info(f"Uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logging.error(f"S3 upload failed: {str(e)}")
            return local_path

    def create_data_catalog(self):
        """Create metadata catalog for stored data"""
        catalog = {
            'datasets': [],
            'last_updated': datetime.now().isoformat()
        }

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    file_info = {
                        'file_name': file,
                        'file_path': file_path,
                        'size_bytes': os.path.getsize(file_path),
                        'created_date': datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat()
                    }
                    catalog['datasets'].append(file_info)

        catalog_path = os.path.join(self.base_path, 'data_catalog.json')
        import json
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logging.info(f"Data catalog created: {catalog_path}")
        return catalog_path

if __name__ == "__main__":
    storage = RawDataStorage()
    catalog = storage.create_data_catalog()
    print(f"Storage setup completed with catalog: {catalog}")
