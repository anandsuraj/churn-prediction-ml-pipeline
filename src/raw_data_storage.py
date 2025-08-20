import os
import shutil
from datetime import datetime
import logging
import boto3
from pathlib import Path

logging.basicConfig(level=logging.INFO)

class RawDataStorage:
    def __init__(self, storage_type="local", base_path="data/raw"):
        self.storage_type = storage_type
        self.base_path = base_path
        self.setup_storage_structure()

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

    def upload_to_cloud(self, local_path, bucket_name="churn-data-lake"):
        """Upload to cloud storage (AWS S3 simulation)"""
        if self.storage_type == "cloud":
            try:
                # Simulate cloud upload
                logging.info(f"Uploading {local_path} to {bucket_name}")
                # s3_client = boto3.client('s3')
                # s3_client.upload_file(local_path, bucket_name, key)
                logging.info("Cloud upload simulated successfully")
                return f"s3://{bucket_name}/{local_path}"
            except Exception as e:
                logging.error(f"Cloud upload failed: {str(e)}")
                raise
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
