import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_s3_object(bucket_name, object_key, output_path):
    """Retrieve an object from S3 and save it locally"""
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get AWS credentials
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            region_name='ap-south-1',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Download the object
        s3_client.download_file(bucket_name, object_key, output_path)
        logging.info(f"Successfully downloaded s3://{bucket_name}/{object_key} to {output_path}")
        return output_path
        
    except ClientError as e:
        logging.error(f"Failed to retrieve S3 object: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        bucket = "churn-data-lake"
        key = "deploy_aws.sh"
        local_path = "downloaded_deploy_aws.sh"
        get_s3_object(bucket, key, local_path)
        print(f"File downloaded to: {local_path}")
    except Exception as e:
        print(f"Error: {str(e)}")