"""
DVC-Based Data Versioning Implementation
=======================================
Standard DVC approach for data versioning in the churn prediction pipeline.
"""

import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Union
import json

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_VERSIONING'])


class DVCVersioning:
    """DVC-based data versioning system for churn prediction pipeline."""

    def __init__(self):
        """Initialize DVC versioning system."""
        self.setup_dvc()
        logger.info("DVC versioning system initialized")

    def setup_dvc(self) -> bool:
        """Set up DVC for data versioning."""
        try:
            # Check if DVC is installed
            result = subprocess.run(
                ['dvc', '--version'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.error("DVC not installed. Install with: pip install dvc")
                return False

            # Initialize DVC if not already done
            if not os.path.exists('.dvc'):
                subprocess.run(['dvc', 'init'], check=True)
                logger.info("DVC initialized")

            # Configure DVC cache
            subprocess.run(['dvc', 'config', 'cache.type', 'copy'], check=True)
            logger.info("DVC cache configured")

            return True

        except subprocess.CalledProcessError as e:
            logger.error("DVC setup failed: %s", str(e))
            return False
        except Exception as e:
            logger.error("DVC setup failed: %s", str(e))
            return False

    def add_data_to_dvc(self, data_path: str) -> bool:
        """Add data directory or file to DVC tracking."""
        try:
            if not os.path.exists(data_path):
                logger.error("Data path does not exist: %s", data_path)
                return False

            # Add to DVC
            subprocess.run(['dvc', 'add', data_path], check=True)
            logger.info("Added %s to DVC tracking", data_path)

            # Add .dvc file to git
            dvc_file = f"{data_path}.dvc"
            if os.path.exists(dvc_file):
                subprocess.run(['git', 'add', dvc_file], check=True)
                logger.info("Added %s to git", dvc_file)

            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to add %s to DVC: %s", data_path, str(e))
            return False

    def create_version(self, message: str, tag: Optional[str] = None) -> bool:
        """Create a new version with current data state."""
        try:
            # Add all .dvc files to git
            subprocess.run(['git', 'add', '*.dvc'], check=True)
            subprocess.run(['git', 'add', '.dvcignore'], check=False)

            # Commit changes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            commit_message = f"Data version {timestamp}: {message}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)

            # Create tag if specified
            if tag:
                subprocess.run(['git', 'tag', '-a', tag, '-m', message], check=True)
                logger.info("Created version with tag: %s", tag)
            else:
                logger.info("Created version: %s", commit_message)

            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to create version: %s", str(e))
            return False

    def checkout_version(self, version: str) -> bool:
        """Checkout a specific version of data."""
        try:
            # Checkout git commit or tag
            subprocess.run(['git', 'checkout', version], check=True)

            # Checkout DVC data
            subprocess.run(['dvc', 'checkout'], check=True)

            logger.info("Checked out version: %s", version)
            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to checkout version %s: %s", version, str(e))
            return False

    def list_versions(self) -> List[Dict]:
        """List all available data versions."""
        try:
            # Get git log with tags
            result = subprocess.run([
                'git', 'log', '--oneline', '--decorate', '--grep=Data version'
            ], capture_output=True, text=True, check=True)

            versions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        commit_hash = parts[0]
                        message = parts[1]
                        versions.append({
                            'commit': commit_hash,
                            'message': message,
                            'timestamp': self._get_commit_timestamp(commit_hash)
                        })

            return versions

        except subprocess.CalledProcessError as e:
            logger.error("Failed to list versions: %s", str(e))
            return []

    def _get_commit_timestamp(self, commit_hash: str) -> str:
        """Get timestamp for a specific commit."""
        try:
            result = subprocess.run([
                'git', 'show', '-s', '--format=%ci', commit_hash
            ], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "Unknown"

    def get_data_status(self) -> Dict:
        """Get current status of DVC-tracked data."""
        try:
            result = subprocess.run([
                'dvc', 'status'
            ], capture_output=True, text=True, check=True)

            status = {
                'clean': 'nothing to commit' in result.stdout.lower(),
                'output': result.stdout,
                'tracked_files': self._get_tracked_files()
            }

            return status

        except subprocess.CalledProcessError as e:
            logger.error("Failed to get DVC status: %s", str(e))
            return {'clean': False, 'output': str(e), 'tracked_files': []}

    def _get_tracked_files(self) -> List[str]:
        """Get list of DVC-tracked files."""
        try:
            dvc_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.dvc'):
                        dvc_files.append(os.path.join(root, file))
            return dvc_files
        except Exception as e:
            logger.error("Failed to get tracked files: %s", str(e))
            return []

    def push_to_remote(self, remote: Optional[str] = None) -> bool:
        """Push data to DVC remote storage."""
        try:
            cmd = ['dvc', 'push']
            if remote:
                cmd.extend(['-r', remote])

            subprocess.run(cmd, check=True)
            logger.info("Data pushed to remote storage")
            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to push to remote: %s", str(e))
            return False

    def pull_from_remote(self, remote: Optional[str] = None) -> bool:
        """Pull data from DVC remote storage."""
        try:
            cmd = ['dvc', 'pull']
            if remote:
                cmd.extend(['-r', remote])

            subprocess.run(cmd, check=True)
            logger.info("Data pulled from remote storage")
            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to pull from remote: %s", str(e))
            return False

    def setup_remote_storage(self, name: str, url: str) -> bool:
        """Setup DVC remote storage."""
        try:
            subprocess.run(['dvc', 'remote', 'add', name, url], check=True)
            subprocess.run(['dvc', 'remote', 'default', name], check=True)
            logger.info("Remote storage configured: %s -> %s", name, url)
            return True

        except subprocess.CalledProcessError as e:
            logger.error("Failed to setup remote storage: %s", str(e))
            return False

    def setup_s3_remote(self) -> bool:
        """Setup S3 remote storage using environment variables."""
        try:
            import os
            from dotenv import load_dotenv
            
            # Load environment variables
            load_dotenv()
            
            bucket_name = os.getenv('S3_BUCKET_NAME')
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            
            if not bucket_name:
                logger.error("S3_BUCKET_NAME not found in environment variables")
                return False
            
            # Setup S3 remote (only if not already configured)
            try:
                result = subprocess.run(['dvc', 'remote', 'list'], capture_output=True, text=True, check=True)
                if 's3remote' not in result.stdout:
                    s3_url = f"s3://{bucket_name}/dvc-storage"
                    subprocess.run(['dvc', 'remote', 'add', '-d', 's3remote', s3_url], check=True)
                    subprocess.run(['dvc', 'remote', 'modify', 's3remote', 'region', aws_region], check=True)
                    logger.info("S3 remote storage configured: %s", s3_url)
                else:
                    logger.info("S3 remote already configured")
            except subprocess.CalledProcessError:
                logger.warning("Could not configure S3 remote, using existing configuration")
            
            # Note: AWS credentials should be set via environment variables or AWS CLI
            # DVC will automatically use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
            # from environment variables, avoiding storing them in config files
            
            return True
            
        except Exception as e:
            logger.error("Failed to setup S3 remote: %s", str(e))
            return False


def setup_pipeline_versioning():
    """Setup DVC versioning for the entire pipeline."""
    versioning = DVCVersioning()

    # Setup S3 remote storage if configured
    import os
    if os.getenv('S3_BUCKET_NAME'):
        logger.info("Setting up S3 remote storage...")
        versioning.setup_s3_remote()

    # For pipeline-managed outputs, we don't add them individually
    # The pipeline itself manages these directories as outputs
    logger.info("DVC pipeline versioning setup complete")

    return versioning


def version_pipeline_step(step_name: str, description: str):
    """Version data after a pipeline step."""
    versioning = DVCVersioning()

    # For pipeline outputs, we don't add them individually to DVC
    # They are managed by the pipeline itself
    # Just create a git commit to track the step completion
    
    # Create version
    tag = f"{step_name.lower().replace(' ', '_')}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Add any changes to git (but not .dvc files for pipeline outputs)
        subprocess.run(['git', 'add', 'logs/', 'reports/'], check=False)
        
        # Commit changes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit_message = f"Pipeline step {timestamp}: {step_name} - {description}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=False)
        
        logger.info("Created version for step: %s", step_name)
    except Exception as e:
        logger.warning("Could not create git commit for step %s: %s", step_name, str(e))
    
    return tag


if __name__ == "__main__":
    # Setup pipeline versioning
    setup_pipeline_versioning()