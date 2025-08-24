#!/usr/bin/env python3
"""
DVC Setup Script
===============
Automated setup for DVC data versioning in the churn prediction pipeline.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dvc_setup')


def check_git_repo():
    """Check if we're in a git repository."""
    try:
        subprocess.run(['git', 'status'], capture_output=True, check=True)
        logger.info("Git repository detected")
        return True
    except subprocess.CalledProcessError:
        logger.info("Initializing git repository...")
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'config', 'user.name', 'Pipeline User'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'pipeline@example.com'], check=True)
        return True


def install_dvc():
    """Install DVC if not already installed."""
    try:
        subprocess.run(['dvc', '--version'], capture_output=True, check=True)
        logger.info("DVC is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing DVC...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'dvc'], check=True)
        logger.info("DVC installed successfully")
        return True


def setup_dvc():
    """Setup DVC in the project."""
    try:
        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            logger.info("DVC initialized")
        else:
            logger.info("DVC already initialized")

        # Configure DVC settings
        subprocess.run(['dvc', 'config', 'cache.type', 'copy'], check=True)
        subprocess.run(['dvc', 'config', 'core.analytics', 'false'], check=True)
        logger.info("DVC configured")

        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to setup DVC: %s", str(e))
        return False


def create_data_directories():
    """Create necessary data directories."""
    directories = [
        'data/raw',
        'data/cleaned',
        'data/processed',
        'data/feature_store',
        'data/models',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", directory)

    # Create .gitkeep files to track empty directories
    for directory in directories:
        gitkeep_path = Path(directory) / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.touch()


def setup_gitignore():
    """Setup .gitignore for DVC project."""
    gitignore_content = """
# DVC
/data/raw/*
/data/cleaned/*
/data/processed/*
/data/feature_store/*
/data/models/*
!*.dvc
!.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log
*.log

# Environment variables
.env

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/
"""

    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content.strip())
    logger.info("Created .gitignore")


def setup_dvcignore():
    """Setup .dvcignore file."""
    dvcignore_content = """
# Ignore log files
*.log
logs/

# Ignore temporary files
*.tmp
*.temp

# Ignore system files
.DS_Store
Thumbs.db
"""

    with open('.dvcignore', 'w', encoding='utf-8') as f:
        f.write(dvcignore_content.strip())
    logger.info("Created .dvcignore")


def create_dvc_pipeline():
    """Create DVC pipeline configuration."""
    pipeline_config = """
stages:
  data_ingestion:
    cmd: python main_pipeline.py --step ingestion
    deps:
    - src/data_ingestion.py
    outs:
    - data/raw/

  data_preparation:
    cmd: python main_pipeline.py --step preparation
    deps:
    - src/data_preparation.py
    - data/raw/
    outs:
    - data/cleaned/

  data_transformation:
    cmd: python main_pipeline.py --step transformation
    deps:
    - src/data_transformation_storage.py
    - data/cleaned/
    outs:
    - data/processed/

  feature_store:
    cmd: python main_pipeline.py --step feature_store
    deps:
    - src/feature_store.py
    - data/processed/
    outs:
    - data/feature_store/
"""

    with open('dvc.yaml', 'w', encoding='utf-8') as f:
        f.write(pipeline_config.strip())
    logger.info("Created DVC pipeline configuration")


def initial_commit():
    """Create initial git commit."""
    try:
        # Add DVC files to git
        subprocess.run(['git', 'add', '.dvc/', '.dvcignore', '.gitignore'], check=True)
        subprocess.run(['git', 'add', 'dvc.yaml'], check=True)
        subprocess.run(['git', 'add', 'setup_dvc.py'], check=True)

        # Check if there are changes to commit
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
        if result.returncode != 0:
            subprocess.run(['git', 'commit', '-m', 'Initial DVC setup'], check=True)
            logger.info("Created initial commit")
        else:
            logger.info("No changes to commit")

    except subprocess.CalledProcessError as e:
        logger.warning("Failed to create initial commit: %s", str(e))


def main():
    """Main setup function."""
    logger.info("Starting DVC setup...")

    try:
        # Check and setup git
        check_git_repo()

        # Install and setup DVC
        install_dvc()
        setup_dvc()

        # Create project structure
        create_data_directories()
        setup_gitignore()
        setup_dvcignore()
        create_dvc_pipeline()

        # Initial commit
        initial_commit()

        logger.info("DVC setup completed successfully!")
        logger.info("You can now run: python test_dvc.py to test the setup")

    except Exception as e:
        logger.error("DVC setup failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()