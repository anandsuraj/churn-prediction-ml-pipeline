# DVC Data Versioning Guide

## Overview

This project uses **Data Version Control (DVC)** for managing data versioning in the customer churn prediction pipeline. DVC provides Git-like versioning for data files, ensuring reproducibility and collaboration.

## What is DVC?

DVC (Data Version Control) is an open-source tool that brings version control to machine learning projects. It:
- Tracks large data files and ML models
- Integrates with Git for metadata versioning
- Supports cloud storage (S3, GCS, Azure, etc.)
- Enables reproducible ML pipelines

## Current Setup

### DVC Configuration
- **Initialized**: DVC is already set up in this project
- **Remote Storage**: S3 bucket `churn-data-lake/dvc-storage`
- **Tracked Directories**: `data/raw/`, `data/cleaned/`, `data/processed/`, `data/feature_store/`

### Pipeline Integration
The pipeline automatically creates data versions at each step:
1. **Data Ingestion** → Version with tag `data_ingestion_vYYYYMMDD_HHMMSS`
2. **Data Preparation** → Version with tag `data_preparation_vYYYYMMDD_HHMMSS`
3. **Data Transformation** → Version with tag `data_transformation_vYYYYMMDD_HHMMSS`
4. **Pipeline Complete** → Version with tag `pipeline_complete_vYYYYMMDD_HHMMSS`

## DVC Commands Reference

### Pipeline Management
```bash
# Run the complete pipeline
dvc repro

# Check pipeline status
dvc status

# Show pipeline DAG
dvc dag

# Show pipeline details
dvc pipeline show

# Force run all stages
dvc repro --force

# Run specific stage
dvc repro stage_name
```

### Data Management
```bash
# Add data to DVC tracking
dvc add data/raw/new_dataset.csv
git add data/raw/new_dataset.csv.dvc .gitignore
git commit -m "Add new dataset"

# Remove from DVC tracking
dvc remove data/raw/old_dataset.csv.dvc

# Check data status
dvc status

# Checkout DVC-tracked files
dvc checkout

# List DVC-tracked files
dvc list . --dvc-only
```

### Remote Storage (S3)
```bash
# List configured remotes
dvc remote list

# Add S3 remote (credentials from environment variables)
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote region your-region
# Note: AWS credentials are read from environment variables:
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

# Test remote connection
dvc remote modify s3remote --test

# Push data to S3
dvc push

# Pull data from S3
dvc pull

# Push specific stage
dvc push full_pipeline

# Fetch metadata only (no checkout)
dvc fetch
```

### Version Control
```bash
# Create version tag
git tag -a v1.0 -m "Initial data version"
git push origin v1.0

# List all versions
git tag -l
git log --oneline --grep="Data version"

# Switch to specific version
git checkout v1.0
dvc checkout

# Compare versions
dvc diff
dvc diff HEAD~1

# Show version differences
dvc metrics diff
dvc plots diff
```

### Debugging & Maintenance
```bash
# Check DVC setup
dvc doctor

# Show configuration
dvc config --list
cat .dvc/config

# Show cache directory
dvc cache dir

# Clean cache
dvc gc --workspace --cloud

# Show DVC version
dvc version

# Verbose output
dvc repro --verbose
dvc push --verbose
```

## Automated Versioning

The pipeline automatically handles versioning through the `DVCVersioning` class in `src/data_versioning.py`:

```python
from src.data_versioning import version_pipeline_step

# Automatically version after each step
tag = version_pipeline_step(
    "Data Ingestion",
    "Raw data from multiple sources"
)
```

## S3 Integration

### Configuration
- **Bucket**: `churn-data-lake`
- **Path**: `dvc-storage/`
- **Region**: `ap-south-1`
- **Credentials**: From `.env` file

### Remote Commands
```bash
dvc remote list                    # Show configured remotes
dvc remote modify s3remote url s3://new-bucket/path    # Change remote URL
```

## File Structure

```
project/
├── .dvc/                    # DVC configuration
├── data/
│   ├── raw/                 # Raw data (DVC tracked)
│   ├── cleaned/             # Cleaned data (DVC tracked)
│   ├── processed/           # Processed data (DVC tracked)
│   └── feature_store/       # Feature store (DVC tracked)
├── src/
│   └── data_versioning.py   # DVC versioning implementation
├── dvc.yaml                 # DVC pipeline definition
├── .dvcignore              # DVC ignore patterns
└── .gitignore              # Git ignore patterns (includes data/)
```

## Best Practices

### 1. Version Naming
- Use semantic versioning: `v1.0.0`, `v1.1.0`, `v2.0.0`
- Include descriptive messages: "Added customer demographics data"
- Use timestamps for automatic versions: `data_ingestion_v20240824_143022`

### 2. Data Organization
- Keep raw data immutable
- Version intermediate processing steps
- Track feature engineering outputs
- Version model artifacts separately

### 3. Collaboration
- Always pull before making changes: `git pull && dvc pull`
- Push both Git and DVC changes: `git push && dvc push`
- Use descriptive commit messages for data changes

### 4. Storage Management
- Monitor S3 storage costs
- Clean up old versions periodically: `dvc gc --workspace --cloud`
- Use appropriate S3 storage classes for archival

## Troubleshooting

### Common Issues & Solutions

1. **Pipeline Reproduction Issues**
   ```bash
   # Force reproduction
   dvc repro --force
   
   # Check dependencies
   dvc dag
   
   # Validate pipeline
   dvc pipeline show
   ```

2. **S3 Permission/Connection Errors**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Verify S3 bucket access
   aws s3 ls s3://churn-data-lake/
   
   # Test DVC remote
   dvc remote modify s3remote --test
   
   # Check DVC S3 configuration
   dvc config --list | grep s3remote
   ```

3. **DVC Status Shows Changes**
   ```bash
   # Restore files to match DVC state
   dvc checkout
   
   # Check what changed
   dvc diff
   
   # Force checkout
   dvc checkout --force
   ```

4. **Large Files in Git**
   ```bash
   # Remove from Git, add to DVC
   git rm --cached large_file.csv
   dvc add large_file.csv
   git add large_file.csv.dvc .gitignore
   git commit -m "Move large file to DVC"
   ```

5. **Cache Issues**
   ```bash
   # Check cache size
   du -sh .dvc/cache
   
   # Clean unused cache
   dvc gc --workspace --cloud
   
   # Remove all cache
   rm -rf .dvc/cache
   dvc pull
   ```

6. **Network/Connectivity Issues**
   ```bash
   # Test with verbose output
   dvc push --verbose
   dvc pull --verbose
   
   # Check remote configuration
   dvc remote list
   dvc config --list
   ```

### Debug Commands
```bash
# Comprehensive health check
dvc doctor

# Show all configuration
dvc config --list

# Show cache information
dvc cache dir
du -sh .dvc/cache

# Validate DVC files
find . -name "*.dvc" -exec dvc status {} \;

# Check Git integration
git status
git log --oneline --grep="Data version"
```

### Emergency Recovery
```bash
# If DVC is completely broken
rm -rf .dvc/cache
dvc pull --force

# If pipeline is corrupted
rm dvc.lock
dvc repro --force

# If remote is inaccessible
dvc remote remove s3remote
dvc remote add -d s3remote s3://backup-bucket/dvc-storage
```

## Integration with Pipeline

The main pipeline (`main_pipeline.py`) automatically:
1. **Tracks data** at each processing step
2. **Creates versions** with descriptive tags
3. **Logs version information** for audit trails
4. **Maintains data lineage** through Git history

## Docker Integration

DVC is automatically configured in Docker containers:
- Environment variables passed from `.env`
- S3 credentials configured automatically
- Data directories properly mounted

## Monitoring and Maintenance

### Regular Tasks
- **Weekly**: Check S3 storage usage and costs
- **Monthly**: Clean up old versions with `dvc gc`
- **Quarterly**: Review and archive old data versions

### Health Checks
```bash
dvc status          # Check data consistency
git log --oneline   # Review version history
dvc remote list     # Verify remote configuration
```

## Security Best Practices

### Credential Management
- **Never commit credentials** to Git repositories
- **Use environment variables** for AWS credentials:
  ```bash
  export AWS_ACCESS_KEY_ID=your_access_key
  export AWS_SECRET_ACCESS_KEY=your_secret_key
  ```
- **Use .env files** for local development (excluded from Git)
- **Use IAM roles** in production environments
- **Rotate credentials** regularly

### File Exclusions
The `.gitignore` file excludes:
- DVC cache and temporary files
- Credential files (`.env`, `*.pem`, `*.key`)
- Large data files (tracked by DVC instead)
- Temporary processing files

### Production Security
- Use AWS IAM roles instead of access keys
- Enable S3 bucket encryption
- Set up proper S3 bucket policies
- Monitor access logs
- Use VPC endpoints for private access

## Summary

DVC provides enterprise-grade data versioning for this ML pipeline:
- **Automatic versioning** at each pipeline step
- **S3 cloud storage** for team collaboration
- **Git integration** for metadata tracking
- **Reproducible experiments** with exact data states
- **Scalable storage** for large datasets
- **Secure credential management** via environment variables

The system is production-ready and requires minimal manual intervention for day-to-day operations.