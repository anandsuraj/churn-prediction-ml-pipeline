# Customer Churn Prediction - End-to-End ML Pipeline

A comprehensive data management pipeline for customer churn prediction, implementing all stages from data ingestion to model deployment with automated versioning and monitoring.

## Project Overview

This project implements a complete end-to-end machine learning pipeline for predicting customer churn in telecommunications. The pipeline automates the entire ML workflow including data collection, validation, feature engineering, model training, and deployment with comprehensive logging and versioning.

## Dataset

**Primary Dataset**: IBM Telco Customer Churn Dataset
- **Source**: Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size**: 7,043 customers with 21 features
- **Target**: Binary churn classification (Yes/No)

### Dataset Features:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account**: Contract, PaperlessBilling, PaymentMethod
- **Financial**: MonthlyCharges, TotalCharges
- **Behavioral**: tenure (months with company)
- **Target**: Churn (Yes/No)

## Complete Pipeline Architecture

### Pipeline Steps (main_pipeline.py):

1. **Step 1: Problem Formulation** - Business problem definition and objectives
2. **Step 2: Data Ingestion** - Fetch data from multiple sources (CSV + Hugging Face API)
3. **Step 3: Raw Data Storage** - Organize and catalog raw data
4. **Step 4: Data Validation** - Validate data quality and integrity
5. **Step 5: Data Preparation** - Clean and preprocess data
6. **Step 6: Data Transformation** - Feature engineering and storage
7. **Step 7: Feature Store** - Manage engineered features
8. **Step 8: Data Versioning** - DVC-based version control for datasets
9. **Step 9: Model Training** - Train and evaluate ML models

## Project Structure

```
churn-prediction-pipeline/
├── config/                        # Configuration files
│   ├── dvc/                       # DVC configuration
│   │   ├── dvc.yaml               # Pipeline definition
│   │   ├── dvc.lock               # Pipeline lock file
│   │   └── .dvcignore             # DVC ignore patterns
│   ├── env/                       # Environment configuration
│   │   └── .env.example           # Environment template
│   └── README.md                  # Configuration guide
├── scripts/                       # Setup and utility scripts
│   ├── setup_project.sh           # Complete project setup
│   ├── setup_dvc.sh               # DVC setup script
│   └── setup_dvc_credentials.sh   # DVC credentials setup
├── src/                           # Source code
│   ├── data_ingestion.py          # Step 2: Data ingestion
│   ├── data_validation.py         # Step 4: Data validation
│   ├── data_preparation.py        # Step 5: Data preparation
│   ├── data_transformation_storage.py # Step 6: Data transformation
│   ├── feature_store.py           # Step 7: Feature store
│   ├── data_versioning.py         # Step 8: Data versioning
│   ├── build_model.py             # Step 9: Model training
│   └── utils/                     # Utility functions
├── data/                          # Data storage (DVC tracked)
│   ├── raw/                       # Raw ingested data
│   ├── cleaned/                   # Cleaned data
│   ├── processed/                 # Transformed data
│   │   └── training_sets/         # ML-ready datasets
│   ├── feature_store/             # Feature store
│   ├── eda/                       # Exploratory data analysis
│   │   ├── raw/                   # Raw data EDA
│   │   └── cleaned/               # Cleaned data EDA
│   └── models/                    # Trained models
├── database/                      # Database setup
│   └── init.sql                   # SQLite schema
├── docs/                          # Documentation
│   ├── DVC_Data_Versioning_Guide.md # DVC guide
│   └── DM4ML_Assignment_Detailed_Instructions.md
├── logs/                          # Pipeline logs
├── reports/                       # Generated reports
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker services
├── requirements.txt               # Python dependencies
├── main_pipeline.py               # Main pipeline runner
├── problem_formulation.md         # Business problem definition
├── dvc.yaml -> config/dvc/dvc.yaml    # Symbolic link
├── dvc.lock -> config/dvc/dvc.lock    # Symbolic link
├── .dvcignore -> config/dvc/.dvcignore # Symbolic link
├── .env.example -> config/env/.env.example # Symbolic link
└── README.md                      # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
git clone <repository-url>
cd churn-prediction-pipeline

# Run complete setup script
bash scripts/setup_project.sh

# Edit environment variables
nano .env

# Run pipeline
python main_pipeline.py
```

#### Option 2: Manual Setup
```bash
git clone <repository-url>
cd churn-prediction-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp config/env/.env.example .env
nano .env  # Edit with your credentials

# Set up DVC
bash scripts/setup_dvc.sh

# Run pipeline
python main_pipeline.py
```

#### Option 3: Docker Setup
```bash
git clone <repository-url>
cd churn-prediction-pipeline

# Set up environment
cp config/env/.env.example .env
nano .env  # Edit with your credentials

# Run with Docker Compose
docker-compose up

# Or run setup first
docker-compose --profile setup up dvc-setup
docker-compose up pipeline
```

## Pipeline Components

### 1. Data Ingestion (`src/data_ingestion.py`)
- **Purpose**: Fetch data from multiple sources
- **Features**: CSV loading, Hugging Face API integration
- **Output**: Raw data in `data/raw/`
- **Logs**: `logs/data_ingestion.log`

### 2. Raw Data Storage (`src/raw_data_storage.py`)
- **Purpose**: Organize and catalog raw data
- **Features**: Directory structure, data catalog
- **Output**: Organized data hierarchy
- **Logs**: `logs/raw_data_storage.log`

### 3. Data Validation (`src/data_validation.py`)
- **Purpose**: Validate data quality and integrity
- **Features**: Schema validation, quality checks, statistical analysis
- **Output**: Validation reports in `reports/validation_reports/`
- **Logs**: `logs/data_validation.log`

### 4. Data Preparation (`src/data_preparation.py`)
- **Purpose**: Clean and preprocess raw data
- **Features**: Missing value handling, categorical encoding, data type conversion
- **Output**: Cleaned data in `data/cleaned/churn_data_cleaned.csv`
- **Logs**: `logs/data_preparation.log`

### 5. Data Transformation (`src/data_transformation_storage.py`)
- **Purpose**: Engineer features and store in database
- **Features**: Feature engineering, scaling, SQLite storage
- **Output**: Transformed features in `data/processed/training_sets/`
- **Logs**: `logs/data_transformation_storage.log`

### 6. Feature Store (`src/feature_store.py`)
- **Purpose**: Manage engineered features
- **Features**: Feature retrieval API, metadata tracking
- **Output**: Feature store in `data/feature_store/`
- **Logs**: `logs/feature_store.log`

### 7. Data Versioning (`src/data_versioning.py`)
- **Purpose**: DVC-based version control for datasets
- **Features**: Git-like versioning, reproducibility, collaboration
- **Output**: DVC-tracked data with `.dvc` files
- **Logs**: `logs/data_versioning.log`

#### DVC Data Versioning Features:
- **Automatic Versioning**: Each pipeline step creates a version
- **Git Integration**: Versions tracked in Git with tags
- **Reproducibility**: Exact data states can be recreated
- **Remote Storage**: Optional cloud storage integration
- **Collaboration**: Team can work with consistent data versions

### 8. Model Training (`src/build_model.py`)
- **Purpose**: Train and evaluate ML models
- **Features**: Multiple algorithms, hyperparameter tuning, model evaluation
- **Output**: Trained models in `src/models/`
- **Logs**: `logs/build_model.log`

## Expected Performance

### Model Performance Targets:
- **Accuracy**: > 85%
- **Precision**: > 80%
- **Recall**: > 75%
- **F1-Score**: > 0.8
- **AUC-ROC**: > 0.85

### Business Impact:
- **Churn Reduction**: 5% decrease in quarterly churn rate
- **Cost Savings**: Reduced customer acquisition costs
- **Revenue Protection**: Maintained customer lifetime value

## Usage Examples

### Run Individual Steps:
```bash
# Data ingestion
python src/data_ingestion.py

# Data validation
python src/data_validation.py

# Data preparation
python src/data_preparation.py

# Data transformation
python src/data_transformation_storage.py

# Feature store
python src/feature_store.py

# Data versioning (DVC-based)
dvc status                     # Check data status
dvc push                       # Push data to S3
dvc pull                       # Pull data from S3

# Model training
python src/build_model.py
```

### Run Complete Pipeline:
```bash
python main_pipeline.py
```

### DVC Data Versioning Commands:
```bash
# Initialize DVC (already done)
dvc init

# Check data status
dvc status

# Run pipeline and track outputs
dvc repro

# Push data to remote storage (S3)
dvc push

# Pull data from remote storage
dvc pull

# Check version history
git log --oneline --grep="Data version"

# List all version tags
git tag -l

# Checkout specific data version
git checkout <version-tag>
dvc checkout

# Add new data to tracking
dvc add data/new_dataset.csv

# Configure S3 remote
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote region us-east-1
```

## Monitoring and Logging

### Log Files:
- All pipeline steps generate detailed logs in `logs/`
- Each log file contains timestamps, error handling, and progress tracking
- Logs are automatically rotated and maintained

### Reports:
- Validation reports: `reports/validation_reports/`
- Model performance: `reports/model_performance/`
- Data quality: `reports/data_quality/`

### Version Tracking:
- Data versions: `data/versions/version_metadata.json`
- Version reports: `data/versions/version_report.md`

## Docker Support

### Build and Run:
```bash
# Build Docker image (includes DVC setup)
docker build -t churn-prediction-pipeline .

# Run pipeline with DVC versioning
docker run -v $(pwd)/data:/app/data churn-prediction-pipeline

# Run with Docker Compose
docker-compose up -d
```

### Docker with DVC:
```bash
# Run pipeline with DVC versioning
docker run -v $(pwd)/data:/app/data churn-pipeline

# Interactive container with DVC
docker run -it -v $(pwd):/app churn-pipeline bash

# Inside container:
dvc status
dvc push
```

### Docker Services:
- Application container with all dependencies
- DVC data versioning automatically configured
- SQLite database (can be upgraded to PostgreSQL)
- Volume mounts for data persistence

## DVC Data Versioning

This project uses DVC (Data Version Control) for data versioning and pipeline management.

### DVC Setup

1. **Install DVC with S3 support:**
```bash
pip install dvc dvc-s3
```

2. **Initialize DVC (already done):**
```bash
dvc init
```

3. **Configure environment variables:**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your credentials
nano .env

# Required variables:
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=your_region
# S3_BUCKET_NAME=your_bucket_name
```

4. **Configure S3 remote storage:**
```bash
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote region your-region
# Credentials are automatically read from environment variables
```

### DVC Pipeline Commands

```bash
# Run the complete pipeline
dvc repro

# Check pipeline status
dvc status

# Push data to S3
dvc push

# Pull data from S3
dvc pull

# Show pipeline DAG
dvc dag

# Show pipeline metrics
dvc metrics show
```

### DVC Data Management

```bash
# Add new data file to DVC tracking
dvc add data/new_file.csv
git add data/new_file.csv.dvc .gitignore
git commit -m "Add new data file"

# Check data status
dvc status

# Compare data versions
dvc diff

# List tracked files
dvc list . --dvc-only
```

### DVC Version Control

```bash
# Create data version tag
git tag -a v1.0 -m "Initial data version"

# Checkout specific version
git checkout v1.0
dvc checkout

# List all versions
git tag -l

# Show version differences
dvc diff HEAD~1
```

### DVC Remote Storage

```bash
# List remotes
dvc remote list

# Test remote connection
dvc remote modify s3remote --test

# Push specific stage
dvc push full_pipeline

# Pull specific stage
dvc pull full_pipeline
```

## Configuration

### Environment Variables:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL=INFO
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=your-region
export S3_BUCKET_NAME=your-bucket-name
```

### Database Configuration:
- **Default**: SQLite for local development
- **Production**: PostgreSQL/MySQL recommended

### Feature Store Configuration:
- CSV-based storage for simplicity
- Extensible to Redis/PostgreSQL for production

## Testing

### Run Tests:
```bash
# Test individual components
python -c "from src.data_ingestion import DataIngestionPipeline; print('Data ingestion ready')"
python -c "from src.data_validation import DataValidator; print('Data validation ready')"
python -c "from src.data_preparation import DataPreparationPipeline; print('Data preparation ready')"
```

### Validate Pipeline:
```bash
# Check all components
python main_pipeline.py
```

## Troubleshooting

### Common Issues:

1. **Import Errors:**
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Data File Not Found:**
   ```bash
   # Ensure dataset is in data/raw/customer_data.csv
   ls -la data/raw/
   ```

3. **Permission Errors:**
   ```bash
   # Fix directory permissions
   chmod -R 755 data/ logs/ reports/
   ```

4. **Memory Issues:**
   ```bash
   # Reduce dataset size for testing
   head -1000 data/raw/customer_data.csv > data/raw/customer_data_sample.csv
   ```

### Debug Mode:
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main_pipeline.py
```

## Performance Optimization

### For Large Datasets:
- Use data chunking in processing
- Implement parallel processing
- Optimize database queries
- Use memory-efficient data structures

### For Production:
- Implement caching mechanisms
- Add monitoring and alerting
- Set up automated retraining
- Implement A/B testing framework

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all pipeline steps work
5. Submit pull request

## Documentation

- **Problem Formulation**: `problem_formulation.md`
- **DVC Data Versioning**: `docs/DVC_Data_Versioning_Guide.md`
- **Feature Store**: `docs/FEATURE_STORE_README.md`
- **Transformation**: `docs/TRANSFORMATION_STORAGE.md`

## License

This project is for educational purposes. Dataset license follows IBM terms.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review logs in `logs/` directory
3. Check documentation in `docs/`
4. Create GitHub issue with details

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)