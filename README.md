# Customer Churn Prediction - End-to-End ML Pipeline

A comprehensive data management pipeline for customer churn prediction, implementing all stages from data ingestion to model deployment with automated versioning and monitoring.

## Project Overview

This project implements a complete end-to-end ML pipeline for predicting customer churn in telecommunications, automating data collection, validation, feature engineering, model training, and deployment with comprehensive logging and versioning.

## Dataset

**Primary Dataset**: IBM Telco Customer Churn Dataset
- **Source**: Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size**: 7,043 customers with 21 features
- **Target**: Binary churn classification (Yes/No)

### Features:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account**: Contract, PaperlessBilling, PaymentMethod
- **Financial**: MonthlyCharges, TotalCharges
- **Behavioral**: tenure (months with company)

## Pipeline Architecture

1. **Problem Formulation** - Business problem definition and objectives
2. **Data Ingestion** - Fetch data from CSV + Hugging Face API
3. **Raw Data Storage** - Organize and catalog raw data
4. **Data Validation** - Validate data quality and integrity
5. **Data Preparation** - Clean and preprocess data
6. **Data Transformation** - Feature engineering and storage
7. **Feature Store** - Manage engineered features
8. **Data Versioning** - DVC-based version control
9. **Model Training** - Train and evaluate ML models

## Project Structure

```
churn-prediction-pipeline/
├── config/                        # DVC and environment configuration
├── scripts/                       # Setup and utility scripts
├── src/                           # Source code
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_preparation.py
│   ├── data_transformation_storage.py
│   ├── feature_store.py
│   ├── data_versioning.py
│   ├── build_model.py
│   └── utils/
├── data/                          # DVC-tracked data storage
│   ├── raw/         cleaned/      processed/
│   ├── feature_store/             eda/
│   └── models/
├── database/                      # SQLite schema
├── docs/                          # Documentation
├── logs/                          # Pipeline logs
├── reports/                       # Generated reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main_pipeline.py
```

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
git clone <repository-url>
cd churn-prediction-pipeline
bash scripts/setup_project.sh
nano .env
python main_pipeline.py
```

### Option 2: Manual Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/env/.env.example .env && nano .env
bash scripts/setup_dvc.sh
python main_pipeline.py
```

### Option 3: Docker
```bash
cp config/env/.env.example .env && nano .env
docker-compose up -d
```

## Pipeline Components

| Step | Script | Output |
|------|--------|--------|
| Data Ingestion | `src/data_ingestion.py` | `data/raw/` |
| Data Validation | `src/data_validation.py` | `reports/validation_reports/` |
| Data Preparation | `src/data_preparation.py` | `data/cleaned/` |
| Data Transformation | `src/data_transformation_storage.py` | `data/processed/training_sets/` |
| Feature Store | `src/feature_store.py` | `data/feature_store/` |
| Data Versioning | `src/data_versioning.py` | DVC `.dvc` files |
| Model Training | `src/build_model.py` | `src/models/` |

### DVC Data Versioning Features:
- Git-like versioning with automatic version creation per pipeline step
- Reproducibility — exact data states can be recreated
- Optional remote/cloud storage integration

## Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | > 85% |
| Precision | > 80% |
| Recall | > 75% |
| F1-Score | > 0.80 |
| AUC-ROC | > 0.85 |

**Business Impact**: 5% quarterly churn reduction, reduced acquisition costs, maintained customer lifetime value.

## Configuration

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL=INFO
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export S3_BUCKET_NAME=your-bucket-name
```

- **Database**: SQLite (local) / PostgreSQL (production)
- **Feature Store**: CSV-based (extensible to Redis/PostgreSQL)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Import Errors | `pip install -r requirements.txt` + set `PYTHONPATH` |
| Data File Not Found | Verify `data/raw/customer_data.csv` exists |
| Permission Errors | `chmod -R 755 data/ logs/ reports/` |
| Memory Issues | Subsample with `head -1000` for testing |

Enable debug logging: `export LOG_LEVEL=DEBUG`

## Documentation

- `problem_formulation.md` — Business problem definition
- `docs/DVC_Data_Versioning_Guide.md` — DVC guide
- `docs/FEATURE_STORE_README.md` — Feature store details
- `docs/TRANSFORMATION_STORAGE.md` — Transformation details

## License

Educational purposes only. Dataset license follows IBM terms.

## References
- [DVC Documentation](https://dvc.org/doc) · [scikit-learn](https://scikit-learn.org/stable/) · [Pandas](https://pandas.pydata.org/docs/) · [SQLite](https://www.sqlite.org/docs.html)