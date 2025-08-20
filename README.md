# Customer Churn Prediction - End-to-End ML Pipeline

This project implements a complete data management pipeline for machine learning, specifically designed for customer churn prediction. The pipeline covers all stages from data ingestion to model deployment and orchestration.

## Project Overview

Customer churn prediction is crucial for businesses to identify at-risk customers and implement retention strategies. This pipeline automates the entire ML workflow including data collection, validation, feature engineering, model training, and deployment.

## Dataset

**Recommended Dataset**: IBM Telco Customer Churn Dataset from Kaggle
- **URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size**: 7,043 customers with 21 features
- **Target**: Binary churn classification (Yes/No)

### Dataset Features:
- Customer demographics (gender, age, dependents)
- Service details (phone, internet, streaming services)
- Account information (tenure, contract, charges)
- Target variable: Churn (Yes/No)

## Project Structure

```
churn-prediction-pipeline/
├── data/
│   ├── raw/                    # Raw ingested data
│   ├── processed/              # Cleaned and processed data
│   ├── versions/               # Data versions
│   └── feature_store/          # Feature store database
├── models/                     # Trained models and artifacts
├── output/                     # Analysis outputs and reports
├── pipeline_logs/              # Pipeline execution logs
├── dags/                       # Airflow DAG files
├── src/                    # Source code modules
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker setup
└── README.md                   # This file
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for Airflow)

### Step 1: Clone Repository and Setup Environment
```bash
git clone <repository-url>
cd churn-prediction-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Visit https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in `data/raw/customer_data.csv`

### Step 3: Initialize Directories
```bash
mkdir -p data/{raw,processed,versions,feature_store}
mkdir -p models output pipeline_logs
```

## Component Overview

### 1. Data Ingestion (`data_ingestion.py`)
- Fetches data from multiple sources (CSV, APIs)
- Handles errors and logging
- Stores raw data with timestamps

### 2. Raw Data Storage (`raw_data_storage.py`)
- Organizes data with partitioning structure
- Creates data catalog metadata
- Supports local and cloud storage

### 3. Data Validation (`data_validation.py`)
- Validates data types, ranges, and formats
- Checks for missing values and duplicates
- Generates comprehensive quality reports

### 4. Data Preparation (`data_preparation.py`)
- Handles missing values and outliers
- Encodes categorical variables
- Performs exploratory data analysis
- Creates cleaned dataset

### 5. Data Transformation (`data_transformation_storage.py`)
- Creates engineered features
- Applies scaling and normalization
- Stores in SQLite database
- Maintains feature metadata

### 6. Feature Store (`feature_store.py`)
- Manages feature definitions and metadata
- Provides online/offline feature serving
- Supports point-in-time correctness
- Compatible with Feast framework

### 7. Data Versioning (`data_versioning.py`)
- Tracks dataset changes over time
- Supports DVC integration
- Enables reproducibility
- Maintains version history

### 8. Model Building (`model_building.py`)
- Trains multiple ML algorithms
- Performs hyperparameter tuning
- Evaluates model performance
- Saves versioned models

### 9. Pipeline Orchestration (`churn_prediction_dag.py`)
- Apache Airflow DAG definition
- Manages task dependencies
- Handles error recovery
- Provides monitoring and logging

## Usage

### Quick Start (Local Execution)

1. **Run Individual Components:**
```bash
# Data ingestion
python data_ingestion.py

# Data validation
python data_validation.py

# Data preparation
python data_preparation.py

# Feature engineering
python data_transformation_storage.py

# Model training
python model_building.py
```

2. **Run Complete Pipeline:**
```bash
python main_pipeline.py
```

### Apache Airflow Setup (Recommended)

1. **Install Airflow:**
```bash
pip install apache-airflow
```

2. **Initialize Airflow:**
```bash
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

3. **Copy DAG file:**
```bash
cp churn_prediction_dag.py ~/airflow/dags/
```

4. **Start Airflow:**
```bash
airflow webserver --port 8080 &
airflow scheduler &
```

5. **Access UI at http://localhost:8080**

### Docker Setup (Alternative)

```bash
docker-compose up -d
```

## Configuration

### Environment Variables
```bash
export AIRFLOW_HOME=~/airflow
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Database Configuration
- **SQLite**: Used for local feature store and metadata
- **Upgrade**: Configure PostgreSQL/MySQL for production

### Feature Store Configuration
Edit `feature_store_config.yaml`:
```yaml
project: churn_prediction
registry: data/feature_store/registry.db
provider: local
online_store:
  path: data/feature_store/online_store.db
```

## Model Performance

Expected performance metrics:
- **Accuracy**: >85%
- **F1-Score**: >0.80
- **AUC-ROC**: >0.85
- **Precision**: >0.80
- **Recall**: >0.75

## Monitoring and Maintenance

### Data Quality Monitoring
- Automated validation checks
- Data drift detection
- Quality score tracking

### Model Performance Monitoring
- Accuracy degradation detection
- Feature importance tracking
- Retraining triggers

### Pipeline Monitoring
- Task success/failure rates
- Execution time tracking
- Resource utilization

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check PYTHONPATH configuration

2. **Data File Not Found**
   - Verify dataset is placed in correct location
   - Check file permissions

3. **Airflow Task Failures**
   - Check logs in Airflow UI
   - Verify database connections
   - Ensure directories exist

4. **Memory Issues**
   - Reduce dataset size for testing
   - Optimize data processing chunks
   - Increase system memory

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is for educational purposes. Dataset license follows IBM terms.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review Airflow logs
3. Create GitHub issue with details

## Next Steps

1. Deploy to cloud environment (AWS/GCP/Azure)
2. Implement real-time inference API
3. Add A/B testing framework
4. Integrate with business systems
5. Add advanced monitoring dashboards

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)