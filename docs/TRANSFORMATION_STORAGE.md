# Data Transformation and Storage

## Overview

The Data Transformation and Storage module transforms cleaned customer data into rich feature sets for machine learning, scales features appropriately, and stores them in a SQLite database for easy querying and training set management.

## Features

### 1. Feature Engineering
- **Aggregated Features**: Total services used, service density
- **Derived Features**: Charges per tenure, total-to-monthly ratio, average monthly charges
- **Customer Segmentation**: Value segments, tenure stability groups
- **Risk Indicators**: High-risk payment method flags
- **Interaction Features**: Tenure-charges interactions, service-charges interactions

### 2. Feature Scaling
- **StandardScaler**: For normally distributed numerical features
- **MinMaxScaler**: For bounded features
- **LabelEncoder**: For categorical features
- **Binary Encoding**: For binary features

### 3. Database Storage
- **SQLite**: Simple local database storage
- **Metadata Tracking**: Feature descriptions, transformation methods, data quality metrics
- **Training Set Management**: Versioned training sets with quality scores

## Database Schema

### Customer Features Table
```sql
CREATE TABLE customer_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    tenure INTEGER,
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(12,2),
    tenure_group INTEGER,
    charges_per_tenure DECIMAL(10,2),
    total_to_monthly_ratio DECIMAL(10,2),
    avg_monthly_charges DECIMAL(10,2),
    -- Encoded categorical features
    gender_encoded INTEGER,
    senior_citizen INTEGER,
    partner_encoded INTEGER,
    dependents_encoded INTEGER,
    phone_service_encoded INTEGER,
    multiple_lines_encoded INTEGER,
    internet_service_encoded INTEGER,
    online_security_encoded INTEGER,
    online_backup_encoded INTEGER,
    device_protection_encoded INTEGER,
    tech_support_encoded INTEGER,
    streaming_tv_encoded INTEGER,
    streaming_movies_encoded INTEGER,
    contract_encoded INTEGER,
    paperless_billing_encoded INTEGER,
    payment_method_encoded INTEGER,
    churn_label INTEGER,
    -- Aggregated features
    total_services INTEGER,
    service_density DECIMAL(10,4),
    customer_value_segment INTEGER,
    tenure_stability INTEGER,
    high_risk_payment INTEGER,
    -- Interaction features
    tenure_monthly_interaction DECIMAL(12,2),
    tenure_total_interaction DECIMAL(14,2),
    services_charges_interaction DECIMAL(12,2),
    contract_payment_interaction INTEGER,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Feature Metadata Table
```sql
CREATE TABLE feature_metadata (
    feature_name VARCHAR(100) PRIMARY KEY,
    feature_type VARCHAR(50),
    description TEXT,
    transformation_applied VARCHAR(200),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Training Sets Table
```sql
CREATE TABLE training_sets (
    set_id VARCHAR(100) PRIMARY KEY,
    set_name VARCHAR(100),
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_count INTEGER,
    record_count INTEGER,
    target_distribution TEXT,
    data_quality_score DECIMAL(5,2)
);
```

## Sample Queries

### 1. Overall Churn Rate
```sql
SELECT 
    COUNT(*) as total_customers,
    SUM(churn_label) as churned_customers,
    ROUND(SUM(churn_label) * 100.0 / COUNT(*), 2) as churn_rate_percent
FROM customer_features
WHERE churn_label IS NOT NULL;
```

### 2. Tenure Analysis
```sql
SELECT 
    tenure_group,
    COUNT(*) as customer_count,
    AVG(monthly_charges) as avg_monthly_charges,
    AVG(total_charges) as avg_total_charges
FROM customer_features
GROUP BY tenure_group
ORDER BY tenure_group;
```

### 3. Service Usage Analysis
```sql
SELECT 
    total_services,
    COUNT(*) as customer_count,
    AVG(monthly_charges) as avg_monthly_charges,
    SUM(churn_label) as churned_count
FROM customer_features
WHERE total_services IS NOT NULL
GROUP BY total_services
ORDER BY total_services;
```

### 4. Customer Value Segments
```sql
SELECT 
    customer_value_segment,
    COUNT(*) as customer_count,
    AVG(tenure) as avg_tenure,
    SUM(churn_label) as churned_count
FROM customer_features
WHERE customer_value_segment IS NOT NULL
GROUP BY customer_value_segment
ORDER BY customer_value_segment;
```

### 5. Feature Statistics View
```sql
SELECT 
    feature_name,
    feature_type,
    description,
    transformation_applied,
    created_date
FROM feature_metadata
ORDER BY feature_name;
```

## Usage Examples

### Basic Usage
```python
from src.data_transformation_storage import DataTransformationStorage

# Initialize transformer
transformer = DataTransformationStorage()

# Run transformation pipeline with automatic file detection
transformed_df, training_path = transformer.run_transformation_pipeline_auto()

# Get feature summary
feature_summary = transformer.get_feature_summary()

# Close connection
transformer.close_connection()
```

### Manual File Processing
```python
# Load your own CSV file
import pandas as pd
df = pd.read_csv("your_data.csv")

# Run transformation pipeline
transformed_df, training_path = transformer.run_transformation_pipeline(df)
```

## Transformation Logic

### 1. Feature Engineering Steps

#### Aggregated Features
- **total_services**: Sum of all service indicators
- **service_density**: Services per tenure month
- **customer_value_segment**: 4-tier segmentation based on total charges
- **tenure_stability**: 4-tier grouping (New, Growing, Stable, Loyal)

#### Derived Features
- **charges_per_tenure**: Average charges per month
- **total_to_monthly_ratio**: Ratio of total to monthly charges
- **avg_monthly_charges**: Average monthly charges over tenure

#### Interaction Features
- **tenure_monthly_interaction**: Tenure × Monthly Charges
- **tenure_total_interaction**: Tenure × Total Charges
- **services_charges_interaction**: Total Services × Monthly Charges

### 2. Scaling Methods

#### StandardScaler (Normal Distribution)
- tenure, monthly_charges, total_charges
- charges_per_tenure, total_to_monthly_ratio, avg_monthly_charges
- tenure_monthly_interaction, tenure_total_interaction, services_charges_interaction

#### MinMaxScaler (Bounded Features)
- total_services, service_density

#### LabelEncoder (Categorical)
- All encoded categorical features (gender_encoded, partner_encoded, etc.)
- tenure_group, customer_value_segment, tenure_stability

#### Binary (No Scaling)
- senior_citizen, high_risk_payment

## Data Quality Metrics

### Completeness Score
- Calculates percentage of non-missing values
- Used for training set quality assessment

### Target Distribution
- Tracks churn label distribution in training sets
- Helps identify class imbalance issues

## File Structure

```
data/
├── processed/
│   ├── churn_data.db          # SQLite database
│   └── training_sets/         # Versioned training sets
│       ├── churn_prediction_v1_20231201_143022.csv
│       └── ...
logs/
└── data_transformation_storage.log
database/
└── init.sql                   # SQLite initialization script
```

## Configuration

### Database Path
- **Default**: `data/processed/churn_data.db`
- **Custom**: Pass `db_path` parameter when initializing

### Example
```python
# Use default path
transformer = DataTransformationStorage()

# Use custom path
transformer = DataTransformationStorage(db_path="my_custom_path.db")
```

## Performance Considerations

### Indexing
- Primary indexes on customer_id
- Secondary indexes on tenure, churn_label, charges
- Composite indexes for common query patterns

### Data Types
- Appropriate decimal precision for monetary values
- Integer types for encoded categorical features
- Timestamp for audit trails

### Query Optimization
- Views for common query patterns
- Materialized views for complex aggregations (if needed)
- Connection pooling for high-throughput scenarios

## Monitoring and Logging

### Log Files
- `logs/data_transformation_storage.log`: Detailed transformation logs
- Console output: Real-time progress updates

### Metrics Tracked
- Transformation execution time
- Data quality scores
- Feature count and record count
- Database operation success/failure rates

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check if the database file path is accessible
   - Ensure the directory exists
   - Verify file permissions

2. **Schema Mismatch Errors**
   - The database tables are created automatically
   - If issues persist, delete the database file and restart

3. **Memory Issues**
   - Use batch processing for large datasets
   - Monitor memory usage during transformations

### Debug Mode
```python
import logging
logging.getLogger('data_transformation_storage').setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Advanced Feature Engineering**
   - Time-series features
   - Customer behavior patterns
   - Seasonal trends

2. **Additional Scaling Methods**
   - RobustScaler for outlier-resistant scaling
   - QuantileTransformer for non-linear transformations

3. **Database Enhancements**
   - Partitioning for large datasets
   - Advanced indexing strategies
   - Backup and recovery procedures

4. **Monitoring and Alerting**
   - Data quality alerts
   - Performance monitoring
   - Automated health checks
