# Feature Store Implementation

## Overview

This is a simplified feature store implementation for the churn prediction pipeline that meets the assignment requirements:

- ✅ **Feature metadata management** (description, source, version)
- ✅ **Simple feature storage and retrieval**
- ✅ **Integration with main pipeline**
- ✅ **Training and inference support**

## Features

### Core Functionality

1. **Feature Registration**: Store metadata for each feature (name, description, source, version, data type)
2. **Feature Storage**: Store feature values for entities (customers)
3. **Feature Retrieval**: Get features for training or inference
4. **Training Dataset**: Create training datasets from stored features
5. **Metadata Management**: Track feature definitions and versions

### Database Schema

The feature store uses SQLite with two main tables:

- **`feature_definitions`**: Stores feature metadata
- **`feature_values`**: Stores actual feature values

## Usage

### Basic Usage

```python
from src.feature_store import ChurnFeatureStore

# Initialize feature store
feature_store = ChurnFeatureStore()

# Register a feature
feature_store.register_feature(
    name="tenure",
    description="Number of months customer has stayed with company",
    source="customer_data",
    data_type="int"
)

# Store feature values
features = {"tenure": 24, "monthly_charges": 49.99}
feature_store.store_features("customer_001", features)

# Retrieve features for inference
customer_features = feature_store.get_features("customer_001")

# Get training dataset
training_df = feature_store.get_training_dataset()

# Close connection
feature_store.close()
```

### Populate from DataFrame

```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Populate feature store
feature_store.populate_from_dataframe(df, entity_id_col='customerID')
```

## Integration

### Main Pipeline

The feature store is integrated into the main pipeline (`main_pipeline.py`) as Step 7.

### Scheduler

The feature store job is added to the scheduler (`scheduler.py`) to run automatically.

## Testing

### Run Tests

```bash
python test_feature_store.py
```

### Run Demo

```bash
python demo_feature_store.py
```

## File Structure

```
src/
├── feature_store.py          # Main feature store implementation
├── data_ingestion.py         # Data ingestion pipeline
├── data_preparation.py       # Data preparation pipeline
├── data_transformation_storage.py  # Data transformation pipeline
└── ...

main_pipeline.py              # Main pipeline runner
scheduler.py                  # Pipeline scheduler
test_feature_store.py         # Feature store tests
demo_feature_store.py         # Feature store demonstration
```

## Requirements Met

### Assignment Deliverables

1. **Feature store configuration/code** ✅
   - `ChurnFeatureStore` class with SQLite backend
   - Simple database schema

2. **Sample API or query demonstrating feature retrieval** ✅
   - `get_features()` method for inference
   - `get_training_dataset()` method for training
   - `demo_feature_store.py` shows usage examples

3. **Documentation of feature metadata and versions** ✅
   - Feature metadata stored in database
   - Version tracking for features
   - `get_feature_metadata()` method

### Key Benefits

- **Simple**: Easy to understand and use
- **Lightweight**: SQLite backend, no external dependencies
- **Integrated**: Works with existing pipeline
- **Extensible**: Easy to add new features and functionality

## Database Details

### Tables

#### feature_definitions
- `feature_name` (TEXT, PRIMARY KEY)
- `description` (TEXT)
- `source` (TEXT)
- `version` (TEXT)
- `data_type` (TEXT)
- `created_date` (TEXT)
- `is_active` (BOOLEAN)

#### feature_values
- `entity_id` (TEXT)
- `feature_name` (TEXT)
- `feature_value` (TEXT)
- `timestamp` (TEXT)
- PRIMARY KEY: (entity_id, feature_name)

## Example Output

```
Feature Store Demonstration
==================================================
✓ Feature store initialized
✓ Created sample data with 100 customers
✓ Feature store populated with sample data

📊 Feature Metadata:
feature_name        description                    source              version data_type created_date is_active
contract_type       Feature: contract_type         data_preparation    1.0     string    2024-01-15T... True
internet_service    Feature: internet_service      data_preparation    1.0     string    2024-01-15T... True
monthly_charges     Feature: monthly_charges       data_preparation    1.0     float     2024-01-15T... True
payment_method      Feature: payment_method        data_preparation    1.0     string    2024-01-15T... True
tech_support        Feature: tech_support          data_preparation    1.0     string    2024-01-15T... True
tenure              Feature: tenure                data_preparation    1.0     int       2024-01-15T... True
total_charges       Feature: total_charges         data_preparation    1.0     float     2024-01-15T... True

🎯 Training Data Retrieval:
Training dataset shape: (100, 8)
First 5 rows:
entity_id  contract_type internet_service monthly_charges payment_method tech_support tenure total_charges
CUST_001   Month-to-month DSL             51.13          Electronic check No           51    2,607.63
CUST_002   One year      Fiber optic      20.25          Bank transfer   No           5     101.25
CUST_003   Two year      No               19.85          Credit card     No           68    1,349.80
CUST_004   Month-to-month DSL             25.36          Mailed check    Yes          33    836.88
CUST_005   One year      Fiber optic      25.19          Bank transfer   No           49    1,234.31
```

## Next Steps

This feature store can be extended with:

1. **Feature versioning**: Track multiple versions of features
2. **Feature lineage**: Track feature dependencies and transformations
3. **Performance optimization**: Add indexes and caching
4. **Monitoring**: Add metrics and health checks
5. **API endpoints**: REST API for feature serving
