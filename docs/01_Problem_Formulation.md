# Customer Churn Prediction Pipeline - Problem Formulation

## Business Problem
**Challenge:** Customer churn is causing significant revenue loss and increased customer acquisition costs for the telecommunications company.

**Goal:** Build an automated end-to-end ML pipeline to predict customer churn and enable proactive retention strategies through data-driven insights.

**Target:** Reduce quarterly churn rate by 5% within 6 months through targeted interventions and personalized retention campaigns.

## Data Sources and Schema
1. **Telco Customer Data (CSV File)**
   - **Customer Demographics:** gender, SeniorCitizen, Partner, Dependents
   - **Service Information:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
   - **Account Details:** Contract, PaperlessBilling, PaymentMethod
   - **Financial Data:** MonthlyCharges, TotalCharges
   - **Behavioral Metrics:** tenure (months with company)
   - **Target Variable:** Churn (Yes/No)

2. **Hugging Face Dataset (JSON via API)**
   - Dataset: `scikit-learn/churn-prediction`
   - Same schema as Telco data for validation and enrichment
   - Retrieved via Hugging Face Datasets Server API
   - Used for data quality validation and cross-reference

## Complete Pipeline Architecture (main_pipeline.py)

### Step 1: Problem Formulation
**Purpose:** Define business problem and objectives
- **Business Context:** Understand churn impact and business goals
- **Success Metrics:** Define measurable KPIs and targets
- **Data Requirements:** Identify required data sources and features
- **Output:** Clear problem definition and success criteria

### Step 2: Data Ingestion Pipeline
**Purpose:** Fetch and consolidate data from multiple sources
- **CSV Ingestion:** Load Telco customer data from local CSV files
- **API Integration:** Retrieve JSON data from Hugging Face API
- **Data Consolidation:** Merge and validate data format consistency
- **Error Handling:** Robust error handling and logging
- **Output:** Raw data files stored in `data/raw/` with timestamps

### Step 3: Raw Data Storage
**Purpose:** Organize and catalog raw data for systematic processing
- **Directory Structure:** Create organized data hierarchy (raw, processed, cleaned)
- **Data Catalog:** Generate metadata and file inventory
- **Version Control:** Track data versions and timestamps
- **Storage Optimization:** Efficient data organization and indexing
- **Output:** Structured data catalog and organized file system

### Step 4: Data Validation
**Purpose:** Ensure data quality and integrity before processing
- **Quality Checks:** Missing values, data types, value ranges, duplicates
- **Schema Validation:** Verify data structure against expected format
- **Statistical Analysis:** Basic statistics and data distribution analysis
- **Data Profiling:** Comprehensive data quality assessment
- **Output:** Validation reports and quality metrics in `logs/`

### Step 5: Data Preparation
**Purpose:** Clean and preprocess raw data for analysis
- **Data Cleaning:** Handle missing values, outliers, and inconsistencies
- **Categorical Encoding:** Convert text variables to numerical format
- **Data Type Conversion:** Ensure proper data types for analysis
- **Feature Engineering:** Create initial derived features
- **Output:** Clean dataset in `data/cleaned/churn_data_cleaned.csv`

### Step 6: Data Transformation & Storage
**Purpose:** Engineer features and store in database for ML training
- **Feature Engineering:**
  - Aggregated features (total services, average charges)
  - Derived features (tenure groups, charge ratios)
  - Interaction features (service-charge interactions)
  - Temporal features (tenure stability, service density)
- **Feature Scaling:** Normalize numerical features using StandardScaler/MinMaxScaler
- **Database Storage:** Store in SQLite with proper schema and indexing
- **Feature Metadata:** Track feature lineage and transformations
- **Output:** Transformed features in `data/processed/training_sets/`

### Step 7: Feature Store
**Purpose:** Manage and serve engineered features for ML training and inference
- **Feature Management:** Store and version engineered features
- **Feature Retrieval API:** Provide methods to access features by customer ID
- **Metadata Tracking:** Track feature lineage and transformations
- **Feature Serving:** Online and offline feature serving capabilities
- **Output:** Feature store with CSV-based storage in `data/feature_store/`

### Step 8: Data Versioning
**Purpose:** Version control for datasets to ensure reproducibility
- **Version Tracking:** Track changes in data using DVC and custom tagging
- **Metadata Storage:** Store version metadata (source, timestamp, change log)
- **Rollback Capability:** Ability to restore previous data versions
- **Reproducibility:** Ensure consistent data states across pipeline runs
- **Output:** Versioned datasets in `data/versions/` with comprehensive metadata

### Step 9: Model Training
**Purpose:** Train and evaluate machine learning models for churn prediction
- **Model Selection:** Logistic Regression and Random Forest classifiers
- **Training Process:** Split data, train models, perform cross-validation
- **Hyperparameter Tuning:** Grid search for optimal parameters
- **Model Evaluation:** Calculate accuracy, precision, recall, F1-score
- **Model Persistence:** Save trained models to `src/models/`

## Technical Implementation Details

### Data Processing Technologies
- **Python Libraries:** pandas, numpy, scikit-learn, sqlite3
- **Database:** SQLite for local storage and feature management
- **ML Framework:** scikit-learn for model training and evaluation
- **Version Control:** DVC for data versioning
- **Logging:** Comprehensive logging system for monitoring and debugging

### Feature Engineering Strategy
- **Numerical Features:** tenure, charges, ratios, interactions
- **Categorical Features:** service types, contract types, payment methods
- **Derived Features:** customer segments, service density, value indicators
- **Temporal Features:** tenure groups, stability metrics
- **Interaction Features:** service-charge combinations, usage patterns

### Model Training Approach
- **Algorithm Selection:** Logistic Regression (interpretable) and Random Forest (robust)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Cross-Validation:** Stratified k-fold for balanced evaluation
- **Hyperparameter Tuning:** Grid search for optimal parameters
- **Model Comparison:** Systematic comparison of multiple algorithms

### Data Versioning Strategy
- **DVC Integration:** Git-based versioning for data files
- **Custom Tagging:** Metadata tracking with timestamps and descriptions
- **Rollback Capability:** Easy restoration of previous data states
- **Reproducibility:** Consistent data states across environments
- **Change Tracking:** Comprehensive audit trail of data modifications

## Success Metrics and Evaluation

### Model Performance Targets
- **Accuracy:** > 85% overall prediction accuracy
- **Precision:** > 80% for churn predictions (minimize false positives)
- **Recall:** > 75% for churn detection (maximize true positives)
- **F1-Score:** > 0.8 balanced performance metric
- **AUC-ROC:** > 0.85 for model discrimination ability

### Business Impact Metrics
- **Churn Reduction:** 5% decrease in quarterly churn rate
- **Cost Savings:** Reduced customer acquisition costs
- **Revenue Protection:** Maintained customer lifetime value
- **ROI:** Positive return on investment for retention campaigns

### Pipeline Reliability
- **Automation:** End-to-end automated execution
- **Monitoring:** Comprehensive logging and error tracking
- **Reproducibility:** Version-controlled data and model artifacts
- **Scalability:** Ability to handle larger datasets
- **Maintainability:** Clean, modular code structure

## Output Deliverables
1. **Clean Datasets:** Validated and preprocessed data ready for analysis
2. **Engineered Features:** Transformed features optimized for ML training
3. **Trained Models:** Serialized models ready for production deployment
4. **Feature Store:** Production-ready feature serving system
5. **Versioned Data:** Complete data versioning with rollback capability
6. **Documentation:** Comprehensive logs, reports, and technical documentation
7. **Monitoring System:** Automated pipeline monitoring and alerting
8. **Reproducible Workflow:** Complete pipeline with version control

## Pipeline Integration and Orchestration

### Main Pipeline Execution
- **Sequential Execution:** Steps 1-9 executed in order
- **Error Handling:** Robust error handling at each step
- **Logging:** Comprehensive logging for monitoring and debugging
- **Progress Tracking:** Clear progress indicators and status updates

### Individual Step Execution
- **Modular Design:** Each step can be run independently
- **Input Validation:** Validate inputs before processing
- **Output Verification:** Verify outputs after processing
- **Integration Testing:** Test step integration and data flow

### Data Flow Between Steps
- **Raw Data:** Step 2 → Step 3 → Step 4
- **Cleaned Data:** Step 4 → Step 5 → Step 6
- **Transformed Data:** Step 6 → Step 7 → Step 8
- **Model Training:** Step 7 → Step 9
- **Versioning:** Integrated throughout all steps

## Quality Assurance and Testing

### Data Quality Checks
- **Completeness:** Check for missing values and data gaps
- **Consistency:** Verify data format and structure consistency
- **Accuracy:** Validate data against business rules
- **Timeliness:** Ensure data freshness and relevance

### Model Quality Assurance
- **Performance Validation:** Cross-validation and holdout testing
- **Feature Importance:** Analyze feature contributions
- **Bias Detection:** Check for model bias and fairness
- **Robustness Testing:** Test model performance on different data subsets

### Pipeline Quality Assurance
- **End-to-End Testing:** Test complete pipeline execution
- **Error Recovery:** Test error handling and recovery mechanisms
- **Performance Testing:** Test pipeline performance and scalability
- **Integration Testing:** Test component integration and data flow