-- SQLite Database Initialization Script
-- This script creates the necessary tables for the churn prediction pipeline

-- Create customer features table
CREATE TABLE IF NOT EXISTS customer_features (
    customer_id TEXT PRIMARY KEY,
    tenure INTEGER,
    MonthlyCharges REAL,
    TotalCharges REAL,
    tenure_group INTEGER,
    charges_per_tenure REAL,
    total_to_monthly_ratio REAL,
    avg_monthly_charges REAL,
    gender_encoded INTEGER,
    SeniorCitizen INTEGER,
    Partner_encoded INTEGER,
    Dependents_encoded INTEGER,
    PhoneService_encoded INTEGER,
    MultipleLines_encoded INTEGER,
    InternetService_encoded INTEGER,
    OnlineSecurity_encoded INTEGER,
    OnlineBackup_encoded INTEGER,
    DeviceProtection_encoded INTEGER,
    TechSupport_encoded INTEGER,
    StreamingTV_encoded INTEGER,
    StreamingMovies_encoded INTEGER,
    Contract_encoded INTEGER,
    PaperlessBilling_encoded INTEGER,
    PaymentMethod_encoded INTEGER,
    Churn INTEGER,
    total_services INTEGER,
    service_density REAL,
    customer_value_segment INTEGER,
    tenure_stability INTEGER,
    high_risk_payment INTEGER,
    tenure_monthly_interaction REAL,
    tenure_total_interaction REAL,
    services_charges_interaction REAL,
    contract_payment_interaction INTEGER,
    created_timestamp TEXT,
    updated_timestamp TEXT
);

-- Create feature metadata table
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name TEXT PRIMARY KEY,
    feature_type TEXT,
    description TEXT,
    transformation_applied TEXT,
    created_date TEXT
);

-- Create training sets table
CREATE TABLE IF NOT EXISTS training_sets (
    set_id TEXT PRIMARY KEY,
    set_name TEXT,
    creation_date TEXT,
    feature_count INTEGER,
    record_count INTEGER,
    target_distribution TEXT,
    data_quality_score REAL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_features_tenure ON customer_features(tenure);
CREATE INDEX IF NOT EXISTS idx_customer_features_churn ON customer_features(Churn);
CREATE INDEX IF NOT EXISTS idx_customer_features_charges ON customer_features(MonthlyCharges, TotalCharges);
CREATE INDEX IF NOT EXISTS idx_customer_features_services ON customer_features(total_services);
CREATE INDEX IF NOT EXISTS idx_training_sets_creation ON training_sets(creation_date);

-- Insert initial feature metadata
INSERT OR IGNORE INTO feature_metadata (feature_name, feature_type, description, transformation_applied) VALUES
('tenure', 'numerical', 'Customer tenure in months', 'StandardScaler'),
('MonthlyCharges', 'numerical', 'Monthly charges amount', 'StandardScaler'),
('TotalCharges', 'numerical', 'Total charges amount', 'StandardScaler'),
('tenure_group', 'categorical', 'Tenure grouped into categories', 'LabelEncoder'),
('charges_per_tenure', 'numerical', 'Average charges per tenure month', 'StandardScaler'),
('total_to_monthly_ratio', 'numerical', 'Ratio of total to monthly charges', 'StandardScaler'),
('avg_monthly_charges', 'numerical', 'Average monthly charges', 'StandardScaler'),
('total_services', 'numerical', 'Total number of services used', 'MinMaxScaler'),
('service_density', 'numerical', 'Services per tenure month', 'MinMaxScaler'),
('customer_value_segment', 'categorical', 'Customer value segmentation', 'LabelEncoder'),
('tenure_stability', 'categorical', 'Tenure stability groups', 'LabelEncoder'),
('high_risk_payment', 'binary', 'High risk payment method indicator', 'None'),
('tenure_monthly_interaction', 'numerical', 'Interaction between tenure and monthly charges', 'StandardScaler'),
('tenure_total_interaction', 'numerical', 'Interaction between tenure and total charges', 'StandardScaler'),
('services_charges_interaction', 'numerical', 'Interaction between services and charges', 'StandardScaler');
