"""
Data Transformation and Storage
-------------------------------
Transforms cleaned data into richer feature sets, scales features, and stores
them in a lightweight SQLite database for downstream querying and training set
management. Also tracks feature metadata and training set summaries.
"""
import pandas as pd
import sqlite3
import logging
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Configure module logger (file + console)
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger('data_transformation_storage')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('logs/data_transformation_storage.log')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Route existing logging.* calls to this module logger
    logging = logger

# Class: applies transformations and persists features/training sets in SQLite
class DataTransformationStorage:
    """Transform features and persist them with simple metadata tracking."""
    def __init__(self, db_path="data/processed/churn_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    # Create tables for features, metadata, and training sets
    def setup_database(self):
        """Create database tables for transformed data"""
        cursor = self.conn.cursor()

        # Create customer features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_features (
                customer_id TEXT PRIMARY KEY,
                tenure INTEGER,
                monthly_charges REAL,
                total_charges REAL,
                tenure_group INTEGER,
                charges_per_tenure REAL,
                total_to_monthly_ratio REAL,
                avg_monthly_charges REAL,
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
                created_timestamp TEXT,
                updated_timestamp TEXT
            )
        """)

        # Create feature metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name TEXT PRIMARY KEY,
                feature_type TEXT,
                description TEXT,
                transformation_applied TEXT,
                created_date TEXT
            )
        """)

        # Create model training sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sets (
                set_id TEXT PRIMARY KEY,
                set_name TEXT,
                creation_date TEXT,
                feature_count INTEGER,
                record_count INTEGER,
                target_distribution TEXT,
                data_quality_score REAL
            )
        """)

        self.conn.commit()
        logging.info("Database tables created")

    # Build higher-level aggregates (e.g., total_services, value segments)
    def create_aggregated_features(self, df):
        """Create aggregated features for better model performance"""
        df_agg = df.copy()

        # Service usage aggregations
        service_columns = [col for col in df.columns if 'service' in col.lower() or 
                          col in ['PhoneService', 'MultipleLines', 'InternetService',
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                 'TechSupport', 'StreamingTV', 'StreamingMovies']]

        if service_columns:
            # Total services used
            df_agg['total_services'] = df[service_columns].sum(axis=1)

            # Service density (services per tenure month)
            df_agg['service_density'] = df_agg['total_services'] / (df_agg['tenure'] + 1)

        # Customer value segments
        df_agg['customer_value_segment'] = pd.cut(df_agg['TotalCharges'], 
                                                bins=4, 
                                                labels=['Low', 'Medium', 'High', 'Premium'])
        df_agg['customer_value_segment'] = df_agg['customer_value_segment'].cat.codes

        # Tenure stability groups
        df_agg['tenure_stability'] = np.where(df_agg['tenure'] <= 12, 0,  # New
                                    np.where(df_agg['tenure'] <= 36, 1,   # Growing
                                    np.where(df_agg['tenure'] <= 60, 2, 3))) # Stable/Loyal

        # Payment behavior indicators
        if 'PaymentMethod' in df.columns:
            # High risk payment methods (typically electronic check)
            df_agg['high_risk_payment'] = (df_agg['PaymentMethod'] == 2).astype(int)  # Assuming 2 is electronic check

        logging.info("Aggregated features created")
        return df_agg

    # Scale features using StandardScaler (and MinMax for remaining)
    def apply_feature_scaling(self, df, features_to_scale=None):
        """Apply different scaling techniques to features"""
        df_scaled = df.copy()

        if features_to_scale is None:
            # Select numerical features for scaling
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary and categorical encoded features
            features_to_scale = [col for col in numerical_features if 
                               not col.endswith('_encoded') and 
                               col not in ['churn_label', 'tenure_group', 'customer_value_segment']]

        # Standard scaling for normal distribution features
        standard_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        standard_features = [col for col in standard_features if col in features_to_scale]

        if standard_features:
            scaler = StandardScaler()
            df_scaled[standard_features] = scaler.fit_transform(df_scaled[standard_features])
            logging.info(f"Applied standard scaling to: {standard_features}")

        # Min-Max scaling for bounded features
        minmax_features = [col for col in features_to_scale if col not in standard_features]
        if minmax_features:
            minmax_scaler = MinMaxScaler()
            df_scaled[minmax_features] = minmax_scaler.fit_transform(df_scaled[minmax_features])
            logging.info(f"Applied min-max scaling to: {minmax_features}")

        return df_scaled

    # Generate interaction terms between important features
    def create_feature_interactions(self, df):
        """Create interaction features"""
        df_interact = df.copy()

        # Tenure and charges interactions
        df_interact['tenure_monthly_interaction'] = df_interact['tenure'] * df_interact['MonthlyCharges']
        df_interact['tenure_total_interaction'] = df_interact['tenure'] * df_interact['TotalCharges']

        # Service level and payment interactions
        if 'total_services' in df.columns:
            df_interact['services_charges_interaction'] = df_interact['total_services'] * df_interact['MonthlyCharges']

        # Contract and payment method interactions
        if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
            df_interact['contract_payment_interaction'] = df_interact['Contract'] * df_interact['PaymentMethod']

        logging.info("Feature interactions created")
        return df_interact

    # Persist transformed data to SQLite, add timestamps, update metadata
    def store_transformed_data(self, df, table_name="customer_features"):
        """Store transformed data in database"""
        # Add timestamps
        timestamp = datetime.now().isoformat()
        if 'created_timestamp' not in df.columns:
            df['created_timestamp'] = timestamp
        df['updated_timestamp'] = timestamp

        # Store data
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        logging.info(f"Transformed data stored in {table_name} table")

        # Update metadata
        self.update_feature_metadata(df)

    # Refresh metadata describing available features
    def update_feature_metadata(self, df):
        """Update feature metadata table"""
        cursor = self.conn.cursor()

        for column in df.columns:
            if column not in ['created_timestamp', 'updated_timestamp']:
                feature_type = 'categorical' if '_encoded' in column else 'numerical'

                cursor.execute("""
                    INSERT OR REPLACE INTO feature_metadata 
                    (feature_name, feature_type, description, transformation_applied, created_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (column, feature_type, f"Feature: {column}", 
                     "StandardScaler/LabelEncoder", datetime.now().isoformat()))

        self.conn.commit()
        logging.info("Feature metadata updated")

    # Snapshot a training set and record its summary
    def create_training_set(self, set_name, feature_columns=None):
        """Create and store training set"""
        cursor = self.conn.cursor()

        # Get data from customer_features table
        if feature_columns:
            columns_str = ', '.join(feature_columns)
            query = f"SELECT {columns_str} FROM customer_features"
        else:
            query = "SELECT * FROM customer_features"

        df = pd.read_sql(query, self.conn)

        # Calculate data quality metrics
        data_quality_score = self.calculate_data_quality_score(df)

        # Calculate target distribution
        if 'churn_label' in df.columns:
            churn_dist = df['churn_label'].value_counts().to_dict()
            target_distribution = str(churn_dist)
        else:
            target_distribution = "No target column"

        # Store training set metadata
        set_id = f"{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cursor.execute("""
            INSERT INTO training_sets 
            (set_id, set_name, creation_date, feature_count, record_count, 
             target_distribution, data_quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (set_id, set_name, datetime.now().isoformat(), 
              len(df.columns), len(df), target_distribution, data_quality_score))

        self.conn.commit()

        # Save training set as CSV
        output_path = f"data/processed/training_sets/{set_id}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"Training set created: {set_id}")
        return set_id, output_path

    # Compute a simple completeness-based data quality score
    def calculate_data_quality_score(self, df):
        """Calculate data quality score"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells

        # Add other quality metrics as needed
        quality_score = completeness_score * 100  # Convert to percentage

        return round(quality_score, 2)

    # Read back feature metadata as a DataFrame
    def get_feature_summary(self):
        """Get summary of all features in database"""
        query = "SELECT * FROM feature_metadata ORDER BY feature_name"
        return pd.read_sql(query, self.conn)

    # Execute full transformation pipeline and store results
    def run_transformation_pipeline(self, input_df):
        """Run complete transformation pipeline"""
        logging.info("Starting data transformation pipeline")

        # Apply transformations
        df_agg = self.create_aggregated_features(input_df)
        df_interact = self.create_feature_interactions(df_agg)
        df_scaled = self.apply_feature_scaling(df_interact)

        # Store transformed data
        self.store_transformed_data(df_scaled)

        # Create training set
        set_id, training_path = self.create_training_set("churn_prediction_v1")

        logging.info("Data transformation pipeline completed")
        return df_scaled, training_path

    # Close SQLite connection
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")

if __name__ == "__main__":
    transformer = DataTransformationStorage()

    # Example usage with sample data
    sample_data = {
        'customerID': ['001', '002', '003'],
        'tenure': [12, 24, 36],
        'MonthlyCharges': [50.0, 75.5, 89.99],
        'TotalCharges': [600.0, 1812.0, 3239.64],
        'Churn': [0, 1, 0]
    }
    df = pd.DataFrame(sample_data)

    print("Data transformation and storage script created")
    # Uncomment to run transformation:
    # result_df, training_path = transformer.run_transformation_pipeline(df)
    # transformer.close_connection()
