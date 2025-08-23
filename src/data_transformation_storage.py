"""
Data Transformation and Storage
-------------------------------
Transforms cleaned data into richer feature sets, scales features, and stores
them in SQLite database for downstream querying and training set management.
Also tracks feature metadata and training set summaries.
"""

import pandas as pd
import sqlite3
import os
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_TRANSFORMATION'])

class DataTransformationStorage:
    """Transform features and persist them in SQLite with metadata tracking."""
    def __init__(self, db_path: str = "data/processed/churn_data.db"):
        """Initialize with database path and connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.setup_database()

    def setup_database(self):
        """Create tables and indexes for features, metadata, and training sets."""
        cursor = self.conn.cursor()
        cursor.execute("""
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
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name TEXT PRIMARY KEY,
                feature_type TEXT,
                description TEXT,
                transformation_applied TEXT,
                created_date TEXT
            )
        """)
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_features_tenure ON customer_features(tenure)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_features_churn ON customer_features(Churn)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_features_charges ON customer_features(MonthlyCharges, TotalCharges)")
        self.conn.commit()
        logger.info("SQLite database initialized")

    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate aggregated features for model performance."""
        df_agg = df.copy()
        service_columns = [col for col in df.columns if 'service' in col.lower() or 
                         col in ['PhoneService', 'MultipleLines', 'InternetService',
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                 'TechSupport', 'StreamingTV', 'StreamingMovies']]
        if service_columns:
            df_agg['total_services'] = df[service_columns].sum(axis=1)
            df_agg['service_density'] = df_agg['total_services'] / (df_agg['tenure'] + 1)
        df_agg['customer_value_segment'] = pd.cut(df_agg['TotalCharges'], 
                                                bins=4, labels=[0, 1, 2, 3]).astype(int)
        df_agg['tenure_stability'] = np.where(df_agg['tenure'] <= 12, 0,
                                    np.where(df_agg['tenure'] <= 36, 1,
                                    np.where(df_agg['tenure'] <= 60, 2, 3)))
        if 'PaymentMethod' in df.columns:
            df_agg['high_risk_payment'] = (df_agg['PaymentMethod'] == 2).astype(int)
        logger.info("Aggregated features created")
        return df_agg

    def apply_feature_scaling(self, df: pd.DataFrame, features_to_scale: list = None) -> pd.DataFrame:
        """Apply StandardScaler and MinMaxScaler to numerical features."""
        df_scaled = df.copy()
        if features_to_scale is None:
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            features_to_scale = [col for col in numerical_features if 
                               not col.endswith('_encoded') and 
                               col not in ['Churn', 'tenure_group', 'customer_value_segment']]
        standard_features = [col for col in ['tenure', 'MonthlyCharges', 'TotalCharges'] if col in features_to_scale]
        if standard_features:
            df_scaled[standard_features] = self.standard_scaler.fit_transform(df_scaled[standard_features])
            logger.info(f"Standard scaled: {standard_features}")
        minmax_features = [col for col in features_to_scale if col not in standard_features]
        if minmax_features:
            df_scaled[minmax_features] = self.minmax_scaler.fit_transform(df_scaled[minmax_features])
            logger.info(f"Min-max scaled: {minmax_features}")
        return df_scaled

    def create_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between key variables."""
        df_interact = df.copy()
        df_interact['tenure_monthly_interaction'] = df_interact['tenure'] * df_interact['MonthlyCharges']
        df_interact['tenure_total_interaction'] = df_interact['tenure'] * df_interact['TotalCharges']
        if 'total_services' in df.columns:
            df_interact['services_charges_interaction'] = df_interact['total_services'] * df_interact['MonthlyCharges']
        if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
            df_interact['contract_payment_interaction'] = df_interact['Contract'] * df_interact['PaymentMethod']
        logger.info("Interaction features created")
        return df_interact

    def store_transformed_data(self, df: pd.DataFrame, table_name: str = "customer_features") -> None:
        """Store transformed data in SQLite database."""
        df = df.rename(columns={'customerID': 'customer_id'}, errors='ignore')
        timestamp = datetime.now().isoformat()
        df['created_timestamp'] = df.get('created_timestamp', timestamp)
        df['updated_timestamp'] = timestamp
        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        logger.info(f"Stored data in {table_name}")
        self.update_feature_metadata(df)

    def update_feature_metadata(self, df: pd.DataFrame) -> None:
        """Update feature metadata in SQLite table."""
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
        logger.info("Feature metadata updated")

    def create_training_set(self, set_name: str, feature_columns: list = None) -> tuple[str, str]:
        """Create and store training set with metadata."""
        query = f"SELECT {', '.join(feature_columns) if feature_columns else '*'} FROM customer_features"
        df = pd.read_sql(query, self.conn)
        data_quality_score = self.calculate_data_quality_score(df)
        target_distribution = str(df['Churn'].value_counts().to_dict()) if 'Churn' in df.columns else "No target"
        set_id = f"{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO training_sets 
            (set_id, set_name, creation_date, feature_count, record_count, target_distribution, data_quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (set_id, set_name, datetime.now().isoformat(), 
              len(df.columns), len(df), target_distribution, data_quality_score))
        self.conn.commit()
        output_path = f"data/processed/training_sets/{set_id}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Training set created: {set_id}")
        return set_id, output_path

    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells
        return round(completeness_score * 100, 2)

    def get_feature_summary(self) -> pd.DataFrame:
        """Retrieve summary of features from metadata table."""
        return pd.read_sql("SELECT * FROM feature_metadata ORDER BY feature_name", self.conn)

    def run_transformation_pipeline(self, input_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Run full transformation pipeline and store results."""
        logger.info("Starting transformation pipeline")
        df = self.create_aggregated_features(input_df)
        df = self.create_feature_interactions(df)
        df = self.apply_feature_scaling(df)
        self.store_transformed_data(df)
        set_id, training_path = self.create_training_set("churn_prediction_v1")
        logger.info("Transformation pipeline completed")
        return df, training_path

    def run_transformation_pipeline_auto(self) -> tuple[pd.DataFrame, str]:
        """Run pipeline with automatic CSV file detection."""
        patterns = ["data/cleaned/*.csv", "data/processed/*.csv"]
        csv_files = []
        for pattern in patterns:
            csv_files.extend(glob.glob(pattern))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data/cleaned or data/processed")
        latest_csv = max(csv_files, key=os.path.getmtime)
        logger.info(f"Using latest CSV: {latest_csv}")
        df = pd.read_csv(latest_csv)
        return self.run_transformation_pipeline(df)

    def close_connection(self) -> None:
        """Close SQLite database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Data Transformation and Storage Pipeline")
    print("=" * 60)
    try:
        transformer = DataTransformationStorage()
        print("Running transformation pipeline...")
        result_df, training_path = transformer.run_transformation_pipeline_auto()
        print(f"Transformation completed! Output shape: {result_df.shape}")
        print(f"Training set: {training_path}")
        print(f"Database: {transformer.db_path}")
        transformer.close_connection()
        print("=" * 60)
        print("Pipeline completed!")
    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during transformation: {str(e)}")
        sys.exit(1)