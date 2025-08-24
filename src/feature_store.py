"""
Simple Feature Store for Churn Prediction
========================================
A lightweight feature store implementation that works with CSV data.
"""

import pandas as pd
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['FEATURE_STORE'])

class SimpleChurnFeatureStore:
    """Simple feature store for managing churn prediction features."""
    
    def __init__(self, store_path="data/feature_store"):
        """Initialize the simple feature store."""
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        os.makedirs(os.path.join(store_path, "offline_store"), exist_ok=True)
        
        logger.info("Simple feature store initialized at %s", store_path)
        
        # Auto-populate from latest data
        self.auto_populate_from_latest_data()

    def find_latest_training_data(self):
        """Find the latest training data CSV file from the training_sets directory."""
        training_sets_dir = "data/processed/training_sets"
        
        if not os.path.exists(training_sets_dir):
            logger.warning("Training sets directory not found: %s", training_sets_dir)
            return None
        
        try:
            # Find all CSV files in training_sets directory
            csv_files = glob.glob(os.path.join(training_sets_dir, "*.csv"))
            if not csv_files:
                logger.warning("No CSV files found in %s", training_sets_dir)
                return None
            
            # Get the latest file by modification time
            latest_file = max(csv_files, key=os.path.getmtime)
            logger.info("Found latest training data: %s", latest_file)
            return latest_file
        
        except Exception as e:
            logger.error("Error finding latest training data: %s", str(e))
            return None

    def populate_from_dataframe(self, df: pd.DataFrame, entity_id_col: str = 'customerID'):
        """Populate feature store from a DataFrame."""
        logger.info("Populating feature store from DataFrame with %d records", len(df))
        
        try:
            # Ensure required timestamps are present
            if 'created_timestamp' not in df.columns:
                df['created_timestamp'] = datetime.now()
            if 'updated_timestamp' not in df.columns:
                df['updated_timestamp'] = datetime.now()
            
            # Save to CSV
            csv_path = os.path.join(self.store_path, "churn_features.csv")
            df.to_csv(csv_path, index=False)
            logger.info("Populated feature store with %d records", len(df))
            
            # Also save a sample for quick access
            sample_path = os.path.join(self.store_path, "churn_features_sample.csv")
            df.head(100).to_csv(sample_path, index=False)
            logger.info("Saved sample data with 100 records")
            
        except Exception as e:
            logger.error("Failed to populate feature store: %s", str(e))
            raise RuntimeError(f"Feature store population failed: {str(e)}")

    def auto_populate_from_latest_data(self):
        """Automatically populate feature store from the latest training data."""
        latest_file = self.find_latest_training_data()
        
        if latest_file:
            try:
                logger.info("Loading data from: %s", latest_file)
                df = pd.read_csv(latest_file)
                logger.info("Loaded data with shape: %s", df.shape)
                
                # Determine entity ID column
                entity_id_col = 'customer_id' if 'customer_id' in df.columns else 'customerID'
                if entity_id_col not in df.columns:
                    entity_id_col = df.columns[0]  # Use first column as entity ID
                
                self.populate_from_dataframe(df, entity_id_col)
                return f"Feature store populated with {len(df)} records from {latest_file}"
                
            except Exception as e:
                logger.error("Error loading data from %s: %s", latest_file, str(e))
                return f"Error: {str(e)}"
        else:
            logger.warning("No training data found, creating sample feature store")
            return self.create_sample_features()

    def create_sample_features(self):
        """Create sample features for demonstration when no data is available."""
        logger.info("Creating sample feature store")
        
        sample_data = pd.DataFrame([
            {
                "customerID": "sample_001",
                "tenure": 12,
                "MonthlyCharges": 29.99,
                "TotalCharges": 359.88,
                "gender_encoded": 0,
                "SeniorCitizen": 0,
                "Partner_encoded": 0,
                "Dependents_encoded": 0,
                "PhoneService_encoded": 1,
                "MultipleLines_encoded": 0,
                "InternetService_encoded": 1,
                "OnlineSecurity_encoded": 0,
                "OnlineBackup_encoded": 0,
                "DeviceProtection_encoded": 0,
                "TechSupport_encoded": 0,
                "StreamingTV_encoded": 0,
                "StreamingMovies_encoded": 0,
                "Contract_encoded": 0,
                "PaperlessBilling_encoded": 1,
                "PaymentMethod_encoded": 0,
                "Churn": 0,
                "tenure_group": "1-12",
                "charges_per_tenure": 2.499,
                "total_to_monthly_ratio": 12.0,
                "avg_monthly_charges": 29.99,
                "total_services": 1,
                "service_density": 0.083,
                "customer_value_segment": "Low",
                "tenure_stability": 1.0,
                "tenure_monthly_interaction": 359.88,
                "tenure_total_interaction": 4318.56,
                "services_charges_interaction": 29.99,
                "created_timestamp": datetime.now(),
                "updated_timestamp": datetime.now()
            }
        ])
        
        try:
            self.populate_from_dataframe(sample_data)
            logger.info("Sample feature store created with 1 record")
            return "Sample feature store created with 1 sample record"
        except Exception as e:
            logger.error("Failed to create sample feature store: %s", str(e))
            return f"Error: {str(e)}"

    def get_features(self, entity_id: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Retrieve features for a customer entity for inference."""
        try:
            csv_path = os.path.join(self.store_path, "churn_features.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Find the row with matching customer_id
                customer_row = df[df['customer_id'] == entity_id]
                if not customer_row.empty:
                    if feature_names:
                        result = {col: customer_row[col].iloc[0] for col in feature_names if col in customer_row.columns}
                    else:
                        result = {col: customer_row[col].iloc[0] for col in customer_row.columns if col not in ['customerID', 'created_timestamp', 'updated_timestamp']}
                    logger.debug("Retrieved features for entity %s: %s", entity_id, result)
                    return result
                else:
                    logger.warning("No features found for entity %s", entity_id)
                    return {}
            else:
                logger.warning("No feature store data found")
                return {}
        
        except Exception as e:
            logger.error("Failed to retrieve features for entity %s: %s", entity_id, str(e))
            return {}

    def get_training_dataset(self) -> pd.DataFrame:
        """Get the complete training dataset."""
        try:
            csv_path = os.path.join(self.store_path, "churn_features.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info("Retrieved training dataset with shape %s", df.shape)
                return df
            else:
                logger.warning("No feature store data found")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error("Failed to retrieve training dataset: %s", str(e))
            return pd.DataFrame()

    def get_feature_metadata(self, output_format: str = "dataframe") -> Any:
        """Get feature metadata in DataFrame or Markdown format."""
        try:
            csv_path = os.path.join(self.store_path, "churn_features.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                metadata = []
                for col in df.columns:
                    if col not in ['created_timestamp', 'updated_timestamp']:
                        feature_type = 'categorical' if '_encoded' in col else 'numerical'
                        metadata.append({
                            "feature_name": col,
                            "description": f"Feature: {col}",
                            "source": "data_preparation",
                            "version": "1.0",
                            "data_type": feature_type,
                            "created_date": datetime.now().isoformat(),
                            "is_active": True
                        })
            else:
                metadata = []

            if output_format == "markdown":
                markdown = "# Feature Metadata\n\n"
                markdown += "| Feature Name | Description | Source | Version | Data Type | Created Date | Is Active |\n"
                markdown += "|--------------|-------------|--------|---------|-----------|--------------|-----------|\n"
                for meta in metadata:
                    markdown += f"| {meta['feature_name']} | {meta['description']} | {meta['source']} | {meta['version']} | {meta['data_type']} | {meta['created_date']} | {meta['is_active']} |\n"
                return markdown
            else:
                return pd.DataFrame(metadata)
        
        except Exception as e:
            logger.error("Failed to retrieve feature metadata: %s", str(e))
            return pd.DataFrame() if output_format == "dataframe" else ""

    def demonstrate_feature_retrieval(self, entity_id: str = "sample_001") -> Dict[str, Any]:
        """Demonstrate feature retrieval for inference with a sample query."""
        feature_names = ["tenure", "MonthlyCharges", "Churn", "tenure_group", "customer_value_segment"]
        try:
            features = self.get_features(entity_id, feature_names)
            logger.info("Sample feature retrieval for entity %s: %s", entity_id, features)
            return features
        except Exception as e:
            logger.error("Sample feature retrieval failed for entity %s: %s", entity_id, str(e))
            return {}

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of the feature store."""
        try:
            csv_path = os.path.join(self.store_path, "churn_features.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                summary = {
                    "total_records": len(df),
                    "total_features": len(df.columns),
                    "feature_columns": list(df.columns),
                    "data_types": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "churn_distribution": df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else {},
                    "last_updated": datetime.now().isoformat()
                }
                return summary
            else:
                return {"error": "No feature store data found"}
        except Exception as e:
            logger.error("Failed to get feature summary: %s", str(e))
            return {"error": str(e)}

    def close(self):
        """Close feature store (no-op for simple implementation)."""
        logger.info("Feature store connection closed")

if __name__ == "__main__":
    # Test the simple feature store
    try:
        print("=" * 60)
        print("Simple Churn Feature Store Test")
        print("=" * 60)
        
        feature_store = SimpleChurnFeatureStore()
        
        # Auto-populate from latest data
        result = feature_store.auto_populate_from_latest_data()
        print(f"Feature store setup result: {result}")
        
        # Show feature metadata
        metadata = feature_store.get_feature_metadata()
        print(f"\nFeature metadata: {metadata.shape}")
        if not metadata.empty:
            print(metadata.to_string(index=False))
        
        # Save metadata as Markdown
        metadata_md = feature_store.get_feature_metadata(output_format="markdown")
        with open("data/feature_store/feature_metadata.md", "w") as f: 
            f.write(metadata_md)
        print("\nFeature metadata saved to data/feature_store/feature_metadata.md")
        
        # Show training dataset
        training_df = feature_store.get_training_dataset()
        print(f"\nTraining dataset: {training_df.shape}")
        if not training_df.empty:
            print(f"Columns: {list(training_df.columns)}")
            print(f"Sample data:")
            print(training_df.head(3).to_string(index=False))
        
        # Get feature summary
        summary = feature_store.get_feature_summary()
        print(f"\nFeature summary:")
        for key, value in summary.items():
            if key != 'feature_columns':
                print(f"  {key}: {value}")
        
        # Demonstrate feature retrieval API
        if 'customerID' in training_df.columns:
            sample_customer = training_df['customerID'].iloc[0]
            sample_features = feature_store.demonstrate_feature_retrieval(sample_customer)
            print(f"\nSample feature retrieval for {sample_customer}:")
            for feature, value in sample_features.items():
                print(f"  {feature}: {value}")
        
        feature_store.close()
        print("\n" + "=" * 60)
        print("Simple feature store test completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error("Feature store test failed: %s", str(e))
        print(f"Error during feature store test: {str(e)}")
