import pandas as pd
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
from feast import FeatureStore, Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.data_format import ParquetFormat
from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['FEATURE_STORE'])

class ChurnFeatureStore:
    """Feature store for managing churn prediction features using Feast."""
    def __init__(self, store_path="data/feature_store"):
        """Initialize the Feast feature store and customer entity."""
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        
        try:
            # Initialize Feast feature store
            self.fs = FeatureStore(repo_path=store_path)
            logger.info("Feast feature store initialized at %s", store_path)
        except Exception as e:
            logger.error("Failed to initialize Feast feature store: %s", str(e))
            raise RuntimeError(f"Feature store initialization failed: {str(e)}")
        
        # Define the customer entity
        self.customer_entity = Entity(
            name="customer",
            join_keys=["customerID"],
            value_type=ValueType.STRING,
            description="Unique identifier for a customer"
        )
        
        # Setup feature store with feature definitions
        self.setup_feature_store()

    def setup_feature_store(self):
        """Define and apply feature views for churn prediction features."""
        latest_file = self.find_latest_training_data()
        if not latest_file:
            logger.warning("No training data found, feature views not created")
            return

        try:
            # Create a file source for Feast
            feature_source = FileSource(
                path=latest_file,
                timestamp_column="created_timestamp",
                created_timestamp_column="updated_timestamp"
            )

            # Define features based on the provided dataset
            feature_definitions = [
                {"name": "tenure", "dtype": ValueType.INT64, "description": "Number of months the customer has been with the company"},
                {"name": "MonthlyCharges", "dtype": ValueType.FLOAT, "description": "Monthly bill amount for the customer"},
                {"name": "TotalCharges", "dtype": ValueType.FLOAT, "description": "Total amount charged to the customer"},
                {"name": "gender_encoded", "dtype": ValueType.INT64, "description": "Encoded gender (0: Male, 1: Female)"},
                {"name": "SeniorCitizen", "dtype": ValueType.INT64, "description": "Indicates if customer is a senior citizen (0: No, 1: Yes)"},
                {"name": "Partner_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has a partner (0: No, 1: Yes)"},
                {"name": "Dependents_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has dependents (0: No, 1: Yes)"},
                {"name": "PhoneService_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has phone service (0: No, 1: Yes)"},
                {"name": "MultipleLines_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has multiple lines (0: No, 1: Yes)"},
                {"name": "InternetService_encoded", "dtype": ValueType.INT64, "description": "Type of internet service (0: None, 1: DSL, 2: Fiber)"},
                {"name": "OnlineSecurity_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has online security (0: No, 1: Yes)"},
                {"name": "OnlineBackup_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has online backup (0: No, 1: Yes)"},
                {"name": "DeviceProtection_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has device protection (0: No, 1: Yes)"},
                {"name": "TechSupport_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has tech support (0: No, 1: Yes)"},
                {"name": "StreamingTV_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has streaming TV (0: No, 1: Yes)"},
                {"name": "StreamingMovies_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer has streaming movies (0: No, 1: Yes)"},
                {"name": "Contract_encoded", "dtype": ValueType.INT64, "description": "Type of contract (0: Month-to-month, 1: One year, 2: Two year)"},
                {"name": "PaperlessBilling_encoded", "dtype": ValueType.INT64, "description": "Indicates if customer uses paperless billing (0: No, 1: Yes)"},
                {"name": "PaymentMethod_encoded", "dtype": ValueType.INT64, "description": "Payment method (0: Electronic check, 1: Mailed check, 2: Bank transfer, 3: Credit card)"},
                {"name": "Churn", "dtype": ValueType.INT64, "description": "Target variable indicating if customer churned (0: No, 1: Yes)"},
                {"name": "tenure_group", "dtype": ValueType.STRING, "description": "Categorized tenure range (e.g., 1-12, 13-24 months)"},
                {"name": "charges_per_tenure", "dtype": ValueType.FLOAT, "description": "Average monthly charges per tenure month"},
                {"name": "total_to_monthly_ratio", "dtype": ValueType.FLOAT, "description": "Ratio of total charges to monthly charges"},
                {"name": "avg_monthly_charges", "dtype": ValueType.FLOAT, "description": "Average monthly charges over tenure"},
                {"name": "total_services", "dtype": ValueType.INT64, "description": "Total number of services subscribed by customer"},
                {"name": "service_density", "dtype": ValueType.FLOAT, "description": "Ratio of services to tenure"},
                {"name": "customer_value_segment", "dtype": ValueType.STRING, "description": "Customer value segment (e.g., Low, Medium, High)"},
                {"name": "tenure_stability", "dtype": ValueType.FLOAT, "description": "Measure of customer tenure stability"},
                {"name": "tenure_monthly_interaction", "dtype": ValueType.FLOAT, "description": "Interaction between tenure and monthly charges"},
                {"name": "tenure_total_interaction", "dtype": ValueType.FLOAT, "description": "Interaction between tenure and total charges"},
                {"name": "services_charges_interaction", "dtype": ValueType.FLOAT, "description": "Interaction between total services and monthly charges"}
            ]

            # Create feature objects
            features = [
                Feature(name=feat["name"], dtype=feat["dtype"], description=feat["description"])
                for feat in feature_definitions
            ]

            # Define feature view
            feature_view = FeatureView(
                name="churn_features",
                entities=[self.customer_entity],
                schema=features,
                source=feature_source,
                ttl=None  # Features don't expire
            )

            # Apply the feature view to the store
            self.fs.apply([self.customer_entity, feature_view])
            logger.info("Feature views for %d features applied to Feast store", len(features))

        except Exception as e:
            logger.error("Error setting up feature store: %s", str(e))
            raise RuntimeError(f"Feature store setup failed: {str(e)}")

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

    def store_features(self, entity_id: str, features: Dict[str, Any]):
        """Store feature values for a customer entity."""
        timestamp = datetime.now()
        feature_data = {
            "customerID": entity_id,
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            **features
        }
        
        try:
            # Convert to DataFrame and save to parquet
            df = pd.DataFrame([feature_data])
            temp_path = os.path.join(self.store_path, f"temp_{entity_id}.parquet")
            df.to_parquet(temp_path)
            
            # Ingest into Feast
            self.fs.write_to_source(df, "churn_features")
            logger.debug("Stored features for entity: %s", entity_id)
        
        except Exception as e:
            logger.error("Failed to store features for entity %s: %s", entity_id, str(e))
            raise RuntimeError(f"Feature storage failed: {str(e)}")

    def get_features(self, entity_id: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """Retrieve features for a customer entity for inference."""
        try:
            entity_df = pd.DataFrame({"customerID": [entity_id], "created_timestamp": [datetime.now()]})
            feature_refs = [f"churn_features:{name}" for name in feature_names] if feature_names else None
            
            features = self.fs.get_online_features(
                entity_rows=entity_df,
                features=feature_refs
            ).to_dict()
            
            # Clean up the response
            result = {k: v[0] for k, v in features.items() if k != 'customerID'}
            logger.debug("Retrieved features for entity %s: %s", entity_id, result)
            return result
        
        except Exception as e:
            logger.error("Failed to retrieve features for entity %s: %s", entity_id, str(e))
            return {}

    def get_training_dataset(self, entity_ids: List[str] = None) -> pd.DataFrame:
        """Create training dataset from stored features."""
        try:
            if not entity_ids:
                latest_file = self.find_latest_training_data()
                if latest_file:
                    df = pd.read_csv(latest_file)
                    entity_ids = df['customerID'].tolist()
                else:
                    logger.warning("No entity IDs provided and no training data found")
                    return pd.DataFrame()

            entity_df = pd.DataFrame({
                "customerID": entity_ids,
                "created_timestamp": [datetime.now() for _ in entity_ids]
            })

            # Retrieve historical features
            feature_refs = [f"churn_features:{feat.name}" for feat in self.fs.get_feature_view("churn_features").features]
            features = self.fs.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            
            logger.info("Retrieved training dataset with shape %s", features.shape)
            return features
        
        except Exception as e:
            logger.error("Failed to retrieve training dataset: %s", str(e))
            return pd.DataFrame()

    def get_feature_metadata(self, output_format: str = "dataframe") -> Any:
        """Get feature metadata in DataFrame or Markdown format."""
        try:
            feature_view = self.fs.get_feature_view("churn_features")
            metadata = [{
                "feature_name": feature.name,
                "description": feature.description,
                "source": "data_preparation",
                "version": "1.0",
                "data_type": str(feature.dtype),
                "created_date": datetime.now().isoformat(),
                "is_active": True
            } for feature in feature_view.features]

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

    def populate_from_dataframe(self, df: pd.DataFrame, entity_id_col: str = 'customerID'):
        """Populate feature store from a DataFrame."""
        logger.info("Populating feature store from DataFrame with %d records", len(df))
        
        try:
            # Ensure required timestamps are present
            if 'created_timestamp' not in df.columns:
                df['created_timestamp'] = datetime.now()
            if 'updated_timestamp' not in df.columns:
                df['updated_timestamp'] = datetime.now()
            
            # Save to parquet and ingest
            temp_path = os.path.join(self.store_path, "temp_data.parquet")
            df.to_parquet(temp_path)
            self.fs.write_to_source(df, "churn_features")
            logger.info("Populated feature store with %d records", len(df))
        
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
                entity_id_col = 'customerID' if 'customerID' in df.columns else 'customer_id'
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
            },
            {
                "customerID": "sample_002",
                "tenure": 24,
                "MonthlyCharges": 49.99,
                "TotalCharges": 1199.76,
                "gender_encoded": 1,
                "SeniorCitizen": 0,
                "Partner_encoded": 1,
                "Dependents_encoded": 0,
                "PhoneService_encoded": 1,
                "MultipleLines_encoded": 1,
                "InternetService_encoded": 1,
                "OnlineSecurity_encoded": 1,
                "OnlineBackup_encoded": 1,
                "DeviceProtection_encoded": 1,
                "TechSupport_encoded": 1,
                "StreamingTV_encoded": 1,
                "StreamingMovies_encoded": 1,
                "Contract_encoded": 1,
                "PaperlessBilling_encoded": 1,
                "PaymentMethod_encoded": 1,
                "Churn": 0,
                "tenure_group": "13-24",
                "charges_per_tenure": 2.083,
                "total_to_monthly_ratio": 24.0,
                "avg_monthly_charges": 49.99,
                "total_services": 6,
                "service_density": 0.25,
                "customer_value_segment": "Medium",
                "tenure_stability": 0.9,
                "tenure_monthly_interaction": 1199.76,
                "tenure_total_interaction": 28794.24,
                "services_charges_interaction": 299.94,
                "created_timestamp": datetime.now(),
                "updated_timestamp": datetime.now()
            }
        ])
        
        try:
            self.populate_from_dataframe(sample_data)
            logger.info("Sample feature store created with 2 records")
            return "Sample feature store created with 2 sample records"
        except Exception as e:
            logger.error("Failed to create sample feature store: %s", str(e))
            return f"Error: {str(e)}"

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

    def close(self):
        """Close feature store (no-op for Feast)."""
        logger.info("Feature store connection closed")

if __name__ == "__main__":
    # Test the feature store
    try:
        feature_store = ChurnFeatureStore()
        
        # Auto-populate from latest data
        result = feature_store.auto_populate_from_latest_data()
        print(f"Feature store setup result: {result}")
        
        # Show feature metadata
        metadata = feature_store.get_feature_metadata()
        print(f"\nFeature metadata: {metadata.shape}")
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
            print(training_df.head())
        
        # Demonstrate feature retrieval API
        sample_features = feature_store.demonstrate_feature_retrieval()
        print(f"\nSample feature retrieval: {sample_features}")
        
        feature_store.close()
        print("\nFeature store test completed")
    
    except Exception as e:
        logger.error("Feature store test failed: %s", str(e))
        print(f"Error during feature store test: {str(e)}")