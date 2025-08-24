"""
Safe Data Preparation Pipeline
-----------------------------
A simplified version of data preparation that avoids segmentation faults
by minimizing matplotlib/seaborn usage and using safer operations.
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_PREPARATION'])

class SafeDataPreparationPipeline:
    def __init__(self):
        """Initialize pipeline with scaler and numerical columns."""
        self.scaler = StandardScaler()
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV file into DataFrame."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values: median for numeric, mode for categorical."""
        df_cleaned = df.copy()
        
        # Convert TotalCharges to numeric
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
                    logger.info(f"Filled {column} missing values with median: {median_value}")
                else:
                    # Fill categorical columns with mode
                    mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
                    df_cleaned[column].fillna(mode_value, inplace=True)
                    logger.info(f"Filled {column} missing values with mode: {mode_value}")
        
        return df_cleaned

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features and map target variable."""
        df_encoded = df.copy()
        
        # Map target variable
        if 'Churn' in df_encoded.columns:
            df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
            logger.info("Mapped Churn: Yes->1, No->0")
        
        # Get categorical columns (excluding target)
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_columns:
            categorical_columns.remove('Churn')
        
        # One-hot encode categorical features
        if categorical_columns:
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
            logger.info(f"One-hot encoded columns: {categorical_columns}")
        
        return df_encoded

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        df_derived = df.copy()
        
        try:
            # Monthly charges per tenure
            if 'MonthlyCharges' in df_derived.columns and 'tenure' in df_derived.columns:
                df_derived['MonthlyCharges_per_tenure'] = df_derived['MonthlyCharges'] / (df_derived['tenure'] + 1)
            
            # Total charges per tenure
            if 'TotalCharges' in df_derived.columns and 'tenure' in df_derived.columns:
                df_derived['TotalCharges_per_tenure'] = df_derived['TotalCharges'] / (df_derived['tenure'] + 1)
            
            # Average monthly charges
            if 'TotalCharges' in df_derived.columns and 'tenure' in df_derived.columns:
                df_derived['AvgMonthlyCharges'] = df_derived['TotalCharges'] / (df_derived['tenure'] + 1)
            
            logger.info("Created derived features successfully")
            
        except Exception as e:
            logger.warning(f"Error creating derived features: {str(e)}")
        
        return df_derived

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method."""
        df_capped = df.copy()
        
        for column in self.numerical_columns:
            if column in df_capped.columns:
                Q1 = df_capped[column].quantile(0.25)
                Q3 = df_capped[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                original_count = len(df_capped)
                df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Capped outliers in {column}: bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_capped

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df_scaled = df.copy()
        
        # Get numerical columns that exist in the dataframe
        numerical_cols_to_scale = [col for col in self.numerical_columns if col in df_scaled.columns]
        
        if numerical_cols_to_scale:
            df_scaled[numerical_cols_to_scale] = self.scaler.fit_transform(df_scaled[numerical_cols_to_scale])
            logger.info(f"Scaled numerical columns: {numerical_cols_to_scale}")
        
        return df_scaled

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save processed data to file."""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/{filename}'
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data: {filepath}")
        return filepath

    def run_preparation_auto(self) -> dict:
        """Run the complete data preparation pipeline automatically."""
        try:
            logger.info("Starting automated data preparation pipeline...")
            
            # Find the latest raw data file
            raw_files = glob.glob('data/raw/customer_churn_*.csv')
            if not raw_files:
                raise FileNotFoundError("No raw data files found")
            
            latest_file = max(raw_files, key=os.path.getctime)
            logger.info(f"Processing file: {latest_file}")
            
            # Load data
            df = self.load_data(latest_file)
            original_shape = df.shape
            
            # Data preparation steps
            df = self.handle_missing_values(df)
            df = self.encode_categorical_features(df)
            df = self.create_derived_features(df)
            df = self.handle_outliers(df)
            df = self.scale_features(df)
            
            # Save processed data
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.save_processed_data(df, f'cleaned_data_{timestamp}.csv')
            
            result = {
                'status': 'success',
                'input_file': latest_file,
                'output_file': output_file,
                'original_shape': original_shape,
                'final_shape': df.shape,
                'timestamp': timestamp
            }
            
            logger.info("Data preparation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

# Create an alias for backward compatibility
DataPreparationPipeline = SafeDataPreparationPipeline