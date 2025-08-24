"""
Data Preparation Pipeline
------------------------
Cleans and preprocesses dataset for modeling:
- Handles missing values (numeric: median, categorical: mode)
- One-hot encodes categorical features; maps 'Churn' to 0/1
- Creates derived features and caps outliers (IQR method)
- Scales numerical features using StandardScaler

Saves:
- EDA outputs: data/eda/raw and data/eda/cleaned
- Cleaned and scaled datasets: data/processed
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Import matplotlib with Agg backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_PREPARATION'])

class DataPreparationPipeline:
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
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['int64', 'float64']:
                    imputer = df_cleaned[column].median()
                    df_cleaned[column].fillna(imputer, inplace=True)
                    logger.info(f"Filled {column} with median: {imputer}")
                else:
                    imputer = df_cleaned[column].mode()[0]
                    df_cleaned[column].fillna(imputer, inplace=True)
                    logger.info(f"Filled {column} with mode: {imputer}")
        return df_cleaned

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns; map Churn to 0/1."""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['customerID', 'Churn']]
        if categorical_cols:
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
            logger.info(f"One-hot encoded {len(categorical_cols)} columns")
        if 'Churn' in df_encoded.columns:
            df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
            logger.info("Encoded 'Churn' to 0/1")
        return df_encoded

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for churn prediction."""
        df_features = df.copy()
        df_features['tenure_group'] = pd.cut(
            df_features['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-12', '12-24', '24-48', '48+']
        ).cat.codes
        df_features['charges_per_tenure'] = df_features['MonthlyCharges'] / (df_features['tenure'] + 1)
        df_features['total_to_monthly_ratio'] = df_features['TotalCharges'] / df_features['MonthlyCharges']
        df_features['avg_monthly_charges'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)
        logger.info("Created derived features")
        return df_features

    def cap_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Cap outliers using IQR method."""
        df_clean = df.copy()
        for column in columns:
            if column in df_clean.columns:
                Q1, Q3 = df_clean[column].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = ((df_clean[column] < lower) | (df_clean[column] > upper)).sum()
                if outliers > 0:
                    df_clean[column] = df_clean[column].clip(lower, upper)
                    logger.info(f"Capped {outliers} outliers in {column}")
        return df_clean

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df_scaled = df.copy()
        num_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Churn', 'customerID']]
        if num_cols:
            df_scaled[num_cols] = self.scaler.fit_transform(df_scaled[num_cols])
            logger.info(f"Scaled {len(num_cols)} numerical features")
        return df_scaled

    def save_eda_plots(self, df: pd.DataFrame, output_dir: str):
        """Generate and save EDA plots."""
        os.makedirs(output_dir, exist_ok=True)
        df.describe().to_csv(f"{output_dir}/summary_stats.csv")
        if 'Churn' in df.columns:
            plt.figure(figsize=(6, 6))
            plt.pie(df['Churn'].value_counts(), labels=['No Churn', 'Churn'], autopct='%1.1f%%')
            plt.title('Churn Distribution')
            plt.savefig(f"{output_dir}/churn_distribution.png")
            plt.close()
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()
        num_cols = df.select_dtypes(include=[np.number]).columns[:6]
        if num_cols.size > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()
            for i, col in enumerate(num_cols):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distributions.png")
            plt.close()
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()
            for i, col in enumerate(num_cols):
                if i < len(axes):
                    sns.boxplot(x=df[col], ax=axes[i])
                    axes[i].set_title(f'{col} Box Plot')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/box_plots.png")
            plt.close()
        logger.info(f"EDA plots saved to {output_dir}")

    def run_pipeline(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Run full data preparation pipeline."""
        logger.info("Starting data preparation pipeline")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs("data/eda/raw", exist_ok=True)
        os.makedirs("data/eda/cleaned", exist_ok=True)
        df = self.load_data(input_file)
        self.save_eda_plots(df, "data/eda/raw")
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)
        df = self.cap_outliers(df, self.numerical_columns)
        df_scaled = self.scale_features(df)
        self.save_eda_plots(df, "data/eda/cleaned")
        df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")
        scaled_output = output_file.replace(".csv", "_scaled.csv")
        df_scaled.to_csv(scaled_output, index=False)
        logger.info(f"Scaled data saved to {scaled_output}")
        return df

    def run_preparation_auto(self):
        """Run preparation pipeline with automatic file detection"""
        input_file = find_latest_csv()
        output_file = "data/processed/cleaned_data.csv"
        logger.info(f"Auto-detected input file: {input_file}")
        return self.run_pipeline(input_file, output_file)

def find_latest_csv() -> str:
    """Find most recent CSV file in data/raw."""
    patterns = [
        "data/raw/customer_churn_*.csv",
        "data/raw/sources/*/churn/*/*/*/*.csv"
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError("No CSV files found in data/raw")
    return max(candidates, key=os.path.getctime)

if __name__ == "__main__":
    pipeline = DataPreparationPipeline()
    input_file = find_latest_csv()
    output_file = "data/processed/cleaned_data.csv"
    print("Running data preparation pipeline...")
    cleaned_data = pipeline.run_pipeline(input_file, output_file)
    print(f"Pipeline completed. Output saved to {output_file}")