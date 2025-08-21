"""
Data Preparation
----------------
Cleans and preprocesses the ingested dataset for modeling:
- Handles missing values (numeric median; categorical mode)
- One-hot encodes categorical features; maps target 'Churn' to 0/1
- Creates derived features and handles outliers (IQR capping)
- Scales numerical features (StandardScaler)

Saves:
- EDA outputs under data/eda/raw and data/eda/cleaned
- Cleaned dataset and scaled variant under data/processed
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
import os
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

# Class: cleans, encodes, engineers features, scales and runs EDA
class DataPreparationPipeline:
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    # Load CSV into DataFrame
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    # Fill missing values (numeric median, categorical mode)
    def handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        df_cleaned = df.copy()

        # Convert TotalCharges to numeric (contains spaces)
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')

        # Fill missing TotalCharges with median
        if df_cleaned['TotalCharges'].isnull().sum() > 0:
            median_charges = df_cleaned['TotalCharges'].median()
            df_cleaned['TotalCharges'].fillna(median_charges, inplace=True)
            logging.info(f"Filled {df['TotalCharges'].isnull().sum()} missing TotalCharges values")

        # Handle other missing values
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype == 'object':
                    # Fill categorical with mode
                    mode_value = df_cleaned[column].mode()[0]
                    df_cleaned[column].fillna(mode_value, inplace=True)
                else:
                    # Fill numerical with median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)

        return df_cleaned

    # One-hot encode categorical columns; map target to 0/1
    def encode_categorical_variables(self, df):
        """One-hot encode categorical variables; map target 'Churn' to 0/1"""
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

        # Remove identifier and target from one-hot
        for col in ['customerID', 'Churn']:
            if col in categorical_columns:
                categorical_columns.remove(col)

        # One-hot encode with drop_first to reduce multicollinearity
        if categorical_columns:
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
            logging.info(f"One-hot encoded {len(categorical_columns)} categorical columns")

        # Encode target variable
        if 'Churn' in df_encoded.columns:
            df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
            logging.info("Target variable 'Churn' encoded")

        return df_encoded

    # Add engineered features helpful for churn modeling
    def create_derived_features(self, df):
        """Create derived features for better prediction"""
        df_features = df.copy()

        # Tenure groups
        df_features['tenure_group'] = pd.cut(df_features['tenure'], 
                                           bins=[0, 12, 24, 48, 72], 
                                           labels=['0-12', '12-24', '24-48', '48+'])
        df_features['tenure_group'] = df_features['tenure_group'].cat.codes

        # Monthly charges per tenure
        df_features['charges_per_tenure'] = df_features['MonthlyCharges'] / (df_features['tenure'] + 1)

        # Total charges per monthly charges ratio
        df_features['total_to_monthly_ratio'] = df_features['TotalCharges'] / df_features['MonthlyCharges']

        # Average monthly charges
        df_features['avg_monthly_charges'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)

        logging.info("Derived features created")
        return df_features

    # Cap outliers using IQR bounds for stability
    def handle_outliers(self, df, columns):
        """Handle outliers using IQR method"""
        df_clean = df.copy()

        for column in columns:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_count = len(df_clean[(df_clean[column] < lower_bound) | 
                                            (df_clean[column] > upper_bound)])

                if outliers_count > 0:
                    df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
                    df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
                    logging.info(f"Handled {outliers_count} outliers in {column}")

        return df_clean

    # Standard scale numeric features (excludes ID/target)
    def scale_numerical_features(self, df):
        """Scale numerical features"""
        df_scaled = df.copy()
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and ID columns
        exclude_columns = ['Churn', 'customerID']
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

        scaler = StandardScaler()
        df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
        self.scalers['standard'] = scaler

        logging.info(f"Scaled {len(numerical_columns)} numerical features")
        return df_scaled

    # Save summary stats, histograms, box plots, correlation heatmap
    def perform_eda(self, df, output_dir="data/eda"):
        """Perform exploratory data analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Basic statistics
        stats = df.describe()
        stats.to_csv(f"{output_dir}/basic_statistics.csv")

        # Churn distribution
        if 'Churn' in df.columns:
            plt.figure(figsize=(8, 6))
            churn_counts = df['Churn'].value_counts()
            plt.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%')
            plt.title('Churn Distribution')
            plt.savefig(f"{output_dir}/churn_distribution.png")
            plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

        # Feature distributions (histograms)
        numeric_columns = df.select_dtypes(include=[np.number]).columns[:6]  # First 6 numeric columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, column in enumerate(numeric_columns):
            if i < len(axes):
                df[column].hist(bins=30, ax=axes[i])
                axes[i].set_title(f'{column} Distribution')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distributions.png")
        plt.close()

        # Box plots for the same numeric columns
        if len(numeric_columns) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            for i, column in enumerate(numeric_columns):
                if i < len(axes):
                    sns.boxplot(x=df[column], ax=axes[i])
                    axes[i].set_title(f'{column} Box Plot')
                    axes[i].set_xlabel(column)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/box_plots.png")
            plt.close()

        logging.info(f"EDA plots saved to {output_dir}")

    # Execute full preparation pipeline and persist outputs
    def run_preparation_pipeline(self, input_file, output_file):
        """Run complete data preparation pipeline"""
        logging.info("Starting data preparation pipeline")

        # Load data
        df = self.load_data(input_file)

        # Perform EDA on raw data
        self.perform_eda(df, "output/raw_data_eda")

        # Data cleaning steps
        df_cleaned = self.handle_missing_values(df)
        df_encoded = self.encode_categorical_variables(df_cleaned)
        df_features = self.create_derived_features(df_encoded)

        # Handle outliers in numerical columns
        numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_outliers_handled = self.handle_outliers(df_features, numerical_columns)

        # Scale features
        df_final = df_outliers_handled
        df_scaled = self.scale_numerical_features(df_outliers_handled)

        # Perform EDA on cleaned data
        self.perform_eda(df_final, "data/eda/cleaned")

        # Save cleaned dataset(s)
        df_final.to_csv(output_file, index=False)
        logging.info(f"Cleaned data saved to {output_file}")
        scaled_output = os.path.splitext(output_file)[0] + "_scaled.csv"
        df_scaled.to_csv(scaled_output, index=False)
        logging.info(f"Scaled cleaned data saved to {scaled_output}")

        return df_final

"""Helper: locate the latest ingested CSV under data/raw structures."""
def find_latest_ingested_csv() -> str:
    """Find the most recent ingested CSV from known locations."""
    candidates = []
    # Legacy flat pattern
    candidates.extend(glob.glob("data/raw/customer_churn_*.csv"))
    # New partitioned storage pattern
    candidates.extend(glob.glob("data/raw/sources/*/churn/*/*/*/*.csv"))
    if not candidates:
        raise FileNotFoundError("No ingested CSV files found under data/raw")
    latest = max(candidates, key=os.path.getctime)
    return latest


if __name__ == "__main__":
    pipeline = DataPreparationPipeline()

    # Auto-detect latest ingested CSV
    input_file = find_latest_ingested_csv()
    output_file = "data/processed/cleaned_data.csv"

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/eda/raw", exist_ok=True)
    os.makedirs("data/eda/cleaned", exist_ok=True)

    print("Data preparation pipeline created")
    # EDA on raw
    raw_df = pipeline.load_data(input_file)
    pipeline.perform_eda(raw_df, "data/eda/raw")

    cleaned_data = pipeline.run_preparation_pipeline(input_file, output_file)
