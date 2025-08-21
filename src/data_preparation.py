import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

class DataPreparationPipeline:
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

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

    def encode_categorical_variables(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Remove customerID from encoding
        if 'customerID' in categorical_columns:
            categorical_columns.remove('customerID')

        for column in categorical_columns:
            if column != 'Churn':  # Don't encode target variable yet
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column])
                self.encoders[column] = le
                logging.info(f"Encoded column: {column}")

        # Encode target variable
        if 'Churn' in df.columns:
            df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
            logging.info("Target variable 'Churn' encoded")

        return df_encoded

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

    def perform_eda(self, df, output_dir="output"):
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

        # Feature distributions
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

        logging.info(f"EDA plots saved to {output_dir}")

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

        # Scale features (optional, comment out if not needed for some models)
        # df_final = self.scale_numerical_features(df_outliers_handled)
        df_final = df_outliers_handled

        # Perform EDA on cleaned data
        self.perform_eda(df_final, "output/cleaned_data_eda")

        # Save cleaned dataset
        df_final.to_csv(output_file, index=False)
        logging.info(f"Cleaned data saved to {output_file}")

        return df_final

if __name__ == "__main__":
    pipeline = DataPreparationPipeline()

    # Example usage
    input_file = "data/raw/customer_data.csv"  # Update with actual path
    output_file = "data/processed/cleaned_data.csv"

    # Ensure output directory exists
    import os
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("Data preparation pipeline created")
    cleaned_data = pipeline.run_preparation_pipeline(input_file, output_file)
