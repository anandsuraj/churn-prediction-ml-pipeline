# standalone_training_script.py
import argparse
import os
from datetime import datetime
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils.logger import get_logger, PIPELINE_NAMES


class TrainCustomModel:
    def __init__(self) -> None:

        # Get logger for this pipeline
        self.logger = get_logger(PIPELINE_NAMES['BUILD_MODEL'])

        # Environment variables for paths. Defaults are provided for convenience.
        self.features_path = os.environ.get('FEATURES_PATH', 'data/raw')

        # init model save dir
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)

        # MLflow configuration for experiment tracking
        self.mlflow_tracking_uri = os.environ.get(
            'MLFLOW_TRACKING_URI', 'file:///tmp/mlflow-runs')
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("Customer Churn Prediction")

        # A dictionary mapping model names to their scikit-learn classifier instances.
        self.available_models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "logistic_regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        }

    def get_latest_training_data(self) -> str:
        """
        Get the latest training data file from the processed/training_sets directory.

        Returns:
            str: Path to the latest training data file
        """
        import glob

        # Check for training sets first (preferred)
        training_sets_dir = "data/processed/training_sets"
        if os.path.exists(training_sets_dir):
            pattern = os.path.join(training_sets_dir, "*.csv")
            files = glob.glob(pattern)
            if files:
                # Get the most recent file
                latest_file = max(files, key=os.path.getctime)
                self.logger.info(f"Using latest training data: {latest_file}")
                return latest_file

        # Fallback to processed directory
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            pattern = os.path.join(processed_dir, "cleaned_data*.csv")
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                self.logger.info(f"Using latest processed data: {latest_file}")
                return latest_file

        # Final fallback to raw directory
        pattern = os.path.join(self.features_path, "customer_churn_*.csv")
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)
            self.logger.info(f"Using latest raw data: {latest_file}")
            return latest_file

        raise FileNotFoundError(
            "No training data files found in any directory")

    def load_and_split_data(self, feature_file: str) -> tuple:
        """
        Loads data from a CSV file, performs basic cleaning, and splits it for training.

        Args: feature_file (str): The full path to the feature data file.
        Returns: tuple: A tuple containing (X_train, X_test, y_train, y_test).
        """
        self.logger.info(f"Loading data from {feature_file}...")
        if not os.path.exists(feature_file):
            self.logger.error(
                f"Feature data file not found at: {feature_file}")
            raise FileNotFoundError(
                f"Feature data file not found at: {feature_file}")

        df = pd.read_csv(feature_file)

        # Drop  non-numeric customerID column
        if 'customerID' in df.columns:
            self.logger.info("Dropping 'customerID' column.")
            df = df.drop('customerID', axis=1)

        # Clean 'TotalCharges': convert to numeric, fill empty values with 0.
        if 'TotalCharges' in df.columns:
            self.logger.info("Cleaning 'TotalCharges' column.")
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)

        # Select only numeric columns for features
        self.logger.info("Selecting only numeric features for training.")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        features = [col for col in numeric_cols if col !=
                    'Churn' and col != 'Churn']
        X = df[features]

        # Identify and encode the target variable 'Churn' from text to binary (1/0).
        target_col = 'Churn' if 'Churn' in df.columns else 'Churn'
        y = df[target_col]
        if y.dtype == 'object':
            self.logger.info(
                f"Converting target variable '{target_col}' to binary (1/0).")
            y = y.apply(lambda x: 1 if x == 'Yes' else 0)

        self.logger.info(
            f"Training with {len(X.columns)} features: {X.columns.tolist()}")

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the trained model on the test set and returns a dictionary of performance metrics.

        Args:
            model: The trained scikit-learn model.
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test target variable.

        Returns: 
            Dict[str, float]: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        self.logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        return metrics

    def train_model(self, model_type: str):
        """
        The main function to orchestrate the model training, evaluation, and logging process.

        Args: model_type (str): The type of model to train (e.g., 'random_forest').
        """
        if model_type not in self.available_models:
            self.logger.error(
                f"Invalid model_type '{model_type}'. Available options are: {list(self.available_models.keys())}")
            return

        # Use the latest processed training data instead of hardcoded filename
        feature_file = self.get_latest_training_data()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"logreg_model_{timestamp}.joblib"
            model_path = os.path.join(self.model_dir, model_filename)

            X_train, X_test, y_train, y_test = self.load_and_split_data(
                feature_file)

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                self.logger.info(f"Starting MLflow Run ID: {run_id}")
                print("\n" + "="*50)
                print(
                    f"  Training Model: {model_type.replace('_', ' ').title()}")
                print("="*50)

                model = self.available_models[model_type]
                self.logger.info(f"Training '{model_type}' model...")
                model.fit(X_train, y_train)

                metrics = self.evaluate_model(model, X_test, y_test)

                self.logger.info("Logging experiment to MLflow...")
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("features_used", list(X_train.columns))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")

                print("\n--- Training Complete ---")
                print(f"Model Type: {model_type}")
                print(f"MLflow Run ID: {run_id}")
                print("\n--- Performance Metrics ---")
                for metric, value in metrics.items():
                    print(f"  {metric.capitalize()}: {value:.4f}")
                print("\n" + "="*50)
                self.logger.info(
                    f"Successfully trained model and logged to MLflow. Weights saved at {model_path}")
                joblib.dump(model, model_path)
                print(
                    f"\nTo view this run, start the MLflow UI with:\nmlflow ui --backend-store-uri {self.mlflow_tracking_uri}")

        except FileNotFoundError as e:
            self.logger.error(f"Process stopped: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during model training: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a customer Churn prediction model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        # choices=list(AVAILABLE_MODELS.keys()),
        help="The type of model to train."
    )

    args = parser.parse_args()
    training_pipeline = TrainCustomModel()
    training_pipeline.train_model(args.model_type)
