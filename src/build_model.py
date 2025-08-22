# standalone_training_script.py
import argparse
import logging
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

# Set up logging to display information in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for paths. Defaults are provided for convenience.
FEATURES_PATH = os.environ.get('FEATURES_PATH', 'data/raw')

# init model save dir
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# MLflow configuration for experiment tracking
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///tmp/mlflow-runs')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Customer Churn Prediction")

# A dictionary mapping model names to their scikit-learn classifier instances.
AVAILABLE_MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "logistic_regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
}

def load_and_split_data(feature_file: str) -> tuple:
    """
    Loads data from a CSV file, performs basic cleaning, and splits it for training.
    
    Args: feature_file (str): The full path to the feature data file.
    Returns: tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    logging.info(f"Loading data from {feature_file}...")
    if not os.path.exists(feature_file):
        logging.error(f"Feature data file not found at: {feature_file}")
        raise FileNotFoundError(f"Feature data file not found at: {feature_file}")

    df = pd.read_csv(feature_file)

    # Drop  non-numeric customerID column
    if 'customerID' in df.columns:
        logging.info("Dropping 'customerID' column.")
        df = df.drop('customerID', axis=1)

    # Clean 'TotalCharges': convert to numeric, fill empty values with 0.
    if 'TotalCharges' in df.columns:
        logging.info("Cleaning 'TotalCharges' column.")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

    # Select only numeric columns for features 
    logging.info("Selecting only numeric features for training.")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    features = [col for col in numeric_cols if col != 'Churn' and col != 'Churn']
    X = df[features]

    # Identify and encode the target variable 'Churn' from text to binary (1/0).
    target_col = 'Churn' if 'Churn' in df.columns else 'Churn'
    y = df[target_col]
    if y.dtype == 'object':
        logging.info(f"Converting target variable '{target_col}' to binary (1/0).")
        y = y.apply(lambda x: 1 if x == 'Yes' else 0)
    
    logging.info(f"Training with {len(X.columns)} features: {X.columns.tolist()}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the trained model on the test set and returns a dictionary of performance metrics.

    Args:
        model: The trained scikit-learn model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

    Returns: 
        Dict[str, float]: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    logging.info("Evaluating model performance...")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics

def train_model(model_type: str):
    """
    The main function to orchestrate the model training, evaluation, and logging process.

    Args: model_type (str): The type of model to train (e.g., 'random_forest').
    """
    if model_type not in AVAILABLE_MODELS:
        logging.error(f"Invalid model_type '{model_type}'. Available options are: {list(AVAILABLE_MODELS.keys())}")
        return

    feature_file = os.path.join(FEATURES_PATH, 'customer_churn_20250821_155642.csv')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"logreg_model_{timestamp}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)

        X_train, X_test, y_train, y_test = load_and_split_data(feature_file)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logging.info(f"Starting MLflow Run ID: {run_id}")
            print("\n" + "="*50)
            print(f"  Training Model: {model_type.replace('_', ' ').title()}")
            print("="*50)

            model = AVAILABLE_MODELS[model_type]
            logging.info(f"Training '{model_type}' model...")
            model.fit(X_train, y_train)
            
            metrics = evaluate_model(model, X_test, y_test)

            logging.info("Logging experiment to MLflow...")
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
            logging.info(f"Successfully trained model and logged to MLflow. Weights saved at {model_path}")
            joblib.dump(model, model_path)
            print(f"\nTo view this run, start the MLflow UI with:\nmlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


    except FileNotFoundError as e:
        logging.error(f"Process stopped: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during model training: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a customer Churn prediction model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=list(AVAILABLE_MODELS.keys()),
        help="The type of model to train."
    )
    
    args = parser.parse_args()
    train_model(args.model_type)
