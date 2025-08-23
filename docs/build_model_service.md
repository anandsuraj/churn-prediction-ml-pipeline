# Model Building Pipeline

##  Purpose

The `build_model.py` script is responsible for training a customer churn prediction model using scikit-learn. It handles data loading, splitting, model training, evaluation, and logging using MLflow.

## Key Features

- Supports two models: `Random Forest` and `Logistic Regression`.
- Loads and cleans customer churn data from a CSV file.
- Splits data into training and testing sets.
- Trains the selected model and evaluates its performance.
- Logs model parameters, metrics, and artifacts to MLflow.
- Saves the trained model to a local directory.

## Usage

```bash
python build_model.py --model_type model_type
```

## Arguments

`--model_type`: Specifies the type of model to train (e.g., 'random_forest').

## Output

- Trained model files are saved in the models directory.
- MLflow experiment runs are logged with performance metrics and model artifacts.

## Technical Details

The script uses the following scikit-learn modules:
- train_test_split
- RandomForestClassifier
- LogisticRegression
- accuracy_score
- precision_score
- recall_score
- f1_score

MLflow is used to track experiments, log metrics, and save models.