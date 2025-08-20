import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)

class DataQualityValidator:
    def __init__(self):
        self.validation_results = {}

    def validate_data_types(self, df, expected_types):
        """Validate column data types"""
        type_issues = []

        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if expected_type not in actual_type:
                    type_issues.append({
                        'column': column,
                        'expected': expected_type,
                        'actual': actual_type
                    })

        self.validation_results['data_types'] = {
            'passed': len(type_issues) == 0,
            'issues': type_issues
        }

        return len(type_issues) == 0

    def validate_missing_values(self, df, critical_columns):
        """Check for missing values in critical columns"""
        missing_data = []

        for column in critical_columns:
            if column in df.columns:
                missing_count = df[column].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100

                if missing_percentage > 0:
                    missing_data.append({
                        'column': column,
                        'missing_count': int(missing_count),
                        'missing_percentage': round(missing_percentage, 2)
                    })

        self.validation_results['missing_values'] = {
            'passed': len(missing_data) == 0,
            'issues': missing_data
        }

        return len(missing_data) == 0

    def validate_value_ranges(self, df, range_constraints):
        """Validate value ranges for numerical columns"""
        range_issues = []

        for column, constraints in range_constraints.items():
            if column in df.columns:
                min_val, max_val = constraints
                out_of_range = df[
                    (df[column] < min_val) | (df[column] > max_val)
                ][column].count()

                if out_of_range > 0:
                    range_issues.append({
                        'column': column,
                        'expected_range': f"{min_val}-{max_val}",
                        'violations': int(out_of_range)
                    })

        self.validation_results['value_ranges'] = {
            'passed': len(range_issues) == 0,
            'issues': range_issues
        }

        return len(range_issues) == 0

    def validate_duplicates(self, df, unique_columns):
        """Check for duplicate records"""
        duplicate_issues = []

        for column in unique_columns:
            if column in df.columns:
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    duplicate_issues.append({
                        'column': column,
                        'duplicate_count': int(duplicates)
                    })

        total_duplicates = df.duplicated().sum()

        self.validation_results['duplicates'] = {
            'passed': total_duplicates == 0,
            'total_duplicate_rows': int(total_duplicates),
            'column_issues': duplicate_issues
        }

        return total_duplicates == 0

    def validate_categorical_values(self, df, allowed_values):
        """Validate categorical column values"""
        categorical_issues = []

        for column, valid_values in allowed_values.items():
            if column in df.columns:
                invalid_values = df[~df[column].isin(valid_values)][column].unique()
                if len(invalid_values) > 0:
                    categorical_issues.append({
                        'column': column,
                        'invalid_values': invalid_values.tolist(),
                        'valid_values': valid_values
                    })

        self.validation_results['categorical_values'] = {
            'passed': len(categorical_issues) == 0,
            'issues': categorical_issues
        }

        return len(categorical_issues) == 0

    def run_complete_validation(self, df):
        """Run all validation checks for customer churn data"""
        logging.info("Starting data quality validation")

        # Define validation rules for churn dataset
        expected_types = {
            'customerID': 'object',
            'tenure': 'int',
            'MonthlyCharges': 'float',
            'TotalCharges': 'object',  # May contain spaces, needs cleaning
            'Churn': 'object'
        }

        critical_columns = ['customerID', 'Churn']

        range_constraints = {
            'tenure': (0, 100),
            'MonthlyCharges': (0, 200)
        }

        unique_columns = ['customerID']

        allowed_values = {
            'Churn': ['Yes', 'No'],
            'gender': ['Male', 'Female'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No']
        }

        # Run validations
        results = {
            'data_types': self.validate_data_types(df, expected_types),
            'missing_values': self.validate_missing_values(df, critical_columns),
            'value_ranges': self.validate_value_ranges(df, range_constraints),
            'duplicates': self.validate_duplicates(df, unique_columns),
            'categorical_values': self.validate_categorical_values(df, allowed_values)
        }

        overall_passed = all(results.values())

        self.validation_results['overall'] = {
            'passed': overall_passed,
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'total_columns': len(df.columns)
        }

        logging.info(f"Validation completed. Overall passed: {overall_passed}")
        return self.validation_results

    def generate_quality_report(self, output_path):
        """Generate comprehensive data quality report"""
        report_path = f"{output_path}/data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        logging.info(f"Quality report generated: {report_path}")
        return report_path

if __name__ == "__main__":
    # Example usage
    validator = DataQualityValidator()

    # Create sample data for testing
    sample_data = {
        'customerID': ['001', '002', '003'],
        'tenure': [12, 24, 36],
        'MonthlyCharges': [50.0, 75.5, 89.99],
        'Churn': ['No', 'Yes', 'No']
    }
    df = pd.DataFrame(sample_data)

    results = validator.run_complete_validation(df)
    print("Validation completed")
