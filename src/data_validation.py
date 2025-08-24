"""
Data Validation
---------------
Performs basic quality checks on ingested CSV and JSON datasets:
- Missing values
- Duplicate records
- Data types and negative values in numeric columns

Generates an Excel report summarizing findings. If one source is missing,
the report is created for the available source.
"""
import pandas as pd
import os
from datetime import datetime
import glob

from utils.logger import get_logger, PIPELINE_NAMES

# Get logger for this pipeline
logger = get_logger(PIPELINE_NAMES['DATA_VALIDATION'])

# Class: validates raw CSV/JSON data and emits an Excel quality report
class DataValidator:
    """Validator for raw data files to ensure minimum quality before prep."""
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        os.makedirs('reports', exist_ok=True)

    # Validate a CSV file: missing values, dtypes, negatives, duplicates
    def validate_csv_data(self, csv_file):
        """Validate CSV data file"""
        try:
            logger.info(f"Validating CSV file: {csv_file}")
            
            df = pd.read_csv(csv_file)
            
            validation_results = {
                'file_name': os.path.basename(csv_file),
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': {},
                'duplicate_records': 0,
                'data_types': {},
                'negative_values': {}
            }
            
            # Check missing values
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    validation_results['missing_values'][column] = int(missing_count)
            
            # Check duplicates
            validation_results['duplicate_records'] = int(df.duplicated().sum())
            
            # Check data types
            for column in df.columns:
                validation_results['data_types'][column] = str(df[column].dtype)
            
            # Check negative values in numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for column in numeric_columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    validation_results['negative_values'][column] = int(negative_count)
            
            logger.info(f"CSV validation completed: {len(df)} records, {len(df.columns)} columns")
            return validation_results
            
        except Exception as e:
            logger.error(f"CSV validation failed: {str(e)}")
            raise

    # Validate a JSON file (HF rows format supported)
    def validate_json_data(self, json_file):
        """Validate JSON data file"""
        try:
            logger.info(f"Validating JSON file: {json_file}")
            
            import json
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract rows from Hugging Face format
            if 'rows' in data:
                rows = data['rows']
                df = pd.DataFrame([row['row'] for row in rows])
            else:
                df = pd.DataFrame(data)
            
            validation_results = {
                'file_name': os.path.basename(json_file),
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': {},
                'duplicate_records': 0,
                'data_types': {},
                'negative_values': {}
            }
            
            # Check missing values
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    validation_results['missing_values'][column] = int(missing_count)
            
            # Check duplicates
            validation_results['duplicate_records'] = int(df.duplicated().sum())
            
            # Check data types
            for column in df.columns:
                validation_results['data_types'][column] = str(df[column].dtype)
            
            # Check negative values in numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for column in numeric_columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    validation_results['negative_values'][column] = int(negative_count)
            
            logger.info(f"JSON validation completed: {len(df)} records, {len(df.columns)} columns")
            return validation_results
            
        except Exception as e:
            logger.error(f"JSON validation failed: {str(e)}")
            raise

    # Run validation on latest available CSV/JSON under raw path
    def run_validation(self):
        """Run validation on all data files"""
        try:
            logger.info("Starting data validation pipeline...")
            
            # Find latest data files
            csv_files = glob.glob(os.path.join(self.raw_data_path, "customer_churn_*.csv"))
            json_files = glob.glob(os.path.join(self.raw_data_path, "huggingface_churn_*.json"))
            
            if not csv_files and not json_files:
                raise Exception("No data files found for validation")
            
            # Get latest files
            latest_csv = max(csv_files, key=os.path.getctime) if csv_files else None
            latest_json = max(json_files, key=os.path.getctime) if json_files else None
            
            # Validate both files
            csv_results = self.validate_csv_data(latest_csv) if latest_csv else None
            json_results = self.validate_json_data(latest_json) if latest_json else None
            
            # Generate report
            report_path = self.generate_validation_report(csv_results, json_results)
            
            logger.info("Data validation completed successfully")
            return {
                'status': 'success',
                'csv_results': csv_results,
                'json_results': json_results,
                'report_path': report_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data validation pipeline failed: {str(e)}")
            raise

    # Create an Excel report summarizing validation metrics
    def generate_validation_report(self, csv_results, json_results):
        """Generate Excel report with validation results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/data_quality_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                
                # Summary sheet
                sources = []
                if csv_results:
                    sources.append(('CSV File', csv_results))
                if json_results:
                    sources.append(('JSON File', json_results))

                summary_data = {
                    'Data Source': [s[0] for s in sources],
                    'File Name': [s[1]['file_name'] for s in sources],
                    'Total Records': [s[1]['total_records'] for s in sources],
                    'Total Columns': [s[1]['total_columns'] for s in sources],
                    'Missing Values Count': [len(s[1]['missing_values']) for s in sources],
                    'Duplicate Records': [s[1]['duplicate_records'] for s in sources],
                    'Negative Values Count': [len(s[1]['negative_values']) for s in sources]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Missing values sheet
                missing_data = []
                for source, results in [(n, r) for n, r in [('CSV', csv_results), ('JSON', json_results)] if r]:
                    for column, count in results['missing_values'].items():
                        missing_data.append({
                            'Source': source,
                            'Column': column,
                            'Missing Count': count,
                            'Percentage': round((count / results['total_records']) * 100, 2)
                        })
                
                if missing_data:
                    missing_df = pd.DataFrame(missing_data)
                    missing_df.to_excel(writer, sheet_name='Missing Values', index=False)
                
                # Data types sheet
                dtype_data = []
                for source, results in [(n, r) for n, r in [('CSV', csv_results), ('JSON', json_results)] if r]:
                    for column, dtype in results['data_types'].items():
                        dtype_data.append({
                            'Source': source,
                            'Column': column,
                            'Data Type': dtype
                        })
                
                dtype_df = pd.DataFrame(dtype_data)
                dtype_df.to_excel(writer, sheet_name='Data Types', index=False)
                
                # Negative values sheet
                negative_data = []
                for source, results in [(n, r) for n, r in [('CSV', csv_results), ('JSON', json_results)] if r]:
                    for column, count in results['negative_values'].items():
                        negative_data.append({
                            'Source': source,
                            'Column': column,
                            'Negative Count': count
                        })
                
                if negative_data:
                    negative_df = pd.DataFrame(negative_data)
                    negative_df.to_excel(writer, sheet_name='Negative Values', index=False)
            
            logger.info(f"Validation report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

if __name__ == "__main__":
    validator = DataValidator()
    result = validator.run_validation()
    print(f"Validation completed: {result}")