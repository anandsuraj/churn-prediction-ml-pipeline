import pandas as pd
import os
import logging
from datetime import datetime
import glob

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()
    ]
)

class DataValidator:
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        os.makedirs('reports', exist_ok=True)

    def validate_csv_data(self, csv_file):
        """Validate CSV data file"""
        try:
            logging.info(f"Validating CSV file: {csv_file}")
            
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
            
            logging.info(f"CSV validation completed: {len(df)} records, {len(df.columns)} columns")
            return validation_results
            
        except Exception as e:
            logging.error(f"CSV validation failed: {str(e)}")
            raise

    def validate_json_data(self, json_file):
        """Validate JSON data file"""
        try:
            logging.info(f"Validating JSON file: {json_file}")
            
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
            
            logging.info(f"JSON validation completed: {len(df)} records, {len(df.columns)} columns")
            return validation_results
            
        except Exception as e:
            logging.error(f"JSON validation failed: {str(e)}")
            raise

    def run_validation(self):
        """Run validation on all data files"""
        try:
            logging.info("Starting data validation pipeline...")
            
            # Find latest data files
            csv_files = glob.glob(os.path.join(self.raw_data_path, "customer_churn_*.csv"))
            json_files = glob.glob(os.path.join(self.raw_data_path, "huggingface_churn_*.json"))
            
            if not csv_files:
                raise Exception("No CSV files found for validation")
            if not json_files:
                raise Exception("No JSON files found for validation")
            
            # Get latest files
            latest_csv = max(csv_files, key=os.path.getctime)
            latest_json = max(json_files, key=os.path.getctime)
            
            # Validate both files
            csv_results = self.validate_csv_data(latest_csv)
            json_results = self.validate_json_data(latest_json)
            
            # Generate report
            report_path = self.generate_validation_report(csv_results, json_results)
            
            logging.info("Data validation completed successfully")
            return {
                'status': 'success',
                'csv_results': csv_results,
                'json_results': json_results,
                'report_path': report_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Data validation pipeline failed: {str(e)}")
            raise

    def generate_validation_report(self, csv_results, json_results):
        """Generate Excel report with validation results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/data_quality_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = {
                    'Data Source': ['CSV File', 'JSON File'],
                    'File Name': [csv_results['file_name'], json_results['file_name']],
                    'Total Records': [csv_results['total_records'], json_results['total_records']],
                    'Total Columns': [csv_results['total_columns'], json_results['total_columns']],
                    'Missing Values Count': [len(csv_results['missing_values']), len(json_results['missing_values'])],
                    'Duplicate Records': [csv_results['duplicate_records'], json_results['duplicate_records']],
                    'Negative Values Count': [len(csv_results['negative_values']), len(json_results['negative_values'])]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Missing values sheet
                missing_data = []
                for source, results in [('CSV', csv_results), ('JSON', json_results)]:
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
                for source, results in [('CSV', csv_results), ('JSON', json_results)]:
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
                for source, results in [('CSV', csv_results), ('JSON', json_results)]:
                    for column, count in results['negative_values'].items():
                        negative_data.append({
                            'Source': source,
                            'Column': column,
                            'Negative Count': count
                        })
                
                if negative_data:
                    negative_df = pd.DataFrame(negative_data)
                    negative_df.to_excel(writer, sheet_name='Negative Values', index=False)
            
            logging.info(f"Validation report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Report generation failed: {str(e)}")
            raise

if __name__ == "__main__":
    validator = DataValidator()
    result = validator.run_validation()
    print(f"Validation completed: {result}")