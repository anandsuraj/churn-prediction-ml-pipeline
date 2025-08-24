#!/usr/bin/env python3
"""
Test Data Preparation
====================

Test the data preparation module to identify segmentation fault issues.
"""

import os
import sys
import traceback

# Set fault handler for better debugging
import faulthandler
faulthandler.enable()

# Set environment variables
os.environ['PYTHONFAULTHANDLER'] = 'true'
os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive matplotlib backend

# Add src to path
sys.path.append('src')

def test_imports():
    """Test importing all modules used in data preparation"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except Exception as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except Exception as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except Exception as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn imported successfully")
    except Exception as e:
        print(f"❌ seaborn import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import StandardScaler
        print("✅ sklearn imported successfully")
    except Exception as e:
        print(f"❌ sklearn import failed: {e}")
        return False
    
    return True

def test_data_preparation_import():
    """Test importing the data preparation module"""
    print("\n🧪 Testing data preparation import...")
    
    try:
        from data_preparation import DataPreparationPipeline
        print("✅ DataPreparationPipeline imported successfully")
        return True
    except Exception as e:
        print(f"❌ DataPreparationPipeline import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without heavy operations"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from data_preparation import DataPreparationPipeline
        
        # Create instance
        pipeline = DataPreparationPipeline()
        print("✅ Pipeline instance created successfully")
        
        # Test with small sample data
        import pandas as pd
        sample_data = pd.DataFrame({
            'tenure': [1, 2, 3, 4, 5],
            'MonthlyCharges': [20.0, 30.0, 40.0, 50.0, 60.0],
            'TotalCharges': ['20.0', '60.0', '120.0', '200.0', '300.0'],
            'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
        })
        
        print("✅ Sample data created successfully")
        
        # Test missing value handling
        cleaned_data = pipeline.handle_missing_values(sample_data)
        print("✅ Missing value handling completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Data Preparation Segmentation Fault Debug")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Check your environment.")
        return False
    
    # Test data preparation import
    dp_import_ok = test_data_preparation_import()
    
    if not dp_import_ok:
        print("\n❌ Data preparation import failed.")
        return False
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    if not basic_ok:
        print("\n❌ Basic functionality test failed.")
        return False
    
    print("\n🎉 All tests passed! Data preparation should work in Airflow.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)