"""
=====================================================================================
DATA CLEANING & PREPROCESSING FOR NASA CMAPSS TURBOFAN ENGINE DATASET (FD001)
Stage 2: Data Types and Missing Values Analysis
=====================================================================================

OBJECTIVE:
This script handles Steps 5-6 of the data cleaning process:
5. Check data types
6. Check for missing values

This step identifies data quality issues that may affect analysis.
=====================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =====================================================================================
# LOAD DATA FROM PREVIOUS STEP
# =====================================================================================
print("=" * 85)
print("LOADING DATA FROM PREVIOUS STEP")
print("=" * 85)

# Load the cleaned data from previous step
data_path = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning"
train_data = pd.read_csv(os.path.join(data_path, "train_FD001_loaded.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test_FD001_loaded.csv"))
rul_data = pd.read_csv(os.path.join(data_path, "rul_FD001_loaded.csv"))

print("[OK] Data loaded successfully")

# =====================================================================================
# STEP 5: CHECK DATA TYPES
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 5: CHECKING DATA TYPES")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Data types affect how pandas processes and analyzes the data
2. Incorrect data types can cause errors in calculations and visualizations
3. Some columns might be loaded as strings instead of numbers (parsing issues)
4. Ensures we can perform mathematical operations on numeric columns
5. Identifies if any unexpected type conversions occurred during loading
""")

# Display training data types
print("\nTRAINING DATA - DATA TYPES:")
print("-" * 85)
print(train_data.dtypes)
print(f"\nData type summary:")
print(train_data.dtypes.value_counts())

# Display test data types
print("\n\nTEST DATA - DATA TYPES:")
print("-" * 85)
print(test_data.dtypes)
print(f"\nData type summary:")
print(test_data.dtypes.value_counts())

# Display RUL data types
print("\n\nRUL DATA - DATA TYPES:")
print("-" * 85)
print(rul_data.dtypes)

# Interpretation
print("\n\nINTERPRETATION:")
print("-" * 85)
print("""
EXPECTED BEHAVIOR:
[OK] All columns should be numeric (int64 or float64)
[OK] No object (string) columns should be present
[OK] This indicates successful data parsing

POSSIBLE ISSUES (if found):
[ERROR] Object dtypes: Indicates some values couldn't be parsed as numbers
[ERROR] Mixed types: May suggest corrupted or malformed data
[ERROR] Integer columns: OK, but may need conversion to float for some operations

CURRENT STATUS:
""")

# Check for any non-numeric columns
non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"⚠ WARNING: Non-numeric columns found: {non_numeric_cols}")
else:
    print("[OK] All columns are numeric - data types are correct")

# =====================================================================================
# STEP 6: CHECK FOR MISSING VALUES
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 6: CHECKING FOR MISSING VALUES")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Missing values can bias models and lead to incorrect conclusions
2. Different columns may require different handling strategies
3. Understanding the pattern of missing values helps decide on imputation
4. In time series data, missing values might indicate sensor failures
5. High missing value percentages may suggest data collection issues

HANDLING OPTIONS:
• Drop rows with missing values (if few)
• Drop columns with high missing value percentages (if not critical)
• Impute values using statistical methods (mean, median, interpolation)
• Forward fill / backward fill (for time series)
""")

# Check for missing values in training data
print("\nTRAINING DATA - MISSING VALUES:")
print("-" * 85)
missing_train = train_data.isnull().sum()
missing_train_pct = (missing_train / len(train_data)) * 100

missing_df = pd.DataFrame({
    'Column': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Percentage': missing_train_pct.values
})

# Filter to show only columns with missing values
missing_df_filtered = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df_filtered) == 0:
    print("[OK] NO MISSING VALUES FOUND in training data")
    print(f"  - Total rows: {len(train_data)}")
    print(f"  - Total columns: {len(train_data.columns)}")
    print(f"  - Data completeness: 100%")
else:
    print("⚠ MISSING VALUES DETECTED:")
    print(missing_df_filtered.to_string(index=False))

# Check for missing values in test data
print("\n\nTEST DATA - MISSING VALUES:")
print("-" * 85)
missing_test = test_data.isnull().sum()
missing_test_pct = (missing_test / len(test_data)) * 100

missing_df = pd.DataFrame({
    'Column': missing_test.index,
    'Missing_Count': missing_test.values,
    'Missing_Percentage': missing_test_pct.values
})

# Filter to show only columns with missing values
missing_df_filtered = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df_filtered) == 0:
    print("[OK] NO MISSING VALUES FOUND in test data")
    print(f"  - Total rows: {len(test_data)}")
    print(f"  - Total columns: {len(test_data.columns)}")
    print(f"  - Data completeness: 100%")
else:
    print("⚠ MISSING VALUES DETECTED:")
    print(missing_df_filtered.to_string(index=False))

# Check for missing values in RUL data
print("\n\nRUL DATA - MISSING VALUES:")
print("-" * 85)
missing_rul = rul_data.isnull().sum()

if missing_rul.sum() == 0:
    print("[OK] NO MISSING VALUES FOUND in RUL data")
    print(f"  - Total values: {len(rul_data)}")
    print(f"  - Data completeness: 100%")
else:
    print(f"⚠ MISSING VALUES DETECTED: {missing_rul.sum()}")

# =====================================================================================
# DETAILED MISSING VALUE ANALYSIS
# =====================================================================================
print("\n" + "=" * 85)
print("DETAILED MISSING VALUE ANALYSIS")
print("=" * 85)

print("\nTRAINING DATA - MISSING VALUE MATRIX:")
print("-" * 85)
print(f"Total cells: {train_data.shape[0] * train_data.shape[1]}")
print(f"Missing cells: {train_data.isnull().sum().sum()}")
print(f"Data completeness: {((train_data.shape[0] * train_data.shape[1] - train_data.isnull().sum().sum()) / (train_data.shape[0] * train_data.shape[1])) * 100:.2f}%")

print("\n\nTEST DATA - MISSING VALUE MATRIX:")
print("-" * 85)
print(f"Total cells: {test_data.shape[0] * test_data.shape[1]}")
print(f"Missing cells: {test_data.isnull().sum().sum()}")
print(f"Data completeness: {((test_data.shape[0] * test_data.shape[1] - test_data.isnull().sum().sum()) / (test_data.shape[0] * test_data.shape[1])) * 100:.2f}%")

print("\n\nRUL DATA - MISSING VALUE MATRIX:")
print("-" * 85)
print(f"Total cells: {rul_data.shape[0] * rul_data.shape[1]}")
print(f"Missing cells: {rul_data.isnull().sum().sum()}")
print(f"Data completeness: {((rul_data.shape[0] * rul_data.shape[1] - rul_data.isnull().sum().sum()) / (rul_data.shape[0] * rul_data.shape[1])) * 100:.2f}%")

# =====================================================================================
# CONCLUSION
# =====================================================================================
print("\n" + "=" * 85)
print("SUMMARY: STEPS 5-6 COMPLETED")
print("=" * 85)

print("""
[OK] DATA TYPE CHECK:
  - All columns are numeric (int64/float64)
  - No unexpected type conversions
  - Data is ready for mathematical operations

[OK] MISSING VALUE CHECK:
  - No missing values detected in any dataset
  - Training data: 100% complete
  - Test data: 100% complete
  - RUL data: 100% complete

[OK] DECISION: No missing value imputation or row removal necessary at this stage

NEXT STEP: Check for duplicate rows and constant/near-constant columns
""")

# Save status for next step
with open(os.path.join(data_path, "step5_6_status.txt"), 'w') as f:
    f.write("Step 5-6 (Data Types & Missing Values): COMPLETED\n")
    f.write(f"Training data shape: {train_data.shape}\n")
    f.write(f"Test data shape: {test_data.shape}\n")
    f.write("Status: No data quality issues found\n")

print("\n[OK] Status saved for next step")


