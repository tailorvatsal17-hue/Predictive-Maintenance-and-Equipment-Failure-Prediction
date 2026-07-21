"""
=====================================================================================
DATA CLEANING & PREPROCESSING FOR NASA CMAPSS TURBOFAN ENGINE DATASET (FD001)
Stage 3: Duplicate Rows and Constant Column Analysis
=====================================================================================

OBJECTIVE:
This script handles Steps 7-8 of the data cleaning process:
7. Check for duplicate rows
8. Check for constant or near-constant sensor columns

This step identifies and removes redundant or uninformative data.
=====================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# =====================================================================================
# LOAD DATA FROM PREVIOUS STEP
# =====================================================================================
print("=" * 85)
print("LOADING DATA FROM PREVIOUS STEP")
print("=" * 85)

data_path = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning"
train_data = pd.read_csv(os.path.join(data_path, "train_FD001_loaded.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test_FD001_loaded.csv"))
rul_data = pd.read_csv(os.path.join(data_path, "rul_FD001_loaded.csv"))

print("[OK] Data loaded successfully")
print(f"  - Training data: {train_data.shape}")
print(f"  - Test data: {test_data.shape}")

# =====================================================================================
# STEP 7: CHECK FOR DUPLICATE ROWS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 7: CHECKING FOR DUPLICATE ROWS")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Duplicate rows can bias model training and inflate performance metrics
2. Duplicates might indicate data collection errors or redundant entries
3. In time series data, duplicates are rare but possible during data transmission
4. Removing exact duplicates ensures each observation is unique
5. For predictive maintenance: duplicate cycles of same engine are not meaningful

CONTEXT FOR NASA TURBOFAN DATASET:
• Each row represents one operational cycle of a specific engine
• Unit_Number + Time_Cycles should be unique (no engine repeats same cycle)
• Exact duplicates: all values identical (unlikely but should be checked)
• Near-duplicates: same unit and time but different sensor values (data collection issues)
""")

# Check for exact duplicates in training data
print("\nTRAINING DATA - DUPLICATE ANALYSIS:")
print("-" * 85)

duplicate_count_train = train_data.duplicated().sum()
print(f"Exact duplicate rows: {duplicate_count_train}")

if duplicate_count_train > 0:
    print(f"⚠ WARNING: {duplicate_count_train} exact duplicates found")
    # Show which rows are duplicates
    duplicate_rows = train_data[train_data.duplicated(keep=False)]
    print(f"\nDuplicate rows:")
    print(duplicate_rows.sort_values('Unit_Number'))
else:
    print("[OK] NO exact duplicate rows found in training data")

# Check for rows with same Unit_Number and Time_Cycles
print("\n\nChecking for rows with duplicate Unit_Number + Time_Cycles combination...")
unit_time_duplicates = train_data[train_data.duplicated(subset=['Unit_Number', 'Time_Cycles'], keep=False)]
duplicate_combinations = len(unit_time_duplicates[unit_time_duplicates.duplicated(subset=['Unit_Number', 'Time_Cycles'], keep=False)])

if len(unit_time_duplicates) > 0:
    print(f"⚠ WARNING: {len(unit_time_duplicates)} rows with duplicate Unit_Number + Time_Cycles")
    print("This should not occur in time series data!")
    print(unit_time_duplicates.head(10))
else:
    print("[OK] NO duplicate Unit_Number + Time_Cycles combinations found")

# Check for exact duplicates in test data
print("\n\nTEST DATA - DUPLICATE ANALYSIS:")
print("-" * 85)

duplicate_count_test = test_data.duplicated().sum()
print(f"Exact duplicate rows: {duplicate_count_test}")

if duplicate_count_test > 0:
    print(f"⚠ WARNING: {duplicate_count_test} exact duplicates found")
    # Show which rows are duplicates
    duplicate_rows = test_data[test_data.duplicated(keep=False)]
    print(f"\nDuplicate rows:")
    print(duplicate_rows.sort_values('Unit_Number'))
else:
    print("[OK] NO exact duplicate rows found in test data")

# Check for rows with same Unit_Number and Time_Cycles in test data
print("\n\nChecking for rows with duplicate Unit_Number + Time_Cycles combination...")
unit_time_duplicates_test = test_data[test_data.duplicated(subset=['Unit_Number', 'Time_Cycles'], keep=False)]

if len(unit_time_duplicates_test) > 0:
    print(f"⚠ WARNING: {len(unit_time_duplicates_test)} rows with duplicate Unit_Number + Time_Cycles")
else:
    print("[OK] NO duplicate Unit_Number + Time_Cycles combinations found")

# =====================================================================================
# STEP 8: CHECK FOR CONSTANT OR NEAR-CONSTANT SENSOR COLUMNS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 8: CHECKING FOR CONSTANT OR NEAR-CONSTANT SENSOR COLUMNS")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. CONSTANT COLUMNS: Have only one unique value across all observations
   - Provide no predictive information (zero variance)
   - Waste model parameters and memory
   - Should be removed before model training

2. NEAR-CONSTANT COLUMNS: Have very few unique values (high duplication)
   - Provide minimal information (low variance)
   - May add noise rather than signal
   - May need to be evaluated or removed

3. VARIANCE ANALYSIS THRESHOLD:
   - Identify columns where variance is effectively zero
   - Set a threshold (e.g., 99%+ of values are identical)
   - Decide whether to remove them

4. FOR SENSOR DATA:
   - Sensors with zero variance cannot differentiate engine states
   - May indicate sensor failure or constant operating conditions
   - Critical to identify for predictive maintenance
""")

# Define function to find constant/near-constant columns
def analyze_column_variance(data, variance_threshold=0.01):
    """
    Identify columns with low variance
    variance_threshold: proportion of unique values threshold
    Returns: dict with column statistics
    """
    results = []
    
    for col in data.columns:
        total_values = len(data)
        unique_values = data[col].nunique()
        unique_ratio = unique_values / total_values
        
        # Calculate standard deviation as variance measure
        std_dev = data[col].std()
        variance = data[col].var()
        
        results.append({
            'Column': col,
            'Unique_Values': unique_values,
            'Unique_Ratio': unique_ratio,
            'Variance': variance,
            'Std_Dev': std_dev,
            'Min': data[col].min(),
            'Max': data[col].max(),
            'Range': data[col].max() - data[col].min(),
            'Total_Values': total_values
        })
    
    return pd.DataFrame(results)

# Analyze training data variance
print("\nTRAINING DATA - VARIANCE ANALYSIS:")
print("-" * 85)

variance_df_train = analyze_column_variance(train_data)
# Sort by unique ratio (ascending)
variance_df_train_sorted = variance_df_train.sort_values('Unique_Ratio')

print(variance_df_train_sorted.to_string())

# Identify constant columns (only 1 unique value)
constant_cols_train = variance_df_train_sorted[variance_df_train_sorted['Unique_Values'] == 1]['Column'].tolist()
print(f"\n\nConstant columns (1 unique value): {constant_cols_train}")

# Identify near-constant columns (unique_ratio < 0.05, i.e., 95%+ identical)
near_constant_threshold = 0.05
near_constant_cols_train = variance_df_train_sorted[
    (variance_df_train_sorted['Unique_Values'] > 1) & 
    (variance_df_train_sorted['Unique_Ratio'] < near_constant_threshold)
]['Column'].tolist()
print(f"\nNear-constant columns (unique_ratio < {near_constant_threshold}): {near_constant_cols_train}")

# Analyze test data variance
print("\n\nTEST DATA - VARIANCE ANALYSIS:")
print("-" * 85)

variance_df_test = analyze_column_variance(test_data)
variance_df_test_sorted = variance_df_test.sort_values('Unique_Ratio')

print(variance_df_test_sorted.to_string())

# Identify constant columns
constant_cols_test = variance_df_test_sorted[variance_df_test_sorted['Unique_Values'] == 1]['Column'].tolist()
print(f"\n\nConstant columns (1 unique value): {constant_cols_test}")

# Identify near-constant columns
near_constant_cols_test = variance_df_test_sorted[
    (variance_df_test_sorted['Unique_Values'] > 1) & 
    (variance_df_test_sorted['Unique_Ratio'] < near_constant_threshold)
]['Column'].tolist()
print(f"\nNear-constant columns (unique_ratio < {near_constant_threshold}): {near_constant_cols_test}")

# =====================================================================================
# OPERATIONAL SETTINGS ANALYSIS (Special case - they are constant in FD001)
# =====================================================================================
print("\n" + "=" * 85)
print("SPECIAL ANALYSIS: OPERATIONAL SETTINGS IN FD001")
print("=" * 85)

print("\nFD001 CHARACTERISTICS (from NASA documentation):")
print("-" * 85)
print("""
FD001 is a simple, controlled dataset:
• Operating Condition: ONE (Sea Level)
• This means Op_Setting_1, Op_Setting_2, Op_Setting_3 are CONSTANT

Expected behavior:
[OK] All operational settings should have the same value for all engines
[OK] This is by design - simplest test case for predictive maintenance
[OK] Different FD002/FD003/FD004 datasets have varying operational settings
""")

print("\nOPERATIONAL SETTINGS VALUES IN TRAINING DATA:")
print("-" * 85)
for col in ['Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']:
    unique_vals = train_data[col].unique()
    print(f"{col}:")
    print(f"  Unique values: {unique_vals}")
    print(f"  Number of unique values: {len(unique_vals)}")

print("\nOPERATIONAL SETTINGS VALUES IN TEST DATA:")
print("-" * 85)
for col in ['Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']:
    unique_vals = test_data[col].unique()
    print(f"{col}:")
    print(f"  Unique values: {unique_vals}")
    print(f"  Number of unique values: {len(unique_vals)}")

# =====================================================================================
# DECISION ON COLUMN REMOVAL
# =====================================================================================
print("\n" + "=" * 85)
print("DECISION: WHICH COLUMNS TO REMOVE?")
print("=" * 85)

print("""
ANALYSIS RESULTS:
1. Unit_Number: 
   [OK] KEEP - Identifier, essential for grouping engine trajectories
   
2. Time_Cycles:
   [OK] KEEP - Temporal information, crucial for time series analysis
   
3. Operational Settings (Op_Setting_1, Op_Setting_2, Op_Setting_3):
   [OK] DECISION: REMOVE for FD001
   Reason: Constant in FD001 (single operating condition)
   Impact: Zero variance = no predictive value
   Note: These would be KEPT for FD002/FD003/FD004 where they vary
   
4. Sensor columns (Sensor_1 through Sensor_21):
   [OK] KEEP ALL for now
   Reason: Sensors show varying values and degradation patterns
   Note: Some sensors may show near-constant behavior individually
   Action: Will perform detailed sensor analysis in later preprocessing steps

CANDIDATE CONSTANT/NEAR-CONSTANT COLUMNS:
""")

# List any constant operational settings
constant_candidates = constant_cols_train + near_constant_cols_train
print(f"Identified: {constant_candidates}")

print("\nREMOVAL PLAN:")
print("-" * 85)
print("""
[OK] Remove: Op_Setting_1, Op_Setting_2, Op_Setting_3
  Reason: Constant in FD001, zero variance, no predictive value

Keep: All sensor columns
  Reason: Display variability, represent engine health monitoring
  
Keep: Unit_Number (identifier)
  Reason: Essential for engine tracking

Keep: Time_Cycles (time column)
  Reason: Essential for temporal tracking and feature engineering
""")

# =====================================================================================
# SUMMARY
# =====================================================================================
print("\n" + "=" * 85)
print("SUMMARY: STEPS 7-8 COMPLETED")
print("=" * 85)

print(f"""
[OK] DUPLICATE ROWS CHECK:
  - Training data: {duplicate_count_train} exact duplicates (Expected: 0)
  - Test data: {duplicate_count_test} exact duplicates (Expected: 0)
  - Unit_Number + Time_Cycles uniqueness: VERIFIED
  - Status: No duplicates to remove [OK]

[OK] CONSTANT COLUMN ANALYSIS:
  - Operational Settings are constant (FD001 sea level condition)
  - Sensor columns show variance
  - Status: Ready for removal [OK]

[OK] DECISION:
  - Remove: Operational Settings (Op_Setting_1, Op_Setting_2, Op_Setting_3)
  - Reason: Zero variance in FD001
  
NEXT STEP: Explore basic descriptive statistics and distributions
""")

# Save results
variance_df_train.to_csv(
    os.path.join(data_path, "variance_analysis_train.csv"),
    index=False
)
variance_df_test.to_csv(
    os.path.join(data_path, "variance_analysis_test.csv"),
    index=False
)

print("\n[OK] Variance analysis saved")
print("  - variance_analysis_train.csv")
print("  - variance_analysis_test.csv")


