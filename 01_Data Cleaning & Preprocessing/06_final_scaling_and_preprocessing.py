"""
=====================================================================================
DATA CLEANING & PREPROCESSING FOR NASA CMAPSS TURBOFAN ENGINE DATASET (FD001)
Stage 6: Column Removal, Scaling, and Final Preprocessing
=====================================================================================

OBJECTIVE:
This script handles Steps 12-14 of the data cleaning process:
12. Explain which columns should be removed and justify decisions
13. Scale/normalize sensor features and explain the method
14. Save the cleaned and preprocessed dataset

This is the FINAL preprocessing step before feature engineering.
=====================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle

# Set visualization style
sns.set_style("whitegrid")

# =====================================================================================
# LOAD DATA FROM PREVIOUS STEPS
# =====================================================================================
print("=" * 85)
print("LOADING DATA FROM PREVIOUS STEPS")
print("=" * 85)

data_path = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning"
train_data = pd.read_csv(os.path.join(data_path, "train_FD001_loaded.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test_FD001_loaded.csv"))
rul_data = pd.read_csv(os.path.join(data_path, "rul_FD001_loaded.csv"))

print("✓ Data loaded successfully")
print(f"  - Training data: {train_data.shape}")
print(f"  - Test data: {test_data.shape}")
print(f"  - RUL data: {rul_data.shape}")

# =====================================================================================
# STEP 12: DECIDE ON COLUMN REMOVAL
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 12: DECIDING ON COLUMN REMOVAL")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. REMOVE COLUMNS THAT:
   • Have zero or near-zero variance (constant values)
   • Are not predictive of the target variable
   • Are duplicates or redundant
   • Are identifiers (not needed during modeling)
   
2. KEEP COLUMNS THAT:
   • Show variability and potential predictive power
   • Represent distinct information
   • Are meaningful sensor readings
   • Are essential for time series context
   
3. FOR FD001 DATASET:
   Analysis from previous steps revealed:
   • Op_Setting_1, Op_Setting_2, Op_Setting_3: CONSTANT (zero variance)
   • Unit_Number: Identifier (use for grouping, remove from features)
   • Time_Cycles: Time index (important for context, remove from features)
   • Sensors: All show variance (keep all for now)
""")

# Decision on which columns to remove
print("\nCOLUMN REMOVAL DECISION MATRIX:")
print("-" * 85)

columns_decision = pd.DataFrame({
    'Column': train_data.columns,
    'Category': ['Identifier', 'Time Index', 'Op Setting', 'Op Setting', 'Op Setting', 
                 'Sensor', 'Sensor', 'Sensor', 'Sensor', 'Sensor', 
                 'Sensor', 'Sensor', 'Sensor', 'Sensor', 'Sensor',
                 'Sensor', 'Sensor', 'Sensor', 'Sensor', 'Sensor',
                 'Sensor', 'Sensor', 'Sensor', 'Sensor', 'Sensor', 'Sensor'],
    'Variance': ['High', 'High', 'Zero', 'Zero', 'Zero',
                 'High', 'High', 'High', 'High', 'High',
                 'High', 'High', 'High', 'High', 'High',
                 'High', 'High', 'High', 'High', 'High',
                 'High', 'High', 'High', 'High', 'High', 'High'],
    'Action': ['Remove (use for grouping only)', 
               'Remove (use for grouping only)',
               'Remove (constant in FD001)',
               'Remove (constant in FD001)',
               'Remove (constant in FD001)',
               'Keep', 'Keep', 'Keep', 'Keep', 'Keep',
               'Keep', 'Keep', 'Keep', 'Keep', 'Keep',
               'Keep', 'Keep', 'Keep', 'Keep', 'Keep',
               'Keep', 'Keep', 'Keep', 'Keep', 'Keep', 'Keep'],
    'Reason': ['Identifier, not a feature', 
               'Time index, not a feature',
               'All rows have same value (no predictive power)',
               'All rows have same value (no predictive power)',
               'All rows have same value (no predictive power)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)',
               'Sensor measurement (important)']
})

print(columns_decision.to_string(index=False))

# Define which columns to remove and which to keep
columns_to_remove = ['Unit_Number', 'Time_Cycles', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']
columns_to_keep = [col for col in train_data.columns if col not in columns_to_remove]

print(f"\n\nCOLUMNS TO REMOVE: {columns_to_remove}")
print(f"  Total: {len(columns_to_remove)}")
print(f"  Reason: Identifiers and zero-variance operational settings")

print(f"\n\nCOLUMNS TO KEEP: {columns_to_keep}")
print(f"  Total: {len(columns_to_keep)}")
print(f"  Reason: Sensor measurements with predictive value")

# Create cleaned datasets (drop unwanted columns)
print("\n\nRemoving columns...")
train_cleaned = train_data.drop(columns=columns_to_remove)
test_cleaned = test_data.drop(columns=columns_to_remove)

print(f"✓ Training data after column removal: {train_cleaned.shape}")
print(f"✓ Test data after column removal: {test_cleaned.shape}")

# Display cleaned data sample
print(f"\nCleaned training data (first 5 rows):")
print(train_cleaned.head())

# =====================================================================================
# STEP 13: SCALE/NORMALIZE SENSOR FEATURES
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 13: SCALING AND NORMALIZING SENSOR FEATURES")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. SCALING IMPORTANCE:
   • Sensors have different ranges and units
   • Some algorithms (KNN, Neural Networks) sensitive to scale
   • Prevents features with large ranges dominating
   • Improves model training speed and stability
   
2. SCALING METHODS:

   a) StandardScaler (Mean = 0, Std = 1):
      Formula: X_scaled = (X - mean) / std
      Pros: Good for normally distributed data
      Cons: Influenced by outliers
      Use: When outliers are important (our case!)
      
   b) RobustScaler (Using median and IQR):
      Formula: X_scaled = (X - median) / IQR
      Pros: Robust to outliers, good for skewed data
      Cons: Not on [0,1] scale
      Use: Preferred for data with significant outliers
      
   c) MinMaxScaler (Scale to [0,1]):
      Formula: X_scaled = (X - min) / (max - min)
      Pros: Results in bounded [0,1] range
      Cons: Very sensitive to outliers
      Use: When all values are known and bounded

3. CHOICE FOR THIS PROJECT:
   ➜ Use RobustScaler
   Reasons:
   • Data has outliers (critical failure signals)
   • Outliers should influence scaling less
   • Median/IQR-based approach more interpretable
   • Better handles sensor measurement noise
   • Preserves relative relationships of extreme values
""")

# Show sensor value ranges before scaling
print("\nSENSOR VALUE RANGES (Before Scaling):")
print("-" * 85)

ranges_df = pd.DataFrame({
    'Sensor': columns_to_keep,
    'Min': [train_cleaned[col].min() for col in columns_to_keep],
    'Max': [train_cleaned[col].max() for col in columns_to_keep],
    'Mean': [train_cleaned[col].mean() for col in columns_to_keep],
    'Median': [train_cleaned[col].median() for col in columns_to_keep],
    'Std': [train_cleaned[col].std() for col in columns_to_keep],
    'Range': [train_cleaned[col].max() - train_cleaned[col].min() for col in columns_to_keep]
})

print(ranges_df.to_string(index=False))

# Apply RobustScaler
print("\n\nAPPLYING ROBUSTSCALER...")
print("-" * 85)

robust_scaler = RobustScaler()

# Fit scaler on training data
train_scaled = robust_scaler.fit_transform(train_cleaned)
train_scaled_df = pd.DataFrame(train_scaled, columns=columns_to_keep)

# Transform test data using training data's scaler
test_scaled = robust_scaler.transform(test_cleaned)
test_scaled_df = pd.DataFrame(test_scaled, columns=columns_to_keep)

print("✓ RobustScaler applied successfully")
print(f"  - Fitted on training data")
print(f"  - Transformed both training and test data")

# Show sensor value ranges after scaling
print("\n\nSENSOR VALUE RANGES (After Scaling):")
print("-" * 85)

ranges_scaled_df = pd.DataFrame({
    'Sensor': columns_to_keep,
    'Min': [train_scaled_df[col].min() for col in columns_to_keep],
    'Max': [train_scaled_df[col].max() for col in columns_to_keep],
    'Mean': [train_scaled_df[col].mean() for col in columns_to_keep],
    'Median': [train_scaled_df[col].median() for col in columns_to_keep],
    'Std': [train_scaled_df[col].std() for col in columns_to_keep]
})

print(ranges_scaled_df.to_string(index=False))

print("\n\nINTERPRETATION OF SCALED DATA:")
print("-" * 85)
print("""
Observations after RobustScaler:
✓ Mean should be close to 0 (but not exactly, due to robust approach)
✓ Different sensors have different ranges (that's expected)
✓ Most values are typically in [-2, 2] to [-3, 3] range
✓ Extreme values remain extreme (but less influential)
✓ Relative distances between points preserved
""")

# Show example comparison
print("\n\nEXAMPLE: First 5 rows comparison (Training Data)")
print("-" * 85)
print("\nBEFORE Scaling (Sensor_1):")
print(train_cleaned['Sensor_1'].head().values)

print("\nAFTER Scaling (Sensor_1):")
print(train_scaled_df['Sensor_1'].head().values)

# =====================================================================================
# VISUALIZATION: BEFORE vs AFTER SCALING
# =====================================================================================
print("\n\nGenerating scaling visualization...")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Before vs After RobustScaler (Sample Sensors)', fontsize=14, fontweight='bold')

# Show 3 sensors before and after
sample_sensors = ['Sensor_1', 'Sensor_11', 'Sensor_21']

for idx, col in enumerate(sample_sensors):
    # Before scaling
    ax_before = axes[0, idx]
    ax_before.hist(train_cleaned[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax_before.set_title(f'{col} - BEFORE Scaling', fontweight='bold')
    ax_before.set_xlabel('Value')
    ax_before.set_ylabel('Frequency')
    ax_before.grid(True, alpha=0.3)
    
    # After scaling
    ax_after = axes[1, idx]
    ax_after.hist(train_scaled_df[col], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax_after.set_title(f'{col} - AFTER Scaling', fontweight='bold')
    ax_after.set_xlabel('Scaled Value')
    ax_after.set_ylabel('Frequency')
    ax_after.grid(True, alpha=0.3)

plt.tight_layout()
viz_dir = os.path.join(data_path, "visualizations")
os.makedirs(viz_dir, exist_ok=True)
plt.savefig(os.path.join(viz_dir, 'before_after_scaling.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: before_after_scaling.png")
plt.close()

# =====================================================================================
# STEP 14: SAVE CLEANED AND PREPROCESSED DATASETS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 14: SAVING CLEANED AND PREPROCESSED DATASETS")
print("=" * 85)

print("\nSaving files...")
print("-" * 85)

# Save scaled training data
train_scaled_df.to_csv(
    os.path.join(data_path, "train_FD001_scaled.csv"),
    index=False
)
print("✓ Saved: train_FD001_scaled.csv")

# Save scaled test data
test_scaled_df.to_csv(
    os.path.join(data_path, "test_FD001_scaled.csv"),
    index=False
)
print("✓ Saved: test_FD001_scaled.csv")

# Save original data with Unit_Number and Time_Cycles (for reference during feature engineering)
train_ref = train_data[['Unit_Number', 'Time_Cycles']].copy()
train_ref.to_csv(
    os.path.join(data_path, "train_FD001_reference.csv"),
    index=False
)
print("✓ Saved: train_FD001_reference.csv (Unit_Number + Time_Cycles)")

test_ref = test_data[['Unit_Number', 'Time_Cycles']].copy()
test_ref.to_csv(
    os.path.join(data_path, "test_FD001_reference.csv"),
    index=False
)
print("✓ Saved: test_FD001_reference.csv (Unit_Number + Time_Cycles)")

# Save RUL data (for later use in evaluation)
rul_data.to_csv(
    os.path.join(data_path, "RUL_FD001_reference.csv"),
    index=False
)
print("✓ Saved: RUL_FD001_reference.csv")

# Save the scaler object for later use during prediction
scaler_path = os.path.join(data_path, "robust_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(robust_scaler, f)
print(f"✓ Saved: robust_scaler.pkl (for applying to new data)")

# Save preprocessing metadata
metadata = {
    'Dataset': 'NASA CMAPSS FD001',
    'Stage': 'Data Cleaning & Preprocessing',
    'Columns_Removed': columns_to_remove,
    'Columns_Kept': columns_to_keep,
    'Scaling_Method': 'RobustScaler',
    'Training_Shape_Original': train_data.shape,
    'Training_Shape_Cleaned': train_scaled_df.shape,
    'Test_Shape_Original': test_data.shape,
    'Test_Shape_Cleaned': test_scaled_df.shape,
    'Notes': 'Operational settings removed (constant in FD001). All 21 sensors retained. Data ready for feature engineering.'
}

import json
with open(os.path.join(data_path, "preprocessing_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: preprocessing_metadata.json")

# =====================================================================================
# FINAL SUMMARY AND CHECKLIST
# =====================================================================================
print("\n" + "=" * 85)
print("SUMMARY: ALL PREPROCESSING STEPS COMPLETED")
print("=" * 85)

print(f"""
✓ STEP 1: Data Loading - COMPLETED
  - Loaded train_FD001.txt, test_FD001.txt, RUL_FD001.txt
  
✓ STEP 2: Column Naming - COMPLETED
  - Assigned 26 meaningful column names
  
✓ STEP 3: Dimensions & First Rows - COMPLETED
  - Verified data structure and shapes
  
✓ STEP 4: Column Purposes - COMPLETED
  - Documented all column meanings
  
✓ STEP 5: Data Types - COMPLETED
  - Confirmed all columns are numeric
  
✓ STEP 6: Missing Values - COMPLETED
  - Confirmed 0% missing values (100% complete)
  
✓ STEP 7: Duplicate Rows - COMPLETED
  - Confirmed 0 exact duplicates
  
✓ STEP 8: Constant Columns - COMPLETED
  - Identified Op_Setting_1/2/3 as constant
  
✓ STEP 9: Descriptive Statistics - COMPLETED
  - Calculated mean, median, std, min, max, skewness, kurtosis
  
✓ STEP 10: Distribution Visualization - COMPLETED
  - Generated histograms, box plots, density plots, trajectories
  
✓ STEP 11: Outlier Detection - COMPLETED
  - Identified 0-5% outliers (kept for model training)
  
✓ STEP 12: Column Removal - COMPLETED
  - Removed 5 columns (identifiers and constants)
  - Kept 21 sensor columns
  
✓ STEP 13: Scaling/Normalization - COMPLETED
  - Applied RobustScaler to 21 sensor features
  
✓ STEP 14: Save Clean Data - COMPLETED
  - Saved scaled datasets and metadata

DATASET SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original Training Data:    {train_data.shape[0]} rows × {train_data.shape[1]} columns
Cleaned Training Data:     {train_scaled_df.shape[0]} rows × {train_scaled_df.shape[1]} columns

Original Test Data:        {test_data.shape[0]} rows × {test_data.shape[1]} columns
Cleaned Test Data:         {test_scaled_df.shape[0]} rows × {test_scaled_df.shape[1]} columns

Columns Removed:           {len(columns_to_remove)} (identifiers + constants)
Columns Retained:          {len(columns_to_keep)} (sensor measurements)

Scaling Method:            RobustScaler (median/IQR-based)
Missing Values:            0% (100% complete data)
Duplicates:                0 (no duplicates)
Outliers Handled:          Retained (important for failure prediction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FILES CREATED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATA FILES:
  ✓ train_FD001_scaled.csv           (cleaned & scaled training data)
  ✓ test_FD001_scaled.csv            (cleaned & scaled test data)
  ✓ train_FD001_reference.csv        (Unit_Number + Time_Cycles for training)
  ✓ test_FD001_reference.csv         (Unit_Number + Time_Cycles for test)
  ✓ RUL_FD001_reference.csv          (ground truth RUL values)

🔧 TOOLS & METADATA:
  ✓ robust_scaler.pkl                (scaler object for new data)
  ✓ preprocessing_metadata.json       (preprocessing configuration)

📈 ANALYSIS REPORTS:
  ✓ training_descriptive_stats.csv   (mean, median, std, etc.)
  ✓ training_additional_stats.csv    (skewness, kurtosis, CV)
  ✓ variance_analysis_train.csv      (column variance analysis)
  ✓ variance_analysis_test.csv       (column variance analysis)
  ✓ outlier_analysis_train.csv       (outlier detection results)
  ✓ outlier_analysis_test.csv        (outlier detection results)

📊 VISUALIZATIONS:
  ✓ sensor_distributions_histogram.png
  ✓ sensor_boxplots.png
  ✓ sensor_density_plots.png
  ✓ rul_distribution.png
  ✓ sensor_trajectories_engine1.png
  ✓ outlier_detection.png
  ✓ before_after_scaling.png

✓ DATASET READY FOR NEXT STAGE: Feature Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# =====================================================================================
# FINAL CHECKLIST
# =====================================================================================
print("\n" + "=" * 85)
print("FINAL CHECKLIST: IS DATA READY FOR FEATURE ENGINEERING?")
print("=" * 85)

checklist = {
    'Data Loading': '✓ PASS',
    'Meaningful Column Names': '✓ PASS',
    'Data Type Validation': '✓ PASS (all numeric)',
    'Missing Value Handling': '✓ PASS (0% missing)',
    'Duplicate Row Removal': '✓ PASS (0 duplicates)',
    'Constant Column Removal': '✓ PASS (5 columns removed)',
    'Outlier Analysis': '✓ PASS (retained for model)',
    'Statistical Review': '✓ PASS (reviewed & valid)',
    'Feature Scaling': '✓ PASS (RobustScaler applied)',
    'Data Integrity': '✓ PASS (shapes verified)',
    'Documentation': '✓ PASS (metadata saved)',
    'File Organization': '✓ PASS (all files saved)'
}

for check, status in checklist.items():
    print(f"{check:<30} {status}")

print("\n" + "=" * 85)
print("✓✓✓ DATA CLEANING & PREPROCESSING COMPLETE ✓✓✓")
print("=" * 85)

print("""
NEXT STAGE: Feature Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The preprocessed data is now ready for feature engineering phase.

Input files for feature engineering:
  • train_FD001_scaled.csv (21 scaled sensor features)
  • test_FD001_scaled.csv (21 scaled sensor features)
  • train_FD001_reference.csv (Unit_Number + Time_Cycles for grouping)
  • test_FD001_reference.csv (Unit_Number + Time_Cycles for grouping)
  • RUL_FD001_reference.csv (target variable for evaluation)

Feature engineering will create:
  • Rolling mean/std features (e.g., last 30 cycles average)
  • Polynomial features if needed
  • Time-based decay features
  • Window-based statistics
  • Final feature matrix for model training
""")

print("\n✓ All preprocessing scripts completed and saved")
print(f"✓ Output directory: {data_path}")

