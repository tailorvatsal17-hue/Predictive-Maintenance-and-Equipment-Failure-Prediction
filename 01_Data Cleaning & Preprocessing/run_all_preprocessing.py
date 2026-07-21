# -*- coding: utf-8 -*-
"""
=====================================================================================
NASA CMAPSS DATA CLEANING & PREPROCESSING - COMPLETE PIPELINE
=====================================================================================
This script consolidates all preprocessing steps (1-14) into one execution:
1. Data Loading
2. Column Naming
3. Data Exploration
4. Data Type Checking
5. Missing Value Analysis
6. Duplicate Detection
7. Constant Column Detection
8. Descriptive Statistics
9. Distribution Visualization
10. Outlier Detection
11. Scaling/Normalization
12. Final Dataset Saving
=====================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Fix stdout encoding for Windows
if sys.platform == 'win32':
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set visualization style
sns.set_style("whitegrid")

print("\n" + "=" * 90)
print("NASA CMAPSS DATA CLEANING & PREPROCESSING - COMPLETE PIPELINE")
print("=" * 90)

# =====================================================================================
# SETUP
# =====================================================================================
data_folder_source = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\CMAPSSData"
data_folder_output = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning"
viz_folder = os.path.join(data_folder_output, "visualizations")
os.makedirs(viz_folder, exist_ok=True)

print("\n[SETUP] Creating output directories...")
print(f"  Output folder: {data_folder_output}")
print(f"  Visualization folder: {viz_folder}")

# =====================================================================================
# STEP 1-4: LOAD AND EXPLORE DATA
# =====================================================================================
print("\n" + "=" * 90)
print("STEPS 1-4: LOADING AND EXPLORING DATA")
print("=" * 90)

train_file = os.path.join(data_folder_source, "train_FD001.txt")
test_file = os.path.join(data_folder_source, "test_FD001.txt")
rul_file = os.path.join(data_folder_source, "RUL_FD001.txt")

# Load data
print("\n[LOAD] Reading NASA CMAPSS FD001 files...")
try:
    train_data = pd.read_csv(train_file, sep=r'\s+', header=None)
    test_data = pd.read_csv(test_file, sep=r'\s+', header=None)
    rul_data = pd.read_csv(rul_file, header=None)
    print(f"[OK] Data loaded successfully")
    print(f"     - train_FD001.txt: {train_data.shape}")
    print(f"     - test_FD001.txt:  {test_data.shape}")
    print(f"     - RUL_FD001.txt:   {rul_data.shape}")
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    sys.exit(1)

# Assign column names
column_names = [
    'Unit_Number', 'Time_Cycles',
    'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3',
    'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5',
    'Sensor_6', 'Sensor_7', 'Sensor_8', 'Sensor_9', 'Sensor_10',
    'Sensor_11', 'Sensor_12', 'Sensor_13', 'Sensor_14', 'Sensor_15',
    'Sensor_16', 'Sensor_17', 'Sensor_18', 'Sensor_19', 'Sensor_20',
    'Sensor_21'
]

train_data.columns = column_names
test_data.columns = column_names
rul_data.columns = ['RUL']

print(f"\n[OK] Column names assigned (26 columns total)")
print(f"     Training data shape: {train_data.shape}")
print(f"     Test data shape: {test_data.shape}")

# Save for future use
train_data.to_csv(os.path.join(data_folder_output, "train_FD001_loaded.csv"), index=False)
test_data.to_csv(os.path.join(data_folder_output, "test_FD001_loaded.csv"), index=False)
rul_data.to_csv(os.path.join(data_folder_output, "rul_FD001_loaded.csv"), index=False)
print(f"[OK] Intermediate CSV files saved")

# =====================================================================================
# STEP 5-6: DATA TYPES AND MISSING VALUES
# =====================================================================================
print("\n" + "=" * 90)
print("STEPS 5-6: DATA TYPE AND MISSING VALUE ANALYSIS")
print("=" * 90)

print("\n[CHECK] Data types:")
dtypes = train_data.dtypes
print(f"     All columns: {dtypes.unique()}")

print("\n[CHECK] Missing values:")
missing_train = train_data.isnull().sum().sum()
missing_test = test_data.isnull().sum().sum()
print(f"     Training data: {missing_train} missing values ({100*missing_train/(train_data.shape[0]*train_data.shape[1]):.1f}%)")
print(f"     Test data: {missing_test} missing values ({100*missing_test/(test_data.shape[0]*test_data.shape[1]):.1f}%)")

# =====================================================================================
# STEP 7-8: DUPLICATES AND CONSTANT COLUMNS
# =====================================================================================
print("\n" + "=" * 90)
print("STEPS 7-8: DUPLICATE AND CONSTANT COLUMN ANALYSIS")
print("=" * 90)

print("\n[CHECK] Duplicate rows:")
duplicates_train = train_data.duplicated().sum()
duplicates_test = test_data.duplicated().sum()
print(f"     Training data: {duplicates_train} duplicates")
print(f"     Test data: {duplicates_test} duplicates")

print("\n[CHECK] Constant columns (zero variance):")
constant_cols = []
for col in train_data.columns:
    if train_data[col].nunique() <= 1:
        constant_cols.append(col)
        print(f"     - {col}: {train_data[col].unique()}")

if not constant_cols:
    print("     - No constant columns found")

print("\n[CHECK] Near-constant columns (very low variance):")
near_constant = []
for col in train_data.columns[2:]:  # Skip Unit_Number and Time_Cycles
    variance = train_data[col].var()
    cv = train_data[col].std() / train_data[col].mean() if train_data[col].mean() != 0 else 0
    if variance < 1.0:
        near_constant.append(col)
        print(f"     - {col}: variance={variance:.4f}, CV={cv:.4f}")

# =====================================================================================
# STEP 9-10: DESCRIPTIVE STATISTICS AND VISUALIZATION
# =====================================================================================
print("\n" + "=" * 90)
print("STEPS 9-10: DESCRIPTIVE STATISTICS AND VISUALIZATION")
print("=" * 90)

print("\n[STAT] Computing descriptive statistics...")
stats_df = train_data.describe().T
stats_df['skewness'] = [train_data[col].skew() for col in train_data.columns]
stats_df['kurtosis'] = [train_data[col].kurtosis() for col in train_data.columns]

stats_df.to_csv(os.path.join(data_folder_output, "descriptive_statistics.csv"))
print(f"[OK] Statistics saved to descriptive_statistics.csv")

print("\n[VIZ] Creating visualizations...")

# Sensor distributions
sensor_cols = [col for col in train_data.columns if col.startswith('Sensor')]
fig, axes = plt.subplots(7, 3, figsize=(15, 14))
axes = axes.ravel()
for idx, col in enumerate(sensor_cols):
    axes[idx].hist(train_data[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(col, fontsize=9)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, "sensor_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: sensor_distributions.png")

# Sensor boxplots
fig, axes = plt.subplots(7, 3, figsize=(15, 14))
axes = axes.ravel()
for idx, col in enumerate(sensor_cols):
    axes[idx].boxplot(train_data[col], vert=True)
    axes[idx].set_title(col, fontsize=9)
    axes[idx].set_ylabel('Value')
plt.tight_layout()
plt.savefig(os.path.join(viz_folder, "sensor_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: sensor_boxplots.png")

# =====================================================================================
# STEP 11: OUTLIER DETECTION
# =====================================================================================
print("\n" + "=" * 90)
print("STEP 11: OUTLIER DETECTION AND ANALYSIS")
print("=" * 90)

print("\n[OUTLIER] Using IQR method...")
outlier_count = 0
for col in sensor_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((train_data[col] < lower) | (train_data[col] > upper)).sum()
    outlier_count += outliers

print(f"[OK] Total outliers detected (IQR method): {outlier_count}")
print(f"     Percentage: {100*outlier_count/(train_data.shape[0]*len(sensor_cols)):.2f}%")
print(f"[DECISION] KEEP all outliers (important for failure prediction)")

# =====================================================================================
# STEP 12: COLUMN REMOVAL
# =====================================================================================
print("\n" + "=" * 90)
print("STEP 12: COLUMN REMOVAL DECISION")
print("=" * 90)

columns_to_remove = ['Unit_Number', 'Time_Cycles', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3']
columns_to_keep = [col for col in train_data.columns if col not in columns_to_remove]

print(f"\n[REMOVE] {len(columns_to_remove)} columns:")
for col in columns_to_remove:
    print(f"     - {col}")
print(f"[KEEP] {len(columns_to_keep)} sensor columns:")
for col in columns_to_keep[:5]:
    print(f"     - {col}")
print(f"     ... and {len(columns_to_keep)-5} more")

train_cleaned = train_data[columns_to_keep].copy()
test_cleaned = test_data[columns_to_keep].copy()

print(f"\n[OK] Training data: {train_data.shape} -> {train_cleaned.shape}")
print(f"[OK] Test data: {test_data.shape} -> {test_cleaned.shape}")

# =====================================================================================
# STEP 13: SCALING AND NORMALIZATION
# =====================================================================================
print("\n" + "=" * 90)
print("STEP 13: SCALING AND NORMALIZATION")
print("=" * 90)

print("\n[SCALE] Applying RobustScaler...")
print("     Reason: Resistant to outliers (important for failure signals)")

robust_scaler = RobustScaler()
train_scaled = robust_scaler.fit_transform(train_cleaned)
train_scaled_df = pd.DataFrame(train_scaled, columns=columns_to_keep)

test_scaled = robust_scaler.transform(test_cleaned)
test_scaled_df = pd.DataFrame(test_scaled, columns=columns_to_keep)

print(f"[OK] Scaling applied successfully")
print(f"     Training data mean: {train_scaled_df.mean().mean():.4f}")
print(f"     Training data std: {train_scaled_df.std().mean():.4f}")

# Visualize before/after scaling
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sample_sensors = ['Sensor_1', 'Sensor_11', 'Sensor_21']
for idx, col in enumerate(sample_sensors):
    axes[0, idx].hist(train_cleaned[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, idx].set_title(f'{col} - BEFORE', fontsize=10, fontweight='bold')
    
    axes[1, idx].hist(train_scaled_df[col], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, idx].set_title(f'{col} - AFTER', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_folder, "before_after_scaling.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: before_after_scaling.png")

# =====================================================================================
# STEP 14: SAVE FINAL DATASETS
# =====================================================================================
print("\n" + "=" * 90)
print("STEP 14: SAVING CLEANED AND PREPROCESSED DATASETS")
print("=" * 90)

print("\n[SAVE] Writing output files...")

# Save scaled data
train_scaled_df.to_csv(os.path.join(data_folder_output, "train_FD001_scaled.csv"), index=False)
test_scaled_df.to_csv(os.path.join(data_folder_output, "test_FD001_scaled.csv"), index=False)
print(f"[OK] train_FD001_scaled.csv ({train_scaled_df.shape})")
print(f"[OK] test_FD001_scaled.csv ({test_scaled_df.shape})")

# Save reference files
train_data[['Unit_Number', 'Time_Cycles']].to_csv(
    os.path.join(data_folder_output, "train_reference.csv"), index=False)
test_data[['Unit_Number', 'Time_Cycles']].to_csv(
    os.path.join(data_folder_output, "test_reference.csv"), index=False)
print(f"[OK] train_reference.csv")
print(f"[OK] test_reference.csv")

# Save RUL data
rul_data.to_csv(os.path.join(data_folder_output, "rul_reference.csv"), index=False)
print(f"[OK] rul_reference.csv")

# Save scaler
with open(os.path.join(data_folder_output, "robust_scaler.pkl"), 'wb') as f:
    pickle.dump(robust_scaler, f)
print(f"[OK] robust_scaler.pkl")

# =====================================================================================
# FINAL SUMMARY
# =====================================================================================
print("\n" + "=" * 90)
print("DATA CLEANING & PREPROCESSING COMPLETE")
print("=" * 90)

print(f"""
PREPROCESSING SUMMARY
=====================
Steps Completed:
  [OK] Step 1-4:   Data loading and exploration
  [OK] Step 5-6:   Data types and missing values (0% missing)
  [OK] Step 7-8:   Duplicates (0) and constant columns (3)
  [OK] Step 9-10:  Descriptive statistics and visualization
  [OK] Step 11:    Outlier detection ({100*outlier_count/(train_data.shape[0]*len(sensor_cols)):.2f}% outliers, KEPT)
  [OK] Step 12:    Column removal (5 columns removed, 21 kept)
  [OK] Step 13:    Scaling/Normalization (RobustScaler applied)
  [OK] Step 14:    Final dataset saving

DATASET STATISTICS
==================
Original Training:  {train_data.shape[0]} rows x {train_data.shape[1]} columns
Cleaned Training:   {train_scaled_df.shape[0]} rows x {train_scaled_df.shape[1]} columns
Original Test:      {test_data.shape[0]} rows x {test_data.shape[1]} columns
Cleaned Test:       {test_scaled_df.shape[0]} rows x {test_scaled_df.shape[1]} columns

Columns Removed:    {len(columns_to_remove)} (Unit_Number, Time_Cycles, Op_Settings)
Columns Retained:   {len(columns_to_keep)} (21 sensors)
Scaling Method:     RobustScaler (median/IQR-based)

OUTPUT FILES CREATED
====================
Data:
  - train_FD001_scaled.csv
  - test_FD001_scaled.csv
  - train_reference.csv (Unit_Number + Time_Cycles)
  - test_reference.csv (Unit_Number + Time_Cycles)
  - rul_reference.csv (ground truth RUL)

Tools:
  - robust_scaler.pkl (for applying to future data)

Visualizations:
  - sensor_distributions.png
  - sensor_boxplots.png
  - before_after_scaling.png

Analysis:
  - descriptive_statistics.csv

NEXT STEPS
==========
The preprocessed data is now ready for Feature Engineering:
  1. Create time-lagged features
  2. Generate rolling statistics
  3. Create polynomial/interaction features
  4. Select most important features
  5. Prepare for model training

""")

print("=" * 90)
print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 90 + "\n")


