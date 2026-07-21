"""
=====================================================================================
DATA CLEANING & PREPROCESSING FOR NASA CMAPSS TURBOFAN ENGINE DATASET (FD001)
Stage 1: Data Loading and Exploration
=====================================================================================

OBJECTIVE:
This script handles Steps 1-4 of the data cleaning process:
1. Load the three NASA dataset files
2. Assign meaningful column names
3. Display dataset dimensions and first few rows
4. Understand the purpose of each column

No data transformations are performed in this step - only exploration.
=====================================================================================
"""

import pandas as pd
import numpy as np
import os

# =====================================================================================
# STEP 1: LOAD THE THREE NASA DATASET FILES
# =====================================================================================
print("=" * 85)
print("STEP 1: LOADING NASA C-MAPSS FD001 DATASET FILES")
print("=" * 85)

# Define paths to the dataset files
data_path = r"C:\Users\Vatsal\OneDrive\Desktop\msc project\CMAPSSData"
train_file = os.path.join(data_path, "train_FD001.txt")
test_file = os.path.join(data_path, "test_FD001.txt")
rul_file = os.path.join(data_path, "RUL_FD001.txt")

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. The NASA dataset files are plain text files (space-separated values)
2. We need to load them into pandas DataFrames for analysis and manipulation
3. Each file serves a specific purpose:
   - train_FD001.txt: Training data with complete engine trajectories (engine runs to failure)
   - test_FD001.txt: Test data with partial trajectories (ends before failure)
   - RUL_FD001.txt: Ground truth RUL values for each test engine (for validation later)
""")

# Load the data files
print("\nLoading data files...")
try:
    # Load with no header, space-separated values
    train_data = pd.read_csv(train_file, sep='\s+', header=None)
    test_data = pd.read_csv(test_file, sep='\s+', header=None)
    rul_data = pd.read_csv(rul_file, header=None)
    print("[OK] Successfully loaded all three files")
except Exception as e:
    print(f"[ERROR] Error loading files: {e}")
    exit()

print(f"\nInitial shapes after loading:")
print(f"  - train_FD001.txt: {train_data.shape}")
print(f"  - test_FD001.txt:  {test_data.shape}")
print(f"  - RUL_FD001.txt:   {rul_data.shape}")

# =====================================================================================
# STEP 2: ASSIGN MEANINGFUL COLUMN NAMES
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 2: ASSIGNING MEANINGFUL COLUMN NAMES")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Raw data has numerical column indices (0, 1, 2, ..., 25)
2. Meaningful names make the data self-documenting and easier to understand
3. According to NASA documentation, each column represents:
   - Column 0: Unit Number (engine ID)
   - Column 1: Time (operational cycles)
   - Columns 2-4: Operational Settings (3 settings affecting engine performance)
   - Columns 5-25: Sensor Measurements (21 sensors monitoring engine health)
4. Named columns enable easier data exploration and reduce errors
""")

# Define column names based on NASA documentation
column_names = [
    'Unit_Number',           # Column 0: Unique engine identifier
    'Time_Cycles',           # Column 1: Operational time (in cycles)
    'Op_Setting_1',          # Column 2: Operational setting 1
    'Op_Setting_2',          # Column 3: Operational setting 2
    'Op_Setting_3',          # Column 4: Operational setting 3
    'Sensor_1',              # Column 5: Sensor 1 measurement
    'Sensor_2',              # Column 6: Sensor 2 measurement
    'Sensor_3',              # Column 7: Sensor 3 measurement
    'Sensor_4',              # Column 8: Sensor 4 measurement
    'Sensor_5',              # Column 9: Sensor 5 measurement
    'Sensor_6',              # Column 10: Sensor 6 measurement
    'Sensor_7',              # Column 11: Sensor 7 measurement
    'Sensor_8',              # Column 12: Sensor 8 measurement
    'Sensor_9',              # Column 13: Sensor 9 measurement
    'Sensor_10',             # Column 14: Sensor 10 measurement
    'Sensor_11',             # Column 15: Sensor 11 measurement
    'Sensor_12',             # Column 16: Sensor 12 measurement
    'Sensor_13',             # Column 17: Sensor 13 measurement
    'Sensor_14',             # Column 18: Sensor 14 measurement
    'Sensor_15',             # Column 19: Sensor 15 measurement
    'Sensor_16',             # Column 20: Sensor 16 measurement
    'Sensor_17',             # Column 21: Sensor 17 measurement
    'Sensor_18',             # Column 22: Sensor 18 measurement
    'Sensor_19',             # Column 23: Sensor 19 measurement
    'Sensor_20',             # Column 24: Sensor 20 measurement
    'Sensor_21'              # Column 25: Sensor 21 measurement
]

# Assign column names to dataframes
train_data.columns = column_names
test_data.columns = column_names
rul_data.columns = ['RUL_Actual']

print(f"\n[OK] Successfully assigned {len(column_names)} column names to training and test data")
print(f"[OK] Assigned 1 column name to RUL data")

# =====================================================================================
# STEP 3: DISPLAY DATASET DIMENSIONS AND FIRST FEW ROWS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 3: DISPLAY DATASET DIMENSIONS AND FIRST FEW ROWS")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Understanding dataset shape helps identify data availability and completeness
2. First few rows provide a concrete view of the actual data values
3. This step confirms that data loading and naming was successful
4. It reveals any obvious data quality issues (e.g., unusual patterns, formatting issues)
""")

# Display training data information
print("\n" + "-" * 85)
print("TRAINING DATA (train_FD001.txt)")
print("-" * 85)
print(f"Shape: {train_data.shape} (rows={train_data.shape[0]}, columns={train_data.shape[1]})")
print(f"\nFirst 5 rows:")
print(train_data.head())
print(f"\nLast 5 rows:")
print(train_data.tail())

# Display test data information
print("\n" + "-" * 85)
print("TEST DATA (test_FD001.txt)")
print("-" * 85)
print(f"Shape: {test_data.shape} (rows={test_data.shape[0]}, columns={test_data.shape[1]})")
print(f"\nFirst 5 rows:")
print(test_data.head())
print(f"\nLast 5 rows:")
print(test_data.tail())

# Display RUL data information
print("\n" + "-" * 85)
print("RUL GROUND TRUTH DATA (RUL_FD001.txt)")
print("-" * 85)
print(f"Shape: {rul_data.shape} (rows={rul_data.shape[0]}, columns={rul_data.shape[1]})")
print(f"\nFirst 5 values:")
print(rul_data.head())
print(f"\nLast 5 values:")
print(rul_data.tail())

# =====================================================================================
# STEP 4: UNDERSTAND THE PURPOSE OF EACH COLUMN
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 4: UNDERSTANDING THE PURPOSE OF EACH COLUMN")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. Different columns serve different purposes and require different handling
2. Understanding column types helps decide on preprocessing strategies
3. Some columns are identifiers (keep as-is)
4. Some are operational conditions (may affect degradation patterns)
5. Some are sensor measurements (potential predictors of failure)
6. This understanding is crucial for feature engineering decisions later
""")

print("\nCOLUMN PURPOSES:")
print("-" * 85)

print("\n1. IDENTIFIER COLUMNS:")
print("   • Unit_Number: Engine identifier (1-100 in training, 1-100 in test)")
print("     Purpose: Track individual engine trajectories")
print("     Type: Categorical/Integer identifier")

print("\n2. TIME COLUMN:")
print("   • Time_Cycles: Operational time measurement")
print("     Purpose: Tracks progression of engine degradation over time")
print("     Type: Integer (continuous cycles)")
print("     Range: Starts at 1, ends at varying values (engine-dependent)")

print("\n3. OPERATIONAL SETTING COLUMNS:")
print("   • Op_Setting_1, Op_Setting_2, Op_Setting_3")
print("     Purpose: Environmental/operational conditions affecting engine performance")
print("     Type: Continuous numeric values")
print("     Note: In FD001, these remain constant (sea level condition)")
print("     Note: In FD002/FD003/FD004, these vary (different operating conditions)")

print("\n4. SENSOR MEASUREMENT COLUMNS (21 sensors):")
print("   • Sensor_1 through Sensor_21")
print("     Purpose: Real-time health indicators monitored during engine operation")
print("     Type: Continuous numeric values")
print("     Note: Values change as engine degrades over time")
print("     Note: Some contain noise (simulated realistic conditions)")
print("     Interpretation: Increasing/decreasing patterns indicate degradation")

print("\n5. RUL DATA:")
print("   • RUL_Actual: Ground truth remaining useful life")
print("     Purpose: Testing metric - number of cycles engine will operate after test ends")
print("     Type: Integer (cycles)")
print("     Use: Model validation and performance evaluation (not created yet)")

# =====================================================================================
# SUMMARY OF STEP 1-4
# =====================================================================================
print("\n" + "=" * 85)
print("SUMMARY: STEPS 1-4 COMPLETED")
print("=" * 85)
print(f"""
[OK] Data loaded successfully:
  - Training data: {train_data.shape[0]} records across {train_data.shape[1]} columns
  - Test data: {test_data.shape[0]} records across {test_data.shape[1]} columns
  - RUL reference: {rul_data.shape[0]} ground truth values

[OK] Meaningful column names assigned

[OK] Data structure understood:
  - 1 identifier column (Unit_Number)
  - 1 time column (Time_Cycles)
  - 3 operational setting columns
  - 21 sensor measurement columns
  - RUL reference data for validation

Next Step: Check data types and missing values
""")

# Save the dataframes for the next preprocessing step
train_data.to_csv(
    r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning\train_FD001_loaded.csv",
    index=False
)
test_data.to_csv(
    r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning\test_FD001_loaded.csv",
    index=False
)
rul_data.to_csv(
    r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning\rul_FD001_loaded.csv",
    index=False
)

print("\n[OK] Intermediate data saved for next step")
print("  Files created:")
print("  - train_FD001_loaded.csv")
print("  - test_FD001_loaded.csv")
print("  - rul_FD001_loaded.csv")


