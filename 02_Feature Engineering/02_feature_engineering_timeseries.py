"""
FEATURE ENGINEERING - PART 2: TIME-SERIES FEATURES
====================================================

MSc Computing Dissertation: Predictive Maintenance & Equipment Failure Prediction
Dataset: NASA CMAPSS FD001

OBJECTIVE:
Create sophisticated time-series features that capture degradation patterns:
- Rolling mean features (sensor trends over time windows)
- Rolling standard deviation features (sensor variability)
- Lag features (past sensor values)
- Delta features (cycle-to-cycle changes)
- Cumulative degradation features

SCOPE:
- Task 1: Create rolling mean features
- Task 2: Create rolling std features
- Task 3: Create lag features
- Task 4: Create delta features
- Task 5: Create cumulative degradation features
- Task 6: Handle missing values from feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

PROJECT_DIR = Path("C:/Users/Vatsal/OneDrive/Desktop/msc project")
DATA_DIR = PROJECT_DIR / "data_cleaning"

# File paths
INPUT_FILE = DATA_DIR / "train_FD001_with_rul.csv"
OUTPUT_FILE = DATA_DIR / "train_FD001_with_timeseries_features.csv"

# Feature engineering parameters
ROLLING_WINDOWS = [3, 5, 10]  # Cycle windows for rolling statistics
LAG_STEPS = [1, 2, 3, 5]      # Number of lags to create
SENSOR_COLUMNS = [f"Sensor_{i}" for i in range(1, 22)]  # 21 sensors

print("=" * 80)
print("FEATURE ENGINEERING - PART 2: TIME-SERIES FEATURES")
print("=" * 80)
print()

# ============================================================================
# TASK 1: LOAD DATA WITH RUL
# ============================================================================

print("TASK 1: LOADING DATA WITH RUL TARGET VARIABLE")
print("-" * 80)

try:
    train_data = pd.read_csv(INPUT_FILE)
    print(f"[OK] Loaded training data: {train_data.shape}")
    print(f"  Columns: {list(train_data.columns[:5])}... + RUL")
except FileNotFoundError as e:
    print(f"[ERROR] Error: {e}")
    print(f"  Expected file: {INPUT_FILE}")
    exit(1)

print()

# ============================================================================
# TASK 2: WHY - TIME-SERIES FEATURES RATIONALE
# ============================================================================

print("TASK 2: WHY TIME-SERIES FEATURES? - RATIONALE")
print("-" * 80)
print("""
Raw sensor values alone capture INSTANTANEOUS state only.
Time-series features capture TEMPORAL PATTERNS and DEGRADATION DYNAMICS.

WHY EACH FEATURE TYPE MATTERS:

1. ROLLING MEAN (3/5/10 cycles):
   - Captures TRENDS in sensor readings
   - Smooths out noise and outliers
   - Example: Avg temperature over last 5 cycles shows heating trend
   - Degradation signal: Gradual increase in mean indicates wear
   - Windows: 3 (short-term), 5 (medium-term), 10 (long-term trend)

2. ROLLING STD (3/5/10 cycles):
   - Captures VARIABILITY in sensor readings
   - High variability = unstable operation = degradation signal
   - Example: Temperature fluctuating widely = cooling issues
   - Degradation signal: Increasing std deviation indicates system instability
   - Windows: Same as rolling mean for consistency

3. LAG FEATURES (t-1, t-2, t-3, t-5):
   - Previous cycle's sensor values
   - Captures TEMPORAL DEPENDENCIES in engine operation
   - Example: Current vibration depends on prior vibration
   - Degradation signal: Historical context helps predict failure point
   - Lags: 1/2/3 (recent history), 5 (longer dependency)

4. DELTA FEATURES (cycle-to-cycle changes):
   - First differences: current - previous
   - Captures RATE OF CHANGE in sensor values
   - Example: Temperature increasing by +2°C per cycle = concerning
   - Degradation signal: Rate of increase indicates failure approach speed
   - Useful for acceleration detection

5. CUMULATIVE DEGRADATION:
   - Sum of absolute deltas up to current cycle
   - Captures TOTAL ACCUMULATED WEAR
   - Example: Total heat accumulated = cumulative thermal stress
   - Degradation signal: Higher cumulative value = more wear
   - Replaces instantaneous with cumulative perspective

FEATURE ENGINEERING WORKFLOW:
1. Group by Unit_Number (each engine independently)
2. Create all features WITHIN each engine group
3. Handle missing values from lagging (fill with forward/backward fill)
4. Preserve RUL and identifiers
5. Save engineered dataset for feature selection
""")
print()

# ============================================================================
# TASK 3: CREATE ROLLING MEAN FEATURES
# ============================================================================

print("TASK 3: CREATING ROLLING MEAN FEATURES")
print("-" * 80)

rolling_mean_features = []

# Group by engine unit
for engine_id in sorted(train_data['Unit_Number'].unique()):
    engine_data = train_data[train_data['Unit_Number'] == engine_id].copy()
    
    # Create rolling mean for each window size and sensor
    for window in ROLLING_WINDOWS:
        for sensor in SENSOR_COLUMNS:
            col_name = f"{sensor}_rolling_mean_{window}"
            engine_data[col_name] = engine_data[sensor].rolling(
                window=window, min_periods=1
            ).mean()
    
    rolling_mean_features.append(engine_data)

train_data = pd.concat(rolling_mean_features, ignore_index=True)
print(f"[OK] Created rolling mean features for {len(ROLLING_WINDOWS)} windows × 21 sensors")
print(f"  Total rolling mean features: {len(ROLLING_WINDOWS) * 21}")
print(f"  Dataset shape: {train_data.shape}")
print()

# ============================================================================
# TASK 4: CREATE ROLLING STD FEATURES
# ============================================================================

print("TASK 4: CREATING ROLLING STANDARD DEVIATION FEATURES")
print("-" * 80)

rolling_std_features = []

# Group by engine unit
for engine_id in sorted(train_data['Unit_Number'].unique()):
    engine_data = train_data[train_data['Unit_Number'] == engine_id].copy()
    
    # Create rolling std for each window size and sensor
    for window in ROLLING_WINDOWS:
        for sensor in SENSOR_COLUMNS:
            col_name = f"{sensor}_rolling_std_{window}"
            engine_data[col_name] = engine_data[sensor].rolling(
                window=window, min_periods=1
            ).std()
    
    rolling_std_features.append(engine_data)

train_data = pd.concat(rolling_std_features, ignore_index=True)
print(f"[OK] Created rolling std features for {len(ROLLING_WINDOWS)} windows × 21 sensors")
print(f"  Total rolling std features: {len(ROLLING_WINDOWS) * 21}")
print(f"  Dataset shape: {train_data.shape}")
print()

# ============================================================================
# TASK 5: CREATE LAG FEATURES
# ============================================================================

print("TASK 5: CREATING LAG FEATURES")
print("-" * 80)

lag_features = []

# Group by engine unit
for engine_id in sorted(train_data['Unit_Number'].unique()):
    engine_data = train_data[train_data['Unit_Number'] == engine_id].copy()
    
    # Create lag features for each lag step and sensor
    for lag_step in LAG_STEPS:
        for sensor in SENSOR_COLUMNS:
            col_name = f"{sensor}_lag_{lag_step}"
            engine_data[col_name] = engine_data[sensor].shift(lag_step)
    
    lag_features.append(engine_data)

train_data = pd.concat(lag_features, ignore_index=True)
print(f"[OK] Created lag features for {len(LAG_STEPS)} lag steps × 21 sensors")
print(f"  Total lag features: {len(LAG_STEPS) * 21}")
print(f"  Dataset shape: {train_data.shape}")
print()

# ============================================================================
# TASK 6: CREATE DELTA (DIFFERENCE) FEATURES
# ============================================================================

print("TASK 6: CREATING DELTA (DIFFERENCE) FEATURES")
print("-" * 80)

delta_features = []

# Group by engine unit
for engine_id in sorted(train_data['Unit_Number'].unique()):
    engine_data = train_data[train_data['Unit_Number'] == engine_id].copy()
    
    # Create delta features (current - previous cycle)
    for sensor in SENSOR_COLUMNS:
        col_name = f"{sensor}_delta"
        engine_data[col_name] = engine_data[sensor].diff()
    
    delta_features.append(engine_data)

train_data = pd.concat(delta_features, ignore_index=True)
print(f"[OK] Created delta features for 21 sensors")
print(f"  Total delta features: 21")
print(f"  Dataset shape: {train_data.shape}")
print()

# ============================================================================
# TASK 7: CREATE CUMULATIVE DEGRADATION FEATURES
# ============================================================================

print("TASK 7: CREATING CUMULATIVE DEGRADATION FEATURES")
print("-" * 80)

cumulative_features = []

# Group by engine unit
for engine_id in sorted(train_data['Unit_Number'].unique()):
    engine_data = train_data[train_data['Unit_Number'] == engine_id].copy()
    
    # Create cumulative absolute delta (total accumulated change)
    for sensor in SENSOR_COLUMNS:
        col_name = f"{sensor}_cumulative_delta"
        delta_col = f"{sensor}_delta"
        # Use cumsum of absolute deltas to capture total wear
        engine_data[col_name] = engine_data[delta_col].abs().cumsum()
    
    cumulative_features.append(engine_data)

train_data = pd.concat(cumulative_features, ignore_index=True)
print(f"[OK] Created cumulative degradation features for 21 sensors")
print(f"  Total cumulative features: 21")
print(f"  Dataset shape: {train_data.shape}")
print()

# ============================================================================
# TASK 8: HANDLE MISSING VALUES FROM FEATURE ENGINEERING
# ============================================================================

print("TASK 8: HANDLING MISSING VALUES FROM FEATURE ENGINEERING")
print("-" * 80)

# Count missing values before handling
missing_before = train_data.isnull().sum().sum()
print(f"Missing values before handling: {missing_before}")

# Fill NaN values using forward fill (propagate valid observation forward)
# Then backward fill for the remaining NaN at the start of each engine sequence
train_data = train_data.bfill().ffill()

missing_after = train_data.isnull().sum().sum()
print(f"Missing values after handling: {missing_after}")
print()

if missing_after > 0:
    print("Remaining missing values by column:")
    missing_cols = train_data.columns[train_data.isnull().any()]
    for col in missing_cols:
        print(f"  {col}: {train_data[col].isnull().sum()}")
else:
    print("[OK] All missing values handled successfully")

print()

# ============================================================================
# TASK 9: FEATURE SUMMARY & STATISTICS
# ============================================================================

print("TASK 9: FEATURE ENGINEERING SUMMARY")
print("-" * 80)

# Count engineered features by type
original_sensor_cols = 21
rolling_mean_count = len(ROLLING_WINDOWS) * 21
rolling_std_count = len(ROLLING_WINDOWS) * 21
lag_count = len(LAG_STEPS) * 21
delta_count = 21
cumulative_count = 21

total_engineered = (rolling_mean_count + rolling_std_count + 
                   lag_count + delta_count + cumulative_count)

print(f"Original sensors: {original_sensor_cols}")
print()
print("Engineered features breakdown:")
print(f"  Rolling mean features (3 windows × 21): {rolling_mean_count}")
print(f"  Rolling std features (3 windows × 21): {rolling_std_count}")
print(f"  Lag features (4 lags × 21): {lag_count}")
print(f"  Delta features (21): {delta_count}")
print(f"  Cumulative degradation (21): {cumulative_count}")
print()
print(f"Total engineered features: {total_engineered}")
print(f"Total features (including RUL): {train_data.shape[1] - 2}")  # -2 for Unit_Number, Time_Cycles
print()

print("Dataset dimensions:")
print(f"  Rows (samples): {train_data.shape[0]}")
print(f"  Columns: {train_data.shape[1]}")
print()

# ============================================================================
# TASK 10: DISPLAY SAMPLE ENGINEERED FEATURES
# ============================================================================

print("TASK 10: SAMPLE ENGINEERED FEATURES")
print("-" * 80)

# Show first engine with sample of different feature types
sample_engine = train_data[train_data['Unit_Number'] == 1].head(10)
display_cols = ['Unit_Number', 'Time_Cycles', 
                'Sensor_1', 'Sensor_1_rolling_mean_5', 'Sensor_1_rolling_std_5',
                'Sensor_1_lag_1', 'Sensor_1_delta', 'Sensor_1_cumulative_delta', 'RUL']

print(f"Sample: Engine Unit_1 (first 10 cycles with selected features)")
print(sample_engine[display_cols].to_string())
print()

# ============================================================================
# TASK 11: SAVE INTERMEDIATE DATASET
# ============================================================================

print("TASK 11: SAVING TIME-SERIES ENGINEERED DATASET")
print("-" * 80)

train_data.to_csv(OUTPUT_FILE, index=False)
print(f"[OK] Saved: {OUTPUT_FILE}")
print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY - TIME-SERIES FEATURES CREATED")
print("=" * 80)
print()

print("FEATURES CREATED:")
print("[OK] Rolling mean features (3/5/10 cycle windows) - captures trends")
print("[OK] Rolling std features (3/5/10 cycle windows) - captures variability")
print("[OK] Lag features (1/2/3/5 cycles) - captures temporal dependencies")
print("[OK] Delta features - captures cycle-to-cycle changes")
print("[OK] Cumulative degradation - captures total accumulated wear")
print()

print("WHAT THESE FEATURES CAPTURE:")
print("• Degradation TRENDS: Rolling mean shows slow degradation progression")
print("• System INSTABILITY: Rolling std shows increasing erratic behavior")
print("• TEMPORAL DEPENDENCIES: Lag features preserve time-series structure")
print("• ACCELERATION: Delta features detect rate of change")
print("• CUMULATIVE DAMAGE: Cumulative deltas show total accumulated stress")
print()

print("NEXT STEP:")
print("-> Feature Engineering Part 3: Feature Selection")
print("  (Correlation analysis, remove highly correlated, statistical importance)")
print()

print("=" * 80)


