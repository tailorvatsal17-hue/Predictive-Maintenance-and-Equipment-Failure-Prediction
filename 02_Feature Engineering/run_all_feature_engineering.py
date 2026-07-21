"""
COMPLETE FEATURE ENGINEERING PIPELINE
========================================

MSc Computing Dissertation: Predictive Maintenance & Equipment Failure Prediction
Dataset: NASA CMAPSS FD001

This script executes the complete feature engineering pipeline:
- Part 1: RUL Target Variable Creation
- Part 2: Time-Series Feature Engineering
- Part 3: Feature Selection (Correlation Analysis)
- Part 4: Dataset Validation & Finalization

EXECUTION: python run_all_feature_engineering.py
"""

import sys
import subprocess
from pathlib import Path

# Project directories
PROJECT_DIR = Path("C:/Users/Vatsal/OneDrive/Desktop/msc project")
DATA_DIR = PROJECT_DIR / "data_cleaning"
SCRIPT_DIR = Path(__file__).parent

# Scripts to execute
SCRIPTS = [
    "01_feature_engineering_rul_creation.py",
    "02_feature_engineering_timeseries.py",
    "03_feature_engineering_selection.py",
    "04_feature_engineering_validation.py"
]

print("=" * 80)
print("FEATURE ENGINEERING - COMPLETE PIPELINE EXECUTION")
print("=" * 80)
print()

print("This script will execute the feature engineering pipeline in sequence:")
print("  1. RUL Target Variable Creation")
print("  2. Time-Series Features (Rolling, Lag, Delta, Cumulative)")
print("  3. Feature Selection (Correlation Analysis)")
print("  4. Dataset Validation & Finalization")
print()

print("Expected outputs:")
print("  - train_FD001_engineered.csv")
print("  - test_FD001_engineered.csv")
print("  - Comprehensive validation reports")
print()

print("=" * 80)
print()

# Execute each script
for i, script_name in enumerate(SCRIPTS, 1):
    script_path = SCRIPT_DIR / script_name
    
    print(f"[PART {i}/4] Running: {script_name}")
    print("-" * 80)
    
    if not script_path.exists():
        print(f"[ERROR] Error: Script not found: {script_path}")
        sys.exit(1)
    
    try:
        # Execute script in subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            capture_output=False,
            check=True
        )
        
        print()
        print(f"[OK] Part {i} completed successfully")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error executing {script_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
print()

print("Output files created:")
output_files = [
    "train_FD001_engineered.csv",
    "test_FD001_engineered.csv",
    "FEATURE_ENGINEERING_VALIDATION.md",
    "ENGINEERED_FEATURES_LIST.txt",
    "feature_importance_statistical.csv"
]

for output_file in output_files:
    output_path = DATA_DIR / output_file
    if output_path.exists():
        size = output_path.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  [OK] {output_file} ({size_str})")
    else:
        print(f"  ? {output_file} (not found)")

print()
print("=" * 80)
print("NEXT PHASE: MODEL TRAINING")
print("=" * 80)
print()

print("Your engineered datasets are ready for model training:")
print("  - train_FD001_engineered.csv: Training data with RUL labels")
print("  - test_FD001_engineered.csv: Test data with RUL ground truth")
print()

print("Ready to train:")
print("  [OK] Random Forest Regressor")
print("  [OK] XGBoost Regressor")
print("  [OK] Neural Network Regressor")
print()

print("=" * 80)






