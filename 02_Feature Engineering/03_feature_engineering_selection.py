"""
FEATURE ENGINEERING - PART 3: FEATURE SELECTION
=================================================

MSc Computing Dissertation: Predictive Maintenance & Equipment Failure Prediction
Dataset: NASA CMAPSS FD001

OBJECTIVE:
Perform statistical feature selection to identify most important sensors
and their engineered features for RUL prediction.

SCOPE:
- Task 1: Load engineered dataset
- Task 2: Perform correlation analysis
- Task 3: Identify highly correlated features
- Task 4: Calculate statistical feature importance
- Task 5: Remove redundant features (if justified)
- Task 6: Create final feature set
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

PROJECT_DIR = Path("C:/Users/Vatsal/OneDrive/Desktop/msc project")
DATA_DIR = PROJECT_DIR / "data_cleaning"

# File paths
INPUT_FILE = DATA_DIR / "train_FD001_with_timeseries_features.csv"
OUTPUT_FILE = DATA_DIR / "train_FD001_engineered_final.csv"
CORRELATION_REPORT = DATA_DIR / "correlation_analysis.csv"
FEATURE_IMPORTANCE_REPORT = DATA_DIR / "feature_importance_statistical.csv"

# Feature selection parameters
CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > 0.95 with others
SENSOR_COLUMNS = [f"Sensor_{i}" for i in range(1, 22)]

print("=" * 80)
print("FEATURE ENGINEERING - PART 3: FEATURE SELECTION")
print("=" * 80)
print()

# ============================================================================
# TASK 1: LOAD ENGINEERED DATASET
# ============================================================================

print("TASK 1: LOADING ENGINEERED DATASET")
print("-" * 80)

try:
    train_data = pd.read_csv(INPUT_FILE)
    print(f"[OK] Loaded engineered data: {train_data.shape}")
except FileNotFoundError as e:
    print(f"[ERROR] Error: {e}")
    exit(1)

# Separate features from identifiers and target
id_cols = ['Unit_Number', 'Time_Cycles']
target_col = ['RUL']
feature_cols = [col for col in train_data.columns if col not in id_cols + target_col]

print(f"  Identifiers: {len(id_cols)}")
print(f"  Features (original + engineered): {len(feature_cols)}")
print(f"  Target variable: {len(target_col)}")
print()

# ============================================================================
# TASK 2: WHY - FEATURE SELECTION RATIONALE
# ============================================================================

print("TASK 2: WHY FEATURE SELECTION? - RATIONALE")
print("-" * 80)
print("""
Feature selection is critical for multiple reasons:

1. CORRELATION ANALYSIS:
   - Identifies redundant features (highly correlated with each other)
   - Removes multicollinearity that can confuse models
   - Keeps features with unique information
   - Example: sensor_01_rolling_mean_5 might be highly correlated with
     sensor_01_rolling_mean_3 (both smooth the same sensor)

2. STATISTICAL IMPORTANCE:
   - Measures correlation of each feature with RUL target
   - Features with high correlation are more predictive
   - Low correlation features contribute little information
   - Example: sensor bearing temperature correlates strongly with RUL,
     but operational pressure might not

3. DIMENSIONALITY REDUCTION:
   - Fewer features = faster training
   - Reduces overfitting risk
   - Makes models more interpretable
   - Reduces computational complexity

4. SPARSITY & EFFICIENCY:
   - Not all engineered features add value
   - Some windows/lags redundant with others
   - Statistical analysis identifies which are useful

FEATURE SELECTION WORKFLOW:
1. Calculate correlation matrix for all features
2. Identify highly correlated feature pairs
3. Calculate correlation of each feature with RUL
4. Remove redundant features (keep more informative one)
5. Retain features with strong RUL correlation
6. Create final lean feature set for modeling
""")
print()

# ============================================================================
# TASK 3: CALCULATE CORRELATION WITH RUL TARGET
# ============================================================================

print("TASK 3: CALCULATING CORRELATION WITH RUL TARGET")
print("-" * 80)

# Calculate Pearson correlation with RUL
correlations_with_rul = pd.DataFrame({
    'Feature': feature_cols,
    'Correlation_with_RUL': [train_data[col].corr(train_data['RUL']) for col in feature_cols]
})

# Sort by absolute correlation
correlations_with_rul['Abs_Correlation'] = correlations_with_rul['Correlation_with_RUL'].abs()
correlations_with_rul = correlations_with_rul.sort_values('Abs_Correlation', ascending=False)

print(f"[OK] Calculated correlation of {len(feature_cols)} features with RUL")
print()

print("TOP 20 FEATURES MOST CORRELATED WITH RUL:")
print(correlations_with_rul.head(20)[['Feature', 'Correlation_with_RUL']].to_string(index=False))
print()

print("FEATURES LEAST CORRELATED WITH RUL:")
print(correlations_with_rul.tail(10)[['Feature', 'Correlation_with_RUL']].to_string(index=False))
print()

# ============================================================================
# TASK 4: IDENTIFY HIGHLY CORRELATED FEATURE PAIRS
# ============================================================================

print("TASK 4: IDENTIFYING HIGHLY CORRELATED FEATURE PAIRS")
print("-" * 80)

# Calculate full correlation matrix (only for features, not target)
feature_data = train_data[feature_cols].copy()
correlation_matrix = feature_data.corr()

# Find highly correlated pairs
highly_correlated_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
            highly_correlated_pairs.append({
                'Feature1': correlation_matrix.columns[i],
                'Feature2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

highly_correlated_df = pd.DataFrame(highly_correlated_pairs)

if len(highly_correlated_df) > 0:
    print(f"Found {len(highly_correlated_df)} highly correlated pairs (r > {CORRELATION_THRESHOLD}):")
    print(highly_correlated_df.head(20).to_string(index=False))
else:
    print(f"No feature pairs with correlation > {CORRELATION_THRESHOLD} found")

print()

# ============================================================================
# TASK 5: DECIDE WHICH FEATURES TO REMOVE
# ============================================================================

print("TASK 5: DECIDING WHICH FEATURES TO REMOVE")
print("-" * 80)

# Strategy: For each highly correlated pair, keep the one with higher RUL correlation
features_to_remove = set()
rul_corr_dict = dict(zip(correlations_with_rul['Feature'], 
                         correlations_with_rul['Correlation_with_RUL']))

for _, row in highly_correlated_df.iterrows():
    feat1 = row['Feature1']
    feat2 = row['Feature2']
    
    corr1_with_rul = abs(rul_corr_dict.get(feat1, 0))
    corr2_with_rul = abs(rul_corr_dict.get(feat2, 0))
    
    # Keep feature with higher RUL correlation, remove the other
    if corr1_with_rul > corr2_with_rul:
        features_to_remove.add(feat2)
    else:
        features_to_remove.add(feat1)

print(f"Features identified for removal: {len(features_to_remove)}")

if len(features_to_remove) > 0:
    print("\nFeatures to be removed (redundant with others):")
    for feat in sorted(list(features_to_remove)):
        rul_corr = rul_corr_dict.get(feat, 0)
        print(f"  - {feat} (RUL correlation: {rul_corr:.4f})")
else:
    print("No redundant features to remove")

print()

# ============================================================================
# TASK 6: CREATE FINAL FEATURE SET
# ============================================================================

print("TASK 6: CREATING FINAL FEATURE SET")
print("-" * 80)

# Determine which features to keep
features_to_keep = [f for f in feature_cols if f not in features_to_remove]

# Also remove features with very low RUL correlation (< 0.01)
# These don't contribute meaningful information
very_low_corr_threshold = 0.01
very_low_corr_features = [f for f in features_to_keep 
                          if abs(rul_corr_dict.get(f, 0)) < very_low_corr_threshold]

print(f"Features with very low RUL correlation (< {very_low_corr_threshold}): {len(very_low_corr_features)}")

# Keep them for now (let model decide), but note them
if len(very_low_corr_features) > 0:
    print("  Note: These features kept as they may have non-linear relationships")

print()

print(f"Initial engineered features: {len(feature_cols)}")
print(f"Features removed (redundant): {len(features_to_remove)}")
print(f"Final features retained: {len(features_to_keep)}")
print()

# Create final dataset
final_columns = id_cols + features_to_keep + target_col
final_data = train_data[final_columns].copy()

print(f"Final dataset shape: {final_data.shape}")
print()

# ============================================================================
# TASK 7: ANALYZE FEATURE TYPES IN FINAL SET
# ============================================================================

print("TASK 7: ANALYZING FEATURE TYPES IN FINAL SET")
print("-" * 80)

# Categorize final features
original_sensors = [f for f in features_to_keep if f.startswith('Sensor_') 
                   and '_rolling' not in f and '_lag' not in f 
                   and '_delta' not in f and '_cumulative' not in f]
rolling_means = [f for f in features_to_keep if '_rolling_mean_' in f]
rolling_stds = [f for f in features_to_keep if '_rolling_std_' in f]
lag_features = [f for f in features_to_keep if '_lag_' in f]
delta_features = [f for f in features_to_keep if '_delta' in f and '_cumulative' not in f]
cumulative_features = [f for f in features_to_keep if '_cumulative_delta' in f]

print(f"Original sensors: {len(original_sensors)}")
print(f"Rolling mean features: {len(rolling_means)}")
print(f"Rolling std features: {len(rolling_stds)}")
print(f"Lag features: {len(lag_features)}")
print(f"Delta features: {len(delta_features)}")
print(f"Cumulative features: {len(cumulative_features)}")
print()

# ============================================================================
# TASK 8: SAVE FEATURE ANALYSIS REPORTS
# ============================================================================

print("TASK 8: SAVING FEATURE ANALYSIS REPORTS")
print("-" * 80)

# Save correlation with RUL
correlations_with_rul.to_csv(FEATURE_IMPORTANCE_REPORT, index=False)
print(f"[OK] Saved: {FEATURE_IMPORTANCE_REPORT}")

# Save final dataset
final_data.to_csv(OUTPUT_FILE, index=False)
print(f"[OK] Saved: {OUTPUT_FILE}")
print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# TASK 9: FEATURE IMPORTANCE INTERPRETATION
# ============================================================================

print("TASK 9: FEATURE IMPORTANCE INTERPRETATION")
print("-" * 80)
print("""
CORRELATION WITH RUL INTERPRETATION:

HIGH POSITIVE CORRELATION (r > 0.3):
- Feature INCREASES as RUL DECREASES (degradation)
- Example: Vibration increases, RUL decreases
- These are direct degradation signals

HIGH NEGATIVE CORRELATION (r < -0.3):
- Feature DECREASES as RUL DECREASES (degradation)
- Example: Efficiency decreases, RUL decreases  
- These are inverse degradation signals

LOW CORRELATION (-0.1 < r < 0.1):
- Feature has weak linear relationship with RUL
- May have non-linear relationship
- Model can still discover patterns

NOTE: Correlation is LINEAR relationship.
Non-linear models (RF, XGBoost, Neural Networks) can discover:
- Polynomial relationships
- Threshold effects
- Interaction effects
- Non-obvious patterns

FEATURE IMPORTANCE TYPES:
1. UNIVARIATE: Correlation of each feature with target (what we calculated)
2. MULTIVARIATE: Importance in context of other features (model-based)
3. PERMUTATION: Importance based on model performance drop

We're using UNIVARIATE (statistical) to avoid data leakage and ML bias.
""")
print()

# ============================================================================
# TASK 10: SUMMARY STATISTICS
# ============================================================================

print("TASK 10: FINAL DATASET SUMMARY STATISTICS")
print("-" * 80)

# Show basic statistics
print(f"Dataset dimensions: {final_data.shape}")
print(f"Memory usage: {final_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print()

# Target variable statistics
print(f"RUL Statistics:")
print(f"  Min: {final_data['RUL'].min():.2f}")
print(f"  Max: {final_data['RUL'].max():.2f}")
print(f"  Mean: {final_data['RUL'].mean():.2f}")
print(f"  Median: {final_data['RUL'].median():.2f}")
print(f"  Std: {final_data['RUL'].std():.2f}")
print()

# Feature statistics
feature_data_final = final_data[features_to_keep]
print(f"Feature Statistics (top 5 highest variance):")
variance_data = pd.DataFrame({
    'Feature': features_to_keep,
    'Variance': [final_data[f].var() for f in features_to_keep]
}).sort_values('Variance', ascending=False)

print(variance_data.head(5).to_string(index=False))
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY - FEATURE SELECTION COMPLETED")
print("=" * 80)
print()

print("WHAT WAS ACCOMPLISHED:")
print("[OK] Calculated correlation of all features with RUL")
print("[OK] Identified highly correlated feature pairs (r > 0.95)")
print("[OK] Removed redundant features (kept more predictive one)")
print("[OK] Analyzed feature importance for interpretation")
print("[OK] Created final lean feature set")
print()

print("FEATURE REDUCTION:")
print(f"  Initial features: {len(feature_cols)}")
print(f"  Features removed: {len(features_to_remove)}")
print(f"  Final features: {len(features_to_keep)}")
print(f"  Reduction: {100*(1 - len(features_to_keep)/len(feature_cols)):.1f}%")
print()

print("FEATURE COMPOSITION:")
print(f"  Original sensors: {len(original_sensors)}")
print(f"  Rolling statistics: {len(rolling_means) + len(rolling_stds)}")
print(f"  Lag features: {len(lag_features)}")
print(f"  Degradation features: {len(delta_features) + len(cumulative_features)}")
print()

print("NEXT STEP:")
print("-> Feature Engineering Part 4: Dataset Validation & Finalization")
print("  (Check missing values, verify dimensions, prepare for modeling)")
print()

print("=" * 80)


