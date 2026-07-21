"""
=====================================================================================
DATA CLEANING & PREPROCESSING FOR NASA CMAPSS TURBOFAN ENGINE DATASET (FD001)
Stage 4: Descriptive Statistics and Distribution Visualization
=====================================================================================

OBJECTIVE:
This script handles Steps 9-10 of the data cleaning process:
9. Explore basic descriptive statistics
10. Visualize feature distributions where appropriate

This step provides deeper understanding of data characteristics.
=====================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

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

# =====================================================================================
# STEP 9: EXPLORE BASIC DESCRIPTIVE STATISTICS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 9: EXPLORING BASIC DESCRIPTIVE STATISTICS")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. CENTRAL TENDENCY: Understand average and median values
   - Identifies if data is symmetric or skewed
   - Reveals typical sensor operating range
   
2. SPREAD: Understand data variability
   - Standard deviation shows how much values vary
   - Min/max show data range and potential outliers
   
3. DISTRIBUTION SHAPE: 
   - Skewness indicates asymmetry (positive/negative skew)
   - Kurtosis indicates tail heaviness (normal vs heavy-tailed)
   
4. FOR PREDICTIVE MAINTENANCE:
   - Sensor ranges indicate healthy vs degraded states
   - High variability suggests sensor noise or real degradation
   - Skewed distributions may indicate rare failure conditions
""")

# Calculate descriptive statistics for training data
print("\nTRAINING DATA - DESCRIPTIVE STATISTICS:")
print("-" * 85)

train_stats = train_data.describe()
print(train_stats.to_string())

# Additional statistics (skewness, kurtosis)
print("\n\nADDITIONAL STATISTICS - TRAINING DATA:")
print("-" * 85)

additional_stats = pd.DataFrame({
    'Column': train_data.columns,
    'Skewness': train_data.skew(),
    'Kurtosis': train_data.kurtosis(),
    'CV (Coeff. of Variation)': (train_data.std() / train_data.mean()).abs()
})

print(additional_stats.to_string())

# Calculate descriptive statistics for test data
print("\n\nTEST DATA - DESCRIPTIVE STATISTICS:")
print("-" * 85)

test_stats = test_data.describe()
print(test_stats.to_string())

# Additional statistics (skewness, kurtosis) for test data
print("\n\nADDITIONAL STATISTICS - TEST DATA:")
print("-" * 85)

test_additional_stats = pd.DataFrame({
    'Column': test_data.columns,
    'Skewness': test_data.skew(),
    'Kurtosis': test_data.kurtosis(),
    'CV (Coeff. of Variation)': (test_data.std() / test_data.mean()).abs()
})

print(test_additional_stats.to_string())

# RUL descriptive statistics
print("\n\nRUL GROUND TRUTH - DESCRIPTIVE STATISTICS:")
print("-" * 85)

rul_stats = rul_data.describe()
print(rul_stats.to_string())

print(f"\nRUL Range: {rul_data['RUL_Actual'].min()} to {rul_data['RUL_Actual'].max()} cycles")
print(f"Mean RUL: {rul_data['RUL_Actual'].mean():.2f} cycles")
print(f"Median RUL: {rul_data['RUL_Actual'].median():.2f} cycles")

# =====================================================================================
# INTERPRETATION OF STATISTICS
# =====================================================================================
print("\n" + "=" * 85)
print("INTERPRETATION OF DESCRIPTIVE STATISTICS")
print("=" * 85)

print("""
KEY INSIGHTS:

Unit_Number & Time_Cycles:
• Unit_Number: Identifier, ranges from 1-100 (100 engines in training)
• Time_Cycles: Starts at 1, varies by engine (depends on failure point)

Operational Settings:
• Op_Setting_1: Constant value (sea level condition in FD001)
• Op_Setting_2: Constant value (no varying conditions)
• Op_Setting_3: Constant value (FD001 has single operating condition)
• ➜ CONFIRM: These should be REMOVED (zero variance)

Sensor Measurements:
• Shows variability across sensors and time
• Different sensors have different ranges and scales
• ➜ ACTION: Need scaling/normalization later

Skewness:
• Positive skew: Right tail (some high outliers possible)
• Negative skew: Left tail (some low outliers possible)
• Near-zero skew: Symmetric distribution

Coefficient of Variation (CV):
• High CV: High variability relative to mean (high noise or range)
• Low CV: Low variability (relatively stable signals)
• Helps identify which sensors are more volatile
""")

# =====================================================================================
# STEP 10: VISUALIZE FEATURE DISTRIBUTIONS
# =====================================================================================
print("\n" + "=" * 85)
print("STEP 10: VISUALIZING FEATURE DISTRIBUTIONS")
print("=" * 85)

# WHY THIS STEP IS NECESSARY:
print("\nWHY THIS STEP IS NECESSARY:")
print("-" * 85)
print("""
1. VISUAL INSPECTION: Patterns not visible in numbers
   - Histogram shape reveals distribution type
   - Box plots show outliers and quartiles
   - Density plots show multi-modal distributions
   
2. OUTLIER DETECTION:
   - Visual identification of extreme values
   - Understand outlier magnitude and frequency
   
3. DATA QUALITY ISSUES:
   - Bimodal distributions may suggest data from two sources
   - Gaps in data reveal data collection issues
   - Artifacts reveal measurement errors
   
4. PREPROCESSING DECISIONS:
   - Distribution shape affects choice of scaling method
   - Outliers may need special handling
   - Normality affects some ML algorithms
""")

# Create visualization directory
viz_dir = os.path.join(data_path, "visualizations")
os.makedirs(viz_dir, exist_ok=True)

# =====================================================================================
# VISUALIZATION 1: Distribution of Sensor Columns (Training Data)
# =====================================================================================
print("\nGenerating visualizations...")
print("-" * 85)

# Select sensor columns only (exclude identifier and time)
sensor_cols = [col for col in train_data.columns if col.startswith('Sensor_')]

# Create histograms for all sensors
fig, axes = plt.subplots(7, 3, figsize=(18, 20))
fig.suptitle('Training Data - Sensor Distribution Histograms', fontsize=16, y=0.995)

for idx, col in enumerate(sensor_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    ax.hist(train_data[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'{col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics to plot
    mean_val = train_data[col].mean()
    std_val = train_data[col].std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
               label=f'±1 Std: {std_val:.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sensor_distributions_histogram.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: sensor_distributions_histogram.png")
plt.close()

# =====================================================================================
# VISUALIZATION 2: Box Plots for Outlier Detection
# =====================================================================================
fig, axes = plt.subplots(7, 3, figsize=(18, 20))
fig.suptitle('Training Data - Sensor Box Plots (Outlier Detection)', fontsize=16, y=0.995)

for idx, col in enumerate(sensor_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    box_plot = ax.boxplot(train_data[col], vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax.set_title(f'{col}', fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate outliers using IQR method
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = len(train_data[(train_data[col] < Q1 - 1.5*IQR) | (train_data[col] > Q3 + 1.5*IQR)])
    ax.text(1.15, train_data[col].max(), f'Outliers: {outliers}', fontsize=8, va='top')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sensor_boxplots.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: sensor_boxplots.png")
plt.close()

# =====================================================================================
# VISUALIZATION 3: Density Plots
# =====================================================================================
fig, axes = plt.subplots(7, 3, figsize=(18, 20))
fig.suptitle('Training Data - Sensor Density Plots', fontsize=16, y=0.995)

for idx, col in enumerate(sensor_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    train_data[col].plot(kind='density', ax=ax, color='steelblue', linewidth=2)
    ax.set_title(f'{col}', fontweight='bold')
    ax.set_xlabel('Value')
    ax.grid(True, alpha=0.3)
    ax.fill_between(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata(), alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sensor_density_plots.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: sensor_density_plots.png")
plt.close()

# =====================================================================================
# VISUALIZATION 4: RUL Distribution
# =====================================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('RUL Ground Truth Distribution (Test Data)', fontsize=14, fontweight='bold')

# Histogram
axes[0].hist(rul_data['RUL_Actual'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0].set_title('Histogram')
axes[0].set_xlabel('RUL (Cycles)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(rul_data['RUL_Actual'].mean(), color='red', linestyle='--', 
                label=f'Mean: {rul_data["RUL_Actual"].mean():.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(rul_data['RUL_Actual'], vert=True, patch_artist=True)
axes[1].set_title('Box Plot')
axes[1].set_ylabel('RUL (Cycles)')
axes[1].grid(True, alpha=0.3, axis='y')

# Density plot
rul_data['RUL_Actual'].plot(kind='density', ax=axes[2], color='green', linewidth=2)
axes[2].set_title('Density Plot')
axes[2].set_xlabel('RUL (Cycles)')
axes[2].fill_between(axes[2].get_lines()[0].get_xdata(), axes[2].get_lines()[0].get_ydata(), alpha=0.3)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'rul_distribution.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: rul_distribution.png")
plt.close()

# =====================================================================================
# VISUALIZATION 5: Sensor Values Over Time (Sample Engine)
# =====================================================================================
fig, axes = plt.subplots(7, 3, figsize=(18, 20))
fig.suptitle('Training Data - Sensor Trajectories Over Time (Engine 1)', fontsize=16, y=0.995)

engine_1_data = train_data[train_data['Unit_Number'] == 1]

for idx, col in enumerate(sensor_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    ax.plot(engine_1_data['Time_Cycles'], engine_1_data[col], linewidth=1.5, color='steelblue')
    ax.set_title(f'{col}')
    ax.set_xlabel('Time (Cycles)')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    ax.fill_between(engine_1_data['Time_Cycles'], engine_1_data[col], alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'sensor_trajectories_engine1.png'), dpi=300, bbox_inches='tight')
print("[OK] Saved: sensor_trajectories_engine1.png")
plt.close()

# =====================================================================================
# SUMMARY
# =====================================================================================
print("\n" + "=" * 85)
print("SUMMARY: STEPS 9-10 COMPLETED")
print("=" * 85)

print(f"""
[OK] DESCRIPTIVE STATISTICS:
  - Calculated mean, median, std, min, max for all columns
  - Identified skewness and kurtosis
  - Computed coefficient of variation
  
[OK] DATA DISTRIBUTIONS:
  - Sensor distributions show variability
  - RUL distribution shows expected range
  - Visualized distributions for all features
  
[OK] VISUALIZATIONS GENERATED:
  - sensor_distributions_histogram.png
  - sensor_boxplots.png (for outlier detection)
  - sensor_density_plots.png
  - rul_distribution.png
  - sensor_trajectories_engine1.png
  
[OK] FILES SAVED in: {viz_dir}

NEXT STEP: Outlier detection and analysis
""")

# Save statistics to CSV
train_stats.to_csv(os.path.join(data_path, 'training_descriptive_stats.csv'))
additional_stats.to_csv(os.path.join(data_path, 'training_additional_stats.csv'), index=False)

print("\n[OK] Statistics saved to CSV files")


