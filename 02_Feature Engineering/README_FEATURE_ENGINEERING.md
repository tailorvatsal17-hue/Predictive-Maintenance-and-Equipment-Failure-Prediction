# Feature Engineering Pipeline - Complete Guide

**MSc Computing Dissertation: Predictive Maintenance & Equipment Failure Prediction**  
**Dataset:** NASA CMAPSS FD001  
**Status:** ✓ COMPLETE

---

## Quick Start

Run the complete feature engineering pipeline:

```bash
python run_all_feature_engineering.py
```

This executes all 4 parts automatically and generates the final engineered datasets.

---

## Individual Scripts

### Part 1: RUL Creation
```bash
python 01_feature_engineering_rul_creation.py
```
**Output:** `train_FD001_with_rul.csv`
- Creates the RUL target variable
- RUL = max_cycles - current_cycle for each engine
- Enables degradation modeling

### Part 2: Time-Series Features
```bash
python 02_feature_engineering_timeseries.py
```
**Output:** `train_FD001_with_timeseries_features.csv`
- Generates 252 engineered features
- Rolling mean/std (3/5/10 windows)
- Lag features (1/2/3/5 cycles)
- Delta and cumulative features

### Part 3: Feature Selection
```bash
python 03_feature_engineering_selection.py
```
**Output:** `train_FD001_engineered_final.csv`
- Analyzes correlation with RUL
- Removes redundant features (r > 0.95)
- Reduces features from 273 → 212

### Part 4: Validation & Finalization
```bash
python 04_feature_engineering_validation.py
```
**Output:** 
- `train_FD001_engineered.csv` (Final training data)
- `test_FD001_engineered.csv` (Final test data)
- Validation reports and feature lists

---

## Output Files

### Primary Outputs (Ready for Modeling)
- **`train_FD001_engineered.csv`** (49.95 MB)
  - 20,631 samples × 215 columns
  - Contains RUL target variable
  - Ready for model training

- **`test_FD001_engineered.csv`** (32.27 MB)
  - 13,096 samples × 215 columns
  - Contains RUL ground truth
  - Ready for model evaluation

### Documentation
- **`ENGINEERED_FEATURES_LIST.txt`** - Complete feature catalog
- **`feature_importance_statistical.csv`** - Feature-RUL correlations
- **`FEATURE_ENGINEERING_VALIDATION.md`** - Validation report

### Intermediate Files
- `train_FD001_with_rul.csv` - After RUL creation
- `train_FD001_with_timeseries_features.csv` - After feature engineering
- `train_FD001_engineered_final.csv` - After feature selection

---

## Feature Engineering Summary

### Features Created

| Feature Type | Count | Windows/Lags |
|---|---|---|
| Original Sensors | 21 | N/A |
| Rolling Mean | 63 | 3, 5, 10 |
| Rolling Std | 63 | 3, 5, 10 |
| Lag Features | 84 | 1, 2, 3, 5 |
| Delta (Difference) | 21 | N/A |
| Cumulative Degradation | 21 | N/A |
| **Total Engineered** | **252** | - |

### Features Removed
- Highly correlated pairs (r > 0.95): 61 removed
- Final feature set: 212 features

### Final Feature Composition
- Original sensors: 18
- Rolling statistics: 89
- Lag features: 76
- Degradation features: 29
- **Total: 212**

---

## Feature Descriptions

### Rolling Mean Features
**Purpose:** Captures sensor trends over time windows  
**Windows:** 3, 5, 10 cycles  
**Why:** Smooths noise, shows degradation progression

**Example:** `Sensor_5_rolling_mean_5`
- Average of Sensor 5 over last 5 cycles
- Increasing mean = heating trend = degradation

### Rolling Std Features
**Purpose:** Captures sensor variability/instability  
**Windows:** 3, 5, 10 cycles  
**Why:** High variability indicates system erratic behavior

**Example:** `Sensor_10_rolling_std_5`
- Standard deviation of Sensor 10 over 5 cycles
- Increasing std = unstable operation = failure approaching

### Lag Features
**Purpose:** Captures temporal dependencies  
**Lags:** 1, 2, 3, 5 cycles  
**Why:** Current state depends on recent history

**Example:** `Sensor_1_lag_3`
- Sensor 1 value from 3 cycles ago
- Models autoregressive behavior

### Delta Features
**Purpose:** Captures rate of change  
**Definition:** Current cycle - previous cycle  
**Why:** Acceleration indicates failure

**Example:** `Sensor_15_delta`
- Change in Sensor 15 from last cycle
- Large delta = abrupt change = concerning

### Cumulative Degradation
**Purpose:** Captures total accumulated wear  
**Definition:** Sum of absolute deltas  
**Why:** Represents integrated stress over time

**Example:** `Sensor_9_cumulative_delta`
- Total thermal stress accumulated
- Monotonically increasing

---

## Data Quality

### Validation Results
- **Missing values:** 0% (both training and test)
- **Duplicate rows:** 0
- **Infinite values:** 0
- **Feature consistency:** 100% (train/test match)
- **Data types:** All numeric (no type conversion needed)

### Dataset Dimensions
| Dataset | Samples | Features | Engines |
|---|---|---|---|
| Training | 20,631 | 215 | 100 |
| Test | 13,096 | 215 | 100 |
| Total | 33,727 | 215 | 100 |

### RUL Target Statistics
| Metric | Training | Test |
|---|---|---|
| Min | 0 | 7 |
| Max | 361 | 145 |
| Mean | 107.81 | 65.40 |
| Median | 103.00 | 65.50 |
| Std | 68.88 | 38.76 |

---

## Feature Selection Methodology

### Why Remove Highly Correlated Features?
1. **Multicollinearity:** Confuses models about feature importance
2. **Efficiency:** Fewer features = faster training
3. **Stability:** More stable model coefficients
4. **Interpretability:** Cleaner feature importance

### Correlation Threshold: r > 0.95
- Features with |correlation| > 0.95 considered redundant
- Keep feature with higher RUL correlation
- Remove the less predictive one

### Results
- Features with correlation > 0.95: Multiple pairs
- Features removed: 61 (22.3% reduction)
- Final features: 212 (high quality)

---

## Ready for Model Training

### Next Phase: Model Development

The engineered datasets are ready for:

1. **Random Forest Regressor**
   - Non-linear model
   - Feature importance from tree splits
   - Robust to outliers

2. **XGBoost Regressor**
   - Gradient boosting
   - Feature interactions
   - Superior accuracy

3. **Neural Network Regressor**
   - Deep learning
   - Complex pattern discovery
   - Hyperparameter tuning

### Recommended Workflow

1. Load datasets:
   ```python
   train_data = pd.read_csv('train_FD001_engineered.csv')
   test_data = pd.read_csv('test_FD001_engineered.csv')
   
   X_train = train_data.drop(['Unit_Number', 'Time_Cycles', 'RUL'], axis=1)
   y_train = train_data['RUL']
   
   X_test = test_data.drop(['Unit_Number', 'Time_Cycles', 'RUL'], axis=1)
   y_test = test_data['RUL']
   ```

2. Train models (avoid data leakage)
   ```python
   from sklearn.ensemble import RandomForestRegressor
   from xgboost import XGBRegressor
   from sklearn.neural_network import MLPRegressor
   
   # Random Forest
   rf_model = RandomForestRegressor(n_estimators=100, max_depth=20)
   rf_model.fit(X_train, y_train)
   rf_pred = rf_model.predict(X_test)
   
   # XGBoost
   xgb_model = XGBRegressor(n_estimators=100, max_depth=6)
   xgb_model.fit(X_train, y_train)
   xgb_pred = xgb_model.predict(X_test)
   
   # Neural Network
   nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32))
   nn_model.fit(X_train, y_train)
   nn_pred = nn_model.predict(X_test)
   ```

3. Evaluate performance
   ```python
   from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
   
   r2 = r2_score(y_test, predictions)
   mae = mean_absolute_error(y_test, predictions)
   rmse = np.sqrt(mean_squared_error(y_test, predictions))
   mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
   ```

---

## Important Notes

### Feature Engineering Principles
- **Time-series aware:** Groups by Unit_Number before feature creation
- **Data leakage prevention:** Test data scaled with training parameters
- **Consistency:** Test features exactly match training features
- **Interpretability:** Each feature has clear physical meaning

### For Dissertation Methodology
All feature engineering decisions justified for dissertation write-up:
- **Why RUL calculation:** Creates degradation label
- **Why time-series features:** Captures temporal patterns
- **Why multiple windows:** Different time scales matter
- **Why statistical selection:** No model bias, interpretable

### Common Pitfalls to Avoid
1. **Don't retrain scaler on test data** ✗
2. **Don't change features between train/test** ✗
3. **Don't use test set for feature selection** ✗
4. **Don't ignore NaN values from lagging** ✗
5. **Don't forget to group by Unit_Number** ✗

---

## File Locations

```
C:\Users\Vatsal\OneDrive\Desktop\msc project\
├── data_cleaning\
│   ├── train_FD001_engineered.csv          [Main output]
│   ├── test_FD001_engineered.csv           [Main output]
│   ├── ENGINEERED_FEATURES_LIST.txt
│   ├── feature_importance_statistical.csv
│   ├── FEATURE_ENGINEERING_VALIDATION.md
│   ├── 01_feature_engineering_rul_creation.py
│   ├── 02_feature_engineering_timeseries.py
│   ├── 03_feature_engineering_selection.py
│   ├── 04_feature_engineering_validation.py
│   └── run_all_feature_engineering.py
├── FEATURE_ENGINEERING_COMPLETE.md         [Summary]
└── CMAPSSData\                             [Original data]
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

## Troubleshooting

### Common Issues

**Issue:** "FileNotFoundError: train_FD001_scaled.csv"
- **Solution:** Run preprocessing pipeline first
- **Check:** Verify `data_cleaning/train_FD001_scaled.csv` exists

**Issue:** "Memory error during feature engineering"
- **Solution:** Run parts sequentially, delete intermediate files
- **Temporary files:** Remove `train_FD001_with_*.csv` files

**Issue:** "DataFrame columns don't match"
- **Solution:** Verify sensor columns are named `Sensor_1` not `Sensor_01`
- **Check:** Load CSV and print `df.columns[:5]`

**Issue:** "RUL values don't match ground truth"
- **Note:** Training RUL = cycles to failure, Test RUL = cycles at monitoring stop
- **Expected:** Training RUL_min=0, Test RUL_min>0

---

## Summary

✓ **Feature Engineering Completed Successfully**

- RUL target variable created
- 252 time-series features engineered
- 212 high-quality features selected
- Training and test datasets validated
- Ready for model training

**Status:** READY FOR PHASE 4 - MODEL TRAINING

---

**Questions?** Refer to `FEATURE_ENGINEERING_COMPLETE.md` for detailed documentation.

