"""
MODEL TRAINING PIPELINE
=======================

Trains three regression models for Remaining Useful Life (RUL) prediction:
1. Random Forest Regressor
2. XGBoost Regressor
3. Simple Feed-Forward Neural Network (sklearn MLPRegressor)

Scope:
- Train only
- No evaluation metrics
- No predictions on test data
- Save trained models to disk
"""

from __future__ import annotations

from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is required for this training pipeline. Install it before running."
    ) from exc

BASE_DIR = Path(r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning")
TRAIN_FILE = BASE_DIR / "train_FD001_engineered.csv"
MODEL_DIR = BASE_DIR / "03_Model Training" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "RUL"
DROP_COLS = ["Unit_Number", "Time_Cycles", TARGET_COL]
RANDOM_STATE = 42

print("=" * 80)
print("MODEL TRAINING PIPELINE - RUL PREDICTION")
print("=" * 80)
print()

print("Loading engineered training dataset...")
df = pd.read_csv(TRAIN_FILE)
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

print(f"Training data shape: {df.shape}")
print(f"Input features: {len(feature_cols)}")
print(f"Target column: {TARGET_COL}")
print()

# Persist metadata for the next phase
meta = {
    "training_file": str(TRAIN_FILE),
    "n_samples": int(df.shape[0]),
    "n_features": int(len(feature_cols)),
    "feature_columns": feature_cols,
    "target_column": TARGET_COL,
    "random_state": RANDOM_STATE,
}
with open(MODEL_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

results = []

# ---------------------------------------------------------------------------
# Model 1: Random Forest Regressor
# ---------------------------------------------------------------------------
print("[1/3] Training Random Forest Regressor")
print("Why suitable: ensemble of decision trees, handles non-linear degradation patterns, robust to noise.")
print("Parameters: n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
start = time.perf_counter()
rf.fit(X, y)
rf_time = time.perf_counter() - start
rf_path = MODEL_DIR / "random_forest_rul.joblib"
joblib.dump(rf, rf_path)
print(f"Saved Random Forest model -> {rf_path}")
print(f"Training time: {rf_time:.2f} seconds")
print()
results.append(("Random Forest", rf_time, rf_path.name))

# ---------------------------------------------------------------------------
# Model 2: XGBoost Regressor
# ---------------------------------------------------------------------------
print("[2/3] Training XGBoost Regressor")
print("Why suitable: gradient boosting captures complex interactions and gradual degradation trends.")
print("Parameters: n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8")
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    reg_alpha=0.0,
    reg_lambda=1.0,
    verbosity=1,
)
start = time.perf_counter()
xgb.fit(X, y)
xgb_time = time.perf_counter() - start
xgb_path = MODEL_DIR / "xgboost_rul.joblib"
joblib.dump(xgb, xgb_path)
print(f"Saved XGBoost model -> {xgb_path}")
print(f"Training time: {xgb_time:.2f} seconds")
print()
results.append(("XGBoost", xgb_time, xgb_path.name))

# ---------------------------------------------------------------------------
# Model 3: Neural Network Regressor
# ---------------------------------------------------------------------------
print("[3/3] Training Neural Network Regressor")
print("Why suitable: learns non-linear mappings from engineered features to RUL.")
print("Architecture: 3 hidden layers (128, 64, 32), ReLU activations, Adam optimizer")
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=100,
    shuffle=True,
    random_state=RANDOM_STATE,
    verbose=True,
    early_stopping=False,
)
start = time.perf_counter()
mlp.fit(X, y)
mlp_time = time.perf_counter() - start
mlp_path = MODEL_DIR / "neural_network_rul.joblib"
joblib.dump(mlp, mlp_path)
print(f"Saved Neural Network model -> {mlp_path}")
print(f"Training time: {mlp_time:.2f} seconds")
print()
results.append(("Neural Network", mlp_time, mlp_path.name))

# Summary
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Input features used: {len(feature_cols)}")
print()
for name, seconds, file_name in results:
    print(f"{name}: {seconds:.2f} seconds -> {file_name}")
print()
print(f"Models saved in: {MODEL_DIR}")
print("Ready for the next phase: RUL Prediction")
print("=" * 80)


