"""
RUL PREDICTION PIPELINE
=======================

Loads the trained models and generates final Remaining Useful Life (RUL)
predictions for the last operating cycle of each engine in the test dataset.

Scope:
- Load saved models
- Predict RUL for each engine's final observed cycle
- Save one CSV per model
- No evaluation metrics
- No retraining
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning")
MODEL_DIR = BASE_DIR / "03_Model Training" / "models"
TEST_FILE = BASE_DIR / "test_FD001_engineered.csv"
METADATA_FILE = MODEL_DIR / "training_metadata.json"

RF_MODEL_FILE = MODEL_DIR / "random_forest_rul.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost_rul.joblib"
NN_MODEL_FILE = MODEL_DIR / "neural_network_rul.joblib"

OUTPUT_RF = BASE_DIR / "04_RUL Prediction" / "random_forest_predictions.csv"
OUTPUT_XGB = BASE_DIR / "04_RUL Prediction" / "xgboost_predictions.csv"
OUTPUT_NN = BASE_DIR / "04_RUL Prediction" / "neural_network_predictions.csv"

print("=" * 80)
print("RUL PREDICTION PIPELINE")
print("=" * 80)
print()

print("Loading engineered test dataset and training metadata...")
test_df = pd.read_csv(TEST_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
feature_cols = metadata["feature_columns"]

print(f"Test dataset shape: {test_df.shape}")
print(f"Number of engine units: {test_df['Unit_Number'].nunique()}")
print(f"Number of input features: {len(feature_cols)}")
print()

# Keep the last observed cycle for each engine.
# In CMAPSS test data, this is the final available operating point used for prediction.
last_cycle = (
    test_df.sort_values(["Unit_Number", "Time_Cycles"])
          .groupby("Unit_Number", as_index=False)
          .tail(1)
          .reset_index(drop=True)
)

print("Final operating cycle per engine selected for prediction.")
print(f"Rows selected: {len(last_cycle)}")
print(f"Unique engines in prediction set: {last_cycle['Unit_Number'].nunique()}")
print()

# Validate one row per engine
if len(last_cycle) != last_cycle["Unit_Number"].nunique():
    raise ValueError("Prediction set does not contain exactly one row per engine.")

missing_features = [c for c in feature_cols if c not in last_cycle.columns]
if missing_features:
    raise ValueError(f"Missing required features in test set: {missing_features[:10]}")

X_pred = last_cycle[feature_cols].copy()
engine_ids = last_cycle[["Unit_Number", "Time_Cycles"]].copy()

# Model loaders
models = {
    "Random Forest": (joblib.load(RF_MODEL_FILE), OUTPUT_RF),
    "XGBoost": (joblib.load(XGB_MODEL_FILE), OUTPUT_XGB),
    "Neural Network": (joblib.load(NN_MODEL_FILE), OUTPUT_NN),
}

results = {}

for model_name, (model, output_file) in models.items():
    print(f"[{model_name}] Loading model and generating predictions...")
    predictions = model.predict(X_pred)
    predictions = np.asarray(predictions, dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError(f"{model_name} produced invalid prediction values.")

    pred_df = engine_ids.copy()
    pred_df["Predicted_RUL"] = predictions
    pred_df["Model"] = model_name

    # Save only one final prediction per engine.
    pred_df = pred_df[["Unit_Number", "Time_Cycles", "Predicted_RUL"]]
    pred_df.to_csv(output_file, index=False)

    results[model_name] = {
        "output_file": output_file,
        "shape": pred_df.shape,
        "predictions": pred_df,
    }

    print(f"Saved predictions -> {output_file}")
    print(f"Prediction file shape: {pred_df.shape}")
    print("First 20 predicted RUL values:")
    print(pred_df[["Unit_Number", "Predicted_RUL"]].head(20).to_string(index=False))
    print()

print("=" * 80)
print("PREDICTION VALIDATION")
print("=" * 80)
print()

for model_name, info in results.items():
    pred_df = info["predictions"]
    n_engines = pred_df["Unit_Number"].nunique()
    n_rows = len(pred_df)
    missing = pred_df.isnull().sum().sum()
    invalid = (~np.isfinite(pred_df["Predicted_RUL"])).sum()
    duplicates = pred_df["Unit_Number"].duplicated().sum()

    print(f"{model_name}:")
    print(f"  Engines predicted: {n_engines}")
    print(f"  Rows in prediction file: {n_rows}")
    print(f"  Missing values: {missing}")
    print(f"  Invalid numeric values: {invalid}")
    print(f"  Duplicate engine rows: {duplicates}")
    print()

summary_lines = [
    "RUL prediction completed successfully.",
    f"Engines predicted: {last_cycle['Unit_Number'].nunique()}",
    f"Input features used: {len(feature_cols)}",
    f"Random Forest file shape: {results['Random Forest']['shape']}",
    f"XGBoost file shape: {results['XGBoost']['shape']}",
    f"Neural Network file shape: {results['Neural Network']['shape']}",
]

print("=" * 80)
print("SUMMARY")
print("=" * 80)
for line in summary_lines:
    print(line)
print()
print("Prediction files saved for the next phase: Performance Evaluation.")
print("=" * 80)





