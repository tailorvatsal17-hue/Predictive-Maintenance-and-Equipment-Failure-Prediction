"""
PERFORMANCE EVALUATION PIPELINE
===============================

Evaluates the saved RUL prediction files against the ground truth RUL values.
This stage computes regression metrics and creates comparison visualizations.

Scope:
- Load prediction files
- Load ground truth RUL values
- Verify alignment
- Compute MAE, MSE, RMSE, R2
- Create evaluation plots
- Save summary tables

No retraining, tuning, or feature importance.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(r"C:\Users\Vatsal\OneDrive\Desktop\msc project\data_cleaning")
GROUND_TRUTH_FILE = Path(r"C:\Users\Vatsal\OneDrive\Desktop\msc project\CMAPSSData\RUL_FD001.txt")

PRED_FILES = {
    "Random Forest": BASE_DIR / "04_RUL Prediction" / "random_forest_predictions.csv",
    "XGBoost": BASE_DIR / "04_RUL Prediction" / "xgboost_predictions.csv",
    "Neural Network": BASE_DIR / "04_RUL Prediction" / "neural_network_predictions.csv",
}

OUTPUT_DIR = BASE_DIR / "05_Performance Evaluation" / "performance_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 80)
print("PERFORMANCE EVALUATION")
print("=" * 80)
print()

print("Loading ground truth RUL values...")
y_true = pd.read_csv(GROUND_TRUTH_FILE, header=None, names=["Actual_RUL"])['Actual_RUL'].astype(float)
print(f"Ground truth shape: {y_true.shape}")
print(f"Number of engines in ground truth: {len(y_true)}")
print()

results = []
model_data = {}

for model_name, pred_file in PRED_FILES.items():
    print(f"[{model_name}] Loading predictions from {pred_file.name}")
    pred_df = pd.read_csv(pred_file)

    if pred_df.shape[0] != len(y_true):
        raise ValueError(
            f"{model_name}: prediction rows ({pred_df.shape[0]}) do not match ground truth ({len(y_true)})"
        )

    if "Unit_Number" not in pred_df.columns or "Predicted_RUL" not in pred_df.columns:
        raise ValueError(f"{model_name}: expected columns Unit_Number and Predicted_RUL")

    pred_df = pred_df.sort_values("Unit_Number").reset_index(drop=True)
    if not pred_df["Unit_Number"].tolist() == list(range(1, len(y_true) + 1)):
        raise ValueError(f"{model_name}: Unit_Number values are not aligned 1..100")

    y_pred = pred_df["Predicted_RUL"].astype(float)

    if y_pred.isnull().any() or not np.isfinite(y_pred).all():
        raise ValueError(f"{model_name}: invalid prediction values detected")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    residuals = y_true - y_pred

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    model_data[model_name] = {
        "actual": y_true.to_numpy(),
        "predicted": y_pred.to_numpy(),
        "residuals": residuals.to_numpy(),
    }

    print(f"  Engines aligned: {len(pred_df)}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    print()

results_df = pd.DataFrame(results).sort_values(["MAE", "RMSE"], ascending=[True, True]).reset_index(drop=True)
results_csv = OUTPUT_DIR / "performance_metrics_summary.csv"
results_df.to_csv(results_csv, index=False)

# ---------- Visualization 1: Actual vs Predicted ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, (model_name, data) in zip(axes, model_data.items()):
    actual = data["actual"]
    pred = data["predicted"]
    ax.scatter(actual, pred, alpha=0.8)
    min_v = min(actual.min(), pred.min())
    max_v = max(actual.max(), pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    ax.set_title(model_name)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
fig.suptitle("Actual vs Predicted RUL")
fig.tight_layout()
fig_path = OUTPUT_DIR / "actual_vs_predicted_rul.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Visualization 2: Prediction Error Distribution ----------
fig, ax = plt.subplots(figsize=(10, 6))
for model_name, data in model_data.items():
    sns.kdeplot(data["residuals"], ax=ax, label=model_name, fill=False)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Prediction Error Distribution (Residuals = Actual - Predicted)")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig_path = OUTPUT_DIR / "prediction_error_distribution.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Visualization 3: Residual Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, (model_name, data) in zip(axes, model_data.items()):
    actual = data["actual"]
    residuals = data["residuals"]
    ax.scatter(actual, residuals, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(model_name)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Residual")
fig.suptitle("Residual Plot")
fig.tight_layout()
fig_path = OUTPUT_DIR / "residual_plot.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Visualization 4: Metric Comparison Bar Chart ----------
metric_long = results_df.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R2"], var_name="Metric", value_name="Value")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=metric_long, x="Model", y="Value", hue="Metric", ax=ax)
ax.set_title("Model Comparison: MAE, RMSE, and R2")
ax.set_xlabel("Model")
ax.set_ylabel("Metric Value")
fig.tight_layout()
fig_path = OUTPUT_DIR / "metric_comparison.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Summary report ----------
results_df["Rank"] = results_df["MAE"].rank(method="dense", ascending=True).astype(int)
summary_md = OUTPUT_DIR / "performance_evaluation_summary.md"

best_row = results_df.sort_values(["MAE", "RMSE"], ascending=[True, True]).iloc[0]

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Performance Evaluation Summary\n\n")
    f.write("## Metric Definitions\n\n")
    f.write("- **MAE**: Average absolute prediction error in RUL cycles. Lower is better.\n")
    f.write("- **MSE**: Average squared error. Penalizes larger mistakes more strongly. Lower is better.\n")
    f.write("- **RMSE**: Square root of MSE, in cycles. Easier to interpret than MSE. Lower is better.\n")
    f.write("- **R2**: Variance explained by the model. Higher is better.\n\n")
    f.write("## Results Table\n\n")
    f.write(results_df.to_markdown(index=False))
    f.write("\n\n")
    f.write("## Model Ranking\n\n")
    f.write(results_df.sort_values(["MAE", "RMSE"], ascending=[True, True])[['Model', 'MAE', 'RMSE', 'R2']].to_markdown(index=False))
    f.write("\n\n")
    f.write(f"**Best-performing model:** {best_row['Model']}\n\n")
    f.write("## Dissertation Discussion\n\n")
    f.write("The models demonstrate that machine learning can predict RUL from engineered sensor features with measurable accuracy. \n")
    f.write("The best model is the one with the lowest MAE and RMSE and the highest R2, indicating it captures degradation patterns most effectively. \n")
    f.write("This supports the research objective of using NASA turbofan data for predictive maintenance and answers the research question on prediction accuracy.\n")

print("=" * 80)
print("RESULTS TABLE")
print("=" * 80)
print(results_df[["Model", "MAE", "MSE", "RMSE", "R2"]].to_string(index=False))
print()
print(f"Best-performing model: {best_row['Model']}")
print(f"Summary saved to: {summary_md}")
print(f"Metrics CSV saved to: {results_csv}")
print("Plots saved to:")
for name in ["actual_vs_predicted_rul.png", "prediction_error_distribution.png", "residual_plot.png", "metric_comparison.png"]:
    print(f"  - {OUTPUT_DIR / name}")
print()
print("Ready for the next phase: Feature Importance Analysis")
print("=" * 80)





