"""Metric definitions, ranking and status logic (no UI dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def asymmetric_score(y_true, y_pred) -> float:
    """PHM08-style score: late predictions (overestimated RUL) are penalised more steeply."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "AsymScore": asymmetric_score(y, p),
    }


def pred_column(model_name: str) -> str:
    """Column name in results/test_predictions.csv, e.g. 'pred_CNN-LSTM_tuned'."""
    return "pred_" + model_name.replace(" ", "_").replace("(", "").replace(")", "")


RANK_OPTIONS = {
    "R² (higher is better)": ("R2", False),
    "RMSE (lower is better)": ("RMSE", True),
    "MAE (lower is better)": ("MAE", True),
    "Asymmetric score (lower is better)": ("AsymScore", True),
}


def best_model(df: pd.DataFrame, rank_label: str) -> tuple[str | None, float | None]:
    if df.empty:
        return None, None
    col, lower_better = RANK_OPTIONS[rank_label]
    idx = df[col].idxmin() if lower_better else df[col].idxmax()
    return df.loc[idx, "Model"], float(df.loc[idx, col])


def rank_position(df: pd.DataFrame, model: str, rank_label: str) -> int:
    col, lower_better = RANK_OPTIONS[rank_label]
    ordered = df.sort_values(col, ascending=lower_better)["Model"].tolist()
    return ordered.index(model) + 1


def status_for(rul: float, critical_below: int, warning_below: int) -> str:
    if rul < critical_below:
        return "Critical"
    if rul < warning_below:
        return "Warning"
    return "Healthy"


# Fallback if results/model_comparison.csv is missing: values from the uploaded run (seed-averaged)
DEFAULT_RESULTS = pd.DataFrame([
    {"Model": "LSTM (tuned)", "R2": 0.9556140700976054, "R2_std": 0.0024652767033704727, "RMSE": 7.751079567614709, "RMSE_std": 0.2140810432157346, "MAE": 3.6204299132029214, "AsymScore": 26827.36021606416},
    {"Model": "CNN-LSTM (tuned)", "R2": 0.9520360827445984, "R2_std": 0.004633893029045963, "RMSE": 8.053256112110446, "RMSE_std": 0.3879658976219278, "MAE": 3.657411813735962, "AsymScore": 20846.348974590142},
    {"Model": "Random Forest", "R2": 0.9437031932025406, "R2_std": np.nan, "RMSE": 8.731558320104062, "RMSE_std": np.nan, "MAE": 3.7533267578785097, "AsymScore": 26509.270248470846},
    {"Model": "Gradient Boosting", "R2": 0.9260973686330615, "R2_std": np.nan, "RMSE": 10.004141576691849, "RMSE_std": np.nan, "MAE": 4.747338834560242, "AsymScore": 32307.638813130867},
    {"Model": "Decision Tree", "R2": 0.9179862128110869, "R2_std": np.nan, "RMSE": 10.538852144526777, "RMSE_std": np.nan, "MAE": 4.5970439141266395, "AsymScore": 38137.95865216155},
])


def leaders_by_metric(df: pd.DataFrame) -> dict[str, str]:
    """Winner for each metric label, e.g. {'R² (higher is better)': 'LSTM (tuned)', ...}."""
    return {label: best_model(df, label)[0] for label in RANK_OPTIONS}


def status_for(rul: float, critical_below: int, warning_below: int) -> str:
    if rul < critical_below:
        return "Critical"
    if rul < warning_below:
        return "Warning"
    return "Healthy"


# Fallback if results/model_comparison.csv is missing: values from the uploaded executed run
DEFAULT_RESULTS = pd.DataFrame([
    {"Model": "CNN-LSTM (tuned)", "R2": 0.956, "R2_std": 0.007, "RMSE": 7.704, "RMSE_std": 0.609, "MAE": 3.783, "AsymScore": 19600.240},
    {"Model": "LSTM (tuned)", "R2": 0.951, "R2_std": 0.008, "RMSE": 8.138, "RMSE_std": 0.699, "MAE": 3.917, "AsymScore": 27592.253},
    {"Model": "Random Forest", "R2": 0.944, "R2_std": np.nan, "RMSE": 8.732, "RMSE_std": np.nan, "MAE": 3.753, "AsymScore": 26509.270},
    {"Model": "Gradient Boosting", "R2": 0.926, "R2_std": np.nan, "RMSE": 10.004, "RMSE_std": np.nan, "MAE": 4.747, "AsymScore": 32307.639},
    {"Model": "Decision Tree", "R2": 0.918, "R2_std": np.nan, "RMSE": 10.539, "RMSE_std": np.nan, "MAE": 4.597, "AsymScore": 38137.959},
])
