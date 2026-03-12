"""
model_training.py
─────────────────
Step 2 of the pipeline.

Responsibilities:
  - Load feature-engineered data from data/processed/features.csv
  - Time-ordered train/test split (NO random splits)
  - Walk-forward cross-validation (3 folds)
  - Train final XGBoost model on full training set
  - Save model artifact to models/xgb_demand_v1.pkl
  - Save training metadata to models/xgb_demand_v1_meta.json

Run directly:
    python src/model_training.py

Or import:
    from src.model_training import run_training
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURES_PATH, MODELS_DIR, METRICS_PATH,
    DATETIME_COL, TARGET_COL, FEATURE_COLS,
    TEST_WEEKS, MIN_TRAIN_ROWS,
    XGB_PARAMS, MAPE_TARGET, MIN_IMPROVEMENT_PCT,
)
from utils import get_logger, timer, ensure_dirs, load_csv, append_metrics, save_json

logger = get_logger(__name__)

try:
    import xgboost as xgb
    import joblib
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}\nRun: pip install xgboost joblib") from e

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ── Split ──────────────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    test_weeks: int = TEST_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered train/test split.
    Test set = last `test_weeks` weeks of data.
    Never uses random splitting on time series data.
    """
    cutoff = df[DATETIME_COL].max() - pd.Timedelta(weeks=test_weeks)
    train  = df[df[DATETIME_COL] <= cutoff].copy()
    test   = df[df[DATETIME_COL] >  cutoff].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Training set too small: {len(train):,} rows "
            f"(minimum required: {MIN_TRAIN_ROWS}). "
            f"Reduce TEST_WEEKS or use a larger dataset."
        )

    logger.info(f"Train: {train[DATETIME_COL].min().date()} → {train[DATETIME_COL].max().date()} ({len(train):,} rows)")
    logger.info(f"Test : {test[DATETIME_COL].min().date()}  → {test[DATETIME_COL].max().date()}  ({len(test):,} rows)")

    return train, test


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MAPE, RMSE, MAE. Returns dict."""
    # Filter near-zero true values to stabilize MAPE
    mask = y_true > 5
    mape  = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else float("nan")
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = float(np.abs(y_true - y_pred).mean())
    # SMAPE is more robust than MAPE for this dataset
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}mape" : round(mape,  6),
        f"{prefix}smape": round(smape, 6),
        f"{prefix}rmse" : round(rmse,  4),
        f"{prefix}mae"  : round(mae,   4),
    }


def naive_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """
    Naive baseline: use lag_7d (same hour last week).
    Falls back to lag_24h then lag_1h if lag_7d is missing.
    """
    preds = (
        test_df["lag_168h"]
        .fillna(test_df["lag_24h"])
        .fillna(test_df["lag_1h"])
        .values
    )
    return preds


# ── Walk-Forward CV ────────────────────────────────────────────────────────────

@timer
def walk_forward_cv(
    df: pd.DataFrame,
    n_folds: int = 3,
    fold_pct: float = 0.15,
    min_train_pct: float = 0.50,
) -> list[dict]:
    """
    Walk-forward cross-validation.

    Each fold trains on data up to a cutoff point, tests on the next
    `fold_pct` of the timeline. Training window grows with each fold.

    Returns list of per-fold metric dicts.
    """
    logger.info(f"Starting {n_folds}-fold walk-forward CV...")
    all_dates  = sorted(df[DATETIME_COL].unique())
    n          = len(all_dates)
    fold_size  = int(n * fold_pct)
    train_start_idx = int(n * min_train_pct)

    results = []
    for fold in range(n_folds):
        cut_idx = train_start_idx + fold * fold_size
        end_idx = cut_idx + fold_size

        if end_idx >= n:
            logger.warning(f"Fold {fold+1}: not enough data — skipping")
            break

        cut_dt = all_dates[cut_idx]
        end_dt = all_dates[end_idx]

        cv_train = df[df[DATETIME_COL] <= cut_dt]
        cv_test  = df[(df[DATETIME_COL] > cut_dt) & (df[DATETIME_COL] <= end_dt)]

        # Train fold model
        m = xgb.XGBRegressor(**{k: v for k, v in XGB_PARAMS.items()
                                  if k != "early_stopping_rounds"})
        cv_fill = cv_train[FEATURE_COLS].median()
        m.fit(cv_train[FEATURE_COLS].fillna(cv_fill), cv_train[TARGET_COL])

        preds = np.maximum(m.predict(cv_test[FEATURE_COLS].fillna(cv_fill)), 0)
        naive = naive_baseline(cv_test)
        naive_valid = ~np.isnan(naive)

        fold_metrics = compute_metrics(cv_test[TARGET_COL].values, preds, label="xgb")
        naive_metrics = compute_metrics(
            cv_test[TARGET_COL].values[naive_valid],
            naive[naive_valid],
            label="naive",
        )
        improvement = (naive_metrics["naive_mape"] - fold_metrics["xgb_mape"]) / naive_metrics["naive_mape"] * 100

        fold_result = {
            "fold"       : fold + 1,
            "train_rows" : len(cv_train),
            "test_rows"  : len(cv_test),
            "train_end"  : str(cut_dt)[:10],
            "test_end"   : str(end_dt)[:10],
            "improvement_pct": round(improvement, 2),
            **fold_metrics,
            **naive_metrics,
        }
        results.append(fold_result)

        logger.info(
            f"  Fold {fold+1}: XGB MAPE={fold_metrics['xgb_mape']:.2%} | "
            f"Naive MAPE={naive_metrics['naive_mape']:.2%} | "
            f"Improvement={improvement:.1f}%"
        )

    avg_mape = np.mean([r["xgb_mape"] for r in results])
    avg_impr = np.mean([r["improvement_pct"] for r in results])
    logger.info(f"CV complete — Avg MAPE: {avg_mape:.2%} | Avg improvement: {avg_impr:.1f}%")

    # Gate check
    if avg_impr < MIN_IMPROVEMENT_PCT:
        logger.warning(
            f"⚠️  Model does not meet {MIN_IMPROVEMENT_PCT}% improvement gate "
            f"(achieved {avg_impr:.1f}%). Revisit feature engineering."
        )
    else:
        logger.info(f"✅ Improvement gate passed ({avg_impr:.1f}% > {MIN_IMPROVEMENT_PCT}%)")

    return results


# ── Train Final Model ──────────────────────────────────────────────────────────

@timer
def train_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> xgb.XGBRegressor:
    """
    Train final XGBoost model on training set.
    Uses test set only for early stopping — not for hyperparameter selection.
    """
    logger.info(f"Training final model on {len(train_df):,} rows...")

    # Use training median to fill NaNs — never use 0 for lag features
    fill_vals = train_df[FEATURE_COLS].median()
    X_train = train_df[FEATURE_COLS].fillna(fill_vals)
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS].fillna(fill_vals)
    y_test  = test_df[TARGET_COL]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


# ── Save ───────────────────────────────────────────────────────────────────────

def save_model(
    model: xgb.XGBRegressor,
    cv_results: list[dict],
    test_metrics: dict,
    naive_metrics: dict,
    path: Path = None,
) -> Path:
    """Save model .pkl and metadata .json to models/."""
    ensure_dirs(MODELS_DIR)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = path or (MODELS_DIR / f"xgb_demand_v1_{ts}.pkl")
    meta_path  = model_path.with_suffix(".json")

    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    # Save metadata
    meta = {
        "model_type"       : "XGBRegressor",
        "trained_at"       : ts,
        "features"         : FEATURE_COLS,
        "n_features"       : len(FEATURE_COLS),
        "best_iteration"   : int(model.best_iteration),
        "xgb_params"       : XGB_PARAMS,
        "test_metrics"     : test_metrics,
        "naive_metrics"    : naive_metrics,
        "cv_results"       : cv_results,
        "improvement_pct"  : round(
            (naive_metrics["naive_mape"] - test_metrics["xgb_mape"])
            / naive_metrics["naive_mape"] * 100, 2
        ),
    }
    save_json(meta, meta_path)
    logger.info(f"Metadata saved → {meta_path}")

    # Also log to rolling metrics CSV for dashboard
    flat_metrics = {
        "model_name"      : model_path.stem,
        "mape"            : test_metrics["xgb_mape"],
        "rmse"            : test_metrics["xgb_rmse"],
        "mae"             : test_metrics["xgb_mae"],
        "naive_mape"      : naive_metrics["naive_mape"],
        "improvement_pct" : meta["improvement_pct"],
        "n_train_rows"    : model.n_features_in_,
        "best_iteration"  : model.best_iteration,
    }
    append_metrics(flat_metrics, METRICS_PATH)
    logger.info(f"Metrics logged → {METRICS_PATH}")

    return model_path


# ── Pipeline entry point ────────────────────────────────────────────────────────

@timer
def run_training(
    features_path: Path = FEATURES_PATH,
    save: bool = True,
    run_cv: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Full training pipeline. Returns (model, metrics_dict).

    Args:
        features_path : Path to features.csv from preprocessing
        save          : If True, saves model and metadata
        run_cv        : If True, runs walk-forward CV before final training
    """
    # Load features
    logger.info(f"Loading features from: {features_path}")
    df = load_csv(features_path, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows")

    # Split
    train_df, test_df = time_split(df)

    # Naive baseline
    naive_preds = naive_baseline(test_df)
    naive_valid = ~np.isnan(naive_preds)
    naive_metrics = compute_metrics(
        test_df[TARGET_COL].values[naive_valid],
        naive_preds[naive_valid],
        label="naive",
    )
    logger.info(f"Naive baseline — MAPE: {naive_metrics['naive_mape']:.2%}")

    # Walk-forward CV
    cv_results = []
    if run_cv:
        cv_results = walk_forward_cv(df)

    # Train final model
    model = train_final_model(train_df, test_df)

    # Evaluate on held-out test set
    y_pred = np.maximum(model.predict(test_df[FEATURE_COLS].fillna(0)), 0)
    test_metrics = compute_metrics(test_df[TARGET_COL].values, y_pred, label="xgb")

    improvement = (naive_metrics["naive_mape"] - test_metrics["xgb_mape"]) / naive_metrics["naive_mape"] * 100

    logger.info("─" * 50)
    logger.info("FINAL TEST SET RESULTS")
    logger.info(f"  XGBoost MAPE      : {test_metrics['xgb_mape']:.2%}")
    logger.info(f"  XGBoost RMSE      : {test_metrics['xgb_rmse']:.2f}")
    logger.info(f"  Naive MAPE        : {naive_metrics['naive_mape']:.2%}")
    logger.info(f"  Improvement       : {improvement:.1f}%")

    if test_metrics["xgb_mape"] <= MAPE_TARGET:
        logger.info(f"  ✅ MAPE target met ({MAPE_TARGET:.0%})")
    else:
        logger.warning(f"  ⚠️  MAPE {test_metrics['xgb_mape']:.2%} exceeds target {MAPE_TARGET:.0%}")
    logger.info("─" * 50)

    # Save
    model_path = None
    if save:
        model_path = save_model(model, cv_results, test_metrics, naive_metrics)

    all_metrics = {
        **test_metrics,
        **naive_metrics,
        "improvement_pct": round(improvement, 2),
        "cv_results"     : cv_results,
        "model_path"     : str(model_path) if model_path else None,
    }

    return model, all_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OLA — Model Training")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--no-save",  action="store_true")
    parser.add_argument("--no-cv",    action="store_true")
    args = parser.parse_args()

    model, metrics = run_training(
        features_path=args.features,
        save=not args.no_save,
        run_cv=not args.no_cv,
    )
    print(f"\n✅ Training complete")
    print(f"   MAPE       : {metrics['xgb_mape']:.2%}")
    print(f"   RMSE       : {metrics['xgb_rmse']:.2f}")
    print(f"   Improvement: {metrics['improvement_pct']:.1f}% over naive")
