"""
model_evaluation.py
───────────────────
Step 3 of the pipeline.

Responsibilities:
  - Load trained model + feature data
  - Produce evaluation plots (actual vs predicted, residuals, feature importance)
  - Run surge recommendation engine
  - Save evaluation report and plots to reports/

Run directly:
    python src/model_evaluation.py --model models/xgb_demand_v1_<timestamp>.pkl

Or import:
    from src.model_evaluation import run_evaluation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURES_PATH, MODELS_DIR, REPORTS_DIR,
    DATETIME_COL, TARGET_COL, FEATURE_COLS,
    TEST_WEEKS, SURGE_CAP, SURGE_STEPS,
    MAPE_TARGET, MIN_IMPROVEMENT_PCT,
)
from utils import get_logger, timer, ensure_dirs, load_csv, save_csv, save_json

logger = get_logger(__name__)

try:
    import joblib
    import xgboost as xgb
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}\nRun: pip install xgboost joblib") from e

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ── Load ───────────────────────────────────────────────────────────────────────

def load_model(model_path: Path) -> xgb.XGBRegressor:
    """Load saved model from .pkl file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from: {model_path}")
    return model


def load_test_data(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load features and return test split only (last TEST_WEEKS weeks)."""
    df = load_csv(features_path, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    cutoff = df[DATETIME_COL].max() - pd.Timedelta(weeks=TEST_WEEKS)
    test_df = df[df[DATETIME_COL] > cutoff].copy()
    logger.info(f"Test set: {len(test_df):,} rows "
                f"({test_df[DATETIME_COL].min().date()} → {test_df[DATETIME_COL].max().date()})")
    return test_df


# ── Predictions ────────────────────────────────────────────────────────────────

def generate_predictions(
    model: xgb.XGBRegressor,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add prediction columns to test_df. Returns copy."""
    df = test_df.copy()
    fill_vals = df[FEATURE_COLS].median()
    df["pred_xgb"]   = np.maximum(model.predict(df[FEATURE_COLS].fillna(fill_vals)), 0)
    df["pred_naive"]  = df["lag_168h"].fillna(df["lag_24h"]).fillna(df["lag_1h"])
    df["residual"]    = df[TARGET_COL] - df["pred_xgb"]
    df["abs_error"]   = df["residual"].abs()
    df["pct_error"]   = df["abs_error"] / df[TARGET_COL].clip(lower=1) * 100
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_full_metrics(df: pd.DataFrame) -> dict:
    """Compute all evaluation metrics. Returns dict."""
    valid_naive = df["pred_naive"].notna()

    xgb_mape   = mean_absolute_percentage_error(df[TARGET_COL], df["pred_xgb"])
    xgb_rmse   = np.sqrt(mean_squared_error(df[TARGET_COL], df["pred_xgb"]))
    xgb_mae    = df["abs_error"].mean()
    naive_mape = mean_absolute_percentage_error(
        df.loc[valid_naive, TARGET_COL], df.loc[valid_naive, "pred_naive"])
    improvement = (naive_mape - xgb_mape) / naive_mape * 100

    # Per-hour MAPE breakdown
    hour_mape = (df.groupby("hour")
                   .apply(lambda g: mean_absolute_percentage_error(g[TARGET_COL], g["pred_xgb"]))
                   .rename("mape_by_hour"))

    # Per-weather MAPE breakdown
    weather_mape = (df.groupby("weather")
                      .apply(lambda g: mean_absolute_percentage_error(g[TARGET_COL], g["pred_xgb"]))
                      .rename("mape_by_weather"))

    metrics = {
        "xgb_mape"        : round(xgb_mape, 6),
        "xgb_rmse"        : round(xgb_rmse, 4),
        "xgb_mae"         : round(xgb_mae,  4),
        "naive_mape"      : round(naive_mape, 6),
        "improvement_pct" : round(improvement, 2),
        "mape_target_met" : xgb_mape <= MAPE_TARGET,
        "gate_passed"     : improvement >= MIN_IMPROVEMENT_PCT,
        "mape_by_hour"    : hour_mape.round(4).to_dict(),
        "mape_by_weather" : weather_mape.round(4).to_dict(),
    }

    # Log summary
    logger.info("─" * 50)
    logger.info("EVALUATION METRICS")
    logger.info(f"  XGBoost MAPE  : {xgb_mape:.2%}")
    logger.info(f"  XGBoost RMSE  : {xgb_rmse:.2f}")
    logger.info(f"  XGBoost MAE   : {xgb_mae:.2f}")
    logger.info(f"  Naive MAPE    : {naive_mape:.2%}")
    logger.info(f"  Improvement   : {improvement:.1f}%")
    logger.info(f"  MAPE target   : {'✅ MET' if metrics['mape_target_met'] else '❌ NOT MET'} ({MAPE_TARGET:.0%})")
    logger.info(f"  20% gate      : {'✅ PASSED' if metrics['gate_passed'] else '❌ FAILED'}")
    logger.info("─" * 50)

    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

@timer
def save_evaluation_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate and save all evaluation plots. Returns list of saved paths."""
    ensure_dirs(output_dir)
    saved = []

    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False

    # ── Plot 1: Actual vs Predicted (last 7 days) ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    sample = df.tail(7 * 24)

    axes[0].plot(sample[DATETIME_COL], sample[TARGET_COL],
                 label="Actual", color="steelblue", alpha=0.8, linewidth=1)
    axes[0].plot(sample[DATETIME_COL], sample["pred_xgb"],
                 label="XGBoost", color="coral", linewidth=1.5, linestyle="--")
    axes[0].plot(sample[DATETIME_COL], sample["pred_naive"],
                 label="Naive (lag_7d)", color="gray", linewidth=1, linestyle=":")
    axes[0].set(title="Actual vs Predicted — Last 7 Days of Test Set",
                ylabel="Rides/Hour")
    axes[0].legend()
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    axes[1].scatter(df["pred_xgb"], df["residual"],
                    alpha=0.3, s=8, color="steelblue")
    axes[1].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set(title="Residuals vs Predicted (should scatter randomly around 0)",
                xlabel="Predicted Rides", ylabel="Residual")

    p = output_dir / "01_actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Saved → {p}")

    # ── Plot 2: MAPE by hour of day ──
    fig, ax = plt.subplots(figsize=(12, 5))
    hour_mape = (df.groupby("hour")
                   .apply(lambda g: mean_absolute_percentage_error(g[TARGET_COL], g["pred_xgb"])))
    ax.bar(hour_mape.index, hour_mape.values * 100,
           color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(MAPE_TARGET * 100, color="red", linestyle="--",
               linewidth=1.5, label=f"Target ({MAPE_TARGET:.0%})")
    ax.set(title="MAPE by Hour of Day", xlabel="Hour", ylabel="MAPE %")
    ax.legend()

    p = output_dir / "02_mape_by_hour.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Plot 3: Feature importance ──
    try:
        from model_training import load_model as _lm  # avoid circular import
    except ImportError:
        pass

    feat_imp = pd.Series(
        model.feature_importances_,   # 'model' injected via closure — see run_evaluation
        index=FEATURE_COLS
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["coral" if any(x in f for x in ["lag", "roll"]) else "steelblue"
              for f in feat_imp.index]
    feat_imp.plot.barh(ax=ax, color=colors, alpha=0.85, edgecolor="white")
    ax.set(title="Feature Importance\n(orange = lag/rolling features)", xlabel="Score")
    ax.axvline(feat_imp.median(), color="gray", linestyle="--", alpha=0.5)

    p = output_dir / "03_feature_importance.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    logger.info(f"Saved {len(saved)} plots to {output_dir}")

    return saved


# ── Surge Engine ───────────────────────────────────────────────────────────────

def compute_surge_multiplier(demand: float, supply: float) -> float:
    """
    Step-function surge multiplier from demand/supply ratio.
    Thresholds defined in config.SURGE_STEPS.
    """
    ratio = demand / max(supply, 1.0)
    for threshold, multiplier in SURGE_STEPS:
        if ratio < threshold:
            return multiplier
    return SURGE_CAP


@timer
def generate_surge_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add surge recommendation columns to predictions DataFrame.
    Supply is simulated as smoothed demand * noise factor.
    In production: replace simulated_supply with real GPS driver count per zone.
    """
    df = df.copy()

    # Simulate supply (replace with real driver availability data in production)
    rng = np.random.default_rng(seed=42)
    df["simulated_supply"] = (
        df["roll_24h_mean"] * rng.uniform(0.7, 1.2, len(df))
    ).clip(lower=1).fillna(df["pred_xgb"] * 0.9)

    df["demand_supply_ratio"] = df["pred_xgb"] / df["simulated_supply"].clip(lower=1)
    df["recommended_surge"]   = df.apply(
        lambda r: compute_surge_multiplier(r["pred_xgb"], r["simulated_supply"]), axis=1
    )
    df["surge_revenue_uplift"] = (df["recommended_surge"] - 1.0) * df["pred_xgb"]

    surge_rate = (df["recommended_surge"] > 1.0).mean()
    avg_surge  = df.loc[df["recommended_surge"] > 1.0, "recommended_surge"].mean()
    logger.info(f"Surge triggered: {surge_rate:.1%} of hours | "
                f"Avg multiplier (when active): {avg_surge:.2f}x")

    return df


# ── Pipeline entry point ────────────────────────────────────────────────────────

@timer
def run_evaluation(
    model_path: Path,
    features_path: Path = FEATURES_PATH,
    save: bool = True,
) -> dict:
    """
    Full evaluation pipeline.
    Returns metrics dict.
    """
    global model   # needed for feature importance plot closure
    ensure_dirs(REPORTS_DIR)

    # Load
    model     = load_model(model_path)
    test_df   = load_test_data(features_path)

    # Predict
    results_df = generate_predictions(model, test_df)

    # Metrics
    metrics = compute_full_metrics(results_df)

    # Surge recommendations
    results_df = generate_surge_recommendations(results_df)

    if save:
        # Save prediction + surge table
        pred_cols = [
            DATETIME_COL, TARGET_COL, "pred_xgb", "pred_naive",
            "residual", "abs_error", "pct_error",
            "recommended_surge", "demand_supply_ratio", "surge_revenue_uplift",
            "hour", "dow", "season", "weather", "temp",
        ]
        out_path = REPORTS_DIR / "predictions.csv"
        save_csv(results_df[[c for c in pred_cols if c in results_df.columns]], out_path)
        logger.info(f"Predictions saved → {out_path}")

        # Save metrics JSON
        metrics_path = REPORTS_DIR / "evaluation_metrics.json"
        save_json(metrics, metrics_path)
        logger.info(f"Metrics saved → {metrics_path}")

        # Save plots
        plots_dir = REPORTS_DIR / "figures"
        save_evaluation_plots(results_df, plots_dir)

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OLA — Model Evaluation")
    parser.add_argument(
        "--model",
        type=Path,
        required=False,
        default=None,
        help="Path to trained model .pkl (defaults to latest in models/)",
    )
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    # Auto-detect latest model if not specified
    if args.model is None:
        candidates = sorted(MODELS_DIR.glob("xgb_demand_v1_*.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"No model found in {MODELS_DIR}. Run model_training.py first."
            )
        args.model = candidates[-1]
        logger.info(f"Auto-detected model: {args.model}")

    metrics = run_evaluation(
        model_path=args.model,
        features_path=args.features,
        save=not args.no_save,
    )
    print(f"\n✅ Evaluation complete")
    print(f"   MAPE       : {metrics['xgb_mape']:.2%}")
    print(f"   Improvement: {metrics['improvement_pct']:.1f}%")
    print(f"   MAPE target: {'✅ MET' if metrics['mape_target_met'] else '❌ NOT MET'}")
