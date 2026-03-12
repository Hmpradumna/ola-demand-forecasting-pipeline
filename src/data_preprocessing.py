"""
data_preprocessing.py
─────────────────────
Step 1 of the pipeline.

Responsibilities:
  - Load raw ola.csv
  - Run data quality checks and fix/flag issues
  - Add all temporal, lag, and rolling features
  - Save cleaned feature table to data/processed/features.csv

Run directly:
    python src/data_preprocessing.py

Or import:
    from src.data_preprocessing import run_preprocessing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly from src/ or from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, FEATURES_PATH,
    DATETIME_COL, TARGET_COL,
    TEMP_MIN, TEMP_MAX, HUMIDITY_MIN, HUMIDITY_MAX,
    WINDSPEED_MIN, WINDSPEED_MAX,
    SEASON_MAP, WEATHER_MAP,
    LAG_HOURS, ROLL_WINDOWS, FEATURE_COLS,
    PROC_DIR,
)
from utils import get_logger, timer, ensure_dirs, load_csv, save_csv, validate_columns

logger = get_logger(__name__)


# ── 1. Load ────────────────────────────────────────────────────────────────────

@timer
def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and parse datetime column."""
    logger.info(f"Loading raw data from: {path}")
    df = load_csv(path, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows | {df.shape[1]} cols")
    logger.info(f"Date range: {df[DATETIME_COL].min().date()} → {df[DATETIME_COL].max().date()}")
    return df


# ── 2. Data Quality ────────────────────────────────────────────────────────────

@timer
def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run all DQ checks. Returns (cleaned_df, report_dict).
    Fixes are applied in-place where safe. Rows are never dropped silently.
    """
    report = {"initial_rows": len(df), "fixes": [], "warnings": []}

    required_cols = [DATETIME_COL, "season", "weather", "temp",
                     "humidity", "windspeed", "casual", "registered", TARGET_COL]
    validate_columns(df, required_cols, label="DQ")

    df = df.copy()

    # ── Nulls ──
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Nulls found:\n{null_counts[null_counts > 0]}")
        report["warnings"].append(f"Null values: {null_counts[null_counts > 0].to_dict()}")
    else:
        logger.info("✅ No nulls")

    # ── Duplicates ──
    dups = df.duplicated(DATETIME_COL).sum()
    if dups > 0:
        logger.warning(f"Removing {dups} duplicate timestamps")
        df = df.drop_duplicates(DATETIME_COL, keep="last")
        report["fixes"].append(f"Removed {dups} duplicate timestamps")

    # ── Value range checks ──
    fixes = {
        "temp"      : (TEMP_MIN,      TEMP_MAX),
        "humidity"  : (HUMIDITY_MIN,  HUMIDITY_MAX),
        "windspeed" : (WINDSPEED_MIN, WINDSPEED_MAX),
    }
    for col, (lo, hi) in fixes.items():
        out_of_range = (~df[col].between(lo, hi)).sum()
        if out_of_range > 0:
            df[col] = df[col].clip(lo, hi)
            msg = f"Clipped {out_of_range} out-of-range values in '{col}' to [{lo}, {hi}]"
            logger.warning(msg)
            report["fixes"].append(msg)

    # ── Season / weather encoding ──
    bad_season  = (~df["season"].isin([1, 2, 3, 4])).sum()
    bad_weather = (~df["weather"].isin([1, 2, 3, 4])).sum()
    if bad_season > 0:
        report["warnings"].append(f"{bad_season} rows with unexpected season values")
    if bad_weather > 0:
        report["warnings"].append(f"{bad_weather} rows with unexpected weather values")

    # ── Count sanity: casual + registered == count ──
    mismatch = (df["casual"] + df["registered"] != df[TARGET_COL]).sum()
    if mismatch > 0:
        logger.warning(f"{mismatch} rows where casual+registered ≠ count. Recomputing count.")
        df[TARGET_COL] = df["casual"] + df["registered"]
        report["fixes"].append(f"Recomputed count for {mismatch} rows")

    # ── Missing hourly slots ──
    expected = pd.date_range(df[DATETIME_COL].min(), df[DATETIME_COL].max(), freq="h")
    missing_slots = len(expected.difference(df[DATETIME_COL]))
    if missing_slots > 0:
        report["warnings"].append(f"{missing_slots} missing hourly slots in timeline")
        logger.warning(f"{missing_slots} missing hourly slots — consider interpolation for production")

    report["final_rows"] = len(df)
    report["rows_removed"] = report["initial_rows"] - report["final_rows"]

    logger.info(f"DQ complete — {len(report['fixes'])} fixes, {len(report['warnings'])} warnings")
    return df, report


# ── 3. Feature Engineering ─────────────────────────────────────────────────────

@timer
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all features needed for modeling.
    IMPORTANT: df must be sorted by datetime before calling this.
    All lag/rolling features use shift(1) minimum to prevent leakage.
    """
    logger.info("Building features...")
    fe = df.copy().sort_values(DATETIME_COL).reset_index(drop=True)

    # ── Temporal ──
    fe["hour"]            = fe[DATETIME_COL].dt.hour
    fe["dow"]             = fe[DATETIME_COL].dt.dayofweek   # 0 = Monday
    fe["month"]           = fe[DATETIME_COL].dt.month
    fe["year"]            = fe[DATETIME_COL].dt.year
    fe["is_weekend"]      = fe["dow"].isin([5, 6]).astype(int)
    fe["is_morning_rush"] = fe["hour"].isin([7, 8, 9]).astype(int)
    fe["is_evening_rush"] = fe["hour"].isin([17, 18, 19]).astype(int)
    fe["is_night"]        = fe["hour"].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    # Cyclical encoding — avoids 23→0 hour discontinuity
    fe["hour_sin"] = np.sin(2 * np.pi * fe["hour"] / 24)
    fe["hour_cos"] = np.cos(2 * np.pi * fe["hour"] / 24)
    fe["dow_sin"]  = np.sin(2 * np.pi * fe["dow"]  / 7)
    fe["dow_cos"]  = np.cos(2 * np.pi * fe["dow"]  / 7)

    # ── Lag features ──
    # shift(n) where n = lag in hours (data is hourly, so 1 row = 1 hour)
    for lag in LAG_HOURS:
        fe[f"lag_{lag}h"] = fe[TARGET_COL].shift(lag)

    # ── Rolling features ──
    # shift(1) before rolling — never include current row in the window
    for window in ROLL_WINDOWS:
        rolled = fe[TARGET_COL].shift(1).rolling(window, min_periods=max(6, window // 8))
        fe[f"roll_{window}h_mean"] = rolled.mean()
        fe[f"roll_{window}h_std"]  = rolled.std()

    # ── Leakage check ──
    lag_vals   = fe["lag_1h"].dropna().values
    count_vals = fe[TARGET_COL].iloc[fe["lag_1h"].dropna().index].values
    assert not (lag_vals == count_vals).all(), "❌ Lag leakage detected — check sort order"
    logger.info("✅ Lag leakage check passed")

    # Drop rows where critical lags are unavailable (first 14 days)
    before = len(fe)
    fe = fe.dropna(subset=["lag_168h", "lag_24h", "lag_1h"]).copy()
    dropped = before - len(fe)
    logger.info(f"Dropped {dropped:,} rows (insufficient lag history) — {len(fe):,} remain")

    return fe


# ── 4. Validate feature set ────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> None:
    """Confirm all expected feature columns are present and non-empty."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")

    null_features = df[FEATURE_COLS].isnull().sum()
    null_features = null_features[null_features > 0]
    if len(null_features) > 0:
        logger.warning(f"Null values in features:\n{null_features.to_string()}")

    logger.info(f"✅ Feature validation passed — {len(FEATURE_COLS)} features, {len(df):,} rows")


# ── 5. Pipeline entry point ────────────────────────────────────────────────────

@timer
def run_preprocessing(
    raw_path: Path = RAW_DATA_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns feature-engineered DataFrame.

    Args:
        raw_path : Path to raw ola.csv
        save     : If True, saves cleaned data and features to data/processed/
    """
    ensure_dirs(PROC_DIR)

    # Step 1: Load
    df = load_raw(raw_path)

    # Step 2: Quality checks
    df_clean, dq_report = run_quality_checks(df)
    if save:
        save_csv(df_clean, CLEAN_DATA_PATH)
        logger.info(f"Saved cleaned data → {CLEAN_DATA_PATH}")

    # Log DQ report
    if dq_report["fixes"]:
        logger.info(f"DQ fixes applied ({len(dq_report['fixes'])}):")
        for fix in dq_report["fixes"]:
            logger.info(f"  • {fix}")
    if dq_report["warnings"]:
        logger.warning(f"DQ warnings ({len(dq_report['warnings'])}):")
        for warn in dq_report["warnings"]:
            logger.warning(f"  ⚠ {warn}")

    # Step 3: Feature engineering
    fe_df = build_features(df_clean)

    # Step 4: Validate
    validate_features(fe_df)

    if save:
        save_csv(fe_df, FEATURES_PATH)
        logger.info(f"Saved features → {FEATURES_PATH}")

    logger.info(f"Preprocessing complete — {len(fe_df):,} rows × {len(FEATURE_COLS)} features")
    return fe_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OLA — Data Preprocessing")
    parser.add_argument("--input",  type=Path, default=RAW_DATA_PATH, help="Path to raw ola.csv")
    parser.add_argument("--no-save", action="store_true", help="Skip saving output files")
    args = parser.parse_args()

    df = run_preprocessing(raw_path=args.input, save=not args.no_save)
    print(f"\n✅ Done — {len(df):,} rows ready for training")
    print(f"   Features: {FEATURE_COLS}")
