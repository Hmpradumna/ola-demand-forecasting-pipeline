"""
config.py
─────────
Single source of truth for all project constants.
Change values here — nowhere else.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
MODELS_DIR  = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH     = RAW_DIR  / "ola.csv"
CLEAN_DATA_PATH   = PROC_DIR / "cleaned_data.csv"
FEATURES_PATH     = PROC_DIR / "features.csv"
METRICS_PATH      = REPORTS_DIR / "model_performance.csv"

# ── Data schema ────────────────────────────────────────────────────────────────
DATETIME_COL  = "datetime"
TARGET_COL    = "count"
CASUAL_COL    = "casual"
REGISTERED_COL = "registered"

SEASON_MAP  = {1: "Spring", 2: "Summer", 3: "Fall",  4: "Winter"}
WEATHER_MAP = {1: "Clear",  2: "Mist",   3: "Light Rain", 4: "Heavy Rain"}

# ── Coordinate / value bounds ──────────────────────────────────────────────────
TEMP_MIN, TEMP_MAX           = 0.0,  40.0
HUMIDITY_MIN, HUMIDITY_MAX   = 0.0, 100.0
WINDSPEED_MIN, WINDSPEED_MAX = 0.0,  67.0

# ── Feature engineering ────────────────────────────────────────────────────────
FORECAST_HORIZON_HOURS = 1   # 1-hour ahead prediction
LAG_HOURS   = [1, 2, 3, 24, 48, 168, 336]  # 1h, 2h, 3h, 1d, 2d, 7d, 14d
ROLL_WINDOWS = [24, 168]     # 24h and 7d rolling windows

FEATURE_COLS = [
    # Temporal
    "hour", "dow", "month", "year",
    "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Weather
    "season", "weather", "temp", "humidity", "windspeed",
    # Lags
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h", "lag_336h",
    # Rolling
    "roll_24h_mean", "roll_168h_mean", "roll_24h_std", "roll_168h_std",
]

# ── Train / test split ─────────────────────────────────────────────────────────
TEST_WEEKS   = 4       # Last N weeks held out as test set
MIN_TRAIN_ROWS = 500   # Safety floor — abort if training set smaller than this

# ── Model hyperparameters (XGBoost) ────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"       : 500,
    "max_depth"          : 6,
    "learning_rate"      : 0.05,
    "subsample"          : 0.8,
    "colsample_bytree"   : 0.8,
    "min_child_weight"   : 5,
    "reg_lambda"         : 1.0,
    "objective"          : "reg:squarederror",
    "random_state"       : 42,
    "n_jobs"             : -1,
    "verbosity"          : 0,
    "early_stopping_rounds": 30,
}

# ── Surge pricing ──────────────────────────────────────────────────────────────
SURGE_CAP = 2.5
SURGE_STEPS = [
    # (demand/supply ratio threshold, multiplier)
    (1.0, 1.00),
    (1.2, 1.25),
    (1.5, 1.50),
    (2.0, 1.75),
    (2.5, 2.00),
    (float("inf"), SURGE_CAP),
]

# ── Evaluation thresholds ──────────────────────────────────────────────────────
MAPE_TARGET          = 0.15   # 15% — model must beat this
MIN_IMPROVEMENT_PCT  = 20.0   # Must beat naive baseline by 20%
MAX_CV_MAPE_VARIANCE = 0.03   # Max allowed MAPE spread across CV folds
