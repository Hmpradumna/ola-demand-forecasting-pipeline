"""
utils.py
────────
Shared utilities: logging setup, timer, directory creation, and
result persistence. Import from here — never duplicate these in other modules.
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

import pandas as pd


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to console with a consistent format.
    Call once per module:  logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — don't add duplicate handlers

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── Timer decorator ────────────────────────────────────────────────────────────

def timer(func):
    """Decorator that logs how long a function takes to run."""
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Directory helpers ──────────────────────────────────────────────────────────

def ensure_dirs(*paths: Path) -> None:
    """Create directories if they don't exist. Safe to call repeatedly."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


# ── Data I/O ───────────────────────────────────────────────────────────────────

def load_csv(path: Path, parse_dates: list | None = None) -> pd.DataFrame:
    """Load CSV with basic validation. Raises FileNotFoundError with clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )
    df = pd.read_csv(path, parse_dates=parse_dates or [])
    return df


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    ensure_dirs(path.parent)
    df.to_csv(path, index=index)


# ── Metrics persistence ────────────────────────────────────────────────────────

def append_metrics(metrics: dict, path: Path) -> None:
    """
    Append a metrics dict to a CSV log file.
    Adds a 'run_timestamp' column automatically.
    Creates the file with headers on first run.
    """
    path = Path(path)
    ensure_dirs(path.parent)

    metrics["run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([metrics])

    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)


def save_json(obj: dict, path: Path) -> None:
    """Save dict as formatted JSON."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def print_df_summary(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print a concise summary — shape, dtypes, null counts."""
    logger = get_logger(__name__)
    logger.info(f"{label} — shape: {df.shape}")
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        logger.warning(f"Null counts:\n{nulls[nulls > 0].to_string()}")
    else:
        logger.info("No nulls found")


def validate_columns(df: pd.DataFrame, required: list[str], label: str = "") -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{'[' + label + '] ' if label else ''}Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
