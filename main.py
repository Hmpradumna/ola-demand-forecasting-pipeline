"""
main.py
───────
Single entry point for the full OLA demand forecasting pipeline.

Usage:
    # Full pipeline (preprocess → train → evaluate)
    python main.py

    # Individual stages
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage evaluate --model models/xgb_demand_v1_<ts>.pkl

    # Skip CV for faster iteration during development
    python main.py --no-cv

    # Dry run (preprocess + train, skip saving files)
    python main.py --no-save
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.config import FEATURES_PATH, MODELS_DIR, REPORTS_DIR
from src.utils  import get_logger, ensure_dirs

logger = get_logger("main")


def run_pipeline(
    stage: str    = "all",
    model_path    = None,
    save: bool    = True,
    run_cv: bool  = True,
) -> None:

    start = time.perf_counter()
    ensure_dirs(MODELS_DIR, REPORTS_DIR)

    # ── Stage 1: Preprocessing ──────────────────────────────────────────────
    fe_df = None
    if stage in ("all", "preprocess"):
        logger.info("=" * 55)
        logger.info("STAGE 1: DATA PREPROCESSING")
        logger.info("=" * 55)
        from src.data_preprocessing import run_preprocessing
        fe_df = run_preprocessing(save=save)

    # ── Stage 2: Training ────────────────────────────────────────────────────
    model, metrics = None, {}
    if stage in ("all", "train"):
        logger.info("=" * 55)
        logger.info("STAGE 2: MODEL TRAINING")
        logger.info("=" * 55)
        from src.model_training import run_training
        model, metrics = run_training(save=save, run_cv=run_cv)

    # ── Stage 3: Evaluation ──────────────────────────────────────────────────
    if stage in ("all", "evaluate"):
        logger.info("=" * 55)
        logger.info("STAGE 3: MODEL EVALUATION")
        logger.info("=" * 55)

        # Auto-detect latest model if not specified
        if model_path is None:
            candidates = sorted(MODELS_DIR.glob("xgb_demand_v1_*.pkl"))
            if not candidates:
                raise FileNotFoundError(
                    "No model found. Run training first:\n"
                    "  python main.py --stage train"
                )
            model_path = candidates[-1]
            logger.info(f"Using model: {model_path.name}")

        from src.model_evaluation import run_evaluation
        eval_metrics = run_evaluation(
            model_path=model_path,
            save=save,
        )
        metrics.update(eval_metrics)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start
    logger.info("=" * 55)
    logger.info(f"PIPELINE COMPLETE — {elapsed:.1f}s")
    if metrics:
        logger.info(f"  MAPE        : {metrics.get('xgb_mape', 'N/A')}")
        logger.info(f"  RMSE        : {metrics.get('xgb_rmse', 'N/A')}")
        logger.info(f"  Improvement : {metrics.get('improvement_pct', 'N/A')}%")
    logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLA Demand Forecasting Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "train", "evaluate"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model .pkl for evaluation stage",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving outputs")
    parser.add_argument("--no-cv",   action="store_true", help="Skip cross-validation")

    args = parser.parse_args()
    run_pipeline(
        stage=args.stage,
        model_path=args.model,
        save=not args.no_save,
        run_cv=not args.no_cv,
    )
