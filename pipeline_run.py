"""
pipeline_run.py — Single-Entrypoint Automated Pipeline Executor
================================================================
Orchestrates the full Downstream Activation Engine pipeline from
synthetic data generation through ingestion, validation, and profiling.

Usage
-----
    python pipeline_run.py              # Run full pipeline
    python pipeline_run.py --phase 1    # Run Phase 1 only (data gen + ingestion)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure the project root is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import ensure_directories, load_config, setup_logger

logger = setup_logger("pipeline")


def run_phase_1(config: Dict[str, Any]) -> None:
    """
    Phase 1: Synthetic Data Generation + Ingestion & Validation.

    Steps
    -----
    1. Generate all 4 raw datasets (CSV + JSON)
    2. Ingest and validate schemas
    3. Profile data quality
    4. Generate data dictionary
    """
    from src.synthetic_data import generate_all_datasets
    from src.ingestion import ingest_all_datasets

    logger.info("=" * 60)
    logger.info("PHASE 1: Data Synthesis, Ingestion & Quality Profiling")
    logger.info("=" * 60)

    # Step 1: Generate synthetic data
    logger.info("\n▶ Step 1/2: Generating synthetic datasets...")
    t0 = time.time()
    output_paths = generate_all_datasets(config)
    gen_time = time.time() - t0
    logger.info("  Data generation completed in %.2fs", gen_time)

    for name, path in output_paths.items():
        size_kb = path.stat().st_size / 1024
        logger.info("    %-12s → %s (%.1f KB)", name, path.name, size_kb)

    # Step 2: Ingest & validate
    logger.info("\n▶ Step 2/2: Ingesting, validating & profiling datasets...")
    t1 = time.time()
    dataframes, profiles = ingest_all_datasets(config)
    ingest_time = time.time() - t1
    logger.info("  Ingestion completed in %.2fs", ingest_time)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 COMPLETE — Summary")
    logger.info("=" * 60)
    total_rows = sum(len(df) for df in dataframes.values())
    logger.info("  Total datasets   : %d", len(dataframes))
    logger.info("  Total rows       : %s", f"{total_rows:,}")
    logger.info("  Total time       : %.2fs", gen_time + ingest_time)
    for key, df in dataframes.items():
        dup_pct = profiles[key].duplicate_row_pct
        null_cols = sum(1 for cp in profiles[key].columns if cp.null_count > 0)
        logger.info(
            "    %-12s : %6d rows | %.1f%% dupes | %d cols with nulls",
            key, len(df), dup_pct, null_cols,
        )


def main() -> None:
    """CLI entry point with phase selection."""
    parser = argparse.ArgumentParser(
        description="Downstream Activation Engine — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific phase (1=Data Gen/Ingestion, 2=Clean/Join/Features, 3=SQL/Dashboard). "
             "Default: run all available phases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data (default: 42).",
    )
    args = parser.parse_args()

    config = load_config(overrides={"random_seed": args.seed})
    ensure_directories(config)

    logger.info("Downstream Activation Engine — Pipeline Run")
    logger.info("  Phase   : %s", args.phase or "ALL")
    logger.info("  Seed    : %d", args.seed)
    logger.info("  Raw dir : %s", config["data_raw_dir"])

    try:
        if args.phase is None or args.phase == 1:
            run_phase_1(config)

        # Phases 2 and 3 will be added in subsequent implementations
        if args.phase and args.phase > 1:
            logger.warning("Phase %d is not yet implemented.", args.phase)

        logger.info("\n🎉 Pipeline run completed successfully!")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
