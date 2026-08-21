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


def run_phase_2(config: Dict[str, Any]) -> None:
    """
    Phase 2: Cleaning, Normalisation & Multi-Source Joining.

    Steps
    -----
    1. Ingest raw datasets (re-uses Phase 1 output)
    2. Clean all datasets (dedup, normalise, impute, outlier filter)
    3. Execute multi-stage joins (union, aggregate, campaign+signups, +events)
    """
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets

    logger.info("=" * 60)
    logger.info("PHASE 2: Cleaning, Normalisation & Multi-Source Joining")
    logger.info("=" * 60)

    # Step 1: Ingest
    logger.info("\n>> Step 1/3: Ingesting raw datasets...")
    t0 = time.time()
    dataframes, _ = ingest_all_datasets(config)
    logger.info("  Ingestion completed in %.2fs", time.time() - t0)

    # Step 2: Clean
    logger.info("\n>> Step 2/3: Cleaning & normalising datasets...")
    t1 = time.time()
    cleaned = clean_all_datasets(dataframes)
    clean_time = time.time() - t1
    logger.info("  Cleaning completed in %.2fs", clean_time)

    # Step 3: Join
    logger.info("\n>> Step 3/3: Executing multi-stage joins...")
    t2 = time.time()
    master_table, audits = join_all_datasets(cleaned)
    join_time = time.time() - t2
    logger.info("  Joining completed in %.2fs", join_time)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE -- Summary")
    logger.info("=" * 60)
    logger.info("  Cleaned datasets:")
    for key, df in cleaned.items():
        raw_count = len(dataframes[key])
        logger.info(
            "    %-15s : %6d -> %6d rows (removed %d)",
            key, raw_count, len(df), raw_count - len(df),
        )
    logger.info("  Master table    : %d rows x %d columns", len(master_table), len(master_table.columns))
    logger.info("  Join audits     : %d joins executed, %d total warnings",
                len(audits), sum(len(a.validate()) for a in audits))
    logger.info("  Total time      : %.2fs", clean_time + join_time)


def run_phase_3(config: Dict[str, Any]) -> None:
    """
    Phase 3: Feature Engineering & SQL Layer.

    Steps
    -----
    1. Re-ingest + clean + join (reuses Phase 2 logic)
    2. Compute user-level activation features
    3. Aggregate campaign-level KPIs (CTR, CPAU, Vanity Ratio)
    4. Push to SQLite and create analytical views
    """
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets
    from src.feature_engineering import engineer_features
    from src.sql_layer import build_sql_layer, query_view

    logger.info("=" * 60)
    logger.info("PHASE 3: Feature Engineering & SQL Layer")
    logger.info("=" * 60)

    # Steps 1-3: Rebuild master table
    logger.info("\n>> Step 1/4: Ingesting raw datasets...")
    t0 = time.time()
    dataframes, _ = ingest_all_datasets(config)
    logger.info("  Ingestion completed in %.2fs", time.time() - t0)

    logger.info("\n>> Step 2/4: Cleaning & joining...")
    t1 = time.time()
    cleaned = clean_all_datasets(dataframes)
    master, _ = join_all_datasets(cleaned)
    logger.info("  Clean + join completed in %.2fs", time.time() - t1)

    # Step 3: Feature engineering
    logger.info("\n>> Step 3/4: Computing activation features & KPIs...")
    t2 = time.time()
    campaigns, users = engineer_features(master, config=config)
    fe_time = time.time() - t2
    logger.info("  Feature engineering completed in %.2fs", fe_time)

    # Step 4: SQL Layer
    logger.info("\n>> Step 4/4: Building SQL layer & analytical views...")
    t3 = time.time()
    engine, views = build_sql_layer(campaigns, users, config=config)
    sql_time = time.time() - t3
    logger.info("  SQL layer completed in %.2fs", sql_time)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 COMPLETE -- Summary")
    logger.info("=" * 60)
    logger.info("  Users processed    : %d", len(users))
    logger.info("  Activated (7-day)  : %d (%.1f%%)",
                int(users["is_activated_7d"].sum()),
                users["is_activated_7d"].mean() * 100)
    logger.info("  Campaigns scored   : %d", len(campaigns))

    vanity_count = int(campaigns["is_vanity_trap"].sum())
    logger.info("  Vanity Traps       : %d", vanity_count)
    logger.info("  SQL views created  : %d (%s)", len(views), ", ".join(views))
    logger.info("  Total time         : %.2fs", fe_time + sql_time)

    # Print campaign leaderboard
    logger.info("\n  Campaign Leaderboard:")
    for _, row in campaigns.iterrows():
        flag = " ** VANITY TRAP **" if row["is_vanity_trap"] else ""
        logger.info(
            "    #%-2d %-12s | CTR: %5.2f%% | Activation: %5.1f%% | CPAU: $%8.2f | %s%s",
            row["cpau_rank"], row["campaign_id"], row["ctr_pct"],
            row["activation_rate_pct"], row["cpau_usd"],
            row["performance_category"], flag,
        )


def main() -> None:
    """CLI entry point with phase selection."""
    parser = argparse.ArgumentParser(
        description="Downstream Activation Engine -- Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific phase (1=Data Gen/Ingestion, 2=Clean/Join, 3=Features/SQL). "
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

    logger.info("Downstream Activation Engine -- Pipeline Run")
    logger.info("  Phase   : %s", args.phase or "ALL")
    logger.info("  Seed    : %d", args.seed)
    logger.info("  Raw dir : %s", config["data_raw_dir"])

    try:
        if args.phase is None or args.phase == 1:
            run_phase_1(config)

        if args.phase is None or args.phase == 2:
            run_phase_2(config)

        if args.phase is None or args.phase == 3:
            run_phase_3(config)

        logger.info("\nPipeline run completed successfully!")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
