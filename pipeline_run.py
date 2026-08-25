"""
pipeline_run.py -- Single-Entrypoint Automated Pipeline Executor
==================================================================
Orchestrates the complete Downstream Activation Engine pipeline:

    synthetic_data -> ingest -> clean -> join -> engineer -> sql_export

Supports CLI arguments for sample sizing, database path override,
seed control, phase selection, and a --validate mode for CI/CD.

Usage
-----
    python pipeline_run.py                          # Full end-to-end run
    python pipeline_run.py --phase 1                # Phase 1 only
    python pipeline_run.py --sample-size 2000       # Smaller dataset for dev
    python pipeline_run.py --db-path ./custom.db    # Custom SQLite path
    python pipeline_run.py --validate               # CI validation mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import ensure_directories, load_config, setup_logger

logger = setup_logger("pipeline")


# ============================================================================
# Phase 1: Data Synthesis + Ingestion
# ============================================================================

def run_phase_1(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: Synthetic Data Generation + Ingestion & Validation.

    Returns
    -------
    dict
        Phase result with dataframes, profiles, and timing.
    """
    from src.synthetic_data import generate_all_datasets
    from src.ingestion import ingest_all_datasets

    logger.info("=" * 60)
    logger.info("PHASE 1: Data Synthesis, Ingestion & Quality Profiling")
    logger.info("=" * 60)

    # Step 1: Generate synthetic data
    logger.info("\n>> Step 1/2: Generating synthetic datasets...")
    t0 = time.time()
    output_paths = generate_all_datasets(config)
    gen_time = time.time() - t0
    logger.info("  Data generation completed in %.2fs", gen_time)

    for name, path in output_paths.items():
        size_kb = path.stat().st_size / 1024
        logger.info("    %-12s -> %s (%.1f KB)", name, path.name, size_kb)

    # Step 2: Ingest & validate
    logger.info("\n>> Step 2/2: Ingesting, validating & profiling datasets...")
    t1 = time.time()
    dataframes, profiles = ingest_all_datasets(config)
    ingest_time = time.time() - t1
    logger.info("  Ingestion completed in %.2fs", ingest_time)

    # Summary
    total_rows = sum(len(df) for df in dataframes.values())
    logger.info("\nPHASE 1 COMPLETE: %d datasets, %s total rows, %.2fs",
                len(dataframes), f"{total_rows:,}", gen_time + ingest_time)

    return {
        "dataframes": dataframes,
        "profiles": profiles,
        "output_paths": output_paths,
        "time_s": gen_time + ingest_time,
    }


# ============================================================================
# Phase 2: Cleaning + Joining
# ============================================================================

def run_phase_2(
    config: Dict[str, Any],
    dataframes: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Phase 2: Cleaning, Normalisation & Multi-Source Joining.

    Parameters
    ----------
    dataframes : dict | None
        Pre-loaded DataFrames from Phase 1. If None, re-ingests from disk.

    Returns
    -------
    dict
        Phase result with cleaned data, master table, and join audits.
    """
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets

    logger.info("=" * 60)
    logger.info("PHASE 2: Cleaning, Normalisation & Multi-Source Joining")
    logger.info("=" * 60)

    # Ingest if not passed
    if dataframes is None:
        logger.info("\n>> Ingesting raw datasets...")
        t0 = time.time()
        dataframes, _ = ingest_all_datasets(config)
        logger.info("  Ingestion completed in %.2fs", time.time() - t0)

    # Clean
    logger.info("\n>> Cleaning & normalising datasets...")
    t1 = time.time()
    cleaned = clean_all_datasets(dataframes)
    clean_time = time.time() - t1
    logger.info("  Cleaning completed in %.2fs", clean_time)

    # Join
    logger.info("\n>> Executing multi-stage joins...")
    t2 = time.time()
    master_table, audits = join_all_datasets(cleaned)
    join_time = time.time() - t2
    logger.info("  Joining completed in %.2fs", join_time)

    # Validate zero join drops
    total_warnings = sum(len(a.validate()) for a in audits)
    logger.info("\nPHASE 2 COMPLETE: %d rows master table, %d join warnings, %.2fs",
                len(master_table), total_warnings, clean_time + join_time)

    return {
        "cleaned": cleaned,
        "master_table": master_table,
        "audits": audits,
        "join_warnings": total_warnings,
        "time_s": clean_time + join_time,
    }


# ============================================================================
# Phase 3: Feature Engineering + SQL Layer
# ============================================================================

def run_phase_3(
    config: Dict[str, Any],
    master_table=None,
) -> Dict[str, Any]:
    """
    Phase 3: Feature Engineering & SQL Layer.

    Parameters
    ----------
    master_table : pd.DataFrame | None
        Joined master table from Phase 2. If None, re-runs Phase 2.

    Returns
    -------
    dict
        Phase result with campaign metrics, user data, SQL views.
    """
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets
    from src.feature_engineering import engineer_features
    from src.sql_layer import build_sql_layer

    logger.info("=" * 60)
    logger.info("PHASE 3: Feature Engineering & SQL Layer")
    logger.info("=" * 60)

    # Rebuild master if not passed
    if master_table is None:
        logger.info("\n>> Rebuilding master table (ingest -> clean -> join)...")
        t0 = time.time()
        dataframes, _ = ingest_all_datasets(config)
        cleaned = clean_all_datasets(dataframes)
        master_table, _ = join_all_datasets(cleaned)
        logger.info("  Rebuild completed in %.2fs", time.time() - t0)

    # Feature engineering
    logger.info("\n>> Computing activation features & KPIs...")
    t1 = time.time()
    campaigns, users = engineer_features(master_table, config=config)
    fe_time = time.time() - t1
    logger.info("  Feature engineering completed in %.2fs", fe_time)

    # SQL Layer
    logger.info("\n>> Building SQL layer & analytical views...")
    t2 = time.time()
    engine, views = build_sql_layer(campaigns, users, config=config)
    sql_time = time.time() - t2
    logger.info("  SQL layer completed in %.2fs", sql_time)

    # Campaign leaderboard
    logger.info("\n  Campaign Leaderboard:")
    for _, row in campaigns.iterrows():
        flag = " ** VANITY TRAP **" if row["is_vanity_trap"] else ""
        logger.info(
            "    #%-2d %-12s | CTR: %5.2f%% | Activation: %5.1f%% | CPAU: $%8.2f | %s%s",
            row["cpau_rank"], row["campaign_id"], row["ctr_pct"],
            row["activation_rate_pct"], row["cpau_usd"],
            row["performance_category"], flag,
        )

    logger.info("\nPHASE 3 COMPLETE: %d campaigns, %d users, %d views, %.2fs",
                len(campaigns), len(users), len(views), fe_time + sql_time)

    return {
        "campaigns": campaigns,
        "users": users,
        "views": views,
        "engine": engine,
        "time_s": fe_time + sql_time,
    }


# ============================================================================
# End-to-End Orchestrator
# ============================================================================

def run_full_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete pipeline end-to-end, passing data between phases
    to avoid redundant re-ingestion.

    Flow
    ----
    synthetic_data -> ingest -> clean -> join -> engineer -> sql_export

    Returns
    -------
    dict
        Combined results from all phases.
    """
    logger.info("#" * 60)
    logger.info("FULL PIPELINE: End-to-End Execution")
    logger.info("#" * 60)
    logger.info("  Started at : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  Sample size: %d signups", config["signup_target"])
    logger.info("  Seed       : %d", config["random_seed"])
    logger.info("")

    pipeline_start = time.time()

    # Phase 1: Generate + Ingest
    p1 = run_phase_1(config)

    # Phase 2: Clean + Join (pass dataframes to avoid re-reading)
    p2 = run_phase_2(config, dataframes=p1["dataframes"])

    # Phase 3: Features + SQL (pass master table)
    p3 = run_phase_3(config, master_table=p2["master_table"])

    total_time = time.time() - pipeline_start

    # Grand summary
    logger.info("")
    logger.info("#" * 60)
    logger.info("PIPELINE COMPLETE -- Grand Summary")
    logger.info("#" * 60)
    logger.info("  Total time      : %.2fs", total_time)
    logger.info("  Phase 1 (Gen)   : %.2fs", p1["time_s"])
    logger.info("  Phase 2 (ETL)   : %.2fs", p2["time_s"])
    logger.info("  Phase 3 (ML+SQL): %.2fs", p3["time_s"])
    logger.info("  ---")
    logger.info("  Raw datasets    : %d", len(p1["dataframes"]))
    logger.info("  Master rows     : %s", f"{len(p2['master_table']):,}")
    logger.info("  Join warnings   : %d", p2["join_warnings"])
    logger.info("  Campaigns       : %d", len(p3["campaigns"]))
    logger.info("  Users scored    : %d", len(p3["users"]))
    activated = int(p3["users"]["is_activated_7d"].sum())
    logger.info("  Activated (7d)  : %d (%.1f%%)",
                activated, activated / len(p3["users"]) * 100 if len(p3["users"]) > 0 else 0)
    logger.info("  SQL views       : %d", len(p3["views"]))
    logger.info("  Vanity Traps    : %d", int(p3["campaigns"]["is_vanity_trap"].sum()))
    logger.info("  Best CPAU       : $%.2f", p3["campaigns"]["cpau_usd"].min())
    logger.info("  Finished at     : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return {"phase1": p1, "phase2": p2, "phase3": p3, "total_time": total_time}


# ============================================================================
# Validation Mode (for CI/CD)
# ============================================================================

def run_validation(config: Dict[str, Any]) -> bool:
    """
    CI/CD validation mode: runs the full pipeline and asserts
    business-critical invariants.

    Returns True if all checks pass, False otherwise.
    """
    logger.info("#" * 60)
    logger.info("VALIDATION MODE: Running pipeline with integrity checks")
    logger.info("#" * 60)

    results = run_full_pipeline(config)
    errors: List[str] = []

    # Check 1: All 4 datasets were generated
    n_datasets = len(results["phase1"]["dataframes"])
    if n_datasets != 4:
        errors.append(f"Expected 4 datasets, got {n_datasets}")
    logger.info("[CHECK] Datasets generated: %d/4 %s", n_datasets, "PASS" if n_datasets == 4 else "FAIL")

    # Check 2: Zero join drop warnings (data leakage)
    join_warnings = results["phase2"]["join_warnings"]
    if join_warnings > 0:
        errors.append(f"Join integrity warnings: {join_warnings} (expected 0)")
    logger.info("[CHECK] Zero join drops: %d warnings %s", join_warnings, "PASS" if join_warnings == 0 else "FAIL")

    # Check 3: Master table has rows
    master_rows = len(results["phase2"]["master_table"])
    if master_rows == 0:
        errors.append("Master table is empty")
    logger.info("[CHECK] Master table non-empty: %d rows %s", master_rows, "PASS" if master_rows > 0 else "FAIL")

    # Check 4: All 10 campaigns present
    n_campaigns = len(results["phase3"]["campaigns"])
    if n_campaigns < 8:
        errors.append(f"Expected >= 8 campaigns, got {n_campaigns}")
    logger.info("[CHECK] Campaigns scored: %d %s", n_campaigns, "PASS" if n_campaigns >= 8 else "FAIL")

    # Check 5: Users were scored
    n_users = len(results["phase3"]["users"])
    if n_users == 0:
        errors.append("No users were scored")
    logger.info("[CHECK] Users scored: %d %s", n_users, "PASS" if n_users > 0 else "FAIL")

    # Check 6: SQL views created
    n_views = len(results["phase3"]["views"])
    if n_views < 4:
        errors.append(f"Expected >= 4 SQL views, got {n_views}")
    logger.info("[CHECK] SQL views created: %d/4 %s", n_views, "PASS" if n_views >= 4 else "FAIL")

    # Check 7: Activation rate is within sane range (0-100%)
    act_rate = results["phase3"]["users"]["is_activated_7d"].mean() * 100
    if not (0 < act_rate < 100):
        errors.append(f"Activation rate {act_rate:.1f}% is outside sane range")
    logger.info("[CHECK] Activation rate sane: %.1f%% %s", act_rate, "PASS" if 0 < act_rate < 100 else "FAIL")

    # Check 8: Parquet output files exist
    from src.utils import DATA_PROCESSED_DIR
    parquet_files = list(DATA_PROCESSED_DIR.glob("*.parquet"))
    if len(parquet_files) < 2:
        errors.append(f"Expected >= 2 Parquet files, found {len(parquet_files)}")
    logger.info("[CHECK] Parquet exports: %d files %s", len(parquet_files), "PASS" if len(parquet_files) >= 2 else "FAIL")

    # Final verdict
    logger.info("")
    if errors:
        logger.error("VALIDATION FAILED: %d error(s)", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        return False
    else:
        logger.info("VALIDATION PASSED: All %d checks passed", 8)
        return True


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI entry point with full argument support."""
    parser = argparse.ArgumentParser(
        description="Downstream Activation Engine -- Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_run.py                          # Full end-to-end run
  python pipeline_run.py --phase 1                # Phase 1 only
  python pipeline_run.py --sample-size 2000       # Smaller dataset
  python pipeline_run.py --db-path ./custom.db    # Custom DB path
  python pipeline_run.py --validate               # CI validation mode
        """,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific phase (1=Data Gen/Ingestion, 2=Clean/Join, "
             "3=Features/SQL). Default: run all phases end-to-end.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of signups to generate (controls overall dataset size). "
             "Default: 5000. Use smaller values for faster dev iterations.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom path for the SQLite database file. "
             "Default: data/processed/activation_engine.db",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data (default: 42).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run in CI validation mode: execute full pipeline and assert "
             "all business-critical integrity checks pass.",
    )
    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides: Dict[str, Any] = {"random_seed": args.seed}
    if args.sample_size is not None:
        overrides["signup_target"] = args.sample_size
    if args.db_path is not None:
        overrides["db_path"] = args.db_path

    config = load_config(overrides=overrides)
    ensure_directories(config)

    logger.info("Downstream Activation Engine -- Pipeline Run")
    logger.info("  Mode          : %s", "VALIDATE" if args.validate else (f"Phase {args.phase}" if args.phase else "FULL"))
    logger.info("  Sample size   : %d signups", config["signup_target"])
    logger.info("  Seed          : %d", args.seed)
    logger.info("  Raw dir       : %s", config["data_raw_dir"])
    if args.db_path:
        logger.info("  Custom DB     : %s", args.db_path)

    try:
        if args.validate:
            # CI validation mode
            passed = run_validation(config)
            sys.exit(0 if passed else 1)

        elif args.phase is None:
            # Full end-to-end pipeline
            run_full_pipeline(config)

        else:
            # Individual phase
            if args.phase == 1:
                run_phase_1(config)
            elif args.phase == 2:
                run_phase_2(config)
            elif args.phase == 3:
                run_phase_3(config)

        logger.info("\nPipeline run completed successfully!")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
