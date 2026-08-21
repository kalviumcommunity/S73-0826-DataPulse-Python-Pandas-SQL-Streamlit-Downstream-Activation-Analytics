"""
src/feature_engineering.py — 7-Day Activation Flags, CPAU & Vanity Ratios
==========================================================================
Derives downstream activation metrics from the joined master table:

User-level features
-------------------
- days_to_first_activation : days from signup to first key action
- is_activated_7d          : boolean (1 if activation within 7 days)

Campaign-level metrics
----------------------
- total_signups, activated_users_7d, activation_rate_pct
- ctr_pct, cpau_usd, vanity_ratio_index
- performance_category ("High Value", "Scaling", "Vanity Trap", "Under-Performer")

Outputs
-------
- User-level DataFrame (one row per user)
- Campaign-level summary DataFrame (one row per campaign)
- Parquet export to data/processed/master_activated_campaigns.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    FeatureEngineeringError,
    load_config,
    setup_logger,
)

logger = setup_logger("feature_engineering")

# Activation events — at least 1 within 7 days of signup = activated
ACTIVATION_EVENTS = frozenset({
    "feature_used",
    "profile_completed",
    "item_added",
    "purchase",
})


# ============================================================================
# 1. User-Level Activation Features
# ============================================================================

def compute_user_activation(
    master: pd.DataFrame,
    activation_window_days: int = 7,
) -> pd.DataFrame:
    """
    Compute per-user activation features from the event-level master table.

    For each user, find the earliest activation event (if any), compute
    ``days_to_first_activation``, and flag ``is_activated_7d``.

    Parameters
    ----------
    master : pd.DataFrame
        Joined master table (one row per user-event).
    activation_window_days : int
        Number of days after signup within which an activation event
        must occur (default: 7).

    Returns
    -------
    pd.DataFrame
        One row per user with columns:
        user_id, campaign_id, platform, signup_timestamp,
        first_activation_event, first_activation_timestamp,
        days_to_first_activation, is_activated_7d
    """
    logger.info("Computing user-level activation features...")

    # --- Filter to activation events only ---
    activation_mask = master["event_type"].isin(ACTIVATION_EVENTS)
    activation_events = master.loc[activation_mask].copy()

    logger.info(
        "  Activation events: %d / %d total events (%.1f%%)",
        len(activation_events), len(master),
        len(activation_events) / len(master) * 100 if len(master) > 0 else 0,
    )

    # --- Find earliest activation event per user (vectorised) ---
    activation_events = activation_events.sort_values("event_timestamp")
    first_activation = (
        activation_events
        .groupby("user_id", as_index=False)
        .agg(
            first_activation_event=("event_type", "first"),
            first_activation_timestamp=("event_timestamp", "min"),
        )
    )

    # --- Build user-level base (one row per user) ---
    user_cols = [
        "user_id", "campaign_id", "platform", "campaign_name",
        "signup_timestamp", "referral_source", "email", "full_name", "country",
    ]
    existing_cols = [c for c in user_cols if c in master.columns]
    users = master[existing_cols].drop_duplicates(subset=["user_id"]).copy()

    # --- Merge first activation onto user table ---
    users = users.merge(first_activation, on="user_id", how="left")

    # --- Compute days_to_first_activation ---
    users["days_to_first_activation"] = np.where(
        users["first_activation_timestamp"].notna(),
        (users["first_activation_timestamp"] - users["signup_timestamp"]).dt.total_seconds() / 86400,
        np.nan,
    )
    users["days_to_first_activation"] = users["days_to_first_activation"].round(2)

    # --- Boolean flag: is_activated_7d ---
    users["is_activated_7d"] = np.where(
        users["days_to_first_activation"].notna()
        & (users["days_to_first_activation"] <= activation_window_days)
        & (users["days_to_first_activation"] >= 0),
        1,
        0,
    ).astype(np.int8)

    total_users = len(users)
    activated = int(users["is_activated_7d"].sum())
    logger.info(
        "  User activation: %d / %d activated within %d days (%.1f%%)",
        activated, total_users, activation_window_days,
        activated / total_users * 100 if total_users > 0 else 0,
    )

    return users


# ============================================================================
# 2. Campaign-Level Metrics
# ============================================================================

def compute_campaign_metrics(
    users: pd.DataFrame,
    master: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Aggregate user-level activation data to campaign-level KPIs.

    Computed metrics
    ----------------
    - total_signups, activated_users_7d
    - activation_rate_pct (target: > 25%)
    - ctr_pct
    - cpau_usd (target: < $45.00)
    - vanity_ratio_index (flagged if > 3.0)
    - performance_category

    Parameters
    ----------
    users : pd.DataFrame
        User-level DataFrame from ``compute_user_activation``.
    master : pd.DataFrame
        Joined master table (for campaign-level ad metrics).
    config : dict | None
        Pipeline configuration with threshold values.

    Returns
    -------
    pd.DataFrame
        One row per campaign with all derived KPIs.
    """
    cfg = config or load_config()
    activation_target = cfg.get("activation_rate_target_pct", 25.0)
    cpau_target = cfg.get("cpau_target_usd", 45.0)
    vanity_threshold = cfg.get("vanity_ratio_flag_threshold", 3.0)

    logger.info("Computing campaign-level metrics...")

    # --- User aggregation per campaign ---
    campaign_users = (
        users
        .groupby(["campaign_id", "platform", "campaign_name"], as_index=False)
        .agg(
            total_signups=("user_id", "nunique"),
            activated_users_7d=("is_activated_7d", "sum"),
            avg_days_to_activation=("days_to_first_activation", "mean"),
            median_days_to_activation=("days_to_first_activation", "median"),
        )
    )

    # --- Pull campaign-level ad metrics (already aggregated in master) ---
    ad_metrics_cols = [
        "campaign_id", "total_impressions", "total_clicks",
        "total_ad_spend", "campaign_days", "avg_daily_spend",
    ]
    existing_ad_cols = [c for c in ad_metrics_cols if c in master.columns]
    ad_metrics = master[existing_ad_cols].drop_duplicates(subset=["campaign_id"])

    # --- Merge ---
    campaigns = campaign_users.merge(ad_metrics, on="campaign_id", how="left")

    # --- CTR (%) ---
    campaigns["ctr_pct"] = np.where(
        campaigns["total_impressions"] > 0,
        (campaigns["total_clicks"] / campaigns["total_impressions"]) * 100,
        0.0,
    )
    campaigns["ctr_pct"] = campaigns["ctr_pct"].round(4)

    # --- Activation Rate (%) ---
    campaigns["activation_rate_pct"] = np.where(
        campaigns["total_signups"] > 0,
        (campaigns["activated_users_7d"] / campaigns["total_signups"]) * 100,
        0.0,
    )
    campaigns["activation_rate_pct"] = campaigns["activation_rate_pct"].round(2)

    # --- CPAU ($) ---
    campaigns["cpau_usd"] = np.where(
        campaigns["activated_users_7d"] > 0,
        campaigns["total_ad_spend"] / campaigns["activated_users_7d"],
        np.inf,  # No activated users = infinite cost
    )
    campaigns["cpau_usd"] = campaigns["cpau_usd"].replace([np.inf], 99999.99).round(2)

    # --- Vanity Ratio Index ---
    campaigns["vanity_ratio_index"] = np.where(
        campaigns["activation_rate_pct"] > 0,
        campaigns["ctr_pct"] / campaigns["activation_rate_pct"],
        np.inf,
    )
    campaigns["vanity_ratio_index"] = campaigns["vanity_ratio_index"].replace([np.inf], 99.99).round(4)

    # --- Is Vanity Trap flag ---
    campaigns["is_vanity_trap"] = (campaigns["vanity_ratio_index"] > vanity_threshold).astype(np.int8)

    # --- Performance Category ---
    campaigns["performance_category"] = _classify_campaign(
        campaigns, activation_target, cpau_target, vanity_threshold
    )

    # --- Sort by CPAU ascending (best performers first) ---
    campaigns = campaigns.sort_values("cpau_usd", ascending=True).reset_index(drop=True)

    # --- Campaign rank ---
    campaigns["cpau_rank"] = campaigns["cpau_usd"].rank(method="min").astype(int)

    # --- Log summary ---
    logger.info("  Campaign metrics computed for %d campaigns:", len(campaigns))
    for _, row in campaigns.iterrows():
        logger.info(
            "    %-12s | CTR: %5.2f%% | Act: %5.1f%% | CPAU: $%8.2f | Vanity: %5.2f | %s",
            row["campaign_id"], row["ctr_pct"], row["activation_rate_pct"],
            row["cpau_usd"], row["vanity_ratio_index"], row["performance_category"],
        )

    # Count categories
    cat_counts = campaigns["performance_category"].value_counts().to_dict()
    logger.info("  Category distribution: %s", cat_counts)

    return campaigns


def _classify_campaign(
    df: pd.DataFrame,
    activation_target: float,
    cpau_target: float,
    vanity_threshold: float,
) -> pd.Series:
    """
    Assign a performance category to each campaign based on business rules.

    Categories
    ----------
    - **High Value**: activation_rate >= target AND cpau <= target
    - **Scaling**: activation_rate >= target but cpau > target (good conversion, high cost)
    - **Vanity Trap**: vanity_ratio > threshold (high CTR, low activation)
    - **Under-Performer**: everything else
    """
    conditions = [
        # High Value: good activation AND cost-efficient
        (df["activation_rate_pct"] >= activation_target) & (df["cpau_usd"] <= cpau_target),
        # Scaling: good activation but expensive
        (df["activation_rate_pct"] >= activation_target) & (df["cpau_usd"] > cpau_target),
        # Vanity Trap: high vanity ratio
        (df["vanity_ratio_index"] > vanity_threshold),
    ]
    choices = ["High Value", "Scaling", "Vanity Trap"]

    return pd.Series(
        np.select(conditions, choices, default="Under-Performer"),
        index=df.index,
    )


# ============================================================================
# 3. Export to Parquet
# ============================================================================

def export_to_parquet(
    campaigns: pd.DataFrame,
    users: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Export the campaign-level and user-level DataFrames to Parquet files.

    Returns
    -------
    dict
        Mapping of output name -> file path.
    """
    cfg = config or load_config()
    processed_dir = Path(cfg["data_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    # Campaign-level
    campaign_path = processed_dir / cfg.get("master_output_file", "master_activated_campaigns.parquet")
    campaigns.to_parquet(campaign_path, index=False, engine="pyarrow")
    outputs["campaign_summary"] = campaign_path
    logger.info("  Exported campaign summary: %s (%.1f KB)", campaign_path.name, campaign_path.stat().st_size / 1024)

    # User-level
    user_path = processed_dir / "user_activation_details.parquet"
    users.to_parquet(user_path, index=False, engine="pyarrow")
    outputs["user_details"] = user_path
    logger.info("  Exported user details: %s (%.1f KB)", user_path.name, user_path.stat().st_size / 1024)

    return outputs


# ============================================================================
# 4. Master Feature Engineering Orchestrator
# ============================================================================

def engineer_features(
    master: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature engineering pipeline.

    Steps
    -----
    1. Compute user-level activation features
    2. Aggregate to campaign-level metrics (CTR, CPAU, Vanity Ratio, etc.)
    3. Export both to Parquet

    Parameters
    ----------
    master : pd.DataFrame
        Joined master table from Phase 2.
    config : dict | None
        Pipeline configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (campaign_summary, user_details)
    """
    cfg = config or load_config()
    activation_window = cfg.get("activation_window_days", 7)

    logger.info("#" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE START")
    logger.info("#" * 60)

    try:
        # Step 1: User-level features
        users = compute_user_activation(master, activation_window_days=activation_window)

        # Step 2: Campaign-level KPIs
        campaigns = compute_campaign_metrics(users, master, config=cfg)

        # Step 3: Export
        logger.info("Exporting processed data to Parquet...")
        output_paths = export_to_parquet(campaigns, users, config=cfg)

        # Summary
        logger.info("#" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
        logger.info("#" * 60)
        logger.info("  Users processed     : %d", len(users))
        logger.info("  Activated (7-day)   : %d (%.1f%%)",
                     int(users["is_activated_7d"].sum()),
                     users["is_activated_7d"].mean() * 100)
        logger.info("  Campaigns scored    : %d", len(campaigns))
        logger.info("  Vanity Traps found  : %d", int(campaigns["is_vanity_trap"].sum()))
        logger.info("  Best CPAU           : $%.2f (%s)",
                     campaigns["cpau_usd"].min(),
                     campaigns.loc[campaigns["cpau_usd"].idxmin(), "campaign_id"])
        logger.info("  Worst CPAU          : $%.2f (%s)",
                     campaigns["cpau_usd"].max(),
                     campaigns.loc[campaigns["cpau_usd"].idxmax(), "campaign_id"])

        return campaigns, users

    except Exception as exc:
        raise FeatureEngineeringError(
            f"Feature engineering failed: {exc}",
            context={"master_shape": str(master.shape)},
        ) from exc


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets

    logger.info("=" * 60)
    logger.info("Downstream Activation Engine -- Feature Engineering (Standalone)")
    logger.info("=" * 60)

    try:
        dfs, _ = ingest_all_datasets()
        cleaned = clean_all_datasets(dfs)
        master, _ = join_all_datasets(cleaned)
        campaigns, users = engineer_features(master)
        logger.info("\nCampaign Summary:\n%s", campaigns.to_string())
    except Exception as exc:
        logger.exception("Feature engineering failed: %s", exc)
        raise
