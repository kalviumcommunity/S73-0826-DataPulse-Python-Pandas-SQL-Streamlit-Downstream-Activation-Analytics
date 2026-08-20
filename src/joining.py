"""
src/joining.py — Multi-Source Entity Resolution & Join Integrity
=================================================================
Executes multi-stage merges across impressions, clicks, signups, and
user events using campaign_id and user_id as join keys. Implements
strict join audit logging to verify row counts, unmatched keys, and
join drop-offs to prevent data leakage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    JoinIntegrityError,
    load_config,
    setup_logger,
)

logger = setup_logger("joining")


# ============================================================================
# Join Audit Infrastructure
# ============================================================================

@dataclass
class JoinAudit:
    """Captures pre/post-join metrics for integrity verification."""
    join_name: str
    left_name: str
    right_name: str
    join_keys: List[str]
    join_type: str
    left_rows_before: int = 0
    right_rows_before: int = 0
    result_rows: int = 0
    left_keys_total: int = 0
    right_keys_total: int = 0
    matched_keys: int = 0
    unmatched_left_keys: int = 0
    unmatched_right_keys: int = 0
    row_expansion_pct: float = 0.0
    row_loss_pct: float = 0.0

    def validate(self, max_expansion_pct: float = 50.0, max_loss_pct: float = 10.0) -> List[str]:
        """Return list of warning/error messages if thresholds are breached."""
        issues: List[str] = []

        if self.row_expansion_pct > max_expansion_pct:
            issues.append(
                f"Row expansion {self.row_expansion_pct:.1f}% exceeds "
                f"threshold {max_expansion_pct:.1f}% -- possible cartesian blow-up"
            )

        if self.row_loss_pct > max_loss_pct:
            issues.append(
                f"Row loss {self.row_loss_pct:.1f}% exceeds "
                f"threshold {max_loss_pct:.1f}% -- possible data leakage"
            )

        if self.unmatched_left_keys > 0:
            issues.append(
                f"{self.unmatched_left_keys} left keys ({self.left_name}) "
                f"had no match in {self.right_name}"
            )

        if self.unmatched_right_keys > 0:
            issues.append(
                f"{self.unmatched_right_keys} right keys ({self.right_name}) "
                f"had no match in {self.left_name}"
            )

        return issues


def _compute_join_audit(
    left: pd.DataFrame,
    right: pd.DataFrame,
    result: pd.DataFrame,
    join_keys: List[str],
    join_name: str,
    left_name: str,
    right_name: str,
    join_type: str,
) -> JoinAudit:
    """Compute comprehensive join audit metrics."""
    # Key sets (handle composite keys by creating tuples)
    if len(join_keys) == 1:
        left_keys_set = set(left[join_keys[0]].dropna().unique())
        right_keys_set = set(right[join_keys[0]].dropna().unique())
    else:
        left_keys_set = set(
            left[join_keys].dropna().apply(tuple, axis=1).unique()
        )
        right_keys_set = set(
            right[join_keys].dropna().apply(tuple, axis=1).unique()
        )

    matched = left_keys_set & right_keys_set
    unmatched_left = left_keys_set - right_keys_set
    unmatched_right = right_keys_set - left_keys_set

    left_rows = len(left)
    expected_baseline = max(left_rows, len(right))
    expansion = (
        ((len(result) - expected_baseline) / expected_baseline * 100)
        if expected_baseline > 0 else 0.0
    )
    loss = (
        ((left_rows - len(result)) / left_rows * 100)
        if left_rows > 0 and len(result) < left_rows else 0.0
    )

    return JoinAudit(
        join_name=join_name,
        left_name=left_name,
        right_name=right_name,
        join_keys=join_keys,
        join_type=join_type,
        left_rows_before=left_rows,
        right_rows_before=len(right),
        result_rows=len(result),
        left_keys_total=len(left_keys_set),
        right_keys_total=len(right_keys_set),
        matched_keys=len(matched),
        unmatched_left_keys=len(unmatched_left),
        unmatched_right_keys=len(unmatched_right),
        row_expansion_pct=max(0.0, expansion),
        row_loss_pct=max(0.0, loss),
    )


def _log_join_audit(audit: JoinAudit) -> None:
    """Log a formatted join audit report."""
    logger.info("  --- Join Audit: %s ---", audit.join_name)
    logger.info("  Type          : %s", audit.join_type)
    logger.info("  Keys          : %s", audit.join_keys)
    logger.info("  Left (%s) : %d rows, %d unique keys", audit.left_name, audit.left_rows_before, audit.left_keys_total)
    logger.info("  Right (%s): %d rows, %d unique keys", audit.right_name, audit.right_rows_before, audit.right_keys_total)
    logger.info("  Matched keys  : %d", audit.matched_keys)
    logger.info("  Unmatched L/R : %d / %d", audit.unmatched_left_keys, audit.unmatched_right_keys)
    logger.info("  Result rows   : %d", audit.result_rows)
    logger.info("  Expansion     : %.1f%%", audit.row_expansion_pct)
    logger.info("  Loss          : %.1f%%", audit.row_loss_pct)

    issues = audit.validate()
    for issue in issues:
        logger.warning("  [!] %s", issue)

    if not issues:
        logger.info("  [OK] Join integrity check passed")


# ============================================================================
# Audited Merge Wrapper
# ============================================================================

def audited_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[List[str]] = None,
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    how: str = "inner",
    join_name: str = "merge",
    left_name: str = "left",
    right_name: str = "right",
    suffixes: Tuple[str, str] = ("_left", "_right"),
    validate: Optional[str] = None,
) -> Tuple[pd.DataFrame, JoinAudit]:
    """
    Perform a pandas merge with full join-audit logging.

    Parameters
    ----------
    left, right : pd.DataFrame
        DataFrames to join.
    on : list[str] | None
        Join keys (if same name in both).
    left_on, right_on : list[str] | None
        Join keys if names differ.
    how : str
        Join type: "inner", "left", "right", "outer".
    join_name : str
        Human-readable label for audit logging.
    left_name, right_name : str
        Labels for the left/right DataFrames.
    suffixes : tuple
        Suffixes for overlapping non-key columns.
    validate : str | None
        Pandas merge validation ("one_to_one", "one_to_many", etc.).

    Returns
    -------
    tuple[pd.DataFrame, JoinAudit]
        Merged DataFrame and its audit report.
    """
    join_keys = on or left_on or []

    logger.info("Executing join: %s (%s on %s)", join_name, how.upper(), join_keys)

    try:
        result = pd.merge(
            left, right,
            on=on, left_on=left_on, right_on=right_on,
            how=how, suffixes=suffixes, validate=validate,
        )
    except pd.errors.MergeError as exc:
        raise JoinIntegrityError(
            f"Merge validation failed for '{join_name}': {exc}",
            context={"join_name": join_name, "how": how, "keys": join_keys},
        )

    audit = _compute_join_audit(
        left, right, result, join_keys,
        join_name, left_name, right_name, how,
    )
    _log_join_audit(audit)

    return result, audit


# ============================================================================
# Stage 1: Combine Google Ads + Meta Ads into Unified Ads Table
# ============================================================================

def combine_ads_platforms(
    google_ads: pd.DataFrame,
    meta_ads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vertically stack Google Ads and Meta Ads data into a single
    unified campaign performance table.

    Both DataFrames share the same schema:
        campaign_id, campaign_name, date, impressions, clicks, ad_spend, platform
    """
    logger.info("=" * 50)
    logger.info("Stage 1: Combining ads platforms (UNION)")
    logger.info("=" * 50)

    # Ensure consistent columns
    common_cols = ["campaign_id", "campaign_name", "date", "impressions", "clicks", "ad_spend", "platform"]
    g_cols = [c for c in common_cols if c in google_ads.columns]
    m_cols = [c for c in common_cols if c in meta_ads.columns]

    unified = pd.concat(
        [google_ads[g_cols], meta_ads[m_cols]],
        ignore_index=True,
    )

    logger.info(
        "  Google Ads: %d rows + Meta Ads: %d rows = %d unified rows",
        len(google_ads), len(meta_ads), len(unified),
    )
    logger.info("  Campaigns in unified set: %s", sorted(unified["campaign_id"].unique()))
    return unified


# ============================================================================
# Stage 2: Aggregate Ads to Campaign-Level Summary
# ============================================================================

def aggregate_ads_to_campaign_level(ads_unified: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily ad performance data to campaign-level totals.

    Output columns
    ------
    campaign_id, campaign_name, platform, total_impressions, total_clicks,
    total_ad_spend, campaign_days, avg_daily_spend
    """
    logger.info("=" * 50)
    logger.info("Stage 2: Aggregating ads to campaign level")
    logger.info("=" * 50)

    agg = (
        ads_unified
        .groupby(["campaign_id", "platform"], as_index=False)
        .agg(
            campaign_name=("campaign_name", "first"),
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
            total_ad_spend=("ad_spend", "sum"),
            campaign_days=("date", "nunique"),
        )
    )
    agg["avg_daily_spend"] = (agg["total_ad_spend"] / agg["campaign_days"]).round(2)

    logger.info("  Aggregated to %d campaign-platform rows", len(agg))
    logger.info("  Total ad spend: $%.2f", agg["total_ad_spend"].sum())
    return agg


# ============================================================================
# Stage 3: Join Campaign Aggregates with Signups
# ============================================================================

def join_campaigns_with_signups(
    campaign_agg: pd.DataFrame,
    signups: pd.DataFrame,
) -> Tuple[pd.DataFrame, JoinAudit]:
    """
    LEFT join campaign-level aggregates with user signups on campaign_id.

    This preserves all campaigns even if they have zero signups,
    which is important for identifying campaigns with traffic but no conversion.
    """
    logger.info("=" * 50)
    logger.info("Stage 3: Joining campaigns with signups")
    logger.info("=" * 50)

    result, audit = audited_merge(
        left=campaign_agg,
        right=signups,
        on=["campaign_id"],
        how="left",
        join_name="campaigns <-> signups",
        left_name="campaign_agg",
        right_name="signups",
    )

    return result, audit


# ============================================================================
# Stage 4: Join Campaign-Signups with User Events
# ============================================================================

def join_signups_with_events(
    campaign_signups: pd.DataFrame,
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, JoinAudit]:
    """
    LEFT join the campaign-signup table with user event logs on user_id.

    This ensures we keep all signups even if a user has zero post-signup events
    (which itself is a signal — they bounced immediately).
    """
    logger.info("=" * 50)
    logger.info("Stage 4: Joining signups with user events")
    logger.info("=" * 50)

    result, audit = audited_merge(
        left=campaign_signups,
        right=events,
        on=["user_id"],
        how="left",
        join_name="campaign_signups <-> events",
        left_name="campaign_signups",
        right_name="events",
    )

    return result, audit


# ============================================================================
# Master Join Orchestrator
# ============================================================================

def join_all_datasets(
    cleaned: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[JoinAudit]]:
    """
    Execute the full multi-stage join pipeline.

    Join Flow
    ---------
    1. UNION:  Google Ads + Meta Ads  ->  ads_unified
    2. AGG:    ads_unified            ->  campaign_agg  (campaign-level totals)
    3. JOIN:   campaign_agg  x signups on campaign_id   ->  campaign_signups
    4. JOIN:   campaign_signups x events on user_id     ->  master_table

    Parameters
    ----------
    cleaned : dict
        Mapping of dataset_key -> cleaned DataFrame.
    config : dict | None
        Pipeline configuration.

    Returns
    -------
    tuple[pd.DataFrame, list[JoinAudit]]
        Master joined table and all audit reports.
    """
    logger.info("#" * 60)
    logger.info("JOINING PIPELINE START")
    logger.info("#" * 60)

    audits: List[JoinAudit] = []

    # Stage 1: Union ads platforms
    ads_unified = combine_ads_platforms(
        cleaned["google_ads"],
        cleaned["meta_ads"],
    )

    # Stage 2: Aggregate to campaign level
    campaign_agg = aggregate_ads_to_campaign_level(ads_unified)

    # Stage 3: Join with signups
    campaign_signups, audit_3 = join_campaigns_with_signups(
        campaign_agg,
        cleaned["signups"],
    )
    audits.append(audit_3)

    # Stage 4: Join with events
    master_table, audit_4 = join_signups_with_events(
        campaign_signups,
        cleaned["events"],
    )
    audits.append(audit_4)

    # Final summary
    logger.info("#" * 60)
    logger.info("JOINING PIPELINE COMPLETE")
    logger.info("#" * 60)
    logger.info("  Master table : %d rows x %d columns", len(master_table), len(master_table.columns))
    logger.info("  Columns      : %s", list(master_table.columns))
    logger.info("  Unique users : %d", master_table["user_id"].nunique() if "user_id" in master_table.columns else 0)
    logger.info("  Unique camps : %d", master_table["campaign_id"].nunique())

    # Aggregate audit summary
    total_warnings = sum(len(a.validate()) for a in audits)
    if total_warnings > 0:
        logger.warning("  Total join warnings: %d (review audit logs above)", total_warnings)
    else:
        logger.info("  All join integrity checks passed")

    return master_table, audits


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets

    logger.info("=" * 60)
    logger.info("Downstream Activation Engine -- Joining Module (Standalone)")
    logger.info("=" * 60)

    try:
        dataframes, _ = ingest_all_datasets()
        cleaned = clean_all_datasets(dataframes)
        master, audits = join_all_datasets(cleaned)
        logger.info("Master table shape: %s", master.shape)
        logger.info("Master table head:\n%s", master.head(3).to_string())
    except Exception as exc:
        logger.exception("Joining failed: %s", exc)
        raise
