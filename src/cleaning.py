"""
src/cleaning.py — Deduplication, Text Normalisation & Timestamp Parsing
========================================================================
Cleans raw ingested datasets by standardising campaign names, normalising
timestamps to ISO 8601, handling missing values with context-aware
imputation, and filtering statistical outliers in spend/duration fields.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DataQualityError,
    load_config,
    setup_logger,
)

logger = setup_logger("cleaning")


# ============================================================================
# 1. Deduplication
# ============================================================================

def deduplicate(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Remove exact-duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    subset : list[str] | None
        Columns to consider for duplicate detection.
        If None, all columns are used.
    dataset_name : str
        Label for logging.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with reset index.
    """
    before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df_clean)
    logger.info(
        "[%s] Deduplication: %d -> %d rows (%d duplicates removed, %.1f%%)",
        dataset_name, before, len(df_clean), removed,
        (removed / before * 100) if before > 0 else 0,
    )
    return df_clean


# ============================================================================
# 2. Campaign Name Standardisation
# ============================================================================

def standardise_campaign_names(
    df: pd.DataFrame,
    column: str = "campaign_name",
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Normalise campaign names: strip whitespace, lowercase, collapse
    multiple spaces, and remove non-printable characters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a campaign name column.
    column : str
        Name of the column to standardise.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned campaign names.
    """
    if column not in df.columns:
        logger.warning("[%s] Column '%s' not found, skipping name standardisation.", dataset_name, column)
        return df

    df = df.copy()
    original_unique = df[column].nunique()

    # Vectorised cleaning pipeline
    cleaned = (
        df[column]
        .astype(str)
        .str.strip()                           # remove leading/trailing whitespace
        .str.lower()                           # lowercase
        .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        .str.replace(r"[^\x20-\x7E]", "", regex=True)  # remove non-printable chars
    )
    df[column] = cleaned

    new_unique = df[column].nunique()
    logger.info(
        "[%s] Campaign name standardisation: %d -> %d unique values",
        dataset_name, original_unique, new_unique,
    )
    return df


# ============================================================================
# 3. Timestamp Normalisation
# ============================================================================

# Common date formats encountered in the synthetic data
_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",           # ISO standard: 2025-06-15
    "%m/%d/%Y",           # US format:    06/15/2025
    "%d-%b-%Y",           # Abbreviated:  15-Jun-2025
    "%Y-%m-%d %H:%M:%S",  # datetime:     2025-06-15 14:30:00
]


def normalise_timestamps(
    df: pd.DataFrame,
    column: str,
    output_format: str = "iso",
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Parse mixed-format date/datetime strings into a consistent format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a date/timestamp column.
    column : str
        Column name to normalise.
    output_format : str
        ``"iso"`` for ISO 8601 strings, ``"datetime"`` for pd.Timestamp objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised timestamp column.
    """
    if column not in df.columns:
        logger.warning("[%s] Column '%s' not found, skipping timestamp normalisation.", dataset_name, column)
        return df

    df = df.copy()

    # pd.to_datetime with infer handles most mixed formats
    parsed = pd.to_datetime(df[column], format="mixed", dayfirst=False, errors="coerce")

    coerced_nulls = int(parsed.isna().sum()) - int(df[column].isna().sum())
    if coerced_nulls > 0:
        logger.warning(
            "[%s] %d values in '%s' could not be parsed and were set to NaT",
            dataset_name, coerced_nulls, column,
        )

    if output_format == "iso":
        # Store as ISO 8601 string for maximum interoperability
        df[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").where(parsed.notna(), None)
    else:
        df[column] = parsed

    logger.info("[%s] Normalised '%s' to %s format", dataset_name, column, output_format)
    return df


# ============================================================================
# 4. Missing Value Handling (Context-Aware Imputation)
# ============================================================================

def impute_numeric_nulls(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "median",
    group_by: Optional[str] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Fill missing numeric values using a group-aware strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Numeric columns to impute.
    strategy : str
        ``"median"`` or ``"mean"``.
    group_by : str | None
        If provided, compute fill values per group (e.g., per campaign_id).

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue

        if group_by and group_by in df.columns:
            # Group-aware imputation
            if strategy == "median":
                fill_values = df.groupby(group_by)[col].transform("median")
            else:
                fill_values = df.groupby(group_by)[col].transform("mean")
            df[col] = df[col].fillna(fill_values)

            # Fall back to global stat for any remaining NaNs
            remaining = int(df[col].isna().sum())
            if remaining > 0:
                global_val = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(global_val)
        else:
            # Global imputation
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)

        logger.info(
            "[%s] Imputed %d nulls in '%s' using %s%s",
            dataset_name, null_count, col, strategy,
            f" (grouped by {group_by})" if group_by else "",
        )

    return df


# ============================================================================
# 5. Statistical Outlier Filtering (IQR Method)
# ============================================================================

def filter_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    factor: float = 3.0,
    action: str = "clip",
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Detect and handle outliers using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Numeric columns to check for outliers.
    factor : float
        IQR multiplier for fence calculation (default 3.0 = very permissive).
    action : str
        ``"clip"`` to cap at fences, ``"drop"`` to remove rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - factor * iqr
        upper_fence = q3 + factor * iqr

        outlier_mask = (df[col] < lower_fence) | (df[col] > upper_fence)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            logger.info("[%s] No outliers in '%s' (IQR x%.1f)", dataset_name, col, factor)
            continue

        if action == "clip":
            df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
            logger.info(
                "[%s] Clipped %d outliers in '%s' to [%.2f, %.2f]",
                dataset_name, outlier_count, col, lower_fence, upper_fence,
            )
        elif action == "drop":
            before = len(df)
            df = df[~outlier_mask].reset_index(drop=True)
            logger.info(
                "[%s] Dropped %d outlier rows from '%s' (%d -> %d)",
                dataset_name, outlier_count, col, before, len(df),
            )

    return df


# ============================================================================
# 6. Per-Dataset Cleaning Orchestrators
# ============================================================================

def clean_ads_data(
    df: pd.DataFrame,
    dataset_name: str = "Ads Data",
) -> pd.DataFrame:
    """
    Full cleaning pipeline for Google Ads / Meta Ads impression/click data.

    Steps
    -----
    1. Deduplicate
    2. Standardise campaign names
    3. Normalise date column to ISO 8601
    4. Impute nulls in clicks/impressions/ad_spend (median, grouped by campaign)
    5. Ensure integer types for impressions and clicks
    6. Filter outliers in ad_spend
    """
    logger.info("=" * 50)
    logger.info("Cleaning: %s", dataset_name)
    logger.info("=" * 50)

    df = deduplicate(df, dataset_name=dataset_name)
    df = standardise_campaign_names(df, dataset_name=dataset_name)
    df = normalise_timestamps(df, column="date", output_format="iso", dataset_name=dataset_name)

    # Impute numeric nulls grouped by campaign
    numeric_cols = ["impressions", "clicks", "ad_spend"]
    df = impute_numeric_nulls(
        df, numeric_cols, strategy="median",
        group_by="campaign_id", dataset_name=dataset_name,
    )

    # Cast impressions and clicks to int after imputation
    for col in ["impressions", "clicks"]:
        if col in df.columns:
            df[col] = df[col].round(0).astype(np.int64)

    # Outlier filtering on ad_spend
    df = filter_outliers_iqr(df, ["ad_spend"], factor=3.0, action="clip", dataset_name=dataset_name)

    logger.info("[%s] Cleaning complete: %d rows, %d columns", dataset_name, len(df), len(df.columns))
    return df


def clean_signups_data(
    df: pd.DataFrame,
    dataset_name: str = "Signups Data",
) -> pd.DataFrame:
    """
    Full cleaning pipeline for platform signup data.

    Steps
    -----
    1. Deduplicate on user_id (keep first signup)
    2. Normalise signup_timestamp to ISO 8601
    3. Standardise email to lowercase
    4. Drop rows with null user_id or campaign_id
    """
    logger.info("=" * 50)
    logger.info("Cleaning: %s", dataset_name)
    logger.info("=" * 50)

    df = deduplicate(df, subset=["user_id"], dataset_name=dataset_name)
    df = normalise_timestamps(df, column="signup_timestamp", output_format="datetime", dataset_name=dataset_name)

    # Lowercase emails
    if "email" in df.columns:
        null_emails = int(df["email"].isna().sum())
        df["email"] = df["email"].str.strip().str.lower()
        logger.info("[%s] Standardised emails to lowercase (%d null emails kept)", dataset_name, null_emails)

    # Drop rows missing critical identifiers
    before = len(df)
    df = df.dropna(subset=["user_id", "campaign_id"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("[%s] Dropped %d rows with null user_id/campaign_id", dataset_name, dropped)

    logger.info("[%s] Cleaning complete: %d rows, %d columns", dataset_name, len(df), len(df.columns))
    return df


def clean_events_data(
    df: pd.DataFrame,
    dataset_name: str = "Events Data",
) -> pd.DataFrame:
    """
    Full cleaning pipeline for user event logs.

    Steps
    -----
    1. Deduplicate on (user_id, event_type, event_timestamp)
    2. Normalise event_timestamp to datetime
    3. Drop rows with null user_id
    4. Standardise event_type to lowercase
    """
    logger.info("=" * 50)
    logger.info("Cleaning: %s", dataset_name)
    logger.info("=" * 50)

    df = deduplicate(
        df,
        subset=["user_id", "event_type", "event_timestamp"],
        dataset_name=dataset_name,
    )
    df = normalise_timestamps(df, column="event_timestamp", output_format="datetime", dataset_name=dataset_name)

    # Standardise event types
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].str.strip().str.lower()
        logger.info("[%s] Standardised event_type values: %s", dataset_name, sorted(df["event_type"].unique()))

    # Drop null user_ids
    before = len(df)
    df = df.dropna(subset=["user_id"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("[%s] Dropped %d rows with null user_id", dataset_name, dropped)

    logger.info("[%s] Cleaning complete: %d rows, %d columns", dataset_name, len(df), len(df.columns))
    return df


# ============================================================================
# 7. Master Cleaning Orchestrator
# ============================================================================

def clean_all_datasets(
    dataframes: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Apply the full cleaning pipeline to all 4 raw datasets.

    Parameters
    ----------
    dataframes : dict
        Mapping of dataset_key -> raw DataFrame from ingestion.
    config : dict | None
        Pipeline configuration.

    Returns
    -------
    dict
        Mapping of dataset_key -> cleaned DataFrame.
    """
    logger.info("#" * 60)
    logger.info("CLEANING PIPELINE START")
    logger.info("#" * 60)

    cleaned: Dict[str, pd.DataFrame] = {}

    # Google Ads
    if "google_ads" in dataframes:
        cleaned["google_ads"] = clean_ads_data(
            dataframes["google_ads"], dataset_name="Google Ads"
        )

    # Meta Ads
    if "meta_ads" in dataframes:
        cleaned["meta_ads"] = clean_ads_data(
            dataframes["meta_ads"], dataset_name="Meta Ads"
        )

    # Signups
    if "signups" in dataframes:
        cleaned["signups"] = clean_signups_data(
            dataframes["signups"], dataset_name="Platform Signups"
        )

    # Events
    if "events" in dataframes:
        cleaned["events"] = clean_events_data(
            dataframes["events"], dataset_name="User Events"
        )

    # Summary
    logger.info("#" * 60)
    logger.info("CLEANING PIPELINE COMPLETE")
    logger.info("#" * 60)
    for key, df in cleaned.items():
        raw_count = len(dataframes.get(key, pd.DataFrame()))
        clean_count = len(df)
        reduction = raw_count - clean_count
        logger.info(
            "  %-15s : %6d -> %6d rows (-%d, -%.1f%%)",
            key, raw_count, clean_count, reduction,
            (reduction / raw_count * 100) if raw_count > 0 else 0,
        )

    return cleaned


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    from src.ingestion import ingest_all_datasets

    logger.info("=" * 60)
    logger.info("Downstream Activation Engine -- Cleaning Module (Standalone)")
    logger.info("=" * 60)

    try:
        dataframes, _ = ingest_all_datasets()
        cleaned = clean_all_datasets(dataframes)
    except Exception as exc:
        logger.exception("Cleaning failed: %s", exc)
        raise
