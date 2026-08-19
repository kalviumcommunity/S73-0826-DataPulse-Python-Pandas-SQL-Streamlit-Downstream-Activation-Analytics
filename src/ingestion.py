"""
src/ingestion.py — Dataset Intake, Schema Enforcement & Quality Profiling
==========================================================================
Reads raw CSV and JSON datasets, validates against strict schemas using
Pydantic-inspired dataclass definitions, profiles data quality, and
auto-generates the docs/data_dictionary.md.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DataIngestionError,
    SchemaValidationError,
    DataQualityError,
    ensure_directories,
    load_config,
    setup_logger,
)

logger = setup_logger("ingestion")


# ============================================================================
# Schema Definitions
# ============================================================================

@dataclass
class ColumnSchema:
    """Schema for a single column in a dataset."""
    name: str
    dtype: str                           # "int64", "float64", "object", "datetime64"
    nullable: bool = False
    description: str = ""
    example: str = ""


@dataclass
class DatasetSchema:
    """Full schema definition for a dataset."""
    name: str
    description: str
    columns: List[ColumnSchema] = field(default_factory=list)
    min_rows: int = 0

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def required_columns(self) -> List[str]:
        return [c.name for c in self.columns if not c.nullable]

    def dtype_map(self) -> Dict[str, str]:
        return {c.name: c.dtype for c in self.columns}


# ---------------------------------------------------------------------------
# Pre-defined schemas for each raw dataset
# ---------------------------------------------------------------------------

GOOGLE_ADS_SCHEMA = DatasetSchema(
    name="Google Ads Impressions",
    description="Daily impression, click, and spend data from Google Ads campaigns.",
    columns=[
        ColumnSchema("campaign_id", "object", False, "Unique campaign identifier", "CAMP_G001"),
        ColumnSchema("campaign_name", "object", False, "Human-readable campaign name", "Summer Flash Sale - Display"),
        ColumnSchema("date", "object", False, "Date of the record (mixed formats possible)", "2025-06-15"),
        ColumnSchema("impressions", "int64", False, "Number of ad impressions served", "12345"),
        ColumnSchema("clicks", "float64", True, "Number of clicks (nullable due to data issues)", "543"),
        ColumnSchema("ad_spend", "float64", True, "Daily ad spend in USD", "350.25"),
        ColumnSchema("platform", "object", False, "Source platform identifier", "google_ads"),
    ],
    min_rows=100,
)

META_ADS_SCHEMA = DatasetSchema(
    name="Meta Ads Clicks",
    description="Daily impression, click, and spend data from Meta (Facebook/Instagram) Ads.",
    columns=[
        ColumnSchema("campaign_id", "object", False, "Unique campaign identifier", "CAMP_M001"),
        ColumnSchema("campaign_name", "object", False, "Human-readable campaign name", "Viral Video Promo - Reels"),
        ColumnSchema("date", "object", False, "Date of the record (mixed formats possible)", "2025-07-20"),
        ColumnSchema("impressions", "float64", True, "Number of ad impressions (nullable)", "8500"),
        ColumnSchema("clicks", "float64", True, "Number of clicks (nullable due to data issues)", "320"),
        ColumnSchema("ad_spend", "float64", False, "Daily ad spend in USD", "425.50"),
        ColumnSchema("platform", "object", False, "Source platform identifier", "meta_ads"),
    ],
    min_rows=100,
)

SIGNUPS_SCHEMA = DatasetSchema(
    name="Platform Signups",
    description="User signup records with nested profile data, originating from the web platform.",
    columns=[
        ColumnSchema("user_id", "object", False, "Unique user identifier", "USR_001234"),
        ColumnSchema("campaign_id", "object", False, "Campaign that drove the signup", "CAMP_G002"),
        ColumnSchema("signup_timestamp", "object", False, "ISO-8601 signup timestamp", "2025-07-15T14:30:00"),
        ColumnSchema("email", "object", True, "User email (extracted from nested profile)", "user@example.com"),
        ColumnSchema("full_name", "object", True, "User full name (extracted from nested profile)", "Jane Doe"),
        ColumnSchema("country", "object", True, "2-letter country code", "US"),
        ColumnSchema("referral_source", "object", False, "Traffic source for the signup", "google_search"),
    ],
    min_rows=100,
)

EVENTS_SCHEMA = DatasetSchema(
    name="User Event Logs",
    description="Post-signup user activity events including page views, feature usage, and purchases.",
    columns=[
        ColumnSchema("user_id", "object", False, "User who triggered the event", "USR_001234"),
        ColumnSchema("event_type", "object", False, "Type of event", "feature_used"),
        ColumnSchema("event_timestamp", "object", False, "Timestamp of the event", "2025-07-16 10:30:00"),
        ColumnSchema("session_id", "object", False, "Short session identifier", "a1b2c3d4"),
        ColumnSchema("metadata", "object", True, "JSON-encoded event metadata", '{"page": "/dashboard", "duration_sec": 45}'),
    ],
    min_rows=100,
)

ALL_SCHEMAS: Dict[str, DatasetSchema] = {
    "google_ads": GOOGLE_ADS_SCHEMA,
    "meta_ads": META_ADS_SCHEMA,
    "signups": SIGNUPS_SCHEMA,
    "events": EVENTS_SCHEMA,
}


# ============================================================================
# Ingestion Functions
# ============================================================================

def ingest_csv(
    filepath: Path,
    schema: DatasetSchema,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read a CSV file, validate against the provided schema, and return a DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    schema : DatasetSchema
        Expected schema definition.
    encoding : str
        File encoding.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    DataIngestionError
        If the file cannot be read.
    SchemaValidationError
        If schema validation fails.
    """
    logger.info("Ingesting CSV: %s", filepath)

    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except FileNotFoundError:
        raise DataIngestionError(
            f"File not found: {filepath}",
            context={"filepath": str(filepath)},
        )
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to read CSV: {exc}",
            context={"filepath": str(filepath), "error": str(exc)},
        )

    validate_schema(df, schema)
    logger.info("  ✓ Ingested %d rows × %d columns from %s", len(df), len(df.columns), filepath.name)
    return df


def ingest_json(
    filepath: Path,
    schema: DatasetSchema,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read a JSON file (list of records with nested objects), flatten, and validate.

    Nested ``user_profile`` objects are flattened into top-level columns:
    ``email``, ``full_name``, ``country``.

    Parameters
    ----------
    filepath : Path
        Path to the JSON file.
    schema : DatasetSchema
        Expected schema definition.

    Returns
    -------
    pd.DataFrame
        Flattened and validated DataFrame.
    """
    logger.info("Ingesting JSON: %s", filepath)

    try:
        with open(filepath, "r", encoding=encoding) as f:
            raw_data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        raise DataIngestionError(
            f"File not found: {filepath}",
            context={"filepath": str(filepath)},
        )
    except json.JSONDecodeError as exc:
        raise DataIngestionError(
            f"Invalid JSON: {exc}",
            context={"filepath": str(filepath), "error": str(exc)},
        )

    # Flatten nested user_profile
    flattened: List[Dict[str, Any]] = []
    for record in raw_data:
        flat: Dict[str, Any] = {
            "user_id": record.get("user_id"),
            "campaign_id": record.get("campaign_id"),
            "signup_timestamp": record.get("signup_timestamp"),
            "referral_source": record.get("referral_source"),
        }
        profile = record.get("user_profile", {})
        flat["email"] = profile.get("email")
        flat["full_name"] = profile.get("full_name")
        flat["country"] = profile.get("country")
        flattened.append(flat)

    df = pd.DataFrame(flattened)
    validate_schema(df, schema)
    logger.info("  ✓ Ingested %d rows × %d columns from %s", len(df), len(df.columns), filepath.name)
    return df


# ============================================================================
# Schema Validation
# ============================================================================

def validate_schema(df: pd.DataFrame, schema: DatasetSchema) -> None:
    """
    Validate a DataFrame against its schema definition.

    Checks
    ------
    1. Required columns are present.
    2. No unexpected columns (warning only).
    3. Minimum row count.
    4. Non-nullable columns do not have nulls.
    5. Data type compatibility.

    Raises
    ------
    SchemaValidationError
        On any critical validation failure.
    """
    errors: List[str] = []
    warnings: List[str] = []

    expected_cols = set(schema.column_names())
    actual_cols = set(df.columns.tolist())

    # 1. Missing columns
    missing = expected_cols - actual_cols
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    # 2. Extra columns (non-fatal)
    extra = actual_cols - expected_cols
    if extra:
        warnings.append(f"Unexpected extra columns (kept): {sorted(extra)}")

    # 3. Minimum rows
    if len(df) < schema.min_rows:
        errors.append(
            f"Row count {len(df)} is below minimum threshold {schema.min_rows}"
        )

    # 4. Non-nullable null checks
    for col_schema in schema.columns:
        if col_schema.name not in df.columns:
            continue
        if not col_schema.nullable:
            null_count = int(df[col_schema.name].isna().sum())
            if null_count > 0:
                warnings.append(
                    f"Column '{col_schema.name}' has {null_count} nulls "
                    f"but is marked non-nullable (data quality issue)"
                )

    # 5. Type compatibility (soft check — compare general category)
    dtype_category_map = {
        "int64": "numeric",
        "float64": "numeric",
        "object": "object",
        "datetime64": "datetime",
    }
    for col_schema in schema.columns:
        if col_schema.name not in df.columns:
            continue
        actual_dtype = str(df[col_schema.name].dtype)
        expected_category = dtype_category_map.get(col_schema.dtype, col_schema.dtype)
        actual_category = dtype_category_map.get(actual_dtype, actual_dtype)
        if expected_category != actual_category:
            warnings.append(
                f"Column '{col_schema.name}': expected {col_schema.dtype} "
                f"({expected_category}), got {actual_dtype} ({actual_category})"
            )

    # Log warnings
    for w in warnings:
        logger.warning("  ⚠ %s [%s]", w, schema.name)

    # Raise on errors
    if errors:
        error_msg = f"Schema validation failed for '{schema.name}': " + "; ".join(errors)
        raise SchemaValidationError(error_msg, context={"schema": schema.name, "errors": errors})

    logger.info("  ✓ Schema validation passed for '%s'", schema.name)


# ============================================================================
# Data Profiling
# ============================================================================

@dataclass
class ColumnProfile:
    """Statistical profile for a single column."""
    name: str
    dtype: str
    total_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: List[str]


@dataclass
class DatasetProfile:
    """Full profiling report for a dataset."""
    dataset_name: str
    filepath: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_pct: float
    columns: List[ColumnProfile]
    profiled_at: str


def profile_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    filepath: str = "",
) -> DatasetProfile:
    """
    Generate a comprehensive profile of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to profile.
    dataset_name : str
        Human-readable dataset name.
    filepath : str
        Source file path for reference.

    Returns
    -------
    DatasetProfile
        Profiling results.
    """
    col_profiles: List[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        total = len(series)
        unique_count = int(series.nunique(dropna=True))

        # Sample up to 3 non-null values
        non_null = series.dropna()
        samples = [str(v) for v in non_null.head(3).tolist()] if len(non_null) > 0 else []

        col_profiles.append(ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            total_count=total,
            null_count=null_count,
            null_pct=round(null_count / total * 100, 2) if total > 0 else 0.0,
            unique_count=unique_count,
            unique_pct=round(unique_count / total * 100, 2) if total > 0 else 0.0,
            sample_values=samples,
        ))

    dup_count = int(df.duplicated().sum())

    profile = DatasetProfile(
        dataset_name=dataset_name,
        filepath=filepath,
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=dup_count,
        duplicate_row_pct=round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0,
        columns=col_profiles,
        profiled_at=datetime.now().isoformat(),
    )

    logger.info("  Profile for '%s': %d rows, %d cols, %d duplicates (%.1f%%)",
                dataset_name, profile.row_count, profile.column_count,
                profile.duplicate_row_count, profile.duplicate_row_pct)

    return profile


def print_profile(profile: DatasetProfile) -> str:
    """Format a DatasetProfile as a readable string report."""
    lines: List[str] = [
        f"{'=' * 60}",
        f"DATA PROFILE: {profile.dataset_name}",
        f"{'=' * 60}",
        f"  Source    : {profile.filepath}",
        f"  Rows      : {profile.row_count:,}",
        f"  Columns   : {profile.column_count}",
        f"  Duplicates: {profile.duplicate_row_count:,} ({profile.duplicate_row_pct}%)",
        f"  Profiled  : {profile.profiled_at}",
        f"{'─' * 60}",
        f"  {'Column':<25} {'Type':<12} {'Nulls':>8} {'Null%':>7} {'Unique':>8} {'Uniq%':>7}",
        f"  {'─' * 25} {'─' * 12} {'─' * 8} {'─' * 7} {'─' * 8} {'─' * 7}",
    ]

    for cp in profile.columns:
        lines.append(
            f"  {cp.name:<25} {cp.dtype:<12} {cp.null_count:>8,} {cp.null_pct:>6.1f}% "
            f"{cp.unique_count:>8,} {cp.unique_pct:>6.1f}%"
        )

    report = "\n".join(lines)
    return report


# ============================================================================
# Data Dictionary Generator
# ============================================================================

def generate_data_dictionary(
    schemas: Dict[str, DatasetSchema],
    profiles: Optional[Dict[str, DatasetProfile]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Auto-generate a Markdown data dictionary from schema definitions
    and optional profiling results.

    Parameters
    ----------
    schemas : dict
        Mapping of dataset key → DatasetSchema.
    profiles : dict | None
        Mapping of dataset key → DatasetProfile (for row counts, null stats).
    output_path : Path | None
        If provided, write the dictionary to this file.

    Returns
    -------
    str
        Markdown content of the data dictionary.
    """
    lines: List[str] = [
        "# 📖 Data Dictionary",
        "",
        f"> Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This document describes all raw datasets consumed by the Downstream "
        "Activation Engine pipeline, including column definitions, data types, "
        "nullability, and sample values.",
        "",
        "---",
        "",
    ]

    for key, schema in schemas.items():
        profile = profiles.get(key) if profiles else None

        lines.append(f"## {schema.name}")
        lines.append("")
        lines.append(f"**Description:** {schema.description}")
        lines.append("")
        if profile:
            lines.append(f"- **Rows:** {profile.row_count:,}")
            lines.append(f"- **Columns:** {profile.column_count}")
            lines.append(f"- **Duplicate Rows:** {profile.duplicate_row_count:,} ({profile.duplicate_row_pct}%)")
        lines.append("")
        lines.append("| # | Column | Data Type | Nullable | Description | Example |")
        lines.append("|---|--------|-----------|----------|-------------|---------|")

        for i, col in enumerate(schema.columns, 1):
            nullable_str = "✅ Yes" if col.nullable else "❌ No"
            example = col.example or "—"
            lines.append(
                f"| {i} | `{col.name}` | `{col.dtype}` | {nullable_str} | "
                f"{col.description} | `{example}` |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    # Business metrics reference
    lines.extend([
        "## 📊 Core Business Metrics Reference",
        "",
        "| Metric | Formula | Target |",
        "|--------|---------|--------|",
        "| Click-Through Rate (CTR) | `(Total Clicks / Total Impressions) × 100` | — |",
        "| 7-Day Activation Rate | `(Signups with 1+ Key Action in 7 Days / Total Signups) × 100` | > 25% |",
        "| Cost Per Activated User (CPAU) | `Total Ad Spend / Total 7-Day Activated Users` | < $45.00 |",
        "| Vanity Ratio Index | `CTR (%) / 7-Day Activation Rate (%)` | Flagged if > 3.0 |",
        "",
        "### Activation Events",
        "",
        "A user is considered **activated** if they perform at least one of the following "
        "events within 7 days of signup:",
        "",
        "- `feature_used`",
        "- `profile_completed`",
        "- `item_added`",
        "- `purchase`",
        "",
    ])

    content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Data dictionary written to %s", output_path)

    return content


# ============================================================================
# Master Ingestion Orchestrator
# ============================================================================

def ingest_all_datasets(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DatasetProfile]]:
    """
    Ingest all 4 raw datasets, validate schemas, profile data quality,
    and generate the data dictionary.

    Returns
    -------
    tuple
        (dict of dataset_key → DataFrame, dict of dataset_key → DatasetProfile)
    """
    cfg = config or load_config()
    raw_dir = Path(cfg["data_raw_dir"])
    docs_dir = Path(cfg["docs_dir"])

    dataframes: Dict[str, pd.DataFrame] = {}
    profiles: Dict[str, DatasetProfile] = {}

    # Define ingestion tasks: (key, filename, schema, reader_func)
    tasks: List[Tuple[str, str, DatasetSchema, str]] = [
        ("google_ads", cfg["google_ads_file"], GOOGLE_ADS_SCHEMA, "csv"),
        ("meta_ads", cfg["meta_ads_file"], META_ADS_SCHEMA, "csv"),
        ("signups", cfg["signups_file"], SIGNUPS_SCHEMA, "json"),
        ("events", cfg["events_file"], EVENTS_SCHEMA, "csv"),
    ]

    for key, filename, schema, file_type in tasks:
        filepath = raw_dir / filename
        logger.info("─" * 50)
        logger.info("Processing: %s (%s)", schema.name, filepath.name)

        try:
            if file_type == "csv":
                df = ingest_csv(filepath, schema)
            else:
                df = ingest_json(filepath, schema)

            profile = profile_dataframe(df, schema.name, str(filepath))
            report = print_profile(profile)
            logger.info("\n%s", report)

            dataframes[key] = df
            profiles[key] = profile

        except (DataIngestionError, SchemaValidationError) as exc:
            logger.error("Failed to ingest '%s': %s", key, exc)
            raise

    # Generate data dictionary
    logger.info("─" * 50)
    logger.info("Generating data dictionary...")
    dict_path = docs_dir / cfg["data_dictionary_file"]
    generate_data_dictionary(ALL_SCHEMAS, profiles, dict_path)

    logger.info("=" * 50)
    logger.info("All datasets ingested and validated successfully.")
    return dataframes, profiles


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Downstream Activation Engine — Data Ingestion & Validation")
    logger.info("=" * 60)

    try:
        dfs, profs = ingest_all_datasets()
        logger.info("\nSummary:")
        for key, df in dfs.items():
            logger.info("  %-12s : %d rows × %d columns", key, len(df), len(df.columns))
    except Exception as exc:
        logger.exception("Ingestion pipeline failed: %s", exc)
        raise
