"""
src/sql_layer.py — SQLite/PostgreSQL View Generator via SQLAlchemy
====================================================================
Pushes processed campaign and user-level data to SQLite, then creates
CTE-based analytical views with window functions for executive reporting.

Views created
-------------
1. v_campaign_executive_summary : RANK() by CPAU, performance flags
2. v_activation_funnel          : Multi-stage funnel (impressions -> clicks -> signups -> activated)
3. v_vanity_trap_alerts         : Campaigns flagged as Vanity Traps with details
4. v_user_activation_timeline   : User-level activation with days-to-activate distribution
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    PipelineError,
    load_config,
    setup_logger,
)

logger = setup_logger("sql_layer")


# ============================================================================
# 1. Database Connection & Table Loading
# ============================================================================

def get_engine(db_path: Optional[Path] = None) -> Engine:
    """
    Create a SQLAlchemy engine for SQLite.

    Parameters
    ----------
    db_path : Path | None
        Path to the SQLite database file.
        Defaults to ``<project>/data/processed/activation_engine.db``.

    Returns
    -------
    sqlalchemy.Engine
    """
    if db_path is None:
        from src.utils import DATA_PROCESSED_DIR
        db_path = DATA_PROCESSED_DIR / "activation_engine.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    logger.info("SQLite engine created: %s", db_path)
    return engine


def push_dataframes_to_sql(
    engine: Engine,
    campaigns: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    """
    Write campaign-level and user-level DataFrames to SQLite tables.

    Tables created/replaced
    -----------------------
    - ``campaign_metrics``  : One row per campaign
    - ``user_activation``   : One row per user
    """
    logger.info("Pushing DataFrames to SQLite...")

    # Campaign metrics table
    campaigns.to_sql("campaign_metrics", engine, if_exists="replace", index=False)
    logger.info("  Table 'campaign_metrics': %d rows written", len(campaigns))

    # User activation table
    users.to_sql("user_activation", engine, if_exists="replace", index=False)
    logger.info("  Table 'user_activation': %d rows written", len(users))

    # Verify
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info("  Tables in database: %s", tables)


# ============================================================================
# 2. CTE-Based Analytical Views
# ============================================================================

# --- View 1: Executive Campaign Summary with RANK ---
VIEW_CAMPAIGN_EXECUTIVE_SUMMARY = """
CREATE VIEW IF NOT EXISTS v_campaign_executive_summary AS
WITH ranked_campaigns AS (
    SELECT
        campaign_id,
        campaign_name,
        platform,
        total_impressions,
        total_clicks,
        total_ad_spend,
        total_signups,
        activated_users_7d,
        ctr_pct,
        activation_rate_pct,
        cpau_usd,
        vanity_ratio_index,
        is_vanity_trap,
        performance_category,
        cpau_rank,
        avg_daily_spend,
        campaign_days,
        -- Window functions for ranking and percentiles
        RANK() OVER (ORDER BY cpau_usd ASC) AS cpau_efficiency_rank,
        RANK() OVER (ORDER BY activation_rate_pct DESC) AS activation_rank,
        RANK() OVER (ORDER BY vanity_ratio_index DESC) AS vanity_risk_rank,
        -- Running totals
        SUM(total_ad_spend) OVER (ORDER BY cpau_usd ASC) AS cumulative_spend_by_efficiency,
        SUM(activated_users_7d) OVER (ORDER BY cpau_usd ASC) AS cumulative_activations_by_efficiency,
        -- Percentage of total
        ROUND(total_ad_spend * 100.0 / SUM(total_ad_spend) OVER (), 2) AS spend_share_pct,
        ROUND(activated_users_7d * 100.0 / NULLIF(SUM(activated_users_7d) OVER (), 0), 2) AS activation_share_pct
    FROM campaign_metrics
)
SELECT
    *,
    CASE
        WHEN spend_share_pct > activation_share_pct * 1.5 THEN 'Over-Invested'
        WHEN spend_share_pct < activation_share_pct * 0.5 THEN 'Under-Invested'
        ELSE 'Balanced'
    END AS budget_alignment
FROM ranked_campaigns
ORDER BY cpau_efficiency_rank ASC
"""

# --- View 2: Multi-Stage Activation Funnel ---
VIEW_ACTIVATION_FUNNEL = """
CREATE VIEW IF NOT EXISTS v_activation_funnel AS
WITH funnel_stages AS (
    SELECT
        campaign_id,
        campaign_name,
        platform,
        performance_category,
        total_impressions AS stage_1_impressions,
        total_clicks AS stage_2_clicks,
        total_signups AS stage_3_signups,
        activated_users_7d AS stage_4_activated,
        -- Conversion rates between stages
        ROUND(total_clicks * 100.0 / NULLIF(total_impressions, 0), 4) AS cvr_impression_to_click,
        ROUND(total_signups * 100.0 / NULLIF(total_clicks, 0), 2) AS cvr_click_to_signup,
        ROUND(activated_users_7d * 100.0 / NULLIF(total_signups, 0), 2) AS cvr_signup_to_activated,
        -- End-to-end conversion
        ROUND(activated_users_7d * 100.0 / NULLIF(total_impressions, 0), 6) AS cvr_end_to_end
    FROM campaign_metrics
)
SELECT
    *,
    -- Biggest drop-off stage
    CASE
        WHEN cvr_impression_to_click <= cvr_click_to_signup
             AND cvr_impression_to_click <= cvr_signup_to_activated
        THEN 'Impression->Click'
        WHEN cvr_click_to_signup <= cvr_signup_to_activated
        THEN 'Click->Signup'
        ELSE 'Signup->Activation'
    END AS biggest_dropoff_stage
FROM funnel_stages
ORDER BY cvr_end_to_end DESC
"""

# --- View 3: Vanity Trap Alerts ---
VIEW_VANITY_TRAP_ALERTS = """
CREATE VIEW IF NOT EXISTS v_vanity_trap_alerts AS
SELECT
    campaign_id,
    campaign_name,
    platform,
    ctr_pct,
    activation_rate_pct,
    vanity_ratio_index,
    cpau_usd,
    total_ad_spend,
    total_signups,
    activated_users_7d,
    -- Wasted spend estimation (spend on non-activated signups)
    ROUND(total_ad_spend - (cpau_usd * activated_users_7d), 2) AS estimated_wasted_spend,
    -- Severity level
    CASE
        WHEN vanity_ratio_index > 10.0 THEN 'CRITICAL'
        WHEN vanity_ratio_index > 5.0 THEN 'HIGH'
        WHEN vanity_ratio_index > 3.0 THEN 'MODERATE'
        ELSE 'LOW'
    END AS alert_severity,
    -- Recommended action
    CASE
        WHEN vanity_ratio_index > 10.0 THEN 'Pause campaign immediately. Review targeting and creative.'
        WHEN vanity_ratio_index > 5.0 THEN 'Reduce budget by 50%. A/B test new landing page.'
        WHEN vanity_ratio_index > 3.0 THEN 'Monitor closely. Optimise post-click experience.'
        ELSE 'No action required.'
    END AS recommended_action
FROM campaign_metrics
WHERE is_vanity_trap = 1
ORDER BY vanity_ratio_index DESC
"""

# --- View 4: User Activation Timeline ---
VIEW_USER_ACTIVATION_TIMELINE = """
CREATE VIEW IF NOT EXISTS v_user_activation_timeline AS
WITH user_buckets AS (
    SELECT
        ua.user_id,
        ua.campaign_id,
        cm.campaign_name,
        cm.platform,
        cm.performance_category,
        ua.is_activated_7d,
        ua.days_to_first_activation,
        -- Day bucket for activation speed
        CASE
            WHEN ua.days_to_first_activation IS NULL THEN 'Never'
            WHEN ua.days_to_first_activation <= 1 THEN 'Day 0-1'
            WHEN ua.days_to_first_activation <= 3 THEN 'Day 2-3'
            WHEN ua.days_to_first_activation <= 7 THEN 'Day 4-7'
            WHEN ua.days_to_first_activation <= 14 THEN 'Day 8-14'
            ELSE 'Day 15+'
        END AS activation_speed_bucket
    FROM user_activation ua
    LEFT JOIN campaign_metrics cm ON ua.campaign_id = cm.campaign_id
)
SELECT
    campaign_id,
    campaign_name,
    platform,
    performance_category,
    activation_speed_bucket,
    COUNT(*) AS user_count,
    SUM(is_activated_7d) AS activated_count,
    ROUND(AVG(days_to_first_activation), 2) AS avg_days_to_activate
FROM user_buckets
GROUP BY campaign_id, campaign_name, platform, performance_category, activation_speed_bucket
ORDER BY campaign_id, activation_speed_bucket
"""

# Collected view definitions
ALL_VIEWS: Dict[str, str] = {
    "v_campaign_executive_summary": VIEW_CAMPAIGN_EXECUTIVE_SUMMARY,
    "v_activation_funnel": VIEW_ACTIVATION_FUNNEL,
    "v_vanity_trap_alerts": VIEW_VANITY_TRAP_ALERTS,
    "v_user_activation_timeline": VIEW_USER_ACTIVATION_TIMELINE,
}


# ============================================================================
# 3. View Creation
# ============================================================================

def create_analytical_views(engine: Engine) -> List[str]:
    """
    Create all CTE-based analytical views in the database.

    Returns
    -------
    list[str]
        Names of views successfully created.
    """
    logger.info("Creating analytical views...")
    created: List[str] = []

    with engine.connect() as conn:
        for view_name, ddl in ALL_VIEWS.items():
            try:
                # Drop if exists (for idempotent re-runs)
                conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
                conn.execute(text(ddl))
                conn.commit()

                # Verify by querying row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {view_name}"))
                row_count = result.scalar()
                logger.info("  Created view '%s' (%d rows)", view_name, row_count)
                created.append(view_name)

            except Exception as exc:
                logger.error("  Failed to create view '%s': %s", view_name, exc)
                raise PipelineError(
                    f"View creation failed: {view_name}",
                    context={"view": view_name, "error": str(exc)},
                ) from exc

    return created


# ============================================================================
# 4. Query Helpers
# ============================================================================

def query_view(engine: Engine, view_name: str) -> pd.DataFrame:
    """Execute a SELECT * on a view and return as DataFrame."""
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {view_name}"), conn)
    return df


def run_custom_query(engine: Engine, sql: str) -> pd.DataFrame:
    """Execute an arbitrary SQL query and return results as DataFrame."""
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


# ============================================================================
# 5. Master SQL Orchestrator
# ============================================================================

def build_sql_layer(
    campaigns: pd.DataFrame,
    users: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Engine, List[str]]:
    """
    Full SQL layer pipeline: push data to SQLite, create analytical views.

    Parameters
    ----------
    campaigns : pd.DataFrame
        Campaign-level metrics from feature engineering.
    users : pd.DataFrame
        User-level activation data from feature engineering.
    config : dict | None
        Pipeline configuration.

    Returns
    -------
    tuple[Engine, list[str]]
        SQLAlchemy engine and list of created view names.
    """
    logger.info("#" * 60)
    logger.info("SQL LAYER PIPELINE START")
    logger.info("#" * 60)

    # Step 1: Create engine and push data
    engine = get_engine()
    push_dataframes_to_sql(engine, campaigns, users)

    # Step 2: Create views
    created_views = create_analytical_views(engine)

    # Step 3: Preview each view
    logger.info("\nView previews:")
    for view_name in created_views:
        df = query_view(engine, view_name)
        logger.info("  %s: %d rows x %d columns", view_name, len(df), len(df.columns))

    # Summary
    logger.info("#" * 60)
    logger.info("SQL LAYER PIPELINE COMPLETE")
    logger.info("#" * 60)
    logger.info("  Database tables : campaign_metrics, user_activation")
    logger.info("  Analytical views: %d created", len(created_views))

    return engine, created_views


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    from src.ingestion import ingest_all_datasets
    from src.cleaning import clean_all_datasets
    from src.joining import join_all_datasets
    from src.feature_engineering import engineer_features

    logger.info("=" * 60)
    logger.info("Downstream Activation Engine -- SQL Layer (Standalone)")
    logger.info("=" * 60)

    try:
        dfs, _ = ingest_all_datasets()
        cleaned = clean_all_datasets(dfs)
        master, _ = join_all_datasets(cleaned)
        campaigns, users = engineer_features(master)
        engine, views = build_sql_layer(campaigns, users)

        # Show executive summary
        exec_summary = query_view(engine, "v_campaign_executive_summary")
        logger.info("\nExecutive Summary:\n%s", exec_summary.to_string())

        # Show vanity alerts
        vanity_alerts = query_view(engine, "v_vanity_trap_alerts")
        logger.info("\nVanity Trap Alerts:\n%s", vanity_alerts.to_string())

    except Exception as exc:
        logger.exception("SQL layer failed: %s", exc)
        raise
