"""
src/synthetic_data.py — Realistic Synthetic Dataset Generator
==============================================================
Generates 4 raw marketing datasets across 10 campaigns with deliberate
archetypes (Vanity Trap, Hidden Gem, Balanced, Poor Performer) and
intentional data-quality issues (duplicates, nulls, format inconsistencies).

Datasets produced
-----------------
1. data/raw/google_ads_impressions.csv
2. data/raw/meta_ads_clicks.csv
3. data/raw/platform_signups.json
4. data/raw/user_event_logs.csv
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# Ensure project root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DataQualityError,
    ensure_directories,
    load_config,
    setup_logger,
)

logger = setup_logger("synthetic_data")
fake = Faker()
Faker.seed(42)

# ============================================================================
# Campaign Archetype Definitions
# ============================================================================

CampaignProfile = Dict[str, Any]

CAMPAIGN_PROFILES: List[CampaignProfile] = [
    # --- Vanity Traps: High CTR, very low activation (<5%) ---
    {
        "campaign_id": "CAMP_G001",
        "campaign_name": "Summer Flash Sale - Display",
        "platform": "google_ads",
        "archetype": "vanity_trap",
        "ctr_range": (0.05, 0.08),        # 5-8 % CTR
        "activation_rate": 0.03,           # 3 % activation
        "daily_spend_range": (200, 500),
        "daily_impression_range": (8000, 15000),
    },
    {
        "campaign_id": "CAMP_M001",
        "campaign_name": "Viral Video Promo - Reels",
        "platform": "meta_ads",
        "archetype": "vanity_trap",
        "ctr_range": (0.06, 0.09),
        "activation_rate": 0.04,
        "daily_spend_range": (300, 600),
        "daily_impression_range": (10000, 20000),
    },
    # --- Hidden Gems: Low CTR, high activation (>40%) ---
    {
        "campaign_id": "CAMP_G002",
        "campaign_name": "B2B Whitepaper Download",
        "platform": "google_ads",
        "archetype": "hidden_gem",
        "ctr_range": (0.005, 0.01),       # 0.5-1 % CTR
        "activation_rate": 0.45,           # 45 % activation
        "daily_spend_range": (50, 120),
        "daily_impression_range": (3000, 6000),
    },
    {
        "campaign_id": "CAMP_M002",
        "campaign_name": "Niche Community Outreach",
        "platform": "meta_ads",
        "archetype": "hidden_gem",
        "ctr_range": (0.004, 0.009),
        "activation_rate": 0.48,
        "daily_spend_range": (40, 100),
        "daily_impression_range": (2500, 5500),
    },
    # --- Balanced Performers ---
    {
        "campaign_id": "CAMP_G003",
        "campaign_name": "Product Launch - Search",
        "platform": "google_ads",
        "archetype": "balanced",
        "ctr_range": (0.025, 0.04),
        "activation_rate": 0.28,
        "daily_spend_range": (150, 350),
        "daily_impression_range": (5000, 10000),
    },
    {
        "campaign_id": "CAMP_M003",
        "campaign_name": "Retargeting - Lookalike Audience",
        "platform": "meta_ads",
        "archetype": "balanced",
        "ctr_range": (0.02, 0.035),
        "activation_rate": 0.30,
        "daily_spend_range": (120, 280),
        "daily_impression_range": (4500, 9000),
    },
    {
        "campaign_id": "CAMP_G004",
        "campaign_name": "Brand Awareness - YouTube",
        "platform": "google_ads",
        "archetype": "balanced",
        "ctr_range": (0.018, 0.030),
        "activation_rate": 0.26,
        "daily_spend_range": (180, 400),
        "daily_impression_range": (7000, 14000),
    },
    # --- Poor Performers ---
    {
        "campaign_id": "CAMP_M004",
        "campaign_name": "Generic Banner Ads",
        "platform": "meta_ads",
        "archetype": "poor_performer",
        "ctr_range": (0.008, 0.015),
        "activation_rate": 0.08,
        "daily_spend_range": (100, 250),
        "daily_impression_range": (6000, 12000),
    },
    {
        "campaign_id": "CAMP_G005",
        "campaign_name": "Expired Promo Remarketing",
        "platform": "google_ads",
        "archetype": "poor_performer",
        "ctr_range": (0.006, 0.012),
        "activation_rate": 0.06,
        "daily_spend_range": (80, 200),
        "daily_impression_range": (4000, 8000),
    },
    {
        "campaign_id": "CAMP_M005",
        "campaign_name": "Seasonal Clearance - Stories",
        "platform": "meta_ads",
        "archetype": "vanity_trap",
        "ctr_range": (0.04, 0.07),
        "activation_rate": 0.02,
        "daily_spend_range": (250, 550),
        "daily_impression_range": (9000, 18000),
    },
]


# ============================================================================
# Helpers
# ============================================================================

def _generate_date_range(
    start: str = "2025-06-01",
    end: str = "2025-08-31",
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex spanning the campaign window."""
    return pd.date_range(start=start, end=end, freq="D")


def _apply_case_noise(name: str, rng: np.random.Generator) -> str:
    """Randomly alter casing to simulate messy source data."""
    choice = rng.integers(0, 4)
    if choice == 0:
        return name.upper()
    elif choice == 1:
        return name.lower()
    elif choice == 2:
        return f"  {name}  "          # leading/trailing whitespace
    return name                        # unchanged


# ============================================================================
# Dataset 1 & 2: Impressions / Clicks (CSV)
# ============================================================================

def generate_ads_data(
    profiles: List[CampaignProfile],
    platform_filter: str,
    rng: np.random.Generator,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Generate daily impression & click records for campaigns on a given platform.

    Returns a DataFrame with columns:
        campaign_id, campaign_name, date, impressions, clicks, ad_spend, platform
    """
    rows: List[Dict[str, Any]] = []
    filtered = [p for p in profiles if p["platform"] == platform_filter]

    for profile in filtered:
        ctr_lo, ctr_hi = profile["ctr_range"]
        spend_lo, spend_hi = profile["daily_spend_range"]
        imp_lo, imp_hi = profile["daily_impression_range"]

        for day in date_range:
            impressions = int(rng.integers(imp_lo, imp_hi + 1))
            ctr = rng.uniform(ctr_lo, ctr_hi)
            clicks = int(round(impressions * ctr))
            ad_spend = round(float(rng.uniform(spend_lo, spend_hi)), 2)

            # Inject occasional date-format inconsistencies
            if rng.random() < 0.05:
                date_str = day.strftime("%m/%d/%Y")     # US format noise
            elif rng.random() < 0.03:
                date_str = day.strftime("%d-%b-%Y")     # 19-Jun-2025 format
            else:
                date_str = day.strftime("%Y-%m-%d")     # ISO standard

            rows.append({
                "campaign_id": profile["campaign_id"],
                "campaign_name": _apply_case_noise(profile["campaign_name"], rng),
                "date": date_str,
                "impressions": impressions,
                "clicks": clicks,
                "ad_spend": ad_spend,
                "platform": platform_filter,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Dataset 3: Platform Signups (JSON)
# ============================================================================

def generate_signups(
    profiles: List[CampaignProfile],
    rng: np.random.Generator,
    date_range: pd.DatetimeIndex,
    target_total: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Generate user signup records distributed across campaigns proportional
    to their CTR (higher CTR → more clicks → more signups).

    Returns a list of dicts (to be serialised as JSON).
    """
    # Compute signup weights from midpoint CTR × midpoint impressions
    weights: List[float] = []
    for p in profiles:
        mid_ctr = np.mean(p["ctr_range"])
        mid_imp = np.mean(p["daily_impression_range"])
        weights.append(mid_ctr * mid_imp * len(date_range))
    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()

    signup_counts = rng.multinomial(target_total, weights_arr)
    signups: List[Dict[str, Any]] = []
    user_counter = 1000

    referral_sources = ["google_search", "direct", "social_media", "email", "referral", "organic"]

    for profile, count in zip(profiles, signup_counts):
        for _ in range(int(count)):
            user_counter += 1
            user_id = f"USR_{user_counter:06d}"

            # Random signup timestamp within the date range
            random_day = rng.choice(date_range)
            hour = int(rng.integers(0, 24))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            signup_ts = pd.Timestamp(random_day) + timedelta(
                hours=hour, minutes=minute, seconds=second
            )

            record: Dict[str, Any] = {
                "user_id": user_id,
                "campaign_id": profile["campaign_id"],
                "signup_timestamp": signup_ts.isoformat(),
                "user_profile": {
                    "email": fake.email(),
                    "full_name": fake.name(),
                    "country": fake.country_code(representation="alpha-2"),
                },
                "referral_source": str(rng.choice(referral_sources)),
            }

            # Inject null email ~1 %
            if rng.random() < 0.01:
                record["user_profile"]["email"] = None

            signups.append(record)

    return signups


# ============================================================================
# Dataset 4: User Event Logs (CSV)
# ============================================================================

EVENT_TYPES: List[str] = [
    "page_view",
    "feature_used",
    "profile_completed",
    "item_added",
    "purchase",
    "support_ticket",
    "settings_changed",
    "invite_sent",
]

# Key activation events (at least 1 within 7 days = activated)
ACTIVATION_EVENTS: set = {"feature_used", "profile_completed", "item_added", "purchase"}


def generate_user_events(
    signups: List[Dict[str, Any]],
    profiles: List[CampaignProfile],
    rng: np.random.Generator,
    event_multiplier: int = 3,
) -> pd.DataFrame:
    """
    Generate post-signup event streams. The probability that a user produces
    an *activation event* within 7 days is governed by their campaign's
    ``activation_rate``.

    Non-activated users will ONLY receive non-activation events within the
    7-day window (page_view, support_ticket, settings_changed, invite_sent),
    ensuring archetype fidelity (Vanity Traps < 5%, Hidden Gems > 40%).
    """
    profile_map = {p["campaign_id"]: p for p in profiles}
    rows: List[Dict[str, Any]] = []

    # Split event types for controlled assignment
    non_activation_events = [e for e in EVENT_TYPES if e not in ACTIVATION_EVENTS]
    activation_event_list = list(ACTIVATION_EVENTS)

    for signup in signups:
        user_id = signup["user_id"]
        campaign_id = signup["campaign_id"]
        signup_ts = pd.Timestamp(signup["signup_timestamp"])
        activation_rate = profile_map[campaign_id]["activation_rate"]

        # Decide if this user will be an "activated" user
        is_activated = rng.random() < activation_rate
        num_events = int(rng.integers(1, event_multiplier * 2 + 1))

        for i in range(num_events):
            # Events within 0-14 days after signup
            day_offset = int(rng.integers(0, 15))
            hour_offset = int(rng.integers(0, 24))
            event_ts = signup_ts + timedelta(days=day_offset, hours=hour_offset)

            if is_activated and i == 0 and day_offset > 7:
                # Ensure at least one activation event falls within the window
                day_offset = int(rng.integers(0, 7))
                event_ts = signup_ts + timedelta(days=day_offset, hours=hour_offset)

            # Pick event type based on activation status and timing
            if is_activated and i == 0:
                # First event for activated user = guaranteed activation event
                event_type = str(rng.choice(activation_event_list))
            elif not is_activated and day_offset <= 7:
                # Non-activated user within 7-day window: ONLY non-activation events
                event_type = str(rng.choice(non_activation_events))
            else:
                # All other cases: any event type is fine
                # (activated users' later events, or non-activated users after day 7)
                event_type = str(rng.choice(EVENT_TYPES))

            rows.append({
                "user_id": user_id,
                "event_type": event_type,
                "event_timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": fake.uuid4()[:8],
                "metadata": json.dumps({"page": fake.uri_path(), "duration_sec": int(rng.integers(5, 300))}),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Data-Quality Injection
# ============================================================================

def inject_duplicates(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Append a random sample of existing rows to simulate duplicate records."""
    n_dupes = max(1, int(len(df) * rate))
    dupe_indices = rng.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_indices].copy()
    logger.info("Injecting %d duplicate rows (%.1f%%)", n_dupes, rate * 100)
    return pd.concat([df, dupes], ignore_index=True)


def inject_nulls(
    df: pd.DataFrame,
    columns: List[str],
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Set random cells to NaN in specified columns."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
        n_nulls = int(mask.sum())
        logger.info("Injected %d nulls into column '%s'", n_nulls, col)
    return df


# ============================================================================
# Orchestrator
# ============================================================================

def generate_all_datasets(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Master function — generates all 4 raw datasets and writes them to disk.

    Returns
    -------
    dict
        Mapping of dataset name → file path.
    """
    cfg = config or load_config()
    rng = np.random.default_rng(cfg["random_seed"])
    ensure_directories(cfg)

    raw_dir = Path(cfg["data_raw_dir"])
    date_range = _generate_date_range()
    output_paths: Dict[str, Path] = {}

    # ---- 1. Google Ads Impressions ----
    logger.info("Generating Google Ads impressions data...")
    google_df = generate_ads_data(CAMPAIGN_PROFILES, "google_ads", rng, date_range)
    google_df = inject_duplicates(google_df, cfg["duplicate_rate"], rng)
    google_df = inject_nulls(google_df, ["clicks", "ad_spend"], cfg["null_injection_rate"], rng)
    google_path = raw_dir / cfg["google_ads_file"]
    google_df.to_csv(google_path, index=False)
    output_paths["google_ads"] = google_path
    logger.info("  → Wrote %d rows to %s", len(google_df), google_path.name)

    # ---- 2. Meta Ads Clicks ----
    logger.info("Generating Meta Ads clicks data...")
    meta_df = generate_ads_data(CAMPAIGN_PROFILES, "meta_ads", rng, date_range)
    meta_df = inject_duplicates(meta_df, cfg["duplicate_rate"], rng)
    meta_df = inject_nulls(meta_df, ["clicks", "impressions"], cfg["null_injection_rate"], rng)
    meta_path = raw_dir / cfg["meta_ads_file"]
    meta_df.to_csv(meta_path, index=False)
    output_paths["meta_ads"] = meta_path
    logger.info("  → Wrote %d rows to %s", len(meta_df), meta_path.name)

    # ---- 3. Platform Signups (JSON) ----
    logger.info("Generating platform signups data...")
    signups = generate_signups(
        CAMPAIGN_PROFILES, rng, date_range, target_total=cfg["signup_target"]
    )
    signups_path = raw_dir / cfg["signups_file"]
    with open(signups_path, "w", encoding="utf-8") as f:
        json.dump(signups, f, indent=2, default=str)
    output_paths["signups"] = signups_path
    logger.info("  → Wrote %d signup records to %s", len(signups), signups_path.name)

    # ---- 4. User Event Logs ----
    logger.info("Generating user event logs...")
    events_df = generate_user_events(
        signups, CAMPAIGN_PROFILES, rng, event_multiplier=cfg["event_multiplier"]
    )
    events_df = inject_duplicates(events_df, cfg["duplicate_rate"], rng)
    events_path = raw_dir / cfg["events_file"]
    events_df.to_csv(events_path, index=False)
    output_paths["events"] = events_path
    logger.info("  → Wrote %d event rows to %s", len(events_df), events_path.name)

    logger.info("All 4 datasets generated successfully.")
    return output_paths


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Downstream Activation Engine — Synthetic Data Generator")
    logger.info("=" * 60)

    try:
        paths = generate_all_datasets()
        logger.info("Output files:")
        for name, path in paths.items():
            size_kb = path.stat().st_size / 1024
            logger.info("  %-12s → %s (%.1f KB)", name, path, size_kb)
    except Exception as exc:
        logger.exception("Synthetic data generation failed: %s", exc)
        raise
