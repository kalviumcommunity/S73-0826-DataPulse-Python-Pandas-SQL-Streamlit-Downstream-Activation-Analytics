# 📖 Data Dictionary

> Auto-generated on 2026-08-19 14:46:28

This document describes all raw datasets consumed by the Downstream Activation Engine pipeline, including column definitions, data types, nullability, and sample values.

---

## Google Ads Impressions

**Description:** Daily impression, click, and spend data from Google Ads campaigns.

- **Rows:** 469
- **Columns:** 7
- **Duplicate Rows:** 9 (1.92%)

| # | Column | Data Type | Nullable | Description | Example |
|---|--------|-----------|----------|-------------|---------|
| 1 | `campaign_id` | `object` | ❌ No | Unique campaign identifier | `CAMP_G001` |
| 2 | `campaign_name` | `object` | ❌ No | Human-readable campaign name | `Summer Flash Sale - Display` |
| 3 | `date` | `object` | ❌ No | Date of the record (mixed formats possible) | `2025-06-15` |
| 4 | `impressions` | `int64` | ❌ No | Number of ad impressions served | `12345` |
| 5 | `clicks` | `float64` | ✅ Yes | Number of clicks (nullable due to data issues) | `543` |
| 6 | `ad_spend` | `float64` | ✅ Yes | Daily ad spend in USD | `350.25` |
| 7 | `platform` | `object` | ❌ No | Source platform identifier | `google_ads` |

---

## Meta Ads Clicks

**Description:** Daily impression, click, and spend data from Meta (Facebook/Instagram) Ads.

- **Rows:** 469
- **Columns:** 7
- **Duplicate Rows:** 9 (1.92%)

| # | Column | Data Type | Nullable | Description | Example |
|---|--------|-----------|----------|-------------|---------|
| 1 | `campaign_id` | `object` | ❌ No | Unique campaign identifier | `CAMP_M001` |
| 2 | `campaign_name` | `object` | ❌ No | Human-readable campaign name | `Viral Video Promo - Reels` |
| 3 | `date` | `object` | ❌ No | Date of the record (mixed formats possible) | `2025-07-20` |
| 4 | `impressions` | `float64` | ✅ Yes | Number of ad impressions (nullable) | `8500` |
| 5 | `clicks` | `float64` | ✅ Yes | Number of clicks (nullable due to data issues) | `320` |
| 6 | `ad_spend` | `float64` | ❌ No | Daily ad spend in USD | `425.50` |
| 7 | `platform` | `object` | ❌ No | Source platform identifier | `meta_ads` |

---

## Platform Signups

**Description:** User signup records with nested profile data, originating from the web platform.

- **Rows:** 5,000
- **Columns:** 7
- **Duplicate Rows:** 0 (0.0%)

| # | Column | Data Type | Nullable | Description | Example |
|---|--------|-----------|----------|-------------|---------|
| 1 | `user_id` | `object` | ❌ No | Unique user identifier | `USR_001234` |
| 2 | `campaign_id` | `object` | ❌ No | Campaign that drove the signup | `CAMP_G002` |
| 3 | `signup_timestamp` | `object` | ❌ No | ISO-8601 signup timestamp | `2025-07-15T14:30:00` |
| 4 | `email` | `object` | ✅ Yes | User email (extracted from nested profile) | `user@example.com` |
| 5 | `full_name` | `object` | ✅ Yes | User full name (extracted from nested profile) | `Jane Doe` |
| 6 | `country` | `object` | ✅ Yes | 2-letter country code | `US` |
| 7 | `referral_source` | `object` | ❌ No | Traffic source for the signup | `google_search` |

---

## User Event Logs

**Description:** Post-signup user activity events including page views, feature usage, and purchases.

- **Rows:** 18,056
- **Columns:** 5
- **Duplicate Rows:** 354 (1.96%)

| # | Column | Data Type | Nullable | Description | Example |
|---|--------|-----------|----------|-------------|---------|
| 1 | `user_id` | `object` | ❌ No | User who triggered the event | `USR_001234` |
| 2 | `event_type` | `object` | ❌ No | Type of event | `feature_used` |
| 3 | `event_timestamp` | `object` | ❌ No | Timestamp of the event | `2025-07-16 10:30:00` |
| 4 | `session_id` | `object` | ❌ No | Short session identifier | `a1b2c3d4` |
| 5 | `metadata` | `object` | ✅ Yes | JSON-encoded event metadata | `{"page": "/dashboard", "duration_sec": 45}` |

---

## 📊 Core Business Metrics Reference

| Metric | Formula | Target |
|--------|---------|--------|
| Click-Through Rate (CTR) | `(Total Clicks / Total Impressions) × 100` | — |
| 7-Day Activation Rate | `(Signups with 1+ Key Action in 7 Days / Total Signups) × 100` | > 25% |
| Cost Per Activated User (CPAU) | `Total Ad Spend / Total 7-Day Activated Users` | < $45.00 |
| Vanity Ratio Index | `CTR (%) / 7-Day Activation Rate (%)` | Flagged if > 3.0 |

### Activation Events

A user is considered **activated** if they perform at least one of the following events within 7 days of signup:

- `feature_used`
- `profile_completed`
- `item_added`
- `purchase`
