"""
app.py — Interactive Streamlit Executive Dashboard
====================================================
Premium B2B SaaS-styled dashboard for the Downstream Activation Engine.
Provides KPI scorecards, vanity trap alerts, funnel visualization,
CTR vs Activation scatter plot, and a sortable campaign table.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import DATA_PROCESSED_DIR, load_config

# ============================================================================
# Theme & Constants
# ============================================================================

# B2B SaaS color palette
COLORS = {
    "navy": "#1E3A8A",
    "navy_light": "#2563EB",
    "navy_dark": "#1E2A4A",
    "teal": "#0D9488",
    "teal_light": "#14B8A6",
    "teal_dark": "#0F766E",
    "red": "#E11D48",
    "red_light": "#FB7185",
    "amber": "#F59E0B",
    "amber_light": "#FCD34D",
    "green": "#10B981",
    "green_light": "#34D399",
    "slate": "#1E293B",
    "slate_light": "#334155",
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "white": "#FFFFFF",
}

CATEGORY_COLORS = {
    "High Value": COLORS["green"],
    "Scaling": COLORS["teal"],
    "Vanity Trap": COLORS["red"],
    "Under-Performer": COLORS["amber"],
}

CATEGORY_ICONS = {
    "High Value": "🏆",
    "Scaling": "📈",
    "Vanity Trap": "⚠️",
    "Under-Performer": "📉",
}


# ============================================================================
# Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="Downstream Activation Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject custom CSS for the premium dark B2B SaaS theme."""
    st.markdown(f"""
    <style>
        /* ---- Global ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp {{
            background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, #0C1222 50%, {COLORS['slate']} 100%);
            font-family: 'Inter', sans-serif;
            color: {COLORS['text_primary']};
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['navy_dark']} 0%, {COLORS['slate']} 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {COLORS['text_primary']};
        }}

        /* ---- Header ---- */
        .dashboard-header {{
            background: linear-gradient(135deg, {COLORS['navy']} 0%, {COLORS['navy_light']} 100%);
            border-radius: 16px;
            padding: 28px 36px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .dashboard-header h1 {{
            color: {COLORS['white']};
            font-size: 28px;
            font-weight: 800;
            margin: 0 0 4px 0;
            letter-spacing: -0.5px;
        }}
        .dashboard-header p {{
            color: {COLORS['text_secondary']};
            font-size: 14px;
            margin: 0;
            font-weight: 400;
        }}

        /* ---- KPI Cards ---- */
        .kpi-card {{
            background: linear-gradient(145deg, {COLORS['bg_card']} 0%, {COLORS['slate_light']} 100%);
            border-radius: 14px;
            padding: 22px 24px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }}
        .kpi-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: {COLORS['text_muted']};
            font-weight: 400;
        }}
        .kpi-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-top: 6px;
        }}

        /* ---- Alert Banner ---- */
        .vanity-alert {{
            background: linear-gradient(135deg, rgba(225,29,72,0.15) 0%, rgba(225,29,72,0.05) 100%);
            border: 1px solid {COLORS['red']};
            border-radius: 12px;
            padding: 16px 24px;
            margin: 16px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .vanity-alert-icon {{
            font-size: 24px;
        }}
        .vanity-alert-text {{
            color: {COLORS['red_light']};
            font-weight: 600;
            font-size: 14px;
        }}
        .vanity-alert-detail {{
            color: {COLORS['text_secondary']};
            font-size: 12px;
            margin-top: 2px;
        }}

        .no-alert {{
            background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.04) 100%);
            border: 1px solid {COLORS['green']};
            border-radius: 12px;
            padding: 16px 24px;
            margin: 16px 0;
        }}

        /* ---- Section headers ---- */
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin: 32px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid {COLORS['teal']};
            display: inline-block;
        }}

        /* ---- Campaign table badges ---- */
        .badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .badge-high-value {{
            background: rgba(16,185,129,0.2);
            color: {COLORS['green_light']};
            border: 1px solid rgba(16,185,129,0.4);
        }}
        .badge-scaling {{
            background: rgba(13,148,136,0.2);
            color: {COLORS['teal_light']};
            border: 1px solid rgba(13,148,136,0.4);
        }}
        .badge-vanity-trap {{
            background: rgba(225,29,72,0.2);
            color: {COLORS['red_light']};
            border: 1px solid rgba(225,29,72,0.4);
        }}
        .badge-under-performer {{
            background: rgba(245,158,11,0.2);
            color: {COLORS['amber_light']};
            border: 1px solid rgba(245,158,11,0.4);
        }}

        /* ---- Plotly charts ---- */
        .stPlotlyChart {{
            background: {COLORS['bg_card']};
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
            padding: 8px;
        }}

        /* ---- Hide default streamlit branding ---- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* ---- Dataframe styling ---- */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=300)
def load_campaign_data() -> pd.DataFrame:
    """Load campaign metrics from Parquet."""
    path = DATA_PROCESSED_DIR / "master_activated_campaigns.parquet"
    if not path.exists():
        st.error(f"Campaign data not found at `{path}`. Run `python pipeline_run.py --phase 3` first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_user_data() -> pd.DataFrame:
    """Load user activation details from Parquet."""
    path = DATA_PROCESSED_DIR / "user_activation_details.parquet"
    if not path.exists():
        st.error(f"User data not found at `{path}`. Run `python pipeline_run.py --phase 3` first.")
        st.stop()
    return pd.read_parquet(path)


# ============================================================================
# Sidebar Filters
# ============================================================================

def render_sidebar(campaigns: pd.DataFrame, users: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters and return filter state."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 20px 0;">
            <span style="font-size: 36px;">🚀</span>
            <h2 style="margin: 8px 0 2px 0; font-size: 18px; font-weight: 800; color: #F1F5F9;">
                Activation Engine
            </h2>
            <p style="font-size: 11px; color: #64748B; margin: 0; text-transform: uppercase; letter-spacing: 1.5px;">
                Executive Dashboard
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Platform filter
        st.markdown("##### 🎯 Platform Filter")
        platforms = ["All"] + sorted(campaigns["platform"].unique().tolist())
        selected_platform = st.selectbox(
            "Select Platform",
            platforms,
            index=0,
            label_visibility="collapsed",
            key="platform_filter",
        )

        st.markdown("")

        # Performance category filter
        st.markdown("##### 📊 Performance Category")
        categories = campaigns["performance_category"].unique().tolist()
        selected_categories = st.multiselect(
            "Filter Categories",
            categories,
            default=categories,
            label_visibility="collapsed",
            key="category_filter",
        )

        st.markdown("")

        # Activation window slider
        st.markdown("##### ⏱️ Activation Window")
        activation_window = st.slider(
            "Days after signup",
            min_value=1,
            max_value=14,
            value=7,
            key="activation_window",
        )

        st.markdown("---")

        # Data summary
        st.markdown("##### 📦 Data Summary")
        st.markdown(f"""
        <div style="font-size: 12px; color: #94A3B8; line-height: 1.8;">
            Campaigns: <b style="color: #F1F5F9;">{len(campaigns)}</b><br>
            Total Users: <b style="color: #F1F5F9;">{len(users):,}</b><br>
            Platforms: <b style="color: #F1F5F9;">{campaigns['platform'].nunique()}</b>
        </div>
        """, unsafe_allow_html=True)

    return {
        "platform": selected_platform,
        "categories": selected_categories,
        "activation_window": activation_window,
    }


def apply_filters(
    campaigns: pd.DataFrame,
    users: pd.DataFrame,
    filters: Dict[str, Any],
) -> tuple:
    """Apply sidebar filters to campaign and user DataFrames."""
    df = campaigns.copy()
    udf = users.copy()

    if filters["platform"] != "All":
        df = df[df["platform"] == filters["platform"]]
        udf = udf[udf["platform"] == filters["platform"]]

    if filters["categories"]:
        df = df[df["performance_category"].isin(filters["categories"])]
        if "campaign_id" in udf.columns:
            udf = udf[udf["campaign_id"].isin(df["campaign_id"])]

    # Recompute activation for custom window
    window = filters["activation_window"]
    if window != 7 and "days_to_first_activation" in udf.columns:
        udf["is_activated_custom"] = np.where(
            udf["days_to_first_activation"].notna()
            & (udf["days_to_first_activation"] <= window)
            & (udf["days_to_first_activation"] >= 0),
            1, 0
        ).astype(int)

        # Reaggregate campaign metrics for custom window
        custom_agg = (
            udf.groupby("campaign_id", as_index=False)
            .agg(
                activated_custom=("is_activated_custom", "sum"),
                total_signups_custom=("user_id", "nunique"),
            )
        )
        df = df.merge(custom_agg, on="campaign_id", how="left")
        df["activation_rate_display"] = np.where(
            df["total_signups_custom"] > 0,
            (df["activated_custom"] / df["total_signups_custom"] * 100).round(2),
            0.0,
        )
        df["cpau_display"] = np.where(
            df["activated_custom"] > 0,
            (df["total_ad_spend"] / df["activated_custom"]).round(2),
            99999.99,
        )
        df["activated_display"] = df["activated_custom"]
    else:
        df["activation_rate_display"] = df["activation_rate_pct"]
        df["cpau_display"] = df["cpau_usd"]
        df["activated_display"] = df["activated_users_7d"]

    return df, udf


# ============================================================================
# Dashboard Components
# ============================================================================

def render_header() -> None:
    """Render the dashboard header."""
    st.markdown("""
    <div class="dashboard-header">
        <h1>🚀 Downstream Activation Engine</h1>
        <p>Translating top-of-funnel vanity traffic into true downstream business value</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render the 4 top-row KPI scorecards."""
    total_spend = df["total_ad_spend"].sum()
    total_impressions = df["total_impressions"].sum()
    total_clicks = df["total_clicks"].sum()
    blended_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    total_signups = df["total_signups"].sum()
    total_activated = int(df["activated_display"].sum())
    activation_rate = (total_activated / total_signups * 100) if total_signups > 0 else 0
    blended_cpau = (total_spend / total_activated) if total_activated > 0 else 0

    window = filters["activation_window"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: {COLORS['text_secondary']};">💰 Total Ad Spend</div>
            <div class="kpi-value" style="color: {COLORS['white']};">${total_spend:,.0f}</div>
            <div class="kpi-sub">Across {len(df)} campaigns</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ctr_color = COLORS["amber"] if blended_ctr > 3 else COLORS["text_secondary"]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: {COLORS['amber']};">👁️ Blended CTR</div>
            <div class="kpi-value" style="color: {ctr_color};">{blended_ctr:.2f}%</div>
            <div class="kpi-sub">{total_clicks:,} clicks / {total_impressions:,} impressions</div>
            <div class="kpi-badge" style="background: rgba(245,158,11,0.15); color: {COLORS['amber_light']};">
                Vanity Metric
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        cfg = load_config()
        target = cfg.get("activation_rate_target_pct", 25.0)
        act_color = COLORS["green"] if activation_rate >= target else COLORS["red"]
        status = "On Target" if activation_rate >= target else "Below Target"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: {COLORS['teal']};">✅ {window}-Day Activation Rate</div>
            <div class="kpi-value" style="color: {act_color};">{activation_rate:.1f}%</div>
            <div class="kpi-sub">{total_activated:,} activated / {total_signups:,} signups</div>
            <div class="kpi-badge" style="background: rgba(13,148,136,0.15); color: {COLORS['teal_light']};">
                Core Value Metric
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        cpau_target = cfg.get("cpau_target_usd", 45.0)
        cpau_color = COLORS["green"] if blended_cpau <= cpau_target else COLORS["red"]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: {COLORS['text_secondary']};">🎯 Blended CPAU</div>
            <div class="kpi-value" style="color: {cpau_color};">${blended_cpau:,.2f}</div>
            <div class="kpi-sub">Target: &lt; ${cpau_target:.0f}</div>
        </div>
        """, unsafe_allow_html=True)


def render_vanity_alert(df: pd.DataFrame) -> None:
    """Render dynamic alert banner for Vanity Trap campaigns."""
    vanity_traps = df[df["is_vanity_trap"] == 1]

    if len(vanity_traps) > 0:
        trap_names = ", ".join(vanity_traps["campaign_name"].str.title().tolist())
        total_wasted = vanity_traps["total_ad_spend"].sum()
        st.markdown(f"""
        <div class="vanity-alert">
            <div class="vanity-alert-icon">🚨</div>
            <div>
                <div class="vanity-alert-text">
                    {len(vanity_traps)} Campaign(s) Flagged as Vanity Trap
                </div>
                <div class="vanity-alert-detail">
                    {trap_names} &mdash; Combined spend at risk: ${total_wasted:,.0f}.
                    These campaigns show high CTR but very low downstream activation.
                    Consider pausing or optimising post-click experience.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-alert">
            <span style="font-size: 18px;">✅</span>
            <span style="color: #34D399; font-weight: 600; font-size: 14px; margin-left: 8px;">
                No Vanity Trap Alerts
            </span>
            <span style="color: #94A3B8; font-size: 12px; margin-left: 8px;">
                All campaigns have healthy activation-to-CTR ratios.
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_funnel_chart(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render multi-stage conversion funnel using Plotly."""
    total_impressions = int(df["total_impressions"].sum())
    total_clicks = int(df["total_clicks"].sum())
    total_signups = int(df["total_signups"].sum())
    total_activated = int(df["activated_display"].sum())

    window = filters["activation_window"]

    stages = [
        f"Impressions ({total_impressions:,})",
        f"Clicks ({total_clicks:,})",
        f"Signups ({total_signups:,})",
        f"{window}-Day Activated ({total_activated:,})",
    ]
    values = [total_impressions, total_clicks, total_signups, total_activated]

    # Compute drop-off percentages
    dropoffs = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            drop = ((values[i - 1] - values[i]) / values[i - 1]) * 100
            dropoffs.append(f"-{drop:.1f}%")
        else:
            dropoffs.append("")

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        textfont=dict(size=14, family="Inter"),
        marker=dict(
            color=[COLORS["navy_light"], COLORS["teal"], COLORS["green"], COLORS["teal_dark"]],
            line=dict(width=1, color="rgba(255,255,255,0.1)"),
        ),
        connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
    ))

    fig.update_layout(
        title=dict(
            text=f"Conversion Funnel ({window}-Day Window)",
            font=dict(size=16, color=COLORS["text_primary"], family="Inter"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"], family="Inter"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render CTR vs Activation Rate scatter plot with performance quadrants."""
    cfg = load_config()
    activation_target = cfg.get("activation_rate_target_pct", 25.0)
    window = filters["activation_window"]

    plot_df = df.copy()
    plot_df["activation_plot"] = plot_df["activation_rate_display"]
    plot_df["bubble_size"] = np.clip(plot_df["total_ad_spend"] / 1000, 5, 50)
    plot_df["campaign_label"] = plot_df["campaign_name"].str.title()

    # Determine CTR midpoint for quadrant line
    ctr_mid = plot_df["ctr_pct"].median()

    fig = go.Figure()

    # Add quadrant background shading
    max_ctr = float(plot_df["ctr_pct"].max()) * 1.3
    max_act = float(plot_df["activation_plot"].max()) * 1.15

    # Quadrant labels
    quadrant_annotations = [
        dict(x=max_ctr * 0.75, y=max_act * 0.85,
             text="<b>High CTR + High Activation</b><br><i>Stars</i>",
             font=dict(size=10, color="rgba(52,211,153,0.5)"), showarrow=False),
        dict(x=ctr_mid * 0.3, y=max_act * 0.85,
             text="<b>Low CTR + High Activation</b><br><i>Hidden Gems</i>",
             font=dict(size=10, color="rgba(20,184,166,0.5)"), showarrow=False),
        dict(x=max_ctr * 0.75, y=activation_target * 0.4,
             text="<b>High CTR + Low Activation</b><br><i>Vanity Traps</i>",
             font=dict(size=10, color="rgba(251,113,133,0.5)"), showarrow=False),
        dict(x=ctr_mid * 0.3, y=activation_target * 0.4,
             text="<b>Low CTR + Low Activation</b><br><i>Under-Performers</i>",
             font=dict(size=10, color="rgba(148,163,184,0.4)"), showarrow=False),
    ]

    # Scatter points by performance category
    for category, color in CATEGORY_COLORS.items():
        mask = plot_df["performance_category"] == category
        subset = plot_df[mask]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=subset["ctr_pct"],
            y=subset["activation_plot"],
            mode="markers+text",
            marker=dict(
                size=subset["bubble_size"],
                color=color,
                line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
                opacity=0.85,
            ),
            text=subset["campaign_id"],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["text_secondary"]),
            name=f"{CATEGORY_ICONS.get(category, '')} {category}",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "CTR: %{x:.2f}%<br>"
                f"{window}-Day Activation: %{{y:.1f}}%<br>"
                "CPAU: $%{customdata[1]:,.2f}<br>"
                "Ad Spend: $%{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
            customdata=subset[["campaign_label", "cpau_display", "total_ad_spend"]].values,
        ))

    # Quadrant lines
    fig.add_hline(y=activation_target, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.add_vline(x=ctr_mid, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"CTR (%) vs {window}-Day Activation Rate (%)",
            font=dict(size=16, color=COLORS["text_primary"], family="Inter"),
        ),
        xaxis=dict(
            title="Click-Through Rate (%)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            color=COLORS["text_secondary"],
        ),
        yaxis=dict(
            title=f"{window}-Day Activation Rate (%)",
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            color=COLORS["text_secondary"],
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"], family="Inter"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11),
        ),
        annotations=quadrant_annotations,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_campaign_table(df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    """Render a sortable, color-coded campaign table."""
    window = filters["activation_window"]

    def make_badge(category: str) -> str:
        css_class = category.lower().replace(" ", "-").replace("-", "-")
        icon = CATEGORY_ICONS.get(category, "")
        return f'<span class="badge badge-{css_class}">{icon} {category}</span>'

    display_df = df[[
        "campaign_id", "campaign_name", "platform",
        "total_impressions", "total_clicks", "ctr_pct",
        "total_signups", "activated_display", "activation_rate_display",
        "total_ad_spend", "cpau_display", "vanity_ratio_index",
        "performance_category",
    ]].copy()

    display_df.columns = [
        "Campaign ID", "Campaign Name", "Platform",
        "Impressions", "Clicks", "CTR %",
        "Signups", f"{window}d Activated", f"{window}d Act. Rate %",
        "Ad Spend ($)", "CPAU ($)", "Vanity Ratio",
        "Category",
    ]

    display_df["Campaign Name"] = display_df["Campaign Name"].str.title()
    display_df = display_df.sort_values("CPAU ($)", ascending=True).reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    # Streamlit dataframe with column formatting
    st.dataframe(
        display_df.style
        .format({
            "Impressions": "{:,.0f}",
            "Clicks": "{:,.0f}",
            "CTR %": "{:.2f}%",
            "Signups": "{:,.0f}",
            f"{window}d Activated": "{:,.0f}",
            f"{window}d Act. Rate %": "{:.1f}%",
            "Ad Spend ($)": "${:,.2f}",
            "CPAU ($)": "${:,.2f}",
            "Vanity Ratio": "{:.2f}",
        })
        .background_gradient(
            subset=["CPAU ($)"],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=float(display_df["CPAU ($)"].max()),
        )
        .background_gradient(
            subset=[f"{window}d Act. Rate %"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
        ),
        use_container_width=True,
        height=420,
    )

    # Category legend below table
    badge_html = " &nbsp; ".join(
        f'<span class="badge badge-{cat.lower().replace(" ", "-")}">'
        f'{CATEGORY_ICONS.get(cat, "")} {cat}</span>'
        for cat in sorted(CATEGORY_COLORS.keys())
    )
    st.markdown(
        f'<div style="text-align: center; margin-top: 8px;">{badge_html}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Main App
# ============================================================================

def main() -> None:
    """Main Streamlit application entry point."""
    inject_custom_css()

    # Load data
    campaigns = load_campaign_data()
    users = load_user_data()

    # Sidebar filters
    filters = render_sidebar(campaigns, users)

    # Apply filters
    filtered_campaigns, filtered_users = apply_filters(campaigns, users, filters)

    if len(filtered_campaigns) == 0:
        st.warning("No campaigns match the current filters. Adjust your selections.")
        return

    # Header
    render_header()

    # KPI Cards
    render_kpi_cards(filtered_campaigns, filters)

    # Vanity Alert Banner
    st.markdown("")
    render_vanity_alert(filtered_campaigns)

    # Middle Grid: Funnel + Scatter
    st.markdown('<div class="section-title">📊 Campaign Performance Analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        render_funnel_chart(filtered_campaigns, filters)
    with col_right:
        render_scatter_plot(filtered_campaigns, filters)

    # Bottom: Campaign Table
    st.markdown(
        '<div class="section-title">📋 Campaign Leaderboard</div>',
        unsafe_allow_html=True,
    )
    render_campaign_table(filtered_campaigns, filters)

    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 32px 0 16px 0; color: {COLORS['text_muted']}; font-size: 11px;">
        Downstream Activation Engine v1.0 &mdash;
        Built with Streamlit + Plotly &mdash;
        Data refreshed from pipeline output
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
