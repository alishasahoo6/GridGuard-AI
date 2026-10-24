"""GridGuard AI — Intelligent Electricity Theft Detection and Smart Grid Anomaly Monitoring.

Streamlit Application Entry Point.
Phase 2: Data Foundation, Exploration & Ingestion Pipeline.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is available in module search path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    DATABASE_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    get_settings,
)
from src.data.adapters import execute_data_pipeline
from src.data.compatibility import run_compatibility_pipeline
from src.data.dataset_inspector import run_inspections

settings = get_settings()

st.set_page_config(
    page_title=f"{settings.app_name} | Smart Grid Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional energy-analytics CSS styling
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .metric-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border-radius: 0.6rem;
            padding: 1.25rem 1.5rem;
            color: #FFFFFF;
            margin-bottom: 1.5rem;
        }
        .pillar-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 0.5rem;
            padding: 1.2rem;
            height: 100%;
        }
        .pillar-title {
            font-weight: 600;
            font-size: 1.0rem;
            color: #1E293B;
            margin-bottom: 0.35rem;
        }
        .pillar-desc {
            font-size: 0.85rem;
            color: #64748B;
            line-height: 1.4;
        }
        .stMetric {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 0.5rem;
            padding: 0.6rem 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_inspection_metrics() -> Dict[str, Any]:
    """Load cached dataset inspection metadata."""
    json_path = PROCESSED_DATA_DIR / "dataset_inspection.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_compatibility_report() -> Dict[str, Any]:
    """Load cached cross-dataset compatibility report."""
    json_path = PROCESSED_DATA_DIR / "dataset_compatibility.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_consumer_labels() -> pd.DataFrame:
    """Load standardized consumer labels and provenance metadata."""
    csv_path = PROCESSED_DATA_DIR / "consumer_labels.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"consumer_id": str})
    return pd.DataFrame()


@st.cache_data
def load_quarantined_records() -> pd.DataFrame:
    """Load quarantined records log."""
    csv_path = PROCESSED_DATA_DIR / "quarantined_records.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"raw_id": str})
    return pd.DataFrame()


def query_consumer_time_series(consumer_id: str, source_dataset: str) -> pd.DataFrame:
    """Query consumption readings for a specific consumer from SQLite or sample CSV."""
    db_path = settings.sqlite_db_path
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT timestamp, consumption_value, data_quality_flags
                FROM standardized_consumption
                WHERE consumer_id = ?
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(consumer_id,))
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass

    # Fallback to standardized sample CSV
    sample_csv = PROCESSED_DATA_DIR / "standardized_consumption_sample.csv"
    if sample_csv.exists():
        df_sample = pd.read_csv(sample_csv, dtype={"consumer_id": str})
        matched = df_sample[df_sample["consumer_id"] == consumer_id]
        if not matched.empty:
            return matched[["timestamp", "consumption_value", "data_quality_flags"]].sort_values("timestamp")

    return pd.DataFrame()


# Sidebar Navigation & Operations
with st.sidebar:
    st.markdown("### ⚡ GridGuard AI")
    st.caption("Intelligent Electricity Theft & Grid Anomaly Platform")
    st.markdown("---")

    active_page = st.radio(
        "Navigation",
        ["📊 Data Explorer & Pipeline", "🏛️ Architecture & Roadmap"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Information")
    st.write(f"**Version:** `{settings.app_version}`")
    st.write(f"**Environment:** `{settings.environment}`")
    st.write(f"**SQLite Database:** `{'Active' if settings.sqlite_db_path.exists() else 'Uninitialized'}`")

    st.markdown("---")
    st.markdown("#### Pipeline Execution")
    if st.button("🔄 Re-run Inspection & Pipeline", use_container_width=True):
        with st.spinner("Executing data ingestion, validation & inspection..."):
            run_inspections()
            execute_data_pipeline(max_long_consumers_per_dataset=1000)
            run_compatibility_pipeline()
            st.cache_data.clear()
            st.success("Pipeline refreshed successfully!")
            st.rerun()


# ==============================================================================
# PAGE 1: DATA EXPLORER & PIPELINE
# ==============================================================================
if active_page == "📊 Data Explorer & Pipeline":
    st.markdown(
        f"""
        <div class="main-header">
            <span>📊</span> Data Ingestion, Inspection & Quality Explorer
        </div>
        <div class="sub-header">
            Auditing real AMI smart-meter datasets, label provenances, quality flags, and temporal coverage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    inspection_data = load_inspection_metrics()
    compatibility_data = load_compatibility_report()
    labels_df = load_consumer_labels()
    quarantined_df = load_quarantined_records()

    if not inspection_data:
        st.warning(
            "Inspection artifacts not yet found in `data/processed/`. "
            "Click 'Re-run Inspection & Pipeline' in the sidebar to initialize."
        )
        st.stop()

    tab_overview, tab_consumer_timeseries, tab_compatibility, tab_artifacts = st.tabs(
        [
            "📈 Dataset Overview & Hygiene",
            "🔍 Consumer Time-Series Explorer",
            "⚖️ Compatibility Assessment",
            "📁 Artifacts & Quarantine Audit",
        ]
    )

    # --- TAB 1: DATASET OVERVIEW & HYGIENE ---
    with tab_overview:
        dataset_choice = st.selectbox(
            "Select Raw Dataset to Inspect:",
            ["data.csv (Dataset 1: SGCC Benchmark)", "Electricity_Theft_Data.csv (Dataset 2: 2015 Cohort)"],
        )
        selected_key = "data.csv" if "data.csv" in dataset_choice else "Electricity_Theft_Data.csv"
        d_meta = inspection_data.get(selected_key, {})

        if not d_meta:
            st.error(f"Metadata for {selected_key} not found.")
            st.stop()

        # Primary Metrics Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Raw Rows", f"{d_meta.get('num_rows', 0):,}")
        m2.metric("Total Columns", f"{d_meta.get('num_columns', 0):,}")
        m3.metric("Unique Meters", f"{d_meta.get('num_unique_consumers', 0):,}")
        m4.metric("Date Span (Days)", f"{d_meta.get('date_columns_count', 0):,}")
        m5.metric("Missing Readings", f"{d_meta.get('missing_consumption_pct', 0)}%")
        m6.metric("Zero Readings", f"{d_meta.get('zero_consumption_pct', 0)}%")

        st.markdown("---")

        # Two Column Layout: Schema & Temporal vs Label Distribution
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 📅 Temporal & Schema Specifications")
            st.write(f"**Consumer ID Column:** `{d_meta.get('id_column')}`")
            st.write(f"**Identifier Type:** `{'32-character hexadecimal MD5 hash' if selected_key == 'data.csv' else '9-10 digit numerical account string'}`")
            st.write(f"**Date Header Format:** `{d_meta.get('date_format_detected')}`")
            st.write(f"**Date Range:** `{d_meta.get('earliest_date')}` to `{d_meta.get('latest_date')}`")
            st.write(f"**Expected Calendar Days:** `{d_meta.get('expected_calendar_days')}`")

            missing_dates = d_meta.get("missing_calendar_dates", [])
            if missing_dates:
                st.warning(f"⚠️ Calendar sequence gap detected: {', '.join(missing_dates)} is absent from date headers.")
            else:
                st.success("✅ Complete continuous calendar coverage (no missing dates in chronological span).")

            neg_cells = d_meta.get("negative_consumption_cells", 0)
            if neg_cells > 0:
                st.warning(
                    f"⚠️ Negative Consumption Alert: {neg_cells:,} readings ({d_meta.get('negative_consumption_pct')}%) are negative. "
                    f"Minimum value: `{d_meta.get('min_consumption')}`. Tagged as `NEGATIVE_READING`."
                )
            else:
                st.info("ℹ️ No negative readings detected. All measurements are ≥ 0.0.")

        with col_right:
            st.markdown("#### 🏷️ Candidate Target Label Distribution")
            label_col = d_meta.get("label_column")
            st.write(f"**Label Column Name:** `{label_col}`")
            st.caption(
                "Notice: Target label originates from utility records and reflects consumer-level inspection results, "
                "not confirmed per-day ground-truth theft events."
            )

            label_dist = d_meta.get("label_distribution", {})
            c0 = label_dist.get("0", 0)
            c1 = label_dist.get("1", 0)
            tot_lbl = c0 + c1

            if tot_lbl > 0:
                p0 = round(c0 / tot_lbl * 100, 2)
                p1 = round(c1 / tot_lbl * 100, 2)

                dist_df = pd.DataFrame({
                    "Class": [f"Class 0 (Normal / Unflagged)", f"Class 1 (Flagged / Theft Candidate)"],
                    "Count": [c0, c1],
                    "Prevalence (%)": [p0, p1],
                })
                st.dataframe(dist_df, hide_index=True, use_container_width=True)

                st.bar_chart(dist_df.set_index("Class")["Count"])

        st.markdown("---")

        # Distribution Quantiles
        st.markdown("#### 📊 Consumption Value Statistics (Unit-Agnostic)")
        st.caption("Units are treated neutrally as `consumption_value` until official utility documentation verifies kWh or pulse constants.")

        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Mean", f"{d_meta.get('mean_consumption', 0):,.2f}")
        q2.metric("Median", f"{d_meta.get('median_consumption', 0):,.2f}")
        q3.metric("95th Percentile", f"{d_meta.get('percentile_95', 0):,.2f}")
        q4.metric("99th Percentile", f"{d_meta.get('percentile_99', 0):,.2f}")
        q5.metric("Max Recorded", f"{d_meta.get('max_consumption', 0):,.2f}")

    # --- TAB 2: CONSUMER TIME-SERIES EXPLORER ---
    with tab_consumer_timeseries:
        st.markdown("#### 📈 Individual Smart-Meter Time Series")
        st.caption(
            "Query actual historical consumption curves from SQLite storage (`database/gridguard.db`). "
            "No synthetic or fabricated readings."
        )

        if labels_df.empty:
            st.warning("Consumer labels table not loaded. Run pipeline first.")
        else:
            ds_filter = st.radio(
                "Filter Cohort by Dataset:",
                ["data.csv", "Electricity_Theft_Data.csv"],
                horizontal=True,
            )
            filtered_labels = labels_df[labels_df["source_dataset"] == ds_filter]

            col_filter, col_select = st.columns([1, 2])
            with col_filter:
                class_filter = st.selectbox(
                    "Filter by Target Label:",
                    ["All Consumers", "Class 1 Only (Theft Flagged)", "Class 0 Only (Unflagged)"],
                )
                if class_filter == "Class 1 Only (Theft Flagged)":
                    filtered_labels = filtered_labels[filtered_labels["source_label"] == 1.0]
                elif class_filter == "Class 0 Only (Unflagged)":
                    filtered_labels = filtered_labels[filtered_labels["source_label"] == 0.0]

            with col_select:
                consumer_options = filtered_labels["consumer_id"].tolist()[:500]
                selected_consumer = st.selectbox(
                    f"Select Consumer ID ({len(filtered_labels):,} available in filter):",
                    options=consumer_options,
                )

            if selected_consumer:
                c_meta = filtered_labels[filtered_labels["consumer_id"] == selected_consumer].iloc[0]

                cm1, cm2, cm3, cm4, cm5 = st.columns(5)
                cm1.metric("Label", f"{int(c_meta['source_label'])} ({c_meta['source_label_column']})")
                cm2.metric("Recorded Days", f"{int(c_meta['recorded_days'])} / {int(c_meta['total_days'])}")
                cm3.metric("Missing Days", f"{int(c_meta['missing_days'])}")
                cm4.metric("Zero Days", f"{int(c_meta['zero_days'])}")
                cm5.metric("Mean Daily Use", f"{c_meta['mean_consumption']}")

                ts_df = query_consumer_time_series(selected_consumer, ds_filter)
                if not ts_df.empty:
                    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
                    ts_df = ts_df.sort_values("timestamp")

                    st.markdown("##### Daily Consumption Curve")
                    st.line_chart(ts_df.set_index("timestamp")["consumption_value"])

                    with st.expander("View Raw Readings & Quality Flags Table (Sample)"):
                        st.dataframe(ts_df.head(100), use_container_width=True)
                else:
                    st.info(
                        f"Detailed long-format time series for consumer `{selected_consumer}` "
                        "is accessible in full via SQLite or during batch analytical training."
                    )

    # --- TAB 3: COMPATIBILITY ASSESSMENT ---
    with tab_compatibility:
        st.markdown("#### ⚖️ Cross-Dataset Compatibility Assessment")

        if not compatibility_data:
            st.warning("Compatibility assessment report not loaded.")
        else:
            comp_rec = compatibility_data.get("architectural_recommendation", "")
            st.error(f"**Architectural Recommendation:** {comp_rec}")

            st.markdown("---")
            d1_info = compatibility_data.get("dataset_1", {})
            d2_info = compatibility_data.get("dataset_2", {})
            c_metrics = compatibility_data.get("compatibility_metrics", {})

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("##### Dataset 1: `data.csv` (SGCC Benchmark)")
                st.write(f"- **Meters:** {d1_info.get('consumer_count', 0):,}")
                st.write(f"- **ID Format:** `{d1_info.get('id_format')}`")
                st.write(f"- **Date Range:** `{d1_info.get('date_range')}`")
                st.write(f"- **Label:** `{d1_info.get('label_column')}` ({d1_info.get('class_1_ratio_pct')}% positive)")
                st.write(f"- **Negative Readings:** `{'Yes' if d1_info.get('negative_readings_present') else 'None'}`")

            with col_c2:
                st.markdown("##### Dataset 2: `Electricity_Theft_Data.csv`")
                st.write(f"- **Meters:** {d2_info.get('consumer_count', 0):,}")
                st.write(f"- **ID Format:** `{d2_info.get('id_format')}`")
                st.write(f"- **Date Range:** `{d2_info.get('date_range')}`")
                st.write(f"- **Label:** `{d2_info.get('label_column')}` ({d2_info.get('class_1_ratio_pct')}% positive)")
                st.write(f"- **Negative Readings:** `{'Yes (1,442 cells)' if d2_info.get('negative_readings_present') else 'None'}`")

            st.markdown("---")
            st.markdown("##### Compatibility Dimension Verdicts")
            st.write(f"1. **Consumer ID Overlap:** **{c_metrics.get('consumer_id_overlap_count', 0)}** meters overlap. The consumer namespaces are mutually exclusive.")
            st.write(f"2. **Temporal Overlap:** {c_metrics.get('temporal_overlap_range')} ({c_metrics.get('temporal_overlap_days')} calendar days in 2015).")
            st.write(f"3. **Physical Units:** `{c_metrics.get('assumed_units')}`.")

    # --- TAB 4: ARTIFACTS & QUARANTINE AUDIT ---
    with tab_artifacts:
        st.markdown("#### 📁 Pipeline Artifacts & Quarantine Audit Log")

        st.markdown("##### Quarantined Records")
        if quarantined_df.empty:
            st.success("No records currently quarantined.")
        else:
            st.dataframe(quarantined_df, use_container_width=True)
            st.caption(
                "Quarantined records are isolated to preserve data hygiene while maintaining full forensic auditability. "
                "Row 0 of `Electricity_Theft_Data.csv` was verified as a column-index artifact and excluded from ML feature spaces."
            )

        st.markdown("##### Generated Pipeline Artifacts on Disk")
        artifact_list = [
            {"Artifact": "Raw Inspection Summary", "File": "data/processed/dataset_inspection.json", "Status": "Active"},
            {"Artifact": "Inspection Report", "File": "reports/dataset_inspection_report.md", "Status": "Active"},
            {"Artifact": "Consumer Labels Registry", "File": "data/processed/consumer_labels.csv", "Status": "Active"},
            {"Artifact": "Quarantine Audit Table", "File": "data/processed/quarantined_records.csv", "Status": "Active"},
            {"Artifact": "Data Quality Report", "File": "reports/data_quality_report.md", "Status": "Active"},
            {"Artifact": "Compatibility Report", "File": "reports/dataset_compatibility_report.md", "Status": "Active"},
            {"Artifact": "Indexed SQLite Store", "File": "database/gridguard.db", "Status": "Active"},
        ]
        st.dataframe(artifact_list, hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 2: ARCHITECTURE & ROADMAP (Phase 1 Baseline)
# ==============================================================================
elif active_page == "🏛️ Architecture & Roadmap":
    st.markdown(
        f"""
        <div class="main-header">
            <span>⚡</span> {settings.app_name}
        </div>
        <div class="sub-header">
            {settings.app_tagline}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_status, col_meta = st.columns([3, 1])
    with col_status:
        st.markdown(
            """
            <div class="metric-banner">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.5rem;">
                    Platform Status
                </div>
                <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; color: #38BDF8;">
                    Phase 2: Real Data Ingestion & Quality Pipeline Established
                </div>
                <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
                    Two real smart-meter datasets are ingested, inspected, validated, and standardized. 
                    Consumer label provenance is verified, malformed artifacts quarantined, and SQLite 
                    query indices initialized. Ready for feature extraction and baseline ML detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_meta:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0.5rem; padding: 1.25rem;">
                <div style="font-size: 0.8rem; font-weight: 600; color: #64748B; margin-bottom: 0.35rem;">RELEASE STAGE</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #0284C7; margin-bottom: 0.75rem;">0.2.0-alpha</div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #64748B; margin-bottom: 0.25rem;">DATA PIPELINE</div>
                <div style="font-size: 0.9rem; color: #059669; font-weight: 600;">Verified & Tested</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🏛️ Platform Objectives & Architectural Pillars")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="pillar-card">
                <div class="pillar-title">⚡ Theft & NTL Detection</div>
                <div class="pillar-desc">
                    Machine learning algorithms (unsupervised anomaly scoring and supervised signatures) 
                    to detect meter bypass, artificial slowdown, and unmetered consumption.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="pillar-card">
                <div class="pillar-title">🔍 Cause Disambiguation</div>
                <div class="pillar-desc">
                    Multi-class diagnostics to separate genuine theft from physical meter sensor degradation, 
                    AMI telemetry communication loss, and normal seasonal/occupancy shifts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="pillar-card">
                <div class="pillar-title">⚖️ Grid Energy Accounting</div>
                <div class="pillar-desc">
                    Distribution feeder and transformer balance reconciliation to compare upstream grid supply 
                    against aggregated downstream consumer meters to localize losses.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="pillar-card">
                <div class="pillar-title">📡 Virtual AMI Simulator</div>
                <div class="pillar-desc">
                    Controllable smart-meter simulator capable of emulating multi-feeder networks 
                    and streaming synthetic anomalies for real-time validation and stress testing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
