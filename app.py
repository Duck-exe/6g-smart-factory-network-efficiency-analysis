
# app.py
# Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories
# Streamlit Dashboard + EDA + KPI Analysis

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="6G Smart Factory Manufacturing Efficiency Dashboard",
    page_icon="🏭",
    layout="wide"
)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_path = "Thales_Group_Manufacturing.csv"
        if not os.path.exists(default_path):
            st.error("CSV file not found. Upload the dataset from the sidebar or place Thales_Group_Manufacturing.csv in the same folder as app.py.")
            st.stop()
        df = pd.read_csv(default_path)

    # Clean column names just in case
    df.columns = df.columns.str.strip()

    # Convert Date + Timestamp to datetime
    if "Date" in df.columns and "Timestamp" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Timestamp"].astype(str),
            errors="coerce",
            dayfirst=True
        )
    elif "Date" in df.columns:
        df["DateTime"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    else:
        df["DateTime"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Drop rows with no datetime only if actual datetime column exists
    if pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        df = df.dropna(subset=["DateTime"])

    # Numeric columns
    numeric_cols = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score",
        "Error_Rate_%"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric missing values with median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    for col in ["Operation_Mode", "Efficiency_Status", "Machine_ID"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------
def add_network_features(df):
    df = df.copy()

    latency = df["Network_Latency_ms"]
    packet_loss = df["Packet_Loss_%"]

    # Network quality bands using quantiles
    latency_low = latency.quantile(0.33)
    latency_high = latency.quantile(0.66)
    packet_low = packet_loss.quantile(0.33)
    packet_high = packet_loss.quantile(0.66)

    def classify_network(row):
        bad_score = 0

        if row["Network_Latency_ms"] > latency_high:
            bad_score += 2
        elif row["Network_Latency_ms"] > latency_low:
            bad_score += 1

        if row["Packet_Loss_%"] > packet_high:
            bad_score += 2
        elif row["Packet_Loss_%"] > packet_low:
            bad_score += 1

        if bad_score <= 1:
            return "High Quality"
        elif bad_score <= 3:
            return "Medium Quality"
        else:
            return "Low Quality"

    df["Network_Quality"] = df.apply(classify_network, axis=1)

    # Normalize latency and packet loss from 0 to 1
    df["Latency_Normalized"] = (
        (df["Network_Latency_ms"] - df["Network_Latency_ms"].min()) /
        (df["Network_Latency_ms"].max() - df["Network_Latency_ms"].min())
    )

    df["Packet_Loss_Normalized"] = (
        (df["Packet_Loss_%"] - df["Packet_Loss_%"].min()) /
        (df["Packet_Loss_%"].max() - df["Packet_Loss_%"].min())
    )

    # Network Stability Index: higher means better network
    df["Network_Stability_Index"] = 100 * (
        1 - (0.6 * df["Latency_Normalized"] + 0.4 * df["Packet_Loss_Normalized"])
    )
    df["Network_Stability_Index"] = df["Network_Stability_Index"].clip(0, 100)

    # Real-time vs delayed periods
    latency_median = df["Network_Latency_ms"].median()
    df["Communication_Period"] = np.where(
        df["Network_Latency_ms"] <= latency_median,
        "Real-Time / Low Latency",
        "Delayed / High Latency"
    )

    # Efficiency numeric mapping for correlation and sensitivity
    efficiency_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Efficiency_Score"] = df["Efficiency_Status"].map(efficiency_map)

    return df


# -------------------------------
# 3. KPI CALCULATION
# -------------------------------
def calculate_kpis(df):
    avg_latency = df["Network_Latency_ms"].mean()
    avg_packet_loss = df["Packet_Loss_%"].mean()
    avg_stability = df["Network_Stability_Index"].mean()
    avg_speed = df["Production_Speed_units_per_hr"].mean()
    avg_error = df["Error_Rate_%"].mean()
    avg_defect = df["Quality_Control_Defect_Rate_%"].mean()

    network_eff_corr = df[["Network_Stability_Index", "Efficiency_Score"]].corr().iloc[0, 1]

    # Latency Sensitivity Score: production speed drop per 1 ms latency increase
    latency_speed_corr = df[["Network_Latency_ms", "Production_Speed_units_per_hr"]].corr().iloc[0, 1]

    # Packet Loss Impact Ratio: error increase relationship with packet loss
    packet_error_corr = df[["Packet_Loss_%", "Error_Rate_%"]].corr().iloc[0, 1]

    return {
        "Average Network Stability Index": avg_stability,
        "Average Latency ms": avg_latency,
        "Average Packet Loss %": avg_packet_loss,
        "Average Production Speed": avg_speed,
        "Average Error Rate %": avg_error,
        "Average Defect Rate %": avg_defect,
        "Network-Efficiency Correlation": network_eff_corr,
        "Latency Sensitivity Score": latency_speed_corr,
        "Packet Loss Impact Ratio": packet_error_corr
    }


# -------------------------------
# 4. SIDEBAR
# -------------------------------
st.sidebar.title("🏭 6G Smart Factory Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Manufacturing CSV", type=["csv"])

df = load_data(uploaded_file)
df = add_network_features(df)

st.sidebar.markdown("### Filters")

network_options = sorted(df["Network_Quality"].dropna().unique().tolist())
selected_network = st.sidebar.multiselect(
    "Select Network Quality",
    network_options,
    default=network_options
)

efficiency_options = sorted(df["Efficiency_Status"].dropna().unique().tolist())
selected_efficiency = st.sidebar.multiselect(
    "Select Efficiency Status",
    efficiency_options,
    default=efficiency_options
)

mode_options = sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Select Operation Mode",
    mode_options,
    default=mode_options
)

# Time filter
if pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
    min_date = df["DateTime"].min().date()
    max_date = df["DateTime"].max().date()
    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_dates = None

filtered_df = df[
    (df["Network_Quality"].isin(selected_network)) &
    (df["Efficiency_Status"].isin(selected_efficiency)) &
    (df["Operation_Mode"].isin(selected_modes))
]

if selected_dates and len(selected_dates) == 2 and pd.api.types.is_datetime64_any_dtype(filtered_df["DateTime"]):
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
        (filtered_df["DateTime"].dt.date >= start_date) &
        (filtered_df["DateTime"].dt.date <= end_date)
    ]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()


# -------------------------------
# 5. HEADER
# -------------------------------
st.title("Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories")
st.markdown(
    """
    This dashboard analyzes how **6G network latency** and **packet loss** affect 
    production speed, defect rates, error rates, and manufacturing efficiency.
    """
)


# -------------------------------
# 6. KPI SCORECARDS
# -------------------------------
st.subheader("Key Performance Indicators")

kpis = calculate_kpis(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Network Stability Index", f"{kpis['Average Network Stability Index']:.2f}/100")
col2.metric("Avg Latency", f"{kpis['Average Latency ms']:.2f} ms")
col3.metric("Avg Packet Loss", f"{kpis['Average Packet Loss %']:.2f}%")
col4.metric("Avg Production Speed", f"{kpis['Average Production Speed']:.2f} units/hr")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Avg Error Rate", f"{kpis['Average Error Rate %']:.2f}%")
col6.metric("Avg Defect Rate", f"{kpis['Average Defect Rate %']:.2f}%")
col7.metric("Network-Efficiency Corr.", f"{kpis['Network-Efficiency Correlation']:.3f}")
col8.metric("Packet Loss Impact", f"{kpis['Packet Loss Impact Ratio']:.3f}")


# -------------------------------
# 7. DASHBOARD TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Network Performance Overview",
    "Network vs Efficiency",
    "Latency Diagnostics",
    "Quality & Error Impact",
    "6G Optimization Insights"
])


# -------------------------------
# TAB 1: NETWORK PERFORMANCE OVERVIEW
# -------------------------------
with tab1:
    st.subheader("Network Performance Overview")

    c1, c2 = st.columns(2)

    with c1:
        fig_latency = px.histogram(
            filtered_df,
            x="Network_Latency_ms",
            nbins=40,
            color="Network_Quality",
            title="Latency Distribution by Network Quality"
        )
        st.plotly_chart(fig_latency, use_container_width=True)

    with c2:
        fig_packet = px.histogram(
            filtered_df,
            x="Packet_Loss_%",
            nbins=40,
            color="Network_Quality",
            title="Packet Loss Distribution by Network Quality"
        )
        st.plotly_chart(fig_packet, use_container_width=True)

    if pd.api.types.is_datetime64_any_dtype(filtered_df["DateTime"]):
        sample_df = filtered_df.sort_values("DateTime").copy()
        if len(sample_df) > 5000:
            sample_df = sample_df.sample(5000, random_state=42).sort_values("DateTime")

        fig_trend = px.line(
            sample_df,
            x="DateTime",
            y=["Network_Latency_ms", "Packet_Loss_%"],
            title="Latency and Packet Loss Trend Over Time"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    fig_stability = px.box(
        filtered_df,
        x="Network_Quality",
        y="Network_Stability_Index",
        color="Network_Quality",
        title="Network Stability Index by Network Quality"
    )
    st.plotly_chart(fig_stability, use_container_width=True)


# -------------------------------
# TAB 2: NETWORK VS EFFICIENCY
# -------------------------------
with tab2:
    st.subheader("Network vs Manufacturing Efficiency")

    c1, c2 = st.columns(2)

    with c1:
        eff_dist = filtered_df.groupby(["Network_Quality", "Efficiency_Status"]).size().reset_index(name="Count")
        fig_eff = px.bar(
            eff_dist,
            x="Network_Quality",
            y="Count",
            color="Efficiency_Status",
            barmode="group",
            title="Efficiency Distribution by Network Quality"
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with c2:
        fig_scatter = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df)), random_state=42),
            x="Network_Latency_ms",
            y="Production_Speed_units_per_hr",
            color="Efficiency_Status",
            size="Packet_Loss_%",
            hover_data=["Machine_ID", "Operation_Mode"],
            title="Latency vs Production Speed"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    fig_stability_eff = px.box(
        filtered_df,
        x="Efficiency_Status",
        y="Network_Stability_Index",
        color="Efficiency_Status",
        title="Network Stability by Efficiency Status"
    )
    st.plotly_chart(fig_stability_eff, use_container_width=True)


# -------------------------------
# TAB 3: LATENCY DIAGNOSTICS
# -------------------------------
with tab3:
    st.subheader("Latency Impact Diagnostics")

    filtered_df["Latency_Band"] = pd.qcut(
        filtered_df["Network_Latency_ms"],
        q=5,
        duplicates="drop"
    )

    latency_summary = filtered_df.groupby("Latency_Band", observed=True).agg(
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Avg_Error_Rate=("Error_Rate_%", "mean"),
        Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
        Count=("Machine_ID", "count")
    ).reset_index()

    latency_summary["Latency_Band"] = latency_summary["Latency_Band"].astype(str)

    fig_latency_speed = px.line(
        latency_summary,
        x="Latency_Band",
        y="Avg_Production_Speed",
        markers=True,
        title="Average Production Speed Across Latency Bands"
    )
    st.plotly_chart(fig_latency_speed, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_real_delayed = px.box(
            filtered_df,
            x="Communication_Period",
            y="Production_Speed_units_per_hr",
            color="Communication_Period",
            title="Real-Time vs Delayed Communication: Production Speed"
        )
        st.plotly_chart(fig_real_delayed, use_container_width=True)

    with c2:
        fig_mode_latency = px.box(
            filtered_df,
            x="Operation_Mode",
            y="Network_Latency_ms",
            color="Efficiency_Status",
            title="Latency by Operation Mode and Efficiency"
        )
        st.plotly_chart(fig_mode_latency, use_container_width=True)

    st.markdown("### Latency Band Summary")
    st.dataframe(latency_summary, use_container_width=True)


# -------------------------------
# TAB 4: QUALITY & ERROR IMPACT
# -------------------------------
with tab4:
    st.subheader("Quality and Error Impact Panel")

    c1, c2 = st.columns(2)

    with c1:
        fig_packet_error = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df)), random_state=42),
            x="Packet_Loss_%",
            y="Error_Rate_%",
            color="Efficiency_Status",
            title="Packet Loss vs Error Rate"
        )
        st.plotly_chart(fig_packet_error, use_container_width=True)

    with c2:
        fig_packet_defect = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df)), random_state=42),
            x="Packet_Loss_%",
            y="Quality_Control_Defect_Rate_%",
            color="Efficiency_Status",
            title="Packet Loss vs Defect Rate"
        )
        st.plotly_chart(fig_packet_defect, use_container_width=True)

    filtered_df["Packet_Loss_Band"] = pd.qcut(
        filtered_df["Packet_Loss_%"],
        q=5,
        duplicates="drop"
    )

    packet_summary = filtered_df.groupby("Packet_Loss_Band", observed=True).agg(
        Avg_Error_Rate=("Error_Rate_%", "mean"),
        Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Count=("Machine_ID", "count")
    ).reset_index()

    packet_summary["Packet_Loss_Band"] = packet_summary["Packet_Loss_Band"].astype(str)

    st.markdown("### Packet Loss Band Summary")
    st.dataframe(packet_summary, use_container_width=True)


# -------------------------------
# TAB 5: OPTIMIZATION INSIGHTS
# -------------------------------
with tab5:
    st.subheader("6G Optimization Insights")

    # Threshold diagnostics
    high_eff_df = filtered_df[filtered_df["Efficiency_Status"] == "High"]
    low_eff_df = filtered_df[filtered_df["Efficiency_Status"] == "Low"]

    if len(high_eff_df) > 0:
        latency_tolerance = high_eff_df["Network_Latency_ms"].quantile(0.75)
        packet_tolerance = high_eff_df["Packet_Loss_%"].quantile(0.75)
    else:
        latency_tolerance = filtered_df["Network_Latency_ms"].median()
        packet_tolerance = filtered_df["Packet_Loss_%"].median()

    risk_latency = low_eff_df["Network_Latency_ms"].median() if len(low_eff_df) > 0 else filtered_df["Network_Latency_ms"].quantile(0.75)
    risk_packet = low_eff_df["Packet_Loss_%"].median() if len(low_eff_df) > 0 else filtered_df["Packet_Loss_%"].quantile(0.75)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency Tolerance Benchmark", f"{latency_tolerance:.2f} ms")
    c2.metric("Packet Loss Safe Benchmark", f"{packet_tolerance:.2f}%")
    c3.metric("Latency Risk Zone Starts Near", f"{risk_latency:.2f} ms")
    c4.metric("Packet Loss Risk Zone Starts Near", f"{risk_packet:.2f}%")

    mode_sensitivity = filtered_df.groupby("Operation_Mode").agg(
        Avg_Latency=("Network_Latency_ms", "mean"),
        Avg_Packet_Loss=("Packet_Loss_%", "mean"),
        Avg_Stability=("Network_Stability_Index", "mean"),
        Avg_Speed=("Production_Speed_units_per_hr", "mean"),
        Avg_Error=("Error_Rate_%", "mean"),
        Avg_Defect=("Quality_Control_Defect_Rate_%", "mean")
    ).reset_index()

    fig_mode = px.bar(
        mode_sensitivity,
        x="Operation_Mode",
        y=["Avg_Stability", "Avg_Speed", "Avg_Error"],
        barmode="group",
        title="Operation Mode Sensitivity to Network Conditions"
    )
    st.plotly_chart(fig_mode, use_container_width=True)

    st.markdown("### Operation Mode Sensitivity Table")
    st.dataframe(mode_sensitivity, use_container_width=True)

    st.markdown("### Recommendations")
    st.success(
        f"""
        1. Maintain latency below approximately **{latency_tolerance:.2f} ms** to preserve high manufacturing efficiency.

        2. Keep packet loss below approximately **{packet_tolerance:.2f}%** to reduce operational errors and quality defects.

        3. Prioritize 6G network slice optimization during high-load operation modes because these periods are more sensitive to latency variation.

        4. Use Network Stability Index as an early warning KPI before production speed and quality degradation become visible.

        5. Investigate machines operating in Low Quality network periods because production issues may be caused by connectivity degradation rather than mechanical faults.
        """
    )


# -------------------------------
# 8. DATA PREVIEW AND DOWNLOAD
# -------------------------------
st.subheader("Processed Dataset Preview")
st.dataframe(filtered_df.head(1000), use_container_width=True)

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Processed Data",
    data=csv,
    file_name="processed_6g_manufacturing_data.csv",
    mime="text/csv"
)
