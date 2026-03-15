"""
Nsight Compute Profiling Dashboard — Naive vs Tiled Matrix Multiplication
=========================================================================
Run:  streamlit run ncu_dashboard.py
"""

import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Project paths: CSVs live in profiling/raw/ (no upload required)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "profiling" / "raw"
NAIVE_CSV = RAW_DIR / "Naive Profiling Raw dump.csv"
TILED_CSV = RAW_DIR / "Tiled Profiling Raw dump.csv"

# ── Page Config ──
st.set_page_config(
    page_title="NCU Profiling Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Light Theme CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background-color: #f8fafc; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        margin-top: 4px;
    }
    .speedup-hero {
        font-family: 'JetBrains Mono', monospace;
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #0891b2, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        line-height: 1.1;
    }
    .speedup-sub {
        text-align: center;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
    }
    .insight-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.6;
    }
    .insight-good {
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.25);
        color: #15803d;
    }
    .insight-warn {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        color: #b45309;
    }
    .insight-info {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.25);
        color: #1d4ed8;
    }
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 24px 0 12px 0;
    }
    /* Hide sidebar completely — no sidebar at all */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] { display: none !important; width: 0 !important; min-width: 0 !important; }
    /* Hide the expand/collapse sidebar button */
    button[kind="header"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    /* Main content full width */
    .block-container { max-width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 8px;
        padding: 8px 20px;
    }
    h1, h2, h3, p, span, td, th, li { color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Colors ──
NAIVE_COLOR = "#dc2626"
TILED_COLOR = "#0891b2"
ACCENT = "#7c3aed"
BG_COLOR = "#f8fafc"
CARD_COLOR = "#ffffff"
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#1e293b"
TEXT_MUTED = "#64748b"

# ── Plotly Layout Template ──
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=TEXT_COLOR, size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_COLOR)),
    bargap=0.3,
)


def _plotly_layout(**overrides):
    """Merge overrides into PLOTLY_LAYOUT so xaxis/yaxis don't get duplicated (avoids Plotly TypeError)."""
    out = dict(PLOTLY_LAYOUT)
    for k in ("xaxis", "yaxis"):
        if k in overrides:
            out[k] = {**out.get(k, {}), **overrides[k]}
            del overrides[k]
    out.update(overrides)
    return out


# ── CSV Parser ──
def parse_ncu_csv(filepath):
    metrics = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                val = ",".join(row[1:]).strip()
                metrics[key] = val
    return metrics


def clean_numeric(val_str):
    if not val_str or val_str.strip() == "":
        return None
    val_str = re.sub(r"\{.*?\}", "", val_str).strip().strip('"').strip()
    if val_str == "":
        return None
    val_str = val_str.replace(",", "")
    try:
        return float(val_str)
    except ValueError:
        return None


def get(metrics, key, numeric=True):
    val = metrics.get(key, "")
    return clean_numeric(val) if numeric else val


# ── Load Data (from profiling/raw/ — no upload) ──
def load_profiles():
    """Load Naive and Tiled CSVs from profiling/raw/. Paths are fixed relative to project root."""
    naive_data = parse_ncu_csv(NAIVE_CSV) if NAIVE_CSV.exists() else None
    tiled_data = parse_ncu_csv(TILED_CSV) if TILED_CSV.exists() else None
    return naive_data, tiled_data


naive, tiled = load_profiles()

if not naive or not tiled:
    st.error(
        "CSV files not found. Place the Nsight Compute raw dumps in:\n\n"
        f"- **Naive:** `{NAIVE_CSV}`\n"
        f"- **Tiled:** `{TILED_CSV}`\n\n"
        "Run the app from the project root: `streamlit run app/profiling_dashboard.py`"
    )
    st.stop()

t_naive = get(naive, "gpu__time_duration.sum [ms]")
t_tiled = get(tiled, "gpu__time_duration.sum [ms]")
speedup = t_naive / t_tiled if t_naive and t_tiled else 1.0
naive_name = get(naive, "Function Name", numeric=False)
tiled_name = get(tiled, "Function Name", numeric=False)
device = get(naive, "Device Name", numeric=False)

st.markdown(f"""
<div style="margin-bottom: 8px;">
    <h1 style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #0f172a !important; margin: 0;">
        🔬 Nsight Compute Profiling Dashboard
    </h1>
    <p style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #64748b !important; margin: 4px 0 0 0;">
        <span style="color: {NAIVE_COLOR} !important">■</span> {naive_name} &nbsp;vs&nbsp;
        <span style="color: {TILED_COLOR} !important">■</span> {tiled_name} &nbsp;·&nbsp; {device}
    </p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_memory, tab_stalls, tab_config = st.tabs(["📊 Overview", "💾 Memory", "🚦 Stalls", "⚙️ Config"])

with tab_overview:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 30px;">
            <p class="speedup-hero">{speedup:.2f}x</p>
            <p class="speedup-sub" style="color: #64748b !important;">{t_naive:.2f} ms → {t_tiled:.2f} ms</p>
        </div>
        """, unsafe_allow_html=True)

    cyc_naive = get(naive, "gpc__cycles_elapsed.max [cycle]")
    cyc_tiled = get(tiled, "gpc__cycles_elapsed.max [cycle]")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Execution Time</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[t_naive], y=["Duration"], orientation="h", name=naive_name, marker_color=NAIVE_COLOR, text=[f"{t_naive:.2f} ms"], textposition="inside", textfont=dict(color="white")))
        fig.add_trace(go.Bar(x=[t_tiled], y=["Duration"], orientation="h", name=tiled_name, marker_color=TILED_COLOR, text=[f"{t_tiled:.2f} ms"], textposition="inside", textfont=dict(color="white")))
        fig.update_layout(**_plotly_layout(height=140, barmode="group", showlegend=False, yaxis=dict(showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<p class="section-header">Elapsed Cycles</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[cyc_naive], y=["Cycles"], orientation="h", name=naive_name, marker_color=NAIVE_COLOR, text=[f"{int(cyc_naive):,}"], textposition="inside", textfont=dict(color="white")))
        fig.add_trace(go.Bar(x=[cyc_tiled], y=["Cycles"], orientation="h", name=tiled_name, marker_color=TILED_COLOR, text=[f"{int(cyc_tiled):,}"], textposition="inside", textfont=dict(color="white")))
        fig.update_layout(**_plotly_layout(height=140, barmode="group", showlegend=False, yaxis=dict(showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Speed of Light — Throughput %</p>', unsafe_allow_html=True)
    throughput_metrics = [
        ("Compute (SM)", "sm__throughput.avg.pct_of_peak_sustained_elapsed [%]"),
        ("Memory", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed [%]"),
        ("L1/TEX", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed [%]"),
        ("L2 Cache", "lts__throughput.avg.pct_of_peak_sustained_elapsed [%]"),
        ("DRAM", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed [%]"),
    ]
    labels = [m[0] for m in throughput_metrics]
    naive_vals = [get(naive, m[1]) or 0 for m in throughput_metrics]
    tiled_vals = [get(tiled, m[1]) or 0 for m in throughput_metrics]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=naive_vals, orientation="h", name=naive_name, marker_color=NAIVE_COLOR, text=[f"{v:.1f}%" for v in naive_vals], textposition="inside", textfont=dict(color="white")))
    fig.add_trace(go.Bar(y=labels, x=tiled_vals, orientation="h", name=tiled_name, marker_color=TILED_COLOR, text=[f"{v:.1f}%" for v in tiled_vals], textposition="inside", textfont=dict(color="white")))
    fig.update_layout(**_plotly_layout(height=300, barmode="group", xaxis=dict(range=[0, 105], title="% of Peak", gridcolor=GRID_COLOR), legend=dict(orientation="h", y=1.15)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Occupancy & Efficiency</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    occ_metrics = [
        ("Active Warps/Cycle", "sm__warps_active.avg.per_cycle_active [warp]", 48, ""),
        ("Achieved Occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active [%]", 100, "%"),
        ("IPC", "sm__inst_executed.avg.per_cycle_active [inst/cycle]", 2.0, ""),
    ]
    for col, (label, key, max_val, unit) in zip([col1, col2, col3], occ_metrics):
        n_val = get(naive, key) or 0
        t_val = get(tiled, key) or 0
        delta = ((t_val - n_val) / n_val * 100) if n_val != 0 else 0
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label" style="color: #64748b !important;">{label}</p>
                <p class="metric-value" style="color: {TILED_COLOR} !important;">{t_val:.2f}{unit}</p>
                <p class="metric-delta" style="color: {'#16a34a' if delta >= 0 else '#dc2626'} !important;">
                    {'↑' if delta >= 0 else '↓'} {abs(delta):.1f}% vs naive ({n_val:.2f}{unit})
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
    insights = [
        ("good", f"⚡ **{speedup:.2f}x speedup** — {t_naive:.2f} ms → {t_tiled:.2f} ms"),
        ("good", "🔥 **LG Throttle stalls eliminated**: 27.50 → 0.01 (global memory bottleneck gone)"),
        ("good", "📦 **Shared memory wavefronts**: 65K → 55M (841x — tiling confirmed working)"),
        ("warn", "⚠️ **MIO Throttle is the new bottleneck**: 0.08 → 20.16 (shared mem pipe saturated)"),
        ("warn", "🚧 **Barrier stalls appeared** at 6.04 (__syncthreads cost from tiling)"),
        ("info", "📉 **L1 hit rate dropped** 87.5% → 7.3% (expected — traffic rerouted to shared mem)"),
    ]
    col1, col2 = st.columns(2)
    for i, (itype, text) in enumerate(insights):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f'<div class="insight-box insight-{itype}">{text}</div>', unsafe_allow_html=True)

with tab_memory:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Cache Hit Rates</p>', unsafe_allow_html=True)
        l1_n, l1_t = get(naive, "l1tex__t_sector_hit_rate.pct [%]") or 0, get(tiled, "l1tex__t_sector_hit_rate.pct [%]") or 0
        l2_n, l2_t = get(naive, "lts__t_sector_hit_rate.pct [%]") or 0, get(tiled, "lts__t_sector_hit_rate.pct [%]") or 0
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["L1 Hit Rate", "L2 Hit Rate"], y=[l1_n, l2_n], name=naive_name, marker_color=NAIVE_COLOR, text=[f"{l1_n:.1f}%", f"{l2_n:.1f}%"], textposition="outside"))
        fig.add_trace(go.Bar(x=["L1 Hit Rate", "L2 Hit Rate"], y=[l1_t, l2_t], name=tiled_name, marker_color=TILED_COLOR, text=[f"{l1_t:.1f}%", f"{l2_t:.1f}%"], textposition="outside"))
        fig.update_layout(**_plotly_layout(height=350, barmode="group", yaxis=dict(range=[0, 110], title="Hit Rate %", gridcolor=GRID_COLOR), legend=dict(orientation="h", y=1.12)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box insight-info">ℹ️ L1 drop is expected — tiled kernel bypasses L1 in favor of shared memory</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="section-header">Data Transfer Volume</p>', unsafe_allow_html=True)
        l2_l1_n, l2_l1_t = get(naive, "l1tex__m_xbar2l1tex_read_bytes.sum [Mbyte]") or 0, get(tiled, "l1tex__m_xbar2l1tex_read_bytes.sum [Mbyte]") or 0
        dram_n, dram_t = get(naive, "dram__bytes_read.sum [Mbyte]") or 0, get(tiled, "dram__bytes_read.sum [Mbyte]") or 0
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["L2 → L1 Transfer (MB)", "DRAM Reads (MB)"], y=[l2_l1_n, dram_n], name=naive_name, marker_color=NAIVE_COLOR, text=[f"{l2_l1_n:.1f}", f"{dram_n:.1f}"], textposition="outside"))
        fig.add_trace(go.Bar(x=["L2 → L1 Transfer (MB)", "DRAM Reads (MB)"], y=[l2_l1_t, dram_t], name=tiled_name, marker_color=TILED_COLOR, text=[f"{l2_l1_t:.1f}", f"{dram_t:.1f}"], textposition="outside"))
        fig.update_layout(**_plotly_layout(height=350, barmode="group", yaxis=dict(title="MB", gridcolor=GRID_COLOR), legend=dict(orientation="h", y=1.12)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box insight-good">✅ 6.4% less data moved from L2 → L1 with tiling</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Shared Memory Activity</p>', unsafe_allow_html=True)
    shmem = {
        "Metric": ["Total Wavefronts", "Shared Loads", "Shared Stores"],
        "Naive": [get(naive, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum") or 0, get(naive, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") or 0, get(naive, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") or 0],
        "Tiled": [get(tiled, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum") or 0, get(tiled, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") or 0, get(tiled, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") or 0],
    }
    col1, col2, col3 = st.columns(3)
    for col, i in zip([col1, col2, col3], range(3)):
        n_val, t_val = shmem["Naive"][i], shmem["Tiled"][i]
        ratio = f"{t_val/n_val:.0f}x" if n_val > 0 else "new"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <p class="metric-label" style="color: #64748b !important;">{shmem["Metric"][i]}</p>
                <div style="display: flex; justify-content: center; gap: 24px; margin: 12px 0;">
                    <div>
                        <p style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: {NAIVE_COLOR} !important; margin: 0;">{n_val:,.0f}</p>
                        <p style="font-size: 11px; color: #64748b !important; margin: 0;">naive</p>
                    </div>
                    <div style="color: #94a3b8 !important; align-self: center;">→</div>
                    <div>
                        <p style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: {TILED_COLOR} !important; margin: 0;">{t_val:,.0f}</p>
                        <p style="font-size: 11px; color: #64748b !important; margin: 0;">tiled</p>
                    </div>
                </div>
                <p style="font-family: 'JetBrains Mono'; font-size: 13px; color: {TILED_COLOR} !important; margin: 0;">{ratio} increase</p>
            </div>
            """, unsafe_allow_html=True)

with tab_stalls:
    stall_keys = [
        ("LG Throttle", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio [inst]"),
        ("MIO Throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio [inst]"),
        ("Barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio [inst]"),
        ("Long Scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio [inst]"),
        ("Not Selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio [inst]"),
        ("Wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio [inst]"),
        ("Dispatch", "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio [inst]"),
        ("Short Scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio [inst]"),
        ("Math Pipe", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio [inst]"),
    ]
    stall_labels = [s[0] for s in stall_keys]
    stall_naive = [get(naive, s[1]) or 0 for s in stall_keys]
    stall_tiled = [get(tiled, s[1]) or 0 for s in stall_keys]

    st.markdown('<p class="section-header">Warp Stall Breakdown</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=stall_labels[::-1], x=stall_naive[::-1], orientation="h", name=naive_name, marker_color=NAIVE_COLOR, text=[f"{v:.2f}" for v in stall_naive[::-1]], textposition="outside"))
    fig.add_trace(go.Bar(y=stall_labels[::-1], x=stall_tiled[::-1], orientation="h", name=tiled_name, marker_color=TILED_COLOR, text=[f"{v:.2f}" for v in stall_tiled[::-1]], textposition="outside"))
    fig.update_layout(**_plotly_layout(height=420, barmode="group", xaxis=dict(title="Avg Warps Stalled per Issue", gridcolor=GRID_COLOR), legend=dict(orientation="h", y=1.08)))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Bottleneck Shift Analysis</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {NAIVE_COLOR};">
            <p style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 600; color: {NAIVE_COLOR} !important; margin: 0 0 8px 0;">Naive: Global Memory Bound</p>
            <p style="font-size: 13px; color: #475569 !important; line-height: 1.7; margin: 0;">
                LG Throttle dominated at <b>27.50</b> — warps constantly stalled waiting for global memory loads from the L1 instruction queue.
            </p>
        </div>
        <div class="metric-card" style="border-left: 3px solid {TILED_COLOR}; margin-top: 12px;">
            <p style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 600; color: {TILED_COLOR} !important; margin: 0 0 8px 0;">Tiled: Shared Memory Bound</p>
            <p style="font-size: 13px; color: #475569 !important; line-height: 1.7; margin: 0;">
                MIO Throttle now leads at <b>20.16</b> — the shared memory pipe is saturated. Barrier stalls (<b>6.04</b>) from __syncthreads() are an additional cost.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="section-header">Stall Distribution</p>', unsafe_allow_html=True)
        tab_pie_n, tab_pie_t = st.tabs(["Naive", "Tiled"])
        with tab_pie_n:
            fig = go.Figure(go.Pie(labels=stall_labels, values=stall_naive, hole=0.45, textinfo="label+percent", textposition="outside", textfont=dict(size=11, color=TEXT_COLOR)))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with tab_pie_t:
            fig = go.Figure(go.Pie(labels=stall_labels, values=stall_tiled, hole=0.45, textinfo="label+percent", textposition="outside", textfont=dict(size=11, color=TEXT_COLOR)))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Next Optimization Targets</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, (title, desc) in zip([col1, col2, col3], [
        ("🔄 Double Buffering", "Overlap shared memory loads with computation to hide MIO latency"),
        ("📐 Larger Tile Size", "More compute per shared memory load — better arithmetic intensity"),
        ("🧵 Thread Coarsening", "Each thread computes multiple output elements to reduce sync overhead"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p style="font-family: 'JetBrains Mono'; font-size: 14px; font-weight: 600; color: #1e293b !important; margin: 0 0 6px 0;">{title}</p>
                <p style="font-size: 13px; color: #64748b !important; margin: 0; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

with tab_config:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Launch Configuration</p>', unsafe_allow_html=True)
        launch_metrics = [
            ("Registers/Thread", "launch__registers_per_thread [register/thread]"),
            ("Shared Mem/Block (KB)", "launch__shared_mem_per_block [Kbyte/block]"),
            ("Block Size", "launch__block_size"),
            ("Grid Size", "launch__grid_size"),
            ("Waves/SM", "launch__waves_per_multiprocessor"),
            ("Occupancy Limit (Regs)", "launch__occupancy_limit_registers [block]"),
            ("Occupancy Limit (Shmem)", "launch__occupancy_limit_shared_mem [block]"),
            ("Occupancy Limit (Warps)", "launch__occupancy_limit_warps [block]"),
        ]
        rows = ""
        for label, key in launch_metrics:
            n_val = re.sub(r"\{.*?\}", "", str(get(naive, key, numeric=False))).strip() or "N/A"
            t_val = re.sub(r"\{.*?\}", "", str(get(tiled, key, numeric=False))).strip() or "N/A"
            rows += f'<tr><td style="padding:8px 12px;color:#475569 !important;border-bottom:1px solid #e2e8f0;">{label}</td><td style="padding:8px 12px;text-align:right;color:#1e293b !important;border-bottom:1px solid #e2e8f0;font-weight:500;">{n_val}</td><td style="padding:8px 12px;text-align:right;color:#1e293b !important;border-bottom:1px solid #e2e8f0;font-weight:500;">{t_val}</td></tr>'
        st.markdown(f"""
        <div class="metric-card">
            <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:13px;">
                <thead><tr style="border-bottom:2px solid #cbd5e1;">
                    <th style="text-align:left;padding:8px 12px;color:#64748b !important;font-weight:500;">Metric</th>
                    <th style="text-align:right;padding:8px 12px;color:{NAIVE_COLOR} !important;font-weight:600;">Naive</th>
                    <th style="text-align:right;padding:8px 12px;color:{TILED_COLOR} !important;font-weight:600;">Tiled</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="section-header">Occupancy Limiters</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size:13px;color:#475569 !important;line-height:1.8;margin:0;">
                Both kernels are <span style="color:#d97706 !important;font-weight:600;">register-limited</span> — max 6 blocks per SM due to 40 registers/thread.<br><br>
                Tiled kernel uses <b>3.07 KB</b> shared memory per block (vs 1.02 KB naive) but this doesn't further limit occupancy — shared memory allows up to 21 blocks vs register limit of 6.<br><br>
                Achieved occupancy is excellent for both at ~<b>98.7%</b> (47.4 of 48 max warps active).
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<p class="section-header">Hardware Context</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <table style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#475569 !important;line-height:2.0;">
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">GPU</td><td style="color:#1e293b !important;">{device}</td></tr>
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">Arch</td><td style="color:#1e293b !important;">Ada Lovelace (SM 8.9)</td></tr>
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">SMs</td><td style="color:#1e293b !important;">20</td></tr>
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">Max Warps/SM</td><td style="color:#1e293b !important;">48</td></tr>
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">Matrix Size</td><td style="color:#1e293b !important;">1024 × 1024</td></tr>
                <tr><td style="color:#94a3b8 !important;padding-right:16px;">Tile Size</td><td style="color:#1e293b !important;">16 × 16</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("This dashboard compares CUDA kernel profiling results exported from **NVIDIA Nsight Compute**.\n\n**How to use:**\n1. Export Raw CSV from Nsight Compute GUI\n2. Upload Naive + Tiled CSVs above\n3. Explore the tabs\n\n**Export from GUI:**\nOpen `.ncu-rep` → Raw tab → Right-click → Export CSV")