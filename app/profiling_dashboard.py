"""
Streamlit dashboard for Nsight Compute profiling raw-dump CSVs.
Loads CSVs from profiling/raw/ and compares naive vs tiled kernel metrics
with analysis and visual comparisons.
"""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# Path to raw dumps (relative to project root when running: streamlit run app/profiling_dashboard.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "profiling" / "raw"

CSV_FILES = {
    "Naive": "Naive Profiling Raw dump.csv",
    "Tiled": "Tiled Profiling Raw dump.csv",
}


def parse_ncu_csv(path: Path) -> dict[str, str]:
    """Parse Nsight Compute key-value CSV (one metric per line, first comma separates key from value)."""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.find(",")
            if idx < 0:
                continue
            key = line[:idx].strip()
            value = line[idx + 1 :].strip()
            out[key] = value
    return out


def safe_float(s: str) -> float | None:
    """Parse string to float, removing comma thousands separators."""
    if s is None or s == "" or (isinstance(s, str) and s.strip() == ""):
        return None
    s = str(s).strip()
    s = re.sub(r"\s*\{[^}]+\}\s*$", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_metrics() -> dict[str, dict]:
    """Load both CSVs and return dict kernel_name -> parsed metrics."""
    data = {}
    for label, filename in CSV_FILES.items():
        path = RAW_DIR / filename
        data[label] = parse_ncu_csv(path)
    return data


def main():
    st.set_page_config(page_title="Matrix Mul Profiling", layout="wide")
    st.title("Matrix multiplication kernel profiling")
    st.caption("Data from Nsight Compute raw-dump CSVs in `profiling/raw/`")

    metrics = load_metrics()
    if not any(metrics.values()):
        st.error(f"No CSV files found under `{RAW_DIR}`. Add the raw dumps and rerun.")
        return

    # ---- Build comparison data ----
    KEY_KEYS = [
        "Function Name",
        "Device Name",
        "gpu__time_duration.sum [ms]",
        "Estimated Speedup [%]",
        "Runtime Improvement [ms]",
        "Issues Detected [issue]",
        "Grid Size",
        "Block Size [block]",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed [%]",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed [%]",
        "gpc__cycles_elapsed.max [cycle]",
    ]

    rows = []
    for label in ["Naive", "Tiled"]:
        m = metrics.get(label, {})
        if not m:
            continue
        runtime_ms = safe_float(m.get("gpu__time_duration.sum [ms]"))
        speedup_pct = safe_float(m.get("Estimated Speedup [%]"))
        issues_str = m.get("Issues Detected [issue]", "—")
        issues_int = safe_float(issues_str) if issues_str != "—" else None
        sm_throughput_str = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed [%]", "—")
        mem_throughput_str = m.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed [%]", "—")
        sm_throughput = safe_float(sm_throughput_str)
        mem_throughput = safe_float(mem_throughput_str)
        rows.append({
            "Kernel": label,
            "Runtime (ms)": runtime_ms,
            "Est. speedup (%)": speedup_pct,
            "Issues": issues_str,
            "Issues (num)": issues_int,
            "SM throughput (%)": sm_throughput,
            "Memory throughput (%)": mem_throughput,
            "SM throughput (str)": sm_throughput_str,
            "Memory throughput (str)": mem_throughput_str,
        })

    if not rows:
        st.warning("No kernel data loaded.")
        return

    df = pd.DataFrame(rows)

    # ----- Summary metrics (KPIs) -----
    st.header("Summary metrics")
    n = len(rows)
    cols = st.columns(max(n, 2))
    for i, r in enumerate(rows):
        with cols[i]:
            st.metric("Kernel", r["Kernel"])
            st.metric("Runtime (ms)", f"{r['Runtime (ms)']:.2f}" if r["Runtime (ms)"] is not None else "—")
            st.metric("Est. speedup headroom (%)", f"{r['Est. speedup (%)']:.1f}" if r["Est. speedup (%)"] is not None else "—")
            st.metric("Issues detected", r["Issues"])
            if r["SM throughput (%)"] is not None:
                st.metric("SM throughput (% peak)", f"{r['SM throughput (%)']:.1f}%")
            if r["Memory throughput (%)"] is not None:
                st.metric("Memory throughput (% peak)", f"{r['Memory throughput (%)']:.1f}%")

    # ----- Comparison table with deltas -----
    st.header("Kernel comparison")
    comp = df[["Kernel", "Runtime (ms)", "Est. speedup (%)", "Issues", "SM throughput (str)", "Memory throughput (str)"]].copy()
    comp.columns = ["Kernel", "Runtime (ms)", "Est. speedup (%)", "Issues", "SM throughput (%)", "Memory throughput (%)"]
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ----- Analysis & comparisons -----
    st.header("Analysis & comparisons")
    naive_row = next((r for r in rows if r["Kernel"] == "Naive"), None)
    tiled_row = next((r for r in rows if r["Kernel"] == "Tiled"), None)

    if naive_row and tiled_row and naive_row.get("Runtime (ms)") and tiled_row.get("Runtime (ms)"):
        t_naive = naive_row["Runtime (ms)"]
        t_tiled = tiled_row["Runtime (ms)"]
        pct_faster = (1 - t_tiled / t_naive) * 100
        runtime_delta_ms = t_tiled - t_naive

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Runtime: Tiled vs Naive", f"{t_tiled:.2f} ms", delta=f"{runtime_delta_ms:.2f} ms")
        with c2:
            st.metric("Speedup (Tiled over Naive)", f"{pct_faster:.1f}%", delta="faster")
        with c3:
            issues_naive = naive_row.get("Issues (num)")
            issues_tiled = tiled_row.get("Issues (num)")
            if issues_naive is not None and issues_tiled is not None:
                delta_issues = int(issues_tiled - issues_naive)
                st.metric("Issues: Tiled vs Naive", str(issues_tiled), delta=f"{delta_issues} vs Naive")

        # Bar chart: runtime comparison
        st.subheader("Runtime comparison")
        runtime_df = pd.DataFrame({"Kernel": [naive_row["Kernel"], tiled_row["Kernel"]], "Runtime (ms)": [t_naive, t_tiled]})
        st.bar_chart(runtime_df.set_index("Kernel"), height=300)

        # Bar chart: throughput comparison
        if naive_row.get("SM throughput (%)") is not None and tiled_row.get("SM throughput (%)") is not None:
            st.subheader("Throughput (% of peak)")
            throughput_df = pd.DataFrame({
                "Kernel": [naive_row["Kernel"], tiled_row["Kernel"]] * 2,
                "Metric": ["SM throughput (%)"] * 2 + ["Memory throughput (%)"] * 2,
                "Value": [
                    naive_row["SM throughput (%)"],
                    tiled_row["SM throughput (%)"],
                    naive_row["Memory throughput (%)"] or 0,
                    tiled_row["Memory throughput (%)"] or 0,
                ],
            })
            st.bar_chart(throughput_df.pivot(index="Kernel", columns="Metric", values="Value"), height=300)

        # Written analysis
        st.subheader("Insights")
        insights = [
            f"**Tiled is ~{pct_faster:.1f}% faster** than naive ({t_tiled:.2f} ms vs {t_naive:.2f} ms) for the same 1024×1024 problem size.",
            "Both kernels are **memory-bound**: SM and memory throughput sit near peak (~96–98%), so gains come from reducing global memory traffic via shared-memory tiling.",
            f"Naive has **higher estimated speedup headroom** ({naive_row.get('Est. speedup (%)') or 0:.1f}% vs {tiled_row.get('Est. speedup (%)') or 0:.1f}%), meaning the tiled kernel is closer to optimal for this GPU and problem size.",
            f"Naive reports **{naive_row.get('Issues') or '?'} issues** vs **{tiled_row.get('Issues') or '?'}** for tiled; tiling improves both performance and typically reduces some Nsight-reported issues.",
        ]
        for bullet in insights:
            st.markdown(f"- {bullet}")
    else:
        st.info("Load both Naive and Tiled dumps to see comparison analysis and charts.")

    # ----- Per-kernel metrics -----
    st.header("Metrics by kernel")
    for label in ["Naive", "Tiled"]:
        m = metrics.get(label, {})
        if not m:
            continue
        with st.expander(f"{label} kernel", expanded=(label == "Naive")):
            table = []
            for key in KEY_KEYS:
                val = m.get(key, "—")
                if val != "—":
                    table.append({"Metric": key, "Value": val})
            if table:
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            else:
                st.write("No key metrics found.")

    # ----- Raw search -----
    st.header("Search raw metrics")
    kernel_sel = st.selectbox("Kernel", list(metrics.keys()), key="kernel_sel")
    search = st.text_input("Filter by metric name (substring)")
    m = metrics.get(kernel_sel, {})
    if m and search:
        filtered = [(k, v) for k, v in m.items() if search.lower() in k.lower()]
        if filtered:
            st.dataframe(
                pd.DataFrame(filtered, columns=["Metric", "Value"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No metrics match the filter.")


if __name__ == "__main__":
    main()
