# Profiling dashboard

Streamlit app that loads Nsight Compute raw-dump CSVs and shows kernel comparison, metrics, and **interactive Plotly charts** (Overview, Memory, Stalls, Config tabs).

**Dependencies:** `streamlit`, `pandas`, `plotly` (see `requirements.txt`).

## Run locally

From the **project root** (not from `app/`):

```bash
# Optional: use a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r app/requirements.txt
streamlit run app/profiling_dashboard.py
```

Then open the URL shown (usually http://localhost:8501). The app loads **Naive Profiling Raw dump.csv** and **Tiled Profiling Raw dump.csv** from `profiling/raw/` automatically—place your Nsight Compute dumps there and run from the project root.

## One-liner (if dependencies are already installed)

```bash
streamlit run app/profiling_dashboard.py
```

Run the command from the repository root so the app finds `profiling/raw/`.
