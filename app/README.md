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

Then open the URL shown (usually http://localhost:8501). Use the **sidebar** to upload the Naive and Tiled CSV files (e.g. from `profiling/raw/`) if the app does not auto-detect them in the current directory.

## One-liner (if dependencies are already installed)

```bash
streamlit run app/profiling_dashboard.py
```

Run the command from the repository root so paths and uploads work as expected.
