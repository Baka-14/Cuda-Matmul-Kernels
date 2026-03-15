# Profiling dashboard

Streamlit app that loads Nsight Compute raw-dump CSVs from `profiling/raw/` and shows kernel comparison and metrics.

## Run locally

From the **project root** (not from `app/`):

```bash
# Optional: use a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r app/requirements.txt
streamlit run app/profiling_dashboard.py
```

Then open the URL shown (usually http://localhost:8501).

## One-liner (if dependencies are already installed)

```bash
streamlit run app/profiling_dashboard.py
```

The app resolves paths relative to the project root, so run the command from the repository root directory.
