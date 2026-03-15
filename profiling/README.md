# Profiling outputs

Nsight Compute profiling data for the matrix multiplication kernels.  
A **Streamlit dashboard** that loads these CSVs and compares kernels lives in **`../app/`** — see the main repo README for run instructions.


| Folder        | Contents                                                                     |
| ------------- | ---------------------------------------------------------------------------- |
| **reports/**  | PDF reports (human-readable summaries and metrics).                          |
| **raw/**      | CSV exports (raw dumps for analysis in spreadsheets or scripts).             |
| **sessions/** | `.ncu-rep` session files (reopen in Nsight Compute for detailed inspection). |


## Kernels profiled

- **naive** — baseline kernel
- **tiled**  — shared-memory tiled kernel
- **unrolled** — unrolled variant (if present)

