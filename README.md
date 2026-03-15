# Matrix Multiplication CUDA Kernels

CUDA implementations of matrix multiplication: **naive** and **tiled** (shared-memory) kernels, with Nsight Compute profiling and a Streamlit dashboard to explore the raw dumps.

Shoutout to my roommate **Aadesh** for lending his laptop to tinker with kernels and their optimisations :)

---

## What is matrix multiplication?

For matrices **A** (M×K) and **B** (K×N), the product **C = A×B** has shape M×N. Each element  
**C[i][j]** is the dot product of row *i* of A and column *j* of B. It’s compute-heavy (O(M·K·N) ops) and highly parallel, so GPUs are a good fit.

---

## Naive method

In the **naive** kernel, each thread computes **one** element of C. It repeatedly reads from **global memory** (VRAM): one full row of A and one full column of B per thread. That causes many redundant, high-latency loads and leaves most of the GPU’s potential unused.

---

## How tiling and loop unrolling help

- **Tiling (shared memory):** Data is loaded in small **tiles** (e.g. 16×16) into **shared memory**, which is much faster than global memory. Threads in a block cooperate to fill the tiles, then reuse that data for many multiplies. That cuts global traffic and improves throughput.
- **Loop unrolling:** The inner loop over the tile dimension can be **unrolled** (manually or by the compiler), reducing branch and loop overhead and improving instruction throughput.

Together, tiling + unrolling typically give a noticeable speedup over the naive kernel (see insights below).

---

## Key insights from profiling (Nsight Compute)

Profiling was done on an **NVIDIA GeForce RTX 4050 Laptop GPU** (1024×1024 matrices, 16×16 blocks/tiles).

| Kernel | Runtime (ms) | Est. speedup vs baseline | Issues reported |
|--------|--------------|---------------------------|-----------------|
| **Naive** | 3.66 | 43.03% | 7 |
| **Tiled** | 2.78 | 3.30% | 6 |

- **Tiled is ~24% faster** than naive (2.78 ms vs 3.66 ms) for the same problem size.
- Both kernels are **memory bound**: SM and memory throughput are near peak (~96–98%), so gains come from reducing global memory traffic via shared-memory tiling.
- The naive kernel has **more issues** (7 vs 6) and a higher estimated speedup headroom (43% vs 3%), meaning the tiled version is already closer to optimal for this setup.
- Tiled uses more **shared memory** (higher occupancy impact) but gains from much better data reuse.

The table above summarizes high-level metrics. **A deeper analysis**—bottleneck diagnosis (memory- vs compute-bound), instruction-pipeline and occupancy comparison, speedup ratios, interactive Plotly charts (throughput, cycles, stalls, memory), and downloadable comparison CSV—**is available when you run the Streamlit app** (see below). The dashboard uses tabs (Overview, Memory, Stalls, Config) and supports either uploading CSVs in the sidebar or auto-loading from `profiling/raw/`. You can also open the raw CSV/PDF reports in `profiling/` for full Nsight Compute detail.

---

## Project structure

```
├── README.md                 # This file — first thing you see when opening the repo
├── src/                      # CUDA source files
│   ├── matmul_naive.cu
│   └── matmul_tiled.cu
├── profiling/                # Nsight Compute outputs
│   ├── reports/              # PDF summaries
│   ├── raw/                  # CSV raw dumps (used by the dashboard)
│   ├── sessions/             # .ncu-rep session files
│   └── README.md
└── app/                      # Streamlit profiling dashboard
    ├── profiling_dashboard.py
    ├── requirements.txt
    └── README.md
```

## Building the CUDA kernels

From the project root:

```bash
nvcc -o matmul_naive src/matmul_naive.cu
nvcc -o matmul_tiled src/matmul_tiled.cu
```

---

## Running the Streamlit profiling dashboard

The dashboard reads the CSV files in `profiling/raw/` (or lets you upload them in the sidebar) and shows comparison tables, key metrics, and interactive Plotly charts. Dependencies include **Streamlit**, **pandas**, and **Plotly** (see `app/requirements.txt`).

**Steps:**

1. Open a terminal and go to the **project root** (the folder that contains `src/`, `profiling/`, and `app/`).
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. Install dependencies and run Streamlit:
   ```bash
   pip install -r app/requirements.txt
   streamlit run app/profiling_dashboard.py
   ```
4. Open the URL printed in the terminal (usually **http://localhost:8501**).

Note: Always run `streamlit run app/profiling_dashboard.py` from the **project root** so the app can find `profiling/raw/`.
