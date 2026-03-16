# pyrecon-cell-tracker

**PyReconstruct plugin for DAPI cell tracking across serial tissue sections.**

This repository is the **connector** layer.  The actual tracking algorithm
lives in [cell-tracker-core](https://github.com/<your-org>/cell-tracker-core),
which is included here as a **git submodule**.

---

## Architecture

```
pyrecon-cell-tracker/          ← this repo (connector)
├── pyrecon_connector/
│   ├── __init__.py            # Public API — exposes PyReconConnector
│   ├── connector.py           # Main connector class
│   └── io_pyrecon.py          # PyReconstruct .jser read/write
├── cell_tracker_core/         ← git submodule (cell-tracker-core repo)
│   └── cell_tracker/          # Core tracking library
├── run_plugin.py              # CLI entry point
├── .gitmodules                # Submodule declaration
├── setup.py
└── README.md
```

---

## Data Flow

```
PyReconstruct .jser
        │
        ▼
load_series_contours()       ← reads contours from all sections
        │
        ▼
normalise_pair()             ← scale coords to [-1,1]
        │
        ▼
BNNBeliefPropagationTracker  ← core algorithm (in submodule)
        │
        ▼
chain_trajectories()         ← assign persistent TrackIDs
        │
        ├──▶  cell_<TrackID>  ← contours renamed in .jser
        ├──▶  cell_trajectories.csv
        └──▶  tracking_summary.csv
```

### What the output means

| Output | Description |
|--------|-------------|
| `*_tracked.jser` | Original series with contours renamed to `cell_<TrackID>`. All sections of the same physical cell share the same name, so PyReconstruct's 3-D renderer links them automatically. |
| `cell_trajectories.csv` | One row per (cell, section): TrackID, Section, Contour_Name, X, Y, Area |
| `tracking_summary.csv` | Per section-pair: Precision, Recall, F1, global shift, elapsed time |

---

## Installation

### 1. Clone with submodule

```bash
git clone --recurse-submodules https://github.com/<your-org>/pyrecon-cell-tracker
cd pyrecon-cell-tracker
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install -e .
```

---

## Usage

### Command line

```bash
python run_plugin.py \
    --jser  /path/to/series.jser \
    --out   /path/to/results \
    --tif-dir /path/to/dapi_tifs   # optional: enables phase alignment
```

or, after `pip install -e .`:

```bash
pyrecon-track \
    --jser  /path/to/series.jser \
    --out   /path/to/results
```

### Python API

```python
from pyrecon_connector import PyReconConnector

conn = PyReconConnector(
    jser_path="/data/myseries.jser",
    out_dir="/results/tracking",
    tif_dir="/data/dapi_tifs",   # optional
)
track_df = conn.run()
```

---

## Module Reference

### `pyrecon_connector.io_pyrecon`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_series_contours(jser_path)` | str | dict[int → DataFrame] | Parse all contours from a `.jser` file; one DataFrame per section with columns Section, Contour_Name, X, Y, Area |
| `rename_contours_in_series(jser_path, track_df, out_jser_path)` | str, DataFrame, str | str | Rename contours to `cell_<TrackID>` and write modified `.jser`; backs up original |
| `write_tracking_csv(track_df, out_dir)` | DataFrame, str | str | Write trajectory CSV to `out_dir/cell_trajectories.csv` |

### `pyrecon_connector.connector.PyReconConnector`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `__init__(jser_path, out_dir, algorithm, tif_dir)` | paths, optional | — | Initialise with series path and optional algorithm override |
| `run()` | — | DataFrame | Execute full pipeline: read contours → track → rename → write |

---

## Submodule Management

### Update core algorithm to latest

```bash
git submodule update --remote
git add cell_tracker_core
git commit -m "update cell-tracker-core to latest"
```

### Pin core to a specific commit

```bash
cd cell_tracker_core
git checkout <commit-hash>
cd ..
git add cell_tracker_core
git commit -m "pin cell-tracker-core to <commit-hash>"
```

### Fix submodule to a branch

```bash
git config -f .gitmodules submodule.cell_tracker_core.branch main
```

---

## Adding a New Algorithm

1. Implement a new algorithm in the **core repo** under
   `cell_tracker/algorithms/my_algo/`.
2. Update the core repo and push.
3. In this connector repo, update the submodule:
   ```bash
   git submodule update --remote
   ```
4. Pass an instance to the connector:
   ```python
   from cell_tracker.algorithms.my_algo import MyAlgoTracker
   conn = PyReconConnector(..., algorithm=MyAlgoTracker())
   ```

---

## License

GPL-3.0 — see `LICENSE`.
