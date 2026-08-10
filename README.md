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

### Multiplex RNA mapping in PyReconstruct

The customized PyReconstruct Plug-In menu now contains a guided **Multiplex RNA mapping** workflow:

1. **Import DAPI/RNA ROI folders** imports ImageJ `.roi` files or ROI ZIPs into the open series. A folder or filename must contain `Section N` or `Sec N` so it cannot confuse, for example, section 3 with section 39. Imported objects receive stable content-hash identities instead of order-dependent ZIP indexes.
2. **Track DAPI nuclei** with the Hungarian or Bayesian command. Set the optional source prefix to `dapi_` (or use the DAPI object group), and use a tracked output prefix such as `cell_`. This keeps anchor RNA traces out of the tracking input. The current Hungarian connector compares aligned ROI centroids and polygon areas, performs one-to-one augmented Hungarian assignment, and writes the persistent TrackID into the trace name.
3. **Map mRNA through DAPI tracks** with the corrected default windows `1:2-3;6:4-14;23:15-31;40:32-39`. Anchor mRNA is associated with a containing/nearby tracked DAPI nucleus. The same DAPI identity on each target section constrains a local inverse-distance-weighted deformation field, so identity comes from tracking while geometry follows nearby tissue deformation. If that exact identity is absent but common neighboring DAPI tracks remain, a local-field-only fallback creates a low-confidence ROI for mandatory review instead of silently dropping the cell.
4. **Review mapped mRNA** filters the PyReconstruct view to all mappings, high-confidence mappings, or mappings that need review.
5. **Validate against expert ROIs** performs one-to-one centroid matching and reports precision, recall, and F1. These are valid only when the expert ROI set is independent and complete.
6. **Measure antibody intensity** reads section-labelled antibody TIFFs, measures signal inside mapped mRNA ROIs, and writes measurement/positive-cell CSVs, a scatter plot, overlays, and a JSON summary. Automatic positivity is explicitly labelled exploratory.

### Fiji-style layers and expert feedback

`Plug-In > Channels / overlays > ROI Manager-style layers and labels` provides
Show All/Show None behavior plus separate DAPI, tracked-DAPI, anchor-mRNA, and
mapped-mRNA layer presets. ROI names can be drawn over visible traces.

`Plug-In > ⚠ Expert feedback (HIGH RISK)` records three correction types:

- anchor mRNA ↔ DAPI association;
- DAPI track link between two sections;
- mapped mRNA ROI approval/rejection.

Every feedback dialog begins with a warning and requires an explicit
acknowledgement. Corrections are stored beside the `.jser` as
`<series>.multiplex_feedback.json`; source ROI geometry is not overwritten.
When “Apply saved expert feedback” is enabled, RNA association feedback
forces/excludes the reviewed pair and tracking feedback splits or joins the
reviewed target-side trajectory. This is deterministic constrained correction,
not automatic neural-network retraining from a single click. The review-report
command exports separate ROI inventories and DAPI trace-line plots so the
record remains auditable.

The mapper writes mapped traces back into the series and produces:

- `multiplex_rna_mapping.csv`, including source identity, anchor/target sections, DAPI TrackID, association method, residual, confidence, and review status;
- `multiplex_rna_mapping_summary.json`;
- one QC plot per requested target section under `mapping_qc/`.

Missing sections, missing anchor ROIs, and target sections without tracked DAPI
are skipped instead of aborting the run. They are listed in the summary JSON,
and their QC plots contain a diagnostic note. The CSV and summary JSON are
still created even when no RNA trace can be mapped.

Users can create DAPI and mRNA traces with the existing U-Net/Cellpose-SAM
commands by choosing distinct prefixes (for example `dapi_` and `rna_`), import
existing Fiji ROI folders, or combine those approaches. Raw image folders
should first be opened as a PyReconstruct series so channel geometry and
section transforms remain authoritative.

### Current in-app DAPI tracking

`Plug-In > Tracking > Hungarian Tracking` uses the existing simple Hungarian
tracking core. For every closed DAPI trace selected by prefix/group, the
connector applies the PyReconstruct section transform and records the aligned
centroid and polygon area. Between successive sections that actually contain
selected DAPI traces, the current cost is a robustly normalized combination of
centroid distance (weight 0.2) and relative area difference (weight 0.1).
Augmented Hungarian assignment then chooses one-to-one matches while allowing
births and deaths (cost 0.6); assignments above 0.95 are rejected. Matched
traces are renamed `cell_<TrackID>`.

Empty or unavailable slices are ignored, so the tracker directly compares the
nearest available DAPI-bearing sections on either side of a gap. The current
simple configuration does not use image intensity, a learned appearance
embedding, cell division, or a motion model. Its output should therefore be
reviewed, especially across large gaps or strong tissue deformation.

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
