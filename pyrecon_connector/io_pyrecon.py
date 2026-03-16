"""
pyrecon_connector.io_pyrecon
============================
Read and write PyReconstruct series data.

PyReconstruct stores data in ``.jser`` (JSON) files.  Each series contains
one JSON file per section describing the contours (ROIs) traced on that
section.

This module provides helpers to:
  - Load contour centroids and polygon vertices from a ``.jser`` series.
  - Write renamed contours back to the series so PyReconstruct can display
    the tracking result.

Functions
---------
load_series_contours    -- Parse all contours from a PyReconstruct series
                           into a dict of per-section DataFrames.
rename_contours_in_series -- Apply TrackID-based names to contours and
                             write the result back to disk.
write_tracking_csv      -- Write the full trajectory CSV alongside the series.

Notes
-----
PyReconstruct `.jser` files are JSON.  The contour objects live under:
  series["sections"][<index>]["contours"]
Each contour has at least: ``name``, ``points`` (list of [x, y]).
"""

import json
import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _centroid_from_points(points: list):
    """
    Compute the centroid of a polygon given as a list of [x, y] pairs.

    Parameters
    ----------
    points : list of [float, float]
        Polygon vertices.

    Returns
    -------
    cx : float
    cy : float
    """
    arr = np.asarray(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _area_from_points(points: list) -> float:
    """
    Compute polygon area (shoelace) from a list of [x, y] pairs.

    Parameters
    ----------
    points : list of [float, float]
        Polygon vertices.

    Returns
    -------
    float
        Absolute area.
    """
    arr = np.asarray(points, dtype=np.float32)
    x, y = arr[:, 0], arr[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


# ---------------------------------------------------------------------------
#  Public functions
# ---------------------------------------------------------------------------

def load_series_contours(jser_path: str,
                          exclude_name: str = "domain1") -> dict:
    """
    Load all contours from a PyReconstruct ``.jser`` series file.

    Reads the JSON, iterates over each section, and extracts every contour
    that has at least 3 points.  Returns one DataFrame per section.

    Parameters
    ----------
    jser_path : str
        Path to the ``.jser`` file.
    exclude_name : str, optional
        Contour name to exclude (default: ``"domain1"`` which is the
        section boundary, not a cell).

    Returns
    -------
    dict
        ``{section_index (int): pd.DataFrame}`` with columns:
        Section, Contour_Name, X, Y, Area, points.

    Raises
    ------
    FileNotFoundError
        If ``jser_path`` does not exist.
    ValueError
        If the file cannot be parsed as a valid PyReconstruct series.
    """
    if not os.path.isfile(jser_path):
        raise FileNotFoundError(f"Series file not found: {jser_path}")

    with open(jser_path, "r", encoding="utf-8") as fh:
        series = json.load(fh)

    sections_data = series.get("sections", {})
    result = {}

    for sec_key, sec_obj in sections_data.items():
        try:
            sec_idx = int(sec_key)
        except ValueError:
            continue

        contours = sec_obj.get("contours", [])
        rows = []
        for c in contours:
            name   = c.get("name", "")
            points = c.get("points", [])
            if name == exclude_name or len(points) < 3:
                continue
            cx, cy = _centroid_from_points(points)
            area   = _area_from_points(points)
            rows.append(dict(
                Section=sec_idx,
                Contour_Name=name,
                X=float(cx),
                Y=float(cy),
                Area=float(area),
                points=points,
            ))

        if rows:
            result[sec_idx] = pd.DataFrame(rows)

    logging.info("Loaded %d sections from %s",
                 len(result), os.path.basename(jser_path))
    return result


def rename_contours_in_series(jser_path: str,
                               track_df: pd.DataFrame,
                               out_jser_path: str | None = None) -> str:
    """
    Rename contours in a PyReconstruct series based on TrackID assignments.

    Each contour that has been assigned a TrackID is renamed to
    ``cell_<TrackID>`` so that all sections belonging to the same cell
    share the same contour name — which is how PyReconstruct links objects
    across sections for 3-D reconstruction.

    Contours that were not matched (untracked) retain their original name
    with a ``untracked_`` prefix so they remain visible but are clearly
    distinguished.

    Parameters
    ----------
    jser_path : str
        Path to the original ``.jser`` file.
    track_df : pd.DataFrame
        Trajectory table with columns Section, Contour_Name, TrackID.
        Produced by ``PyReconConnector.run()``.
    out_jser_path : str or None, optional
        Where to write the modified series.  If None, overwrites the
        original (a ``.bak`` backup is made first).

    Returns
    -------
    str
        Path to the written ``.jser`` file.
    """
    if not os.path.isfile(jser_path):
        raise FileNotFoundError(f"Series file not found: {jser_path}")

    # Build rename lookup: (section, original_name) -> "cell_<TrackID>"
    rename_map: dict[tuple, str] = {}
    for row in track_df.itertuples():
        key = (int(row.Section), str(row.Contour_Name))
        rename_map[key] = f"cell_{int(row.TrackID):05d}"

    with open(jser_path, "r", encoding="utf-8") as fh:
        series = json.load(fh)

    for sec_key, sec_obj in series.get("sections", {}).items():
        try:
            sec_idx = int(sec_key)
        except ValueError:
            continue
        for c in sec_obj.get("contours", []):
            old_name = c.get("name", "")
            new_name = rename_map.get((sec_idx, old_name))
            if new_name:
                c["name"] = new_name
            elif old_name not in ("domain1",):
                c["name"] = f"untracked_{old_name}"

    # Determine output path
    if out_jser_path is None:
        bak = jser_path + ".bak"
        os.replace(jser_path, bak)
        out_jser_path = jser_path

    with open(out_jser_path, "w", encoding="utf-8") as fh:
        json.dump(series, fh, indent=2)

    logging.info("Renamed series written to %s", out_jser_path)
    return out_jser_path


def write_tracking_csv(track_df: pd.DataFrame, out_dir: str) -> str:
    """
    Write the full trajectory DataFrame to a CSV file.

    Parameters
    ----------
    track_df : pd.DataFrame
        Trajectory table with columns:
        TrackID, Section, Contour_Name, X, Y, Area.
    out_dir : str
        Directory to write into.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "cell_trajectories.csv")
    track_df.to_csv(out_path, index=False)
    logging.info("Trajectory CSV written to %s", out_path)
    return out_path
