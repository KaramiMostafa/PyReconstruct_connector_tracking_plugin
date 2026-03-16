"""
pyrecon_connector.connector
============================
Main connector class that bridges PyReconstruct's data format with the
cell-tracker-core library.

Classes
-------
PyReconConnector  -- Reads PyReconstruct ROIs, runs tracking via
                     cell-tracker-core, renames contours, and writes outputs.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Import cell-tracker-core.
#  The core library is expected either:
#   (a) installed via pip, OR
#   (b) present as a git submodule at ./cell_tracker_core/
# ---------------------------------------------------------------------------
try:
    from cell_tracker import CellTracker
    from cell_tracker.geometry import normalise_pair
    from cell_tracker.alignment import phase_shift_dxdy
    from cell_tracker.io import read_tif_as_2d
    from cell_tracker.algorithms.bnn_bp import BNNBeliefPropagationTracker
except ImportError:
    _submodule = os.path.join(os.path.dirname(__file__), "..", "cell_tracker_core")
    sys.path.insert(0, os.path.abspath(_submodule))
    from cell_tracker import CellTracker
    from cell_tracker.geometry import normalise_pair
    from cell_tracker.alignment import phase_shift_dxdy
    from cell_tracker.io import read_tif_as_2d
    from cell_tracker.algorithms.bnn_bp import BNNBeliefPropagationTracker

from .io_pyrecon import (
    load_series_contours,
    rename_contours_in_series,
    write_tracking_csv,
)


class PyReconConnector:
    """
    Plugin connector between PyReconstruct and the cell-tracker-core library.

    Workflow
    --------
    1. Read contours from a PyReconstruct ``.jser`` series file.
    2. Extract centroid coordinates per section.
    3. Run the BNN-BP tracking algorithm (or any custom BaseTracker).
    4. Rename contours in the series:  each tracked cell gets the name
       ``cell_<TrackID>`` across every section it appears in — this is the
       convention PyReconstruct uses to link objects across sections for 3-D
       rendering.
    5. Write outputs:
       - Modified ``.jser`` with renamed contours.
       - ``cell_trajectories.csv`` with full track table.
       - ``tracking_summary.csv`` with per-pair Precision / Recall / F1.

    Parameters
    ----------
    jser_path : str
        Path to the input PyReconstruct ``.jser`` series file.
    out_dir : str
        Directory for all output files.
    algorithm : BaseTracker or None
        Tracking algorithm.  Defaults to
        :class:`~cell_tracker.algorithms.bnn_bp.BNNBeliefPropagationTracker`.
    tif_dir : str or None
        Directory containing per-section DAPI TIF images for phase-alignment.
        If None, phase alignment is skipped (global shift = 0).

    Examples
    --------
    >>> conn = PyReconConnector(
    ...     jser_path="/data/myseries.jser",
    ...     out_dir="/results/tracking",
    ... )
    >>> track_df = conn.run()
    """

    def __init__(self,
                 jser_path: str,
                 out_dir: str,
                 algorithm=None,
                 tif_dir: str | None = None):
        self.jser_path  = jser_path
        self.out_dir    = out_dir
        self.algorithm  = algorithm or BNNBeliefPropagationTracker()
        self.tif_dir    = tif_dir

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full connector pipeline.

        Returns
        -------
        pd.DataFrame
            Trajectory table with columns:
            TrackID, Section, Contour_Name, X, Y, Area.
        """
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # ── Step 1: load contours from PyReconstruct series ────────────────
        logging.info("Reading series: %s", self.jser_path)
        sections_data = load_series_contours(self.jser_path)
        section_list  = sorted(sections_data.keys())

        if len(section_list) < 2:
            logging.error("Need >= 2 sections with contours. Aborting.")
            return pd.DataFrame()

        logging.info("Found %d sections: %s ... %s",
                     len(section_list), section_list[0], section_list[-1])

        # ── Step 2: pairwise tracking ──────────────────────────────────────
        all_pair_matches: dict = {}
        summary_rows: list    = []

        for sec_a, sec_b in zip(section_list[:-1], section_list[1:]):
            df_a = sections_data[sec_a]
            df_b = sections_data[sec_b]
            xy1  = df_a[["X", "Y"]].to_numpy(np.float32)
            xy2  = df_b[["X", "Y"]].to_numpy(np.float32)

            # Optional phase alignment using TIF images
            gdx, gdy = 0.0, 0.0
            if self.tif_dir:
                gdx, gdy = self._phase_align(sec_a, sec_b)
            xy1_shifted = xy1 + np.array([gdx, gdy], np.float32)

            # Normalise coordinates
            xy1_norm, xy2_norm = normalise_pair(xy1_shifted, xy2)

            logging.info("Tracking  %02d -> %02d  (n1=%d  n2=%d) ...",
                         sec_a, sec_b, len(xy1), len(xy2))
            matched = self.algorithm.match_pair(xy1_norm, xy2_norm)
            all_pair_matches[(sec_a, sec_b)] = matched

            n1, n2 = len(xy1), len(xy2)
            prec   = len(matched) / n1 if n1 else 0.0
            rec    = len(matched) / n2 if n2 else 0.0
            f1     = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
            summary_rows.append(dict(
                SectionA=sec_a, SectionB=sec_b,
                CellsA=n1, CellsB=n2, Matched=len(matched),
                Precision=round(prec, 4), Recall=round(rec, 4),
                F1=round(f1, 4),
                gdx=round(gdx, 2), gdy=round(gdy, 2)))
            logging.info("  -> %d matched", len(matched))

        # ── Step 3: chain into global tracks ──────────────────────────────
        logging.info("Building global trajectories ...")
        track_df = self._chain(all_pair_matches, sections_data, section_list)

        # ── Step 4: rename contours in the series ─────────────────────────
        out_jser = os.path.join(self.out_dir,
                                os.path.basename(self.jser_path).replace(
                                    ".jser", "_tracked.jser"))
        rename_contours_in_series(self.jser_path, track_df, out_jser)
        logging.info("Tracked series written: %s", out_jser)

        # ── Step 5: write CSVs ─────────────────────────────────────────────
        write_tracking_csv(track_df, self.out_dir)
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(self.out_dir, "tracking_summary.csv"), index=False)

        n_tracks = track_df["TrackID"].nunique()
        logging.info("Done. %d unique cell tracks. Results -> %s",
                     n_tracks, self.out_dir)
        return track_df

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _phase_align(self, sec_a: int, sec_b: int):
        """
        Estimate global (dx, dy) shift between two sections using TIF images.

        Looks for files named ``*<sec_a>*.tif`` and ``*<sec_b>*.tif`` in
        ``self.tif_dir``.

        Parameters
        ----------
        sec_a : int
            Source section number.
        sec_b : int
            Target section number.

        Returns
        -------
        dx : float
        dy : float
        """
        import glob
        def _find_tif(sec):
            cands = glob.glob(os.path.join(self.tif_dir, f"*{sec}*.tif"))
            return cands[0] if cands else None

        tif_a = _find_tif(sec_a)
        tif_b = _find_tif(sec_b)
        if tif_a and tif_b:
            img_a = read_tif_as_2d(tif_a)
            img_b = read_tif_as_2d(tif_b)
            return phase_shift_dxdy(img_a, img_b)
        logging.warning("TIF not found for sections %d or %d; skipping phase alignment",
                        sec_a, sec_b)
        return 0.0, 0.0

    @staticmethod
    def _chain(all_pair_matches: dict,
               sections_data: dict,
               section_list: list) -> pd.DataFrame:
        """
        Chain pairwise matches into persistent global TrackIDs.

        Parameters
        ----------
        all_pair_matches : dict
            ``{(sec_a, sec_b): [(idx_a, idx_b), ...]}``
        sections_data : dict
            ``{section_index: pd.DataFrame}`` as returned by
            ``load_series_contours``.
        section_list : list of int
            Sorted list of section numbers.

        Returns
        -------
        pd.DataFrame
            Columns: TrackID, Section, Contour_Name, X, Y, Area.
        """
        first_sec = section_list[0]
        df0       = sections_data[first_sec]
        track_map = {first_sec: {i: i for i in range(len(df0))}}
        next_tid  = len(df0)
        rows      = []

        for i, row in enumerate(df0.itertuples()):
            rows.append(dict(
                TrackID=track_map[first_sec][i],
                Section=int(row.Section),
                Contour_Name=str(row.Contour_Name),
                X=float(row.X), Y=float(row.Y), Area=float(row.Area)))

        for sec_a, sec_b in zip(section_list[:-1], section_list[1:]):
            matched   = all_pair_matches.get((sec_a, sec_b), [])
            df_b      = sections_data[sec_b]
            prev      = track_map[sec_a]
            matched_b = {ib: ia for ia, ib in matched}
            track_map[sec_b] = {}

            for ib in range(len(df_b)):
                if ib in matched_b:
                    tid = prev.get(matched_b[ib], next_tid)
                    if tid == next_tid:
                        next_tid += 1
                else:
                    tid = next_tid
                    next_tid += 1
                track_map[sec_b][ib] = tid

            for ib, row in enumerate(df_b.itertuples()):
                rows.append(dict(
                    TrackID=track_map[sec_b][ib],
                    Section=int(row.Section),
                    Contour_Name=str(row.Contour_Name),
                    X=float(row.X), Y=float(row.Y), Area=float(row.Area)))

        return pd.DataFrame(rows)
