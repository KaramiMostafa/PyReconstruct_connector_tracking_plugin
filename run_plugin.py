#!/usr/bin/env python3
"""
run_plugin.py
=============
Command-line entry point for the PyReconstruct cell-tracker plugin.

Usage
-----
    python run_plugin.py \\
        --jser  /path/to/series.jser \\
        --out   /path/to/results \\
        [--tif-dir /path/to/dapi_tifs]

Arguments
---------
--jser      Path to the PyReconstruct .jser series file.
--out       Output directory (created if it does not exist).
--tif-dir   Optional: directory of per-section DAPI TIF images for
            phase-alignment between sections.
"""

import argparse
import sys

from pyrecon_connector import PyReconConnector


def main():
    parser = argparse.ArgumentParser(
        description="PyReconstruct DAPI cell tracker plugin")
    parser.add_argument("--jser",    required=True,
                        help="Path to .jser series file")
    parser.add_argument("--out",     required=True,
                        help="Output directory")
    parser.add_argument("--tif-dir", default=None,
                        help="Directory of DAPI TIF images (optional)")
    args = parser.parse_args()

    conn = PyReconConnector(
        jser_path=args.jser,
        out_dir=args.out,
        tif_dir=args.tif_dir,
    )
    track_df = conn.run()

    if track_df is None or track_df.empty:
        print("[ERROR] Tracking produced no output.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[OK] Tracking complete.")
    print(f"     Unique tracks : {track_df['TrackID'].nunique()}")
    print(f"     Sections      : {track_df['Section'].nunique()}")
    print(f"     Output        : {args.out}")


if __name__ == "__main__":
    main()
