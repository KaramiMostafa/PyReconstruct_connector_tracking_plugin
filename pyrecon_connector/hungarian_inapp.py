from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from tracking_hungarian.roi import ROIFrameTable
from tracking_hungarian.pipeline import HungarianConfig, track_series


def _poly_area(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    area2 = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area2 += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area2) * 0.5


@dataclass
class _Ref:
    section_num: int
    trace: object


def run_hungarian_tracking_on_series(series, start_sec: int, end_sec: int, prefix: str = "cell_") -> int:
    sec_nums = [s for s in sorted(series.sections.keys()) if start_sec <= s <= end_sec]
    if len(sec_nums) < 2:
        raise ValueError("Need at least 2 sections in range.")

    rows = []
    refs: Dict[int, List[_Ref]] = {}
    sections_by_frame: Dict[int, object] = {}

    for frame_idx, snum in enumerate(sec_nums):
        section = series.loadSection(snum)
        sections_by_frame[frame_idx] = section
        tform = section.tform

        frame_refs: List[_Ref] = []
        local_label = 0

        for cname, contour in section.contours.items():
            if cname == "domain1":
                continue
            for tr in contour.traces:
                if (not tr.closed) or (len(tr.points) < 3):
                    continue

                cx, cy = tr.getCentroid(tform=tform)
                pts_t = tform.map(tr.points)
                area = _poly_area(pts_t)

                rows.append(
                    {
                        "FrameID": frame_idx,
                        "Label": local_label,
                        "Centroid_X": float(cx),
                        "Centroid_Y": float(cy),
                        "Area": float(area),
                    }
                )
                frame_refs.append(_Ref(section_num=snum, trace=tr))
                local_label += 1

        refs[frame_idx] = frame_refs

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No closed traces found in selected range.")

    tbl = ROIFrameTable(df)
    cfg = HungarianConfig()
    tracks_df = track_series(tbl, cfg)

    renamed = 0
    for frame_idx, sub in tracks_df.groupby("FrameID"):
        section = sections_by_frame[int(frame_idx)]

        for r in sub.itertuples():
            idx = int(r.Label)
            tid = int(r.TrackID)
            new_name = f"{prefix}{tid:05d}"
            ref = refs[int(frame_idx)][idx]

            section.editTraceAttributes(
                traces=[ref.trace],
                name=new_name,
                color=None,
                tags=None,
                mode=None,
                add_tags=False,
                log_event=True,
            )
            renamed += 1

        section.save(update_series_data=True)

    return renamed