from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from tracking_hungarian.roi import ROIFrameTable
from tracking_hungarian.pipeline import HungarianConfig, track_series

TRACKED_DAPI_GROUP = "multiplex_tracked_dapi"


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


def _nearest_ref_index(frame_refs: List[_Ref], name: str, centroid) -> int | None:
    exact = [index for index, ref in enumerate(frame_refs) if str(ref.trace.name) == str(name)]
    candidates = exact or list(range(len(frame_refs)))
    if not candidates:
        return None
    if centroid is None:
        return candidates[0] if len(candidates) == 1 else None
    cx, cy = float(centroid[0]), float(centroid[1])
    return min(
        candidates,
        key=lambda index: (
            float(frame_refs[index].trace.getCentroid()[0]) - cx
        ) ** 2 + (
            float(frame_refs[index].trace.getCentroid()[1]) - cy
        ) ** 2,
    )


def _apply_link_feedback(tracks_df, refs, frame_for_section, records) -> int:
    """Apply explicit link/unlink constraints to the computed TrackIDs."""
    applied = 0
    next_track_id = int(tracks_df["TrackID"].max()) + 1
    for record in records:
        source_frame = frame_for_section.get(int(record.get("section", -1)))
        target_frame = frame_for_section.get(int(record.get("secondary_section", -1)))
        if source_frame is None or target_frame is None or source_frame == target_frame:
            continue
        if source_frame > target_frame:
            source_frame, target_frame = target_frame, source_frame
            source_name, target_name = record.get("secondary_name"), record.get("primary_name")
            source_centroid, target_centroid = record.get("secondary_centroid"), record.get("primary_centroid")
        else:
            source_name, target_name = record.get("primary_name"), record.get("secondary_name")
            source_centroid, target_centroid = record.get("primary_centroid"), record.get("secondary_centroid")
        source_label = _nearest_ref_index(refs[source_frame], source_name, source_centroid)
        target_label = _nearest_ref_index(refs[target_frame], target_name, target_centroid)
        if source_label is None or target_label is None:
            continue
        source_rows = tracks_df[
            (tracks_df["FrameID"] == source_frame) & (tracks_df["Label"] == source_label)
        ]
        target_rows = tracks_df[
            (tracks_df["FrameID"] == target_frame) & (tracks_df["Label"] == target_label)
        ]
        if source_rows.empty or target_rows.empty:
            continue
        source_tid = int(source_rows.iloc[0]["TrackID"])
        target_tid = int(target_rows.iloc[0]["TrackID"])
        verdict = str(record.get("verdict"))
        if verdict == "incorrect" and source_tid == target_tid:
            mask = (tracks_df["TrackID"] == target_tid) & (tracks_df["FrameID"] >= target_frame)
            tracks_df.loc[mask, "TrackID"] = next_track_id
            next_track_id += 1
            applied += 1
        elif verdict == "correct" and source_tid != target_tid:
            # Merge the target-side segment only, avoiding changes to earlier reviewed sections.
            mask = (tracks_df["TrackID"] == target_tid) & (tracks_df["FrameID"] >= target_frame)
            occupied = set(
                tracks_df.loc[
                    (tracks_df["TrackID"] == source_tid) & (tracks_df["FrameID"] >= target_frame),
                    "FrameID",
                ].astype(int)
            )
            mask &= ~tracks_df["FrameID"].isin(occupied)
            tracks_df.loc[mask, "TrackID"] = source_tid
            applied += 1
    return applied


def run_hungarian_tracking_on_series(
    series,
    start_sec: int,
    end_sec: int,
    prefix: str = "cell_",
    source_prefix: str = "",
    source_group: str = "",
    apply_feedback: bool = True,
) -> int:
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

        allowed = set(series.object_groups.getGroupObjects(source_group)) if source_group else None
        for cname, contour in section.contours.items():
            if cname == "domain1":
                continue
            if source_prefix and not str(cname).startswith(source_prefix):
                continue
            if allowed is not None and cname not in allowed:
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
    if apply_feedback:
        try:
            from .feedback import dapi_link_constraints, load_feedback
            records = dapi_link_constraints(load_feedback(series))
            _apply_link_feedback(
                tracks_df,
                refs,
                {section_num: frame for frame, section_num in enumerate(sec_nums)},
                records,
            )
        except ValueError:
            pass

    renamed = 0
    for frame_idx, sub in tracks_df.groupby("FrameID"):
        section = sections_by_frame[int(frame_idx)]

        for r in sub.itertuples():
            idx = int(r.Label)
            tid = int(r.TrackID)
            new_name = f"{prefix}{tid:05d}"
            ref = refs[int(frame_idx)][idx]
            old_name = str(ref.trace.name)
            old_groups = set(series.object_groups.getObjectGroups(old_name))

            section.editTraceAttributes(
                traces=[ref.trace],
                name=new_name,
                color=None,
                tags=None,
                mode=None,
                add_tags=False,
                log_event=True,
            )
            series.object_groups.add(TRACKED_DAPI_GROUP, new_name)
            for group in old_groups:
                series.object_groups.add(group, new_name)
            renamed += 1

        section.save(update_series_data=True)

    return renamed
