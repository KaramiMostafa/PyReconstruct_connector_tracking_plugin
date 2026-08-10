from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tracking_BayesianTransformer import BayesianTransformerForCellTracking, train_bnn, load_model, match_pair


def _poly_area(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    area2 = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area2 += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area2) * 0.5


def _poly_perimeter(pts: List[Tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        dx = pts[i][0] - pts[j][0]
        dy = pts[i][1] - pts[j][1]
        total += (dx * dx + dy * dy) ** 0.5
    return float(total)


@dataclass
class _Ref:
    section_num: int
    trace: object


class _Table:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["FrameID", "Label"]).reset_index(drop=True)

    def at(self, frame: int) -> pd.DataFrame:
        return self.df[self.df["FrameID"] == int(frame)].sort_values("Label").reset_index(drop=True)


def _parse_idx(token: str) -> int:
    return int(str(token).split("_")[1])


def _make_feature_table(series, sec_nums: List[int], source_prefix: str = "", source_group: str = ""):
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
                pts_t = list(tform.map(tr.points))
                xs = [float(p[0]) for p in pts_t]
                ys = [float(p[1]) for p in pts_t]
                area = _poly_area(pts_t)
                perimeter = _poly_perimeter(pts_t)
                width = max(xs) - min(xs) if xs else 0.0
                height = max(ys) - min(ys) if ys else 0.0
                compactness = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-8)
                rows.append(
                    {
                        "FrameID": frame_idx,
                        "Label": local_label,
                        "Centroid_X": float(cx),
                        "Centroid_Y": float(cy),
                        "Area": float(area),
                        "Perimeter": float(perimeter),
                        "BBox_Width": float(width),
                        "BBox_Height": float(height),
                        "Compactness": float(compactness),
                    }
                )
                frame_refs.append(_Ref(section_num=snum, trace=tr))
                local_label += 1

        refs[frame_idx] = frame_refs

    df = pd.DataFrame(rows)
    return df, refs, sections_by_frame


def _build_track_table(df: pd.DataFrame, matches_by_frame: Dict[int, list], frame_ids: List[int]) -> pd.DataFrame:
    next_track_id = 0
    track_by_key: Dict[Tuple[int, int], int] = {}

    if not frame_ids:
        return pd.DataFrame(columns=["FrameID", "Label", "TrackID"])

    first = frame_ids[0]
    first_df = df[df["FrameID"] == first].sort_values("Label").reset_index(drop=True)
    for r in first_df.itertuples():
        track_by_key[(first, int(r.Label))] = next_track_id
        next_track_id += 1

    for frame_idx in frame_ids[:-1]:
        cur_df = df[df["FrameID"] == frame_idx].sort_values("Label").reset_index(drop=True)
        nxt_df = df[df["FrameID"] == frame_idx + 1].sort_values("Label").reset_index(drop=True)
        assigned_children = set()

        for parent, children in matches_by_frame.get(frame_idx, []):
            pidx = _parse_idx(parent)
            if pidx < 0 or pidx >= len(cur_df):
                continue
            parent_label = int(cur_df.iloc[pidx]["Label"])
            parent_key = (frame_idx, parent_label)
            if parent_key not in track_by_key:
                track_by_key[parent_key] = next_track_id
                next_track_id += 1
            parent_track = track_by_key[parent_key]
            child_tokens = children if isinstance(children, list) else [children]

            for child_order, child in enumerate(child_tokens):
                cidx = _parse_idx(child)
                if cidx < 0 or cidx >= len(nxt_df):
                    continue
                child_label = int(nxt_df.iloc[cidx]["Label"])
                if child_label in assigned_children:
                    continue
                if child_order == 0:
                    child_track = parent_track
                else:
                    child_track = next_track_id
                    next_track_id += 1
                track_by_key[(frame_idx + 1, child_label)] = child_track
                assigned_children.add(child_label)

        for r in nxt_df.itertuples():
            key = (frame_idx + 1, int(r.Label))
            if key not in track_by_key:
                track_by_key[key] = next_track_id
                next_track_id += 1

    rows = []
    for (frame, label), track in sorted(track_by_key.items()):
        rows.append({"FrameID": int(frame), "Label": int(label), "TrackID": int(track)})
    return pd.DataFrame(rows)


def _rename_from_tracks(sections_by_frame, refs, tracks_df: pd.DataFrame, prefix: str) -> int:
    renamed = 0
    for frame_idx, sub in tracks_df.groupby("FrameID"):
        section = sections_by_frame[int(frame_idx)]
        for r in sub.itertuples():
            label = int(r.Label)
            track_id = int(r.TrackID)
            if label < 0 or label >= len(refs[int(frame_idx)]):
                continue
            ref = refs[int(frame_idx)][label]
            section.editTraceAttributes(
                traces=[ref.trace],
                name=f"{prefix}{track_id:05d}",
                color=None,
                tags=None,
                mode=None,
                add_tags=False,
                log_event=True,
            )
            renamed += 1
        section.save(update_series_data=True)
    return renamed


def _load_or_train_model(df: pd.DataFrame, frame_ids: List[int], model_path: str | None, train_epochs: int):
    if model_path:
        model, mean, std, features = load_model(model_path)
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"The checkpoint requires features that are not available from PyReconstruct traces: {missing}")
        return model, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32), list(features)

    features = ["Centroid_X", "Centroid_Y", "Area", "Perimeter", "BBox_Width", "BBox_Height", "Compactness"]
    mean = df[features].to_numpy(np.float32).mean(axis=0)
    std = df[features].to_numpy(np.float32).std(axis=0) + 1e-8
    model = BayesianTransformerForCellTracking(input_dim=len(features), embed_dim=64, num_heads=2, ff_hidden_dim=256, num_layers=2, output_dim=2)
    frame_pairs = [(int(f), int(f) + 1) for f in frame_ids[:-1]]
    model = train_bnn(model, frame_pairs, df, features, num_epochs=max(1, int(train_epochs)), batch_size=128, early_stopping_patience=5, reduce_lr_patience=3, device="auto")
    return model, mean, std, features


def run_bayesian_tracking_on_series(series, start_sec: int, end_sec: int, prefix: str = "bt_cell_", model_path: str | None = None, train_epochs: int = 20, motion_threshold: float = 200.0, source_prefix: str = "", source_group: str = "") -> int:
    sec_nums = [s for s in sorted(series.sections.keys()) if int(start_sec) <= s <= int(end_sec)]
    if len(sec_nums) < 2:
        raise ValueError("Need at least 2 sections in range.")

    df, refs, sections_by_frame = _make_feature_table(series, sec_nums, source_prefix, source_group)
    if df.empty:
        raise ValueError("No closed traces found in selected range.")

    frame_ids = sorted(df["FrameID"].unique().astype(int).tolist())
    if len(frame_ids) < 2:
        raise ValueError("Need closed traces on at least 2 sections.")

    model, mean, std, features = _load_or_train_model(df, frame_ids, model_path, train_epochs)
    table = _Table(df)
    matches_by_frame = {}

    for frame_idx in frame_ids[:-1]:
        if table.at(frame_idx).empty or table.at(frame_idx + 1).empty:
            matches_by_frame[frame_idx] = []
            continue
        matches, _, _ = match_pair(table, int(frame_idx), model, mean, std, features, motion_threshold=float(motion_threshold))
        matches_by_frame[frame_idx] = matches

    tracks_df = _build_track_table(df, matches_by_frame, frame_ids)
    renamed = _rename_from_tracks(sections_by_frame, refs, tracks_df, prefix)
    series.save()
    return renamed
