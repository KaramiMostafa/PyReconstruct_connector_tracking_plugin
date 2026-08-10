from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MAPPED_GROUP = "multiplex_mapped_rna"
REVIEW_GROUP = "multiplex_mapping_review"
HIGH_CONFIDENCE_GROUP = "multiplex_mapping_high_confidence"


@dataclass
class _TraceRecord:
    section: int
    name: str
    trace: object
    local_points: np.ndarray
    aligned_points: np.ndarray
    centroid: np.ndarray
    ordinal: int


def parse_mapping_windows(value: str) -> Dict[int, List[int]]:
    """Parse ``anchor:start-end,...`` mappings used by the GUI."""
    windows: Dict[int, List[int]] = {}
    for block in str(value or "").split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Invalid mapping window '{block}'. Use anchor:start-end.")
        anchor_text, targets_text = block.split(":", 1)
        anchor = int(anchor_text.strip())
        targets: List[int] = []
        for token in targets_text.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo_text, hi_text = token.split("-", 1)
                lo, hi = int(lo_text), int(hi_text)
                step = 1 if hi >= lo else -1
                targets.extend(range(lo, hi + step, step))
            else:
                targets.append(int(token))
        targets = list(dict.fromkeys(targets))
        if not targets:
            raise ValueError(f"Anchor {anchor} has no target sections.")
        if anchor in targets:
            raise ValueError(f"Anchor {anchor} cannot also be one of its targets.")
        windows[anchor] = targets
    if not windows:
        raise ValueError("Enter at least one mapping window.")
    return windows


def _slug(value: str, limit: int = 45) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value or "roi")).strip("_")
    return (text or "roi")[:limit]


def _section_from_path(value: str) -> int | None:
    matches = re.findall(r"(?i)(?:section|sec)[ _-]*0*(\d+)(?=\D|$)", str(value))
    return int(matches[-1]) if matches else None


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(2, dtype=float)
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area6 = 3.0 * float(cross.sum())
    if abs(area6) < 1e-10:
        return points.mean(axis=0)
    return np.array([
        float(((x + np.roll(x, -1)) * cross).sum() / area6),
        float(((y + np.roll(y, -1)) * cross).sum() / area6),
    ])


def _point_in_polygon(point: Sequence[float], polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    poly = np.asarray(polygon, dtype=float)
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi:
            inside = not inside
        j = i
    return inside


def _idw_displacement(points: np.ndarray, origins: np.ndarray, displacements: np.ndarray, k: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(origins) == 0:
        return np.zeros_like(points)
    out = []
    use_k = max(1, min(int(k), len(origins)))
    for point in points:
        distances = np.linalg.norm(origins - point[None, :], axis=1)
        indexes = np.argsort(distances)[:use_k]
        if distances[indexes[0]] < 1e-9:
            out.append(displacements[indexes[0]])
            continue
        weights = 1.0 / (distances[indexes] + 1e-6)
        out.append((weights[:, None] * displacements[indexes]).sum(axis=0) / weights.sum())
    return np.asarray(out, dtype=float)


def _trace_records(series, section_num: int, prefix: str = "", group: str = "") -> List[_TraceRecord]:
    section = series.loadSection(int(section_num))
    allowed = set(series.object_groups.getGroupObjects(group)) if group else None
    records: List[_TraceRecord] = []
    ordinal = 0
    for name, contour in section.contours.items():
        if prefix and not str(name).startswith(prefix):
            continue
        if allowed is not None and name not in allowed:
            continue
        for trace in contour.traces:
            if not trace.closed or len(trace.points) < 3:
                continue
            local = np.asarray(trace.points, dtype=float)
            aligned = np.asarray(section.tform.map(local.tolist()), dtype=float)
            records.append(_TraceRecord(
                section=int(section_num), name=str(name), trace=trace,
                local_points=local, aligned_points=aligned,
                centroid=_polygon_centroid(aligned), ordinal=ordinal,
            ))
            ordinal += 1
    return records


def _associate_rna_to_dapi(rna: _TraceRecord, dapi: List[_TraceRecord], max_distance: float):
    contained = [record for record in dapi if _point_in_polygon(record.centroid, rna.aligned_points)]
    candidates = contained or dapi
    if not candidates:
        return None
    distances = [float(np.linalg.norm(record.centroid - rna.centroid)) for record in candidates]
    index = int(np.argmin(distances))
    record, distance = candidates[index], distances[index]
    if not contained and distance > float(max_distance):
        return None
    return record, distance, bool(contained)


def _stable_mapped_name(prefix: str, anchor: int, rna: _TraceRecord) -> Tuple[str, str]:
    raw = f"{anchor}|{rna.name}|{rna.ordinal}|{rna.centroid[0]:.6f}|{rna.centroid[1]:.6f}"
    uid = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}a{anchor:03d}_{_slug(rna.name, 28)}_{uid}", uid


def _remove_existing(section, name: str) -> None:
    contour = section.contours.get(name)
    if contour is None:
        return
    for trace in list(contour.traces):
        section.removeTrace(trace, log_event=True)


def _mapping_geometry(
    rna: _TraceRecord,
    source_dapi: _TraceRecord,
    target_dapi: _TraceRecord,
    source_dapi_by_name: Dict[str, _TraceRecord],
    target_dapi_by_name: Dict[str, _TraceRecord],
    neighbor_count: int,
):
    common = sorted(set(source_dapi_by_name) & set(target_dapi_by_name))
    origins = np.asarray([source_dapi_by_name[name].centroid for name in common], dtype=float)
    displacements = np.asarray([
        target_dapi_by_name[name].centroid - source_dapi_by_name[name].centroid for name in common
    ], dtype=float)
    associated_disp = target_dapi.centroid - source_dapi.centroid

    if len(origins):
        field = _idw_displacement(rna.aligned_points, origins, displacements, neighbor_count)
        at_source = _idw_displacement(source_dapi.centroid[None, :], origins, displacements, neighbor_count)[0]
        mapped = rna.aligned_points + field + (associated_disp - at_source)[None, :]
        nearest = np.argsort(np.linalg.norm(origins - source_dapi.centroid[None, :], axis=1))[:max(1, min(neighbor_count, len(origins)))]
        residual = float(np.median(np.linalg.norm(displacements[nearest] - associated_disp[None, :], axis=1)))
    else:
        mapped = rna.aligned_points + associated_disp[None, :]
        residual = 0.0
    return mapped, len(common), residual


def _write_qc_plots(output_dir: Path, rows: List[dict], polygons: Dict[Tuple[int, str], np.ndarray], dapi_points: Dict[int, np.ndarray]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon
    except Exception:
        return

    plot_dir = output_dir / "mapping_qc"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for section_num in sorted({int(row["target_section"]) for row in rows}):
        section_rows = [row for row in rows if int(row["target_section"]) == section_num]
        fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
        points = dapi_points.get(section_num, np.zeros((0, 2)))
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=5, color="#777777", alpha=0.5, label="DAPI")
        for row in section_rows:
            poly = polygons[(section_num, row["mapped_name"])]
            color = "#28c76f" if row["status"] == "high_confidence" else "#ff9f43"
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=0.8))
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"Section {section_num}: mapped RNA ({len(section_rows)})")
        ax.set_xlabel("Aligned X")
        ax.set_ylabel("Aligned Y")
        fig.tight_layout()
        fig.savefig(plot_dir / f"section_{section_num:03d}_mapping.png")
        plt.close(fig)


def run_multiplex_rna_mapping(
    series,
    mapping_windows,
    dapi_prefix: str = "cell_",
    dapi_group: str = "",
    rna_prefix: str = "rna_",
    rna_group: str = "",
    mapped_prefix: str = "mapped_rna_",
    output_dir: str | None = None,
    association_max_distance: float = 15.0,
    neighbor_count: int = 7,
    high_confidence_threshold: float = 0.70,
    overwrite: bool = False,
) -> dict:
    """Associate anchor RNA traces with tracked DAPI and map them to target sections."""
    windows = parse_mapping_windows(mapping_windows) if isinstance(mapping_windows, str) else mapping_windows
    if not dapi_prefix and not dapi_group:
        raise ValueError("Provide a tracked DAPI name prefix or object group.")
    if not rna_prefix and not rna_group:
        raise ValueError("Provide an RNA name prefix or object group.")
    if not mapped_prefix:
        raise ValueError("Provide a mapped RNA name prefix.")
    existing_sections = {int(value) for value in series.sections.keys()}
    missing = sorted(({int(a) for a in windows} | {int(t) for v in windows.values() for t in v}) - existing_sections)
    if missing:
        raise ValueError(f"Sections are not present in the open series: {missing}")

    rows: List[dict] = []
    mapped_polygons: Dict[Tuple[int, str], np.ndarray] = {}
    dapi_plot_points: Dict[int, np.ndarray] = {}
    skipped_unassociated = 0
    skipped_missing_track = 0

    for anchor, targets in windows.items():
        anchor_dapi = _trace_records(series, anchor, dapi_prefix, dapi_group)
        anchor_rna = _trace_records(series, anchor, rna_prefix, rna_group)
        if not anchor_dapi:
            raise ValueError(f"No tracked DAPI traces found on anchor section {anchor}.")
        if not anchor_rna:
            raise ValueError(f"No RNA traces found on anchor section {anchor}.")
        anchor_dapi_by_name = {record.name: record for record in anchor_dapi}

        associations = []
        for rna in anchor_rna:
            association = _associate_rna_to_dapi(rna, anchor_dapi, association_max_distance)
            if association is None:
                skipped_unassociated += 1
                continue
            associations.append((rna, *association))

        for target_num in targets:
            target_section = series.loadSection(int(target_num))
            target_dapi = _trace_records(series, target_num, dapi_prefix, dapi_group)
            target_dapi_by_name = {record.name: record for record in target_dapi}
            dapi_plot_points[int(target_num)] = np.asarray([record.centroid for record in target_dapi], dtype=float)

            for rna, source_dapi, association_distance, contained in associations:
                target_track = target_dapi_by_name.get(source_dapi.name)
                if target_track is None:
                    skipped_missing_track += 1
                    continue
                mapped_aligned, common_neighbors, residual = _mapping_geometry(
                    rna, source_dapi, target_track,
                    anchor_dapi_by_name, target_dapi_by_name, neighbor_count,
                )
                mapped_local = np.asarray(target_section.tform.map(mapped_aligned.tolist(), inverted=True), dtype=float)
                mapped_name, source_uid = _stable_mapped_name(mapped_prefix, int(anchor), rna)
                if mapped_name in target_section.contours:
                    if not overwrite:
                        continue
                    _remove_existing(target_section, mapped_name)

                association_score = 1.0 if contained else max(0.0, 1.0 - association_distance / max(association_max_distance, 1e-6))
                neighbor_score = min(1.0, common_neighbors / max(1.0, float(neighbor_count)))
                residual_score = math.exp(-residual / max(float(association_max_distance), 1e-6))
                confidence = float(0.50 * association_score + 0.20 * neighbor_score + 0.30 * residual_score)
                status = "high_confidence" if confidence >= float(high_confidence_threshold) else "review"

                from PyReconstruct.modules.datatypes import Trace
                trace = Trace(mapped_name, (40, 200, 100) if status == "high_confidence" else (255, 160, 40), closed=True)
                trace.points = [tuple(map(float, point)) for point in mapped_local]
                trace.fill_mode = ("transparent", "unselected")
                trace.tags.update({MAPPED_GROUP, status, f"anchor_{int(anchor):03d}", f"source_{source_uid}"})
                target_section.addTrace(trace, log_event=True)
                series.object_groups.add(MAPPED_GROUP, mapped_name)
                series.object_groups.add(HIGH_CONFIDENCE_GROUP if status == "high_confidence" else REVIEW_GROUP, mapped_name)

                mapped_polygons[(int(target_num), mapped_name)] = mapped_aligned
                rows.append({
                    "source_uid": source_uid,
                    "anchor_section": int(anchor),
                    "target_section": int(target_num),
                    "source_rna_name": rna.name,
                    "dapi_track_name": source_dapi.name,
                    "mapped_name": mapped_name,
                    "association_method": "containment" if contained else "nearest",
                    "association_distance": float(association_distance),
                    "common_neighbor_tracks": int(common_neighbors),
                    "local_displacement_residual": float(residual),
                    "confidence": confidence,
                    "status": status,
                })
            target_section.save(update_series_data=True)

    series.save()
    summary = {
        "created": len(rows),
        "high_confidence": sum(row["status"] == "high_confidence" for row in rows),
        "review": sum(row["status"] == "review" for row in rows),
        "skipped_unassociated": skipped_unassociated,
        "skipped_missing_track": skipped_missing_track,
    }

    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "multiplex_rna_mapping.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["source_uid"])
            writer.writeheader()
            writer.writerows(rows)
        (output / "multiplex_rna_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_qc_plots(output, rows, mapped_polygons, dapi_plot_points)
        summary["csv"] = str(csv_path)
        summary["qc_dir"] = str(output / "mapping_qc")
    return summary


def _roi_payloads(folder: str) -> Iterable[Tuple[int, str, bytes]]:
    root = Path(folder)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        section_num = _section_from_path(str(path.relative_to(root)))
        if path.suffix.lower() == ".roi" and section_num is not None:
            yield section_num, path.stem, path.read_bytes()
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    for entry in sorted(
                        n for n in archive.namelist()
                        if n.lower().endswith(".roi") and not Path(n).name.startswith("._")
                    ):
                        entry_section = _section_from_path(entry) or section_num
                        if entry_section is not None:
                            yield entry_section, Path(entry).stem, archive.read(entry)
            except (OSError, zipfile.BadZipFile):
                continue


def import_multiplex_roi_folders(
    series,
    dapi_folder: str | None = None,
    rna_folder: str | None = None,
    dapi_prefix: str = "dapi_",
    rna_prefix: str = "rna_",
    dapi_group: str = "multiplex_dapi",
    rna_group: str = "multiplex_rna_anchor",
) -> dict:
    """Import section-labelled ImageJ ROI files/ZIPs into an open series."""
    from roifile import ImagejRoi
    from PyReconstruct.modules.datatypes import Trace

    section_numbers = {int(value) for value in series.sections.keys()}
    counts = {"dapi": 0, "rna": 0, "skipped": 0}
    sources = [
        (dapi_folder, dapi_prefix, dapi_group, "dapi", (80, 140, 255)),
        (rna_folder, rna_prefix, rna_group, "rna", (255, 70, 70)),
    ]
    loaded_sections = {}
    for folder, prefix, group, kind, color in sources:
        if not folder:
            continue
        for section_num, source_name, payload in _roi_payloads(folder):
            if section_num not in section_numbers:
                counts["skipped"] += 1
                continue
            section = loaded_sections.setdefault(section_num, series.loadSection(section_num))
            try:
                roi = ImagejRoi.frombytes(payload)
            except Exception:
                counts["skipped"] += 1
                continue
            coordinates = np.asarray(roi.coordinates(), dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[0] < 3:
                counts["skipped"] += 1
                continue
            image_height = int(section.img_dims[0])
            points = np.column_stack((coordinates[:, 0] * section.mag, (image_height - coordinates[:, 1]) * section.mag))
            identity = hashlib.sha1(payload).hexdigest()[:10]
            name = f"{prefix}s{section_num:03d}_{_slug(getattr(roi, 'name', None) or source_name, 28)}_{identity}"
            if name in section.contours:
                counts["skipped"] += 1
                continue
            trace = Trace(name, color, closed=True)
            trace.points = [tuple(map(float, point)) for point in points]
            trace.fill_mode = ("transparent", "unselected")
            trace.tags.update({group, f"source_{identity}"})
            section.addTrace(trace, log_event=True)
            series.object_groups.add(group, name)
            counts[kind] += 1

    for section in loaded_sections.values():
        section.save(update_series_data=True)
    series.save()
    return counts
