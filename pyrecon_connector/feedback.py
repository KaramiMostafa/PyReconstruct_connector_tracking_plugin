from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
FEEDBACK_SUFFIX = ".multiplex_feedback.json"
VALID_KINDS = {"rna_dapi_pair", "dapi_track_link", "mapped_rna"}
VALID_VERDICTS = {"correct", "incorrect"}


def feedback_path_for_series(series) -> Path:
    """Return the auditable feedback sidecar path for an open PyReconstruct series."""
    raw_path = str(getattr(series, "jser_fp", "") or "").strip()
    if not raw_path:
        raise ValueError("Save the PyReconstruct series before recording feedback.")
    jser_path = Path(raw_path)
    return jser_path.with_name(f"{jser_path.stem}{FEEDBACK_SUFFIX}")


def empty_feedback() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "warning": (
            "Expert corrections alter later tracking and RNA mapping. "
            "Only record a correction after checking cell identity, section, and ROI type."
        ),
        "records": [],
    }


def load_feedback(path_or_series) -> dict:
    path = (
        feedback_path_for_series(path_or_series)
        if not isinstance(path_or_series, (str, Path))
        else Path(path_or_series)
    )
    if not path.exists():
        return empty_feedback()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid feedback file: {path}")
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("warning", empty_feedback()["warning"])
    return payload


def save_feedback(path_or_series, payload: dict) -> str:
    path = (
        feedback_path_for_series(path_or_series)
        if not isinstance(path_or_series, (str, Path))
        else Path(path_or_series)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _centroid(trace) -> list[float]:
    try:
        x, y = trace.getCentroid()
    except Exception:
        points = list(getattr(trace, "points", []) or [])
        if not points:
            return [0.0, 0.0]
        x = sum(float(point[0]) for point in points) / len(points)
        y = sum(float(point[1]) for point in points) / len(points)
    return [float(x), float(y)]


def add_feedback_record(
    series,
    kind: str,
    verdict: str,
    section: int,
    primary_trace,
    secondary_trace=None,
    secondary_section: int | None = None,
    notes: str = "",
) -> dict:
    """Append one explicit correction without modifying the source ROI geometry."""
    kind = str(kind)
    verdict = str(verdict)
    if kind not in VALID_KINDS:
        raise ValueError(f"Unsupported feedback kind: {kind}")
    if verdict not in VALID_VERDICTS:
        raise ValueError("Feedback verdict must be 'correct' or 'incorrect'.")
    payload = load_feedback(series)
    record = {
        "kind": kind,
        "verdict": verdict,
        "section": int(section),
        "primary_name": str(primary_trace.name),
        "primary_centroid": _centroid(primary_trace),
        "secondary_section": int(secondary_section if secondary_section is not None else section),
        "secondary_name": str(secondary_trace.name) if secondary_trace is not None else "",
        "secondary_centroid": _centroid(secondary_trace) if secondary_trace is not None else None,
        "notes": str(notes or ""),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Replace the same assertion instead of accumulating contradictory stale clicks.
    key_fields = ("kind", "section", "primary_name", "secondary_section", "secondary_name")
    payload["records"] = [
        old for old in payload["records"]
        if any(old.get(field) != record.get(field) for field in key_fields)
    ]
    payload["records"].append(record)
    save_feedback(series, payload)
    return record


def association_constraints(payload: dict, section: int) -> dict[str, dict[str, set[str]]]:
    """Return forced/excluded DAPI names keyed by RNA name for one anchor section."""
    constraints: dict[str, dict[str, set[str]]] = {}
    for record in payload.get("records", []):
        if record.get("kind") != "rna_dapi_pair" or int(record.get("section", -1)) != int(section):
            continue
        rna_name = str(record.get("primary_name", ""))
        dapi_name = str(record.get("secondary_name", ""))
        if not rna_name or not dapi_name:
            continue
        entry = constraints.setdefault(rna_name, {"correct": set(), "incorrect": set()})
        entry[str(record.get("verdict", "incorrect"))].add(dapi_name)
    return constraints


def dapi_link_constraints(payload: dict) -> list[dict]:
    return [
        record for record in payload.get("records", [])
        if record.get("kind") == "dapi_track_link"
    ]


def mapped_roi_verdicts(payload: dict) -> dict[tuple[int, str], str]:
    return {
        (int(record.get("section", -1)), str(record.get("primary_name", ""))): str(record.get("verdict"))
        for record in payload.get("records", [])
        if record.get("kind") == "mapped_rna"
    }


def feedback_summary(path_or_series) -> dict:
    payload = load_feedback(path_or_series)
    counts = {}
    for record in payload.get("records", []):
        key = f"{record.get('kind')}:{record.get('verdict')}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "path": str(
            feedback_path_for_series(path_or_series)
            if not isinstance(path_or_series, (str, Path))
            else Path(path_or_series)
        ),
        "total": len(payload.get("records", [])),
        "counts": counts,
    }


def export_feedback_csv(path_or_series, output_path: str | Path) -> str:
    payload = load_feedback(path_or_series)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind", "verdict", "section", "primary_name", "primary_centroid",
        "secondary_section", "secondary_name", "secondary_centroid", "notes", "created_utc",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload.get("records", []):
            row = {field: record.get(field, "") for field in fields}
            row["primary_centroid"] = json.dumps(row["primary_centroid"])
            row["secondary_centroid"] = json.dumps(row["secondary_centroid"])
            writer.writerow(row)
    return str(output_path)


def write_feedback_review(series, output_dir: str | Path, prefixes: Iterable[str] = ("cell_", "rna_", "mapped_rna_")) -> dict:
    """Write separate ROI inventories and DAPI trajectory plots for visual review."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix_values = tuple(str(value) for value in prefixes)
    rows = []
    for section_num in sorted(int(value) for value in series.sections):
        section = series.loadSection(section_num)
        for name, contour in section.contours.items():
            roi_type = "other"
            if str(name).startswith(prefix_values[2]):
                roi_type = "mapped_mrna"
            elif str(name).startswith(prefix_values[1]):
                roi_type = "anchor_mrna"
            elif str(name).startswith(prefix_values[0]):
                roi_type = "dapi"
            if roi_type == "other":
                continue
            for trace in contour.traces:
                cx, cy = trace.getCentroid(tform=section.tform)
                rows.append({
                    "section": section_num,
                    "roi_type": roi_type,
                    "name": str(name),
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                })

    csv_path = output / "expert_feedback_roi_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["section", "roi_type", "name", "centroid_x", "centroid_y"]
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output / "expert_feedback_dapi_tracks.png"
    mrna_plot_path = output / "expert_feedback_mrna_rois.png"
    plot_written = False
    mrna_plot_written = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        dapi_rows = [row for row in rows if row["roi_type"] == "dapi"]
        fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
        names = sorted({row["name"] for row in dapi_rows})
        for name in names:
            track = sorted((row for row in dapi_rows if row["name"] == name), key=lambda row: row["section"])
            sections = [row["section"] for row in track]
            ax_x.plot(sections, [row["centroid_x"] for row in track], linewidth=0.8, marker=".", markersize=2)
            ax_y.plot(sections, [row["centroid_y"] for row in track], linewidth=0.8, marker=".", markersize=2)
        ax_x.set_ylabel("Aligned X")
        ax_y.set_ylabel("Aligned Y")
        ax_y.set_xlabel("Section")
        ax_x.set_title(f"DAPI track review ({len(names)} named tracks)")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        plot_written = True

        mrna_rows = [row for row in rows if row["roi_type"] in {"anchor_mrna", "mapped_mrna"}]
        fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
        for roi_type, color, label in (
            ("anchor_mrna", "#e63c3c", "anchor mRNA"),
            ("mapped_mrna", "#28a95b", "mapped mRNA"),
        ):
            selected = [row for row in mrna_rows if row["roi_type"] == roi_type]
            ax_x.scatter(
                [row["section"] for row in selected],
                [row["centroid_x"] for row in selected],
                s=8, alpha=0.7, color=color, label=label,
            )
            ax_y.scatter(
                [row["section"] for row in selected],
                [row["centroid_y"] for row in selected],
                s=8, alpha=0.7, color=color, label=label,
            )
        ax_x.set_ylabel("Aligned X")
        ax_y.set_ylabel("Aligned Y")
        ax_y.set_xlabel("Section")
        ax_x.set_title("mRNA ROI review (shown separately from DAPI)")
        ax_x.legend()
        fig.tight_layout()
        fig.savefig(mrna_plot_path)
        plt.close(fig)
        mrna_plot_written = True
    except Exception:
        pass

    counts = {}
    for row in rows:
        counts[row["roi_type"]] = counts.get(row["roi_type"], 0) + 1
    summary = {
        "inventory_csv": str(csv_path),
        "dapi_track_plot": str(plot_path) if plot_written else "",
        "mrna_roi_plot": str(mrna_plot_path) if mrna_plot_written else "",
        "counts": counts,
        **feedback_summary(series),
    }
    summary_path = output / "expert_feedback_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
