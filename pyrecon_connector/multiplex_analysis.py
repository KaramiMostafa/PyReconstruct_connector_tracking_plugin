from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .multiplex_mapping import _polygon_centroid, _section_from_path, _trace_records


def _match_centroids(predicted, expert, max_distance: float):
    if not predicted or not expert:
        return [], list(range(len(predicted))), list(range(len(expert)))
    from scipy.optimize import linear_sum_assignment

    distances = np.asarray([
        [np.linalg.norm(p.centroid - e.centroid) for e in expert] for p in predicted
    ], dtype=float)
    rows, columns = linear_sum_assignment(distances)
    matches = [
        (int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns)
        if distances[row, column] <= float(max_distance)
    ]
    matched_predicted = {row for row, _, _ in matches}
    matched_expert = {column for _, column, _ in matches}
    return (
        matches,
        [index for index in range(len(predicted)) if index not in matched_predicted],
        [index for index in range(len(expert)) if index not in matched_expert],
    )


def validate_mapped_rna(
    series,
    output_dir: str,
    predicted_prefix: str = "mapped_rna_",
    predicted_group: str = "multiplex_mapped_rna",
    expert_prefix: str = "expert_rna_",
    expert_group: str = "",
    max_centroid_distance: float = 15.0,
) -> dict:
    """Compare mapped RNA ROIs with independent expert ROIs without treating them as training data."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for section_num in sorted(int(value) for value in series.sections):
        predicted = _trace_records(series, section_num, predicted_prefix, predicted_group)
        expert = _trace_records(series, section_num, expert_prefix, expert_group)
        if not predicted and not expert:
            continue
        matches, unmatched_predicted, unmatched_expert = _match_centroids(
            predicted, expert, max_centroid_distance
        )
        totals["tp"] += len(matches)
        totals["fp"] += len(unmatched_predicted)
        totals["fn"] += len(unmatched_expert)
        for p_index, e_index, distance in matches:
            rows.append({
                "section": section_num,
                "predicted_name": predicted[p_index].name,
                "expert_name": expert[e_index].name,
                "centroid_distance": distance,
                "result": "matched",
            })
        for p_index in unmatched_predicted:
            rows.append({
                "section": section_num,
                "predicted_name": predicted[p_index].name,
                "expert_name": "",
                "centroid_distance": "",
                "result": "false_positive",
            })
        for e_index in unmatched_expert:
            rows.append({
                "section": section_num,
                "predicted_name": "",
                "expert_name": expert[e_index].name,
                "centroid_distance": "",
                "result": "missed_expert_roi",
            })

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    csv_path = output / "mapped_rna_expert_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "predicted_name", "expert_name", "centroid_distance", "result"],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "max_centroid_distance": float(max_centroid_distance),
        "csv": str(csv_path),
        "warning": "These metrics are meaningful only if the expert ROIs are independent and complete.",
    }
    summary_path = output / "mapped_rna_expert_validation_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _find_section_tiffs(folder: str) -> dict[int, Path]:
    found = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        section = _section_from_path(str(path))
        if section is not None:
            found.setdefault(int(section), path)
    return found


def _select_channel(image: np.ndarray, channel: int) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 2:
        return image.astype(float)
    if image.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D TIFF, found shape {image.shape}.")
    # Prefer a short channel axis; otherwise treat the last axis as channels.
    channel_axis = next((axis for axis, size in enumerate(image.shape) if size <= 8), 2)
    if not 0 <= int(channel) < image.shape[channel_axis]:
        raise ValueError(f"Channel {channel} is outside TIFF shape {image.shape}.")
    return np.take(image, int(channel), axis=channel_axis).astype(float)


def measure_antibody_intensity(
    series,
    antibody_tiff_folder: str,
    output_dir: str,
    roi_prefix: str = "mapped_rna_",
    roi_group: str = "multiplex_mapped_rna",
    channel: int = 0,
    threshold_method: str = "median + 3 MAD",
) -> dict:
    """Measure antibody signal inside mapped RNA ROIs and write reviewable outputs."""
    import tifffile
    from skimage.draw import polygon

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_paths = _find_section_tiffs(antibody_tiff_folder)
    rows = []
    skipped_sections = []
    overlays = output / "antibody_overlays"
    overlays.mkdir(exist_ok=True)

    for section_num in sorted(int(value) for value in series.sections):
        rois = _trace_records(series, section_num, roi_prefix, roi_group)
        if not rois:
            continue
        image_path = image_paths.get(section_num)
        if image_path is None:
            skipped_sections.append(section_num)
            continue
        image = _select_channel(tifffile.imread(image_path), channel)
        section = series.loadSection(section_num)
        height, width = image.shape
        section_rows = []
        for roi in rois:
            local = roi.local_points
            px = local[:, 0] / float(section.mag)
            py = height - local[:, 1] / float(section.mag)
            yy, xx = polygon(py, px, image.shape)
            values = image[yy, xx]
            if not len(values):
                continue
            centroid = _polygon_centroid(roi.aligned_points)
            row = {
                "section": section_num,
                "roi_name": roi.name,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "pixel_count": int(len(values)),
                "mean_intensity": float(np.mean(values)),
                "median_intensity": float(np.median(values)),
                "max_intensity": float(np.max(values)),
            }
            rows.append(row)
            section_rows.append((row, px, py))

        if section_rows:
            try:
                import matplotlib
                matplotlib.use("Agg")
                from matplotlib import pyplot as plt
                from matplotlib.patches import Polygon

                fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
                lo, hi = np.percentile(image, [1, 99.5])
                ax.imshow(image, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9))
                for row, px, py in section_rows:
                    ax.add_patch(Polygon(np.column_stack((px, py)), fill=False, edgecolor="#ff4040", linewidth=0.8))
                ax.set_title(f"Section {section_num}: antibody with mapped mRNA ROIs")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(overlays / f"section_{section_num:03d}_antibody_overlay.png")
                plt.close(fig)
            except Exception:
                pass

    intensities = np.asarray([row["mean_intensity"] for row in rows], dtype=float)
    threshold_method = str(threshold_method)
    if len(intensities):
        if threshold_method == "95th percentile":
            threshold = float(np.percentile(intensities, 95))
        elif threshold_method == "median + 2 MAD":
            median = float(np.median(intensities))
            threshold = median + 2.0 * 1.4826 * float(np.median(np.abs(intensities - median)))
        else:
            median = float(np.median(intensities))
            threshold = median + 3.0 * 1.4826 * float(np.median(np.abs(intensities - median)))
    else:
        threshold = math.nan
    for row in rows:
        row["threshold"] = threshold
        row["positive"] = bool(row["mean_intensity"] > threshold) if math.isfinite(threshold) else False

    fields = [
        "section", "roi_name", "centroid_x", "centroid_y", "pixel_count",
        "mean_intensity", "median_intensity", "max_intensity", "threshold", "positive",
    ]
    csv_path = output / "antibody_intensity_by_mapped_rna_roi.csv"
    positive_path = output / "antibody_positive_cells.csv"
    for path, selected in (
        (csv_path, rows),
        (positive_path, [row for row in rows if row["positive"]]),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    scatter_path = output / "antibody_intensity_scatter.png"
    scatter_written = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        x = np.arange(len(rows))
        colors = ["#d62728" if row["positive"] else "#4c78a8" for row in rows]
        ax.scatter(x, intensities, c=colors, s=14)
        if math.isfinite(threshold):
            ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1, label=f"threshold {threshold:.3g}")
            ax.legend()
        ax.set_xlabel("Mapped mRNA ROI")
        ax.set_ylabel("Mean antibody intensity")
        ax.set_title("Antibody intensity inside mapped mRNA ROIs")
        fig.tight_layout()
        fig.savefig(scatter_path)
        plt.close(fig)
        scatter_written = True
    except Exception:
        pass

    summary = {
        "measured_rois": len(rows),
        "positive_rois": sum(bool(row["positive"]) for row in rows),
        "threshold": threshold if math.isfinite(threshold) else None,
        "threshold_method": threshold_method,
        "missing_tiff_sections": sorted(set(skipped_sections)),
        "measurements_csv": str(csv_path),
        "positive_csv": str(positive_path),
        "scatter_plot": str(scatter_path) if scatter_written else "",
        "overlay_dir": str(overlays),
        "warning": (
            "Automatic positivity is exploratory. Confirm the threshold with controls "
            "or expert labels before biological interpretation."
        ),
    }
    summary_path = output / "antibody_intensity_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
