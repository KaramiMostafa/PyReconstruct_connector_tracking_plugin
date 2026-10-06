"""Serial landmark registration in the first section's pixel coordinate system.

Each original section is fitted to the preceding section's REGISTERED landmarks.
Both inverse image sampling and forward propagation follow mainEMToEM_Ostroff.py.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import numpy as np
from tifffile import imwrite

from .registration import _read_plane, _section, _write_preview
from .registration_core import (
    RegistrationCancelled, check_cancelled, fit_transforms, validate_points, warp_pair,
)


def discover_images(image_series):
    """Accept a folder of numbered TIFF planes or an explicit slice -> path dict."""
    if isinstance(image_series, dict):
        pairs = list(image_series.items())
    else:
        folder = Path(image_series)
        if not folder.is_dir():
            raise ValueError("Choose an existing image-series folder.")
        pairs = []
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
                match = re.search(r"(\d+)$", path.stem)
                if match is None:
                    raise ValueError(f"No trailing slice number in TIFF filename: {path.name}")
                pairs.append((match[1], path))
    images = {}
    for raw_section, path in pairs:
        section = _section(raw_section)
        if section in images:
            raise ValueError(f"Slice {section}: more than one image has this slice number.")
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Slice {section}: image cannot be loaded: {path}")
        images[section] = path
    if not images:
        raise ValueError("No numbered TIFF images found.")
    if len(set(images.values())) != len(images):
        raise ValueError("Each slice must refer to a different image file.")
    return dict(sorted(images.items()))


def read_serial_landmarks(path, sections):
    """Retain all IDs per section, not only those shared with the previous one."""
    points = {section: {} for section in sections}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = [str(h).strip().lower() for h in (reader.fieldnames or [])]
        id_key = next((key for key in ("point", "point_id", "id", "", "unnamed: 0")
                       if key in headers), None)
        if (id_key is None or len(headers) != len(set(headers))
                or not {"slice", "x", "y"}.issubset(headers)):
            raise ValueError("Serial CSV requires unique point,slice,x,y columns (unnamed point column accepted).")
        for row_number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV row {row_number}: too many columns.")
            row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items()}
            if not any(row.values()):
                continue
            section = _section(row["slice"])
            if section not in points:
                raise ValueError(f"CSV row {row_number}: slice {section} has no corresponding image.")
            name = row[id_key]
            if not name or name in points[section]:
                raise ValueError(f"Slice {section}, CSV row {row_number}: empty or duplicate landmark ID {name!r}.")
            try:
                xy = np.array([float(row["x"]), float(row["y"])])
            except ValueError as exc:
                raise ValueError(f"Slice {section}, CSV row {row_number}: invalid coordinates.") from exc
            if not np.isfinite(xy).all():
                raise ValueError(f"Slice {section}, CSV row {row_number}: coordinates must be finite.")
            points[section][name] = xy
    for section, values in points.items():
        if not values:
            raise ValueError(f"Slice {section}: no landmarks in the CSV.")
    return points


def run_serial_registration(*, image_series, landmarks_csv, output_dir, method="tps",
                            coordinate_origin="top-left", coordinate_base=0,
                            progress=None, cancelled=None, block_rows=64):
    """Publish a complete registered stack, all transformed landmarks, and QC.

    TIFFs are ordered by numeric slice ID. The lowest ID is the unchanged
    reference. Raw points must lie inside their source image; propagated points
    may lie outside the output canvas and are retained for the next pair.
    A failed/cancelled run never publishes a partial stack or alters its inputs.
    """
    if method not in ("tps", "affine"):
        raise ValueError(f"Unknown registration method: {method!r}")
    if coordinate_origin not in ("top-left", "bottom-left") or coordinate_base not in (0, 1):
        raise ValueError("Select top-left/bottom-left origin and zero/one-based pixel coordinates.")
    if not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("block_rows must be a positive integer.")
    parent = Path(output_dir).resolve()
    if not parent.is_dir():
        raise ValueError("Choose an existing output parent folder.")
    check_cancelled(cancelled)
    images = discover_images(image_series)
    raw = read_serial_landmarks(landmarks_csv, images)
    sections = list(images)
    originals = {}
    # Validate every input before any warping; hold only one image at a time.
    for index, section in enumerate(sections):
        check_cancelled(cancelled)
        try:
            plane = _read_plane(images[section], f"Slice {section}")
            names = sorted(raw[section])
            xy = np.array([raw[section][name] for name in names]) - coordinate_base
            if coordinate_origin == "bottom-left":
                xy[:, 1] = plane.shape[0] - 1 - xy[:, 1]
            validate_points(xy, plane.shape, f"Slice {section}")
            originals[section] = dict(zip(names, xy))
        except Exception as exc:
            raise ValueError(f"Slice {section}: input validation failed: {exc}") from exc
        if progress:
            progress(int(10 * (index + 1) / len(sections)), f"Validated slice {section}")
    del plane
    for previous, current in zip(sections, sections[1:]):
        common = sorted(originals[previous].keys() & originals[current].keys())
        for section in (previous, current):
            xy = np.array([originals[section][name] for name in common]).reshape(-1, 2)
            validate_points(xy, None, f"Slice {current} -> {previous}, shared points on {section}")

    first = sections[0]
    reference = _read_plane(images[first], f"Slice {first}")
    registered_points = {first: originals[first]}
    report = {
        "status": "complete", "method": method, "reference_section": first,
        "section_order": sections, "output_shape": list(reference.shape),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_coordinate_origin": coordinate_origin, "input_coordinate_base": coordinate_base,
        "output_coordinates": "zero-based top-left pixels in the first section's coordinate system",
        "inputs": {str(s): str(p) for s, p in images.items()},
        "landmarks_csv": str(Path(landmarks_csv).resolve()), "pairs": [],
        "image_interpolation": "bilinear, constant zero padding, original dtype",
        "transform_settings": ({"kernel": "thin_plate_spline", "degree": 1, "smoothing": 0}
                               if method == "tps" else {"fit": "least-squares affine"}),
        "warnings": ["Landmark fitting residuals are not independent accuracy measurements.",
                     "Serial registration may accumulate drift; review the stack and areas far from landmarks."],
    }
    destination = parent / ("serial_registration_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    width = max(4, len(str(max(sections))))
    outputs = {"registered_landmarks": "registered_landmarks.csv", "report": "registration.json"}
    with tempfile.TemporaryDirectory(prefix=".serial_registration_", dir=parent) as temporary:
        staging = Path(temporary)
        for folder in ("images", "coverage", "previews"):
            (staging / folder).mkdir()
        def image_name(section):
            return f"images/section_{section:0{width}d}.tif"
        shutil.copyfile(images[first], staging / image_name(first))
        previous_image = reference
        for index, (previous, current) in enumerate(zip(sections, sections[1:])):
            check_cancelled(cancelled)
            try:
                moving = _read_plane(images[current], f"Slice {current}")
                common = sorted(registered_points[previous].keys() & originals[current].keys())
                fixed = np.array([registered_points[previous][name] for name in common])
                source = np.array([originals[current][name] for name in common])
                validate_points(fixed, None, f"Registered slice {previous}")
                inverse, forward = fit_transforms(fixed, source, method=method)
                def pair_progress(value, message):
                    if progress:
                        progress(int(10 + 85 * (index + value / 100) / max(1, len(sections) - 1)),
                                 f"Slice {current} -> registered slice {previous}: {message}")
                registered, _, coverage = warp_pair(
                    moving, None, reference.shape, inverse, progress=pair_progress,
                    cancelled=cancelled, block_rows=block_rows, prototype_sampling=True,
                )
                names = sorted(originals[current])
                transformed = forward(np.array([originals[current][name] for name in names]))
                if not np.isfinite(transformed).all():
                    raise ValueError("Transformation produced non-finite registered landmarks.")
                registered_points[current] = dict(zip(names, transformed))
                residual = np.linalg.norm(forward(source) - fixed, axis=1)
                inverse_residual = np.linalg.norm(inverse(fixed) - source, axis=1)
                pair = {
                    "slice": current, "previous_slice": previous, "shared_point_ids": common,
                    "input_shape": list(moving.shape), "image_dtype": str(moving.dtype),
                    "previous_only_ids": sorted(registered_points[previous].keys() - originals[current].keys()),
                    "current_only_ids": sorted(originals[current].keys() - registered_points[previous].keys()),
                    "registered_reference_points": fixed.tolist(), "original_moving_points": source.tolist(),
                    "landmark_fit_rmse_pixels": float(np.sqrt(np.mean(residual ** 2))),
                    "inverse_fit_rmse_pixels": float(np.sqrt(np.mean(inverse_residual ** 2))),
                    "outside_moving_fraction": float(1 - coverage.mean()),
                    "image": image_name(current),
                    "preview": f"previews/section_{current:0{width}d}.png",
                }
                check_cancelled(cancelled)
                imwrite(staging / image_name(current), registered, photometric="minisblack", metadata={"axes": "YX"})
                imwrite(staging / f"coverage/section_{current:0{width}d}.tif", coverage, photometric="minisblack")
                _write_preview(staging / pair["preview"], previous_image, moving, registered,
                               fixed, source, None, None)
                report["pairs"].append(pair)
                previous_image = registered
            except RegistrationCancelled:
                raise
            except Exception as exc:
                raise ValueError(f"Slice {current} -> registered slice {previous} failed: {exc}") from exc
        with (staging / outputs["registered_landmarks"]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point", "slice", "x", "y", "original_x", "original_y"])
            for section in sections:
                for name in sorted(registered_points[section]):
                    writer.writerow([name, section, *registered_points[section][name], *originals[section][name]])
        if report["pairs"]:
            outputs["preview"] = report["pairs"][-1]["preview"]
        outputs["images"] = "images"
        report["outputs"] = outputs
        (staging / outputs["report"]).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        check_cancelled(cancelled)
        staging.rename(destination)
    if progress:
        progress(100, f"Saved {len(sections)} registered sections")
    return {"output_dir": str(destination), "report": report,
            "registered_landmarks": registered_points,
            "files": {key: str(destination / relative) for key, relative in outputs.items()}}
