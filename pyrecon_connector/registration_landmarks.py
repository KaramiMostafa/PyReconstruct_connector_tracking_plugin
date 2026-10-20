"""Pixel landmark preparation and PyReconstruct calibration conversion.

The editor and registration CSV use zero-based, top-left raw-image pixels.
PyReconstruct's unaligned trace coordinates use micrometres and upward Y.
"""

import csv
import os
from pathlib import Path
import tempfile

import numpy as np
from tifffile import TiffFile

from .registration import _read_plane, _section, read_landmarks
from .registration_core import validate_points
from .serial_registration import discover_images


def get_registration_images(image_series):
    images = discover_images(image_series)
    if len(images) < 2:
        raise ValueError("Choose at least two different section images.")
    _image_shapes(images)
    return images


def _image_shapes(images):
    shapes = {}
    for section, path in images.items():
        with TiffFile(path) as tif:
            shape = tif.series[0].shape
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError(f"Slice {section}: choose a single 2D grayscale TIFF.")
        shapes[section] = shape
    return shapes


def get_registration_image(path):
    """Return an 8-bit display image at the original pixel dimensions."""
    plane = _read_plane(path, "Landmark image")
    stride = max(1, max(plane.shape) // 512)
    low, high = np.percentile(plane[::stride, ::stride], [1, 99])
    if high <= low:
        low, high = float(plane.min()), float(plane.max())
    return np.ascontiguousarray(
        np.uint8(np.clip((plane.astype(np.float64) - low) / max(high - low, 1e-12), 0, 1) * 255)
    )


def _coordinate_array(points, pixel_size, image_height):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite (x, y) pairs.")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("Image calibration must be positive micrometres per pixel.")
    if not np.isfinite(image_height) or image_height < 2:
        raise ValueError("Image height must be at least two pixels.")
    return points.copy()


def _affine(transform):
    values = np.asarray(transform, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Alignment must contain six finite PyReconstruct transform values.")
    matrix = values.reshape(2, 3)
    if np.linalg.matrix_rank(matrix[:, :2]) < 2:
        raise ValueError("Cannot convert coordinates using a singular alignment.")
    return matrix


def pr_to_pixels(points, pixel_size, image_height, transform=None):
    """Convert µm coordinates, optionally undoing a specified forward alignment.

    Uses PyReconstruct's point_2_pix convention: y_px = H - y_um / mag.
    This is deliberately H, not the H-1 used to flip pixel-index coordinates.
    """
    result = _coordinate_array(points, pixel_size, image_height)
    if transform is not None:
        matrix = _affine(transform)
        result = np.linalg.solve(matrix[:, :2], (result - matrix[:, 2]).T).T
    result /= pixel_size
    result[:, 1] = image_height - result[:, 1]
    return result


def pixels_to_pr(points, pixel_size, image_height, transform=None):
    """Reverse pr_to_pixels; optionally return aligned project coordinates."""
    result = _coordinate_array(points, pixel_size, image_height)
    result[:, 1] = image_height - result[:, 1]
    result *= pixel_size
    if transform is not None:
        matrix = _affine(transform)
        result = result @ matrix[:, :2].T + matrix[:, 2]
    return result


def _validate_entries(points, images, require_ready=True, section_order=None):
    shapes = _image_shapes(images)
    if set(points) - set(images):
        raise ValueError("Landmarks contain a slice outside the chosen image set.")
    for section in images:
        entries = points.get(section, {})
        for name, value in entries.items():
            xy = np.asarray(value, dtype=float)
            height, width = shapes[section]
            if (not str(name).strip() or xy.shape != (2,) or not np.isfinite(xy).all()
                    or np.any(xy < 0) or xy[0] > width - 1 or xy[1] > height - 1):
                raise ValueError(f"Slice {section}, landmark {name!r}: invalid ID or out-of-image coordinates.")
        if require_ready:
            validate_points(list(entries.values()), shapes[section], f"Slice {section}")
    if require_ready:
        order = list(section_order) if section_order is not None else sorted(images)
        for previous, current in zip(order, order[1:]):
            common = sorted(points[previous].keys() & points[current].keys())
            for section in (previous, current):
                validate_points(
                    [points[section][name] for name in common], shapes[section],
                    f"Slices {previous}/{current}, shared landmarks on {section}",
                )


def load_registration_landmarks(path, image_series, *, coordinate_origin="top-left",
                                coordinate_base=0, pair_sections=None):
    """Read existing pixel CSVs into the editor's canonical coordinate system."""
    images = get_registration_images(image_series)
    if coordinate_origin not in ("top-left", "bottom-left") or coordinate_base not in (0, 1):
        raise ValueError("Select a pixel origin and zero/one-based indexing.")
    points = {section: {} for section in images}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = [str(h).strip().lower() for h in (reader.fieldnames or [])]
        if {"fixed_x", "fixed_y", "moving_x", "moving_y"}.issubset(headers):
            if pair_sections is None:
                raise ValueError("Paired-column CSVs require single-pair mode.")
            paired = read_landmarks(path, *pair_sections)
            for section, key in zip(pair_sections, ("fixed", "moving")):
                points[section] = dict(zip(paired["point_ids"], paired[key]))
        else:
            name_key = next((key for key in ("point", "point_id", "id", "", "unnamed: 0")
                             if key in headers), None)
            if (name_key is None or len(headers) != len(set(headers))
                    or not {"slice", "x", "y"}.issubset(headers)):
                raise ValueError("Expected point,slice,X,Y in raw pixels. For project traces, use 'Use project landmarks'.")
            for index, raw in enumerate(reader, 2):
                if None in raw:
                    raise ValueError(f"CSV row {index}: too many columns.")
                row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items()}
                if not any(row.values()):
                    continue
                section = _section(row["slice"])
                if section not in images:
                    if pair_sections is not None:
                        continue
                    raise ValueError(f"CSV row {index}: slice {section} has no selected image.")
                name = row[name_key]
                if not name or name in points[section]:
                    raise ValueError(f"CSV row {index}: empty or duplicate landmark ID.")
                try:
                    points[section][name] = (float(row["x"]), float(row["y"]))
                except ValueError as exc:
                    raise ValueError(f"CSV row {index}: invalid coordinates.") from exc
    shapes = _image_shapes(images)
    for section, entries in points.items():
        for name, value in entries.items():
            xy = np.asarray(value, dtype=float) - coordinate_base
            if coordinate_origin == "bottom-left":
                xy[1] = shapes[section][0] - 1 - xy[1]
            entries[name] = tuple(xy)
    _validate_entries(points, images, require_ready=False)
    return points


def save_registration_landmarks(path, points, image_series, *, overwrite=False):
    """Validate correspondence and atomically save a registration-ready CSV."""
    images = get_registration_images(image_series)
    _validate_entries(points, images)
    path = Path(path).resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv filename.")
    if not path.parent.is_dir():
        raise ValueError("Choose an existing output folder.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"CSV already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(["point", "slice", "X", "Y"])
            for section in sorted(images):
                for name in sorted(points[section]):
                    writer.writerow([name, section, *points[section][name]])
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Fails safely if another writer created it.
        return str(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_project_landmarks(series, image_series, *, prefix):
    """Read named trace centroids in raw µm, convert them using actual source TIFFs.

    Match image paths, not section numbers: PR may renumber imported sections.
    Stored traces are unaligned, so the active alignment must NOT be undone again.
    """
    if not prefix or not prefix.strip():
        raise ValueError("Enter a nonempty landmark name prefix, such as lm_.")
    images = get_registration_images(image_series)
    shapes = _image_shapes(images)
    by_path = {Path(path).resolve(): section for section, path in images.items()}
    matched = {}
    for section_number in sorted(series.sections):
        section = series.loadSection(section_number)
        source = Path(section.src_fp).resolve()
        if source not in by_path:
            continue
        target = by_path[source]
        if target in matched:
            raise ValueError(f"The project uses the same source image on multiple sections: {source}")
        entries = {}
        for name, contour in section.contours.items():
            if not str(name).startswith(prefix):
                continue
            traces = list(contour)
            if len(traces) != 1:
                raise ValueError(f"Slice {target}, {name}: use exactly one landmark trace with this name per image.")
            raw = [traces[0].getCentroid()]
            entries[str(name)] = tuple(pr_to_pixels(raw, section.mag, shapes[target][0])[0])
        matched[target] = entries
    missing = sorted(set(images) - set(matched))
    if missing:
        raise ValueError(
            f"No matching project source TIFF for slices {missing}. Select the exact original "
            "TIFFs used by the open project, or mark landmarks directly in the pixel window."
        )
    if not any(matched.values()):
        raise ValueError(f"No traces start with {prefix!r}. Name small landmark traces lm_1, lm_2, etc.")
    _validate_entries(matched, images, require_ready=False)
    return matched
