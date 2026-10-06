"""File adapter for paired EM images, masks, and named landmark CSVs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
from PIL import Image, ImageDraw
import scipy
from scipy.spatial import ConvexHull
from tifffile import imread, imwrite

from .registration_core import check_cancelled, fit_transforms, validate_points, warp_pair


def _section(value):
    try:
        number = float(value)
        if not np.isfinite(number) or not number.is_integer() or number < 0:
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid nonnegative slice number: {value!r}") from exc


def read_landmarks(path, fixed_section, moving_section):
    """Match IDs (never row order), accepting long or paired-column CSVs.

    Long: point,slice,X,Y; the original unnamed first column is accepted.
    Paired: point,fixed_x,fixed_y,moving_x,moving_y.
    """
    fixed_section, moving_section = _section(fixed_section), _section(moving_section)
    if fixed_section == moving_section:
        raise ValueError("Fixed and moving slice numbers must be different.")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = [str(h).strip().lower() for h in (reader.fieldnames or [])]
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Landmark CSV needs unique column headers.")
        id_key = next((key for key in ("point", "point_id", "id", "", "unnamed: 0")
                       if key in headers), None)
        wide = {"fixed_x", "fixed_y", "moving_x", "moving_y"}.issubset(headers)
        if id_key is None or not (wide or {"slice", "x", "y"}.issubset(headers)):
            raise ValueError(
                "CSV columns must be point,slice,X,Y (an unnamed point column is OK) "
                "or point,fixed_x,fixed_y,moving_x,moving_y."
            )
        fixed, moving = {}, {}
        for row_number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV row {row_number} has too many columns.")
            row = {str(key).strip().lower(): (value or "").strip() for key, value in raw.items()}
            if not any(row.values()):
                continue
            if not wide:
                section = _section(row["slice"])
                if section not in (fixed_section, moving_section):
                    continue
            name = row[id_key]
            if not name:
                raise ValueError(f"CSV row {row_number}: landmark ID is empty.")
            destinations = [(fixed, "fixed_x", "fixed_y"), (moving, "moving_x", "moving_y")] if wide else [
                (fixed if section == fixed_section else moving, "x", "y")
            ]
            for collection, x_key, y_key in destinations:
                if name in collection:
                    raise ValueError(f"CSV row {row_number}: duplicate landmark ID {name!r} on one slice.")
                try:
                    xy = (float(row[x_key]), float(row[y_key]))
                except ValueError as exc:
                    raise ValueError(f"CSV row {row_number}: invalid landmark coordinates.") from exc
                if not np.isfinite(xy).all():
                    raise ValueError(f"CSV row {row_number}: coordinates must be finite.")
                collection[name] = xy
    common = sorted(fixed.keys() & moving.keys())
    if len(common) < 3:
        raise ValueError(f"Found {len(common)} shared landmark IDs; at least 3 are required.")
    return {
        "point_ids": common,
        "fixed": np.asarray([fixed[name] for name in common]),
        "moving": np.asarray([moving[name] for name in common]),
        "fixed_only": sorted(fixed.keys() - moving.keys()),
        "moving_only": sorted(moving.keys() - fixed.keys()),
    }


def _read_plane(path, label, mask=False):
    array = imread(path)
    if array.ndim != 2:
        raise ValueError(f"{label}: select a single 2D grayscale TIFF, not RGB or a stack (shape {array.shape}).")
    if min(array.shape) < 2 or array.dtype.kind not in ("buif" if mask else "uif"):
        raise ValueError(f"{label}: unsupported image shape or dtype ({array.shape}, {array.dtype}).")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: image contains NaN or infinite values.")
    if mask and (np.any(array < 0) or (array.dtype.kind == "f" and np.any(array != np.floor(array)))):
        raise ValueError(f"{label}: masks must contain nonnegative integer labels; 0 is background.")
    return array


def _gray(array, shape):
    # Only the preview is reduced/scaled; exported data retain their intensity range.
    stride = max(1, int(max(array.shape) / 512))
    low, high = np.percentile(array[::stride, ::stride], [1, 99])
    sampled = np.asarray(Image.fromarray(array.astype(np.float32)).resize(shape))
    return np.uint8(np.clip((sampled - low) / max(high - low, 1e-8), 0, 1) * 255)


def _write_preview(path, fixed, moving, registered, fixed_points, moving_points,
                   fixed_mask, registered_mask):
    height, width = fixed.shape
    scale = min(1.0, 650 / width, 600 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    ref = _gray(fixed, size)
    # The before panel uses the same pixel canvas, not independent image resizing.
    before = np.zeros(fixed.shape, dtype=moving.dtype)
    hh, ww = min(height, moving.shape[0]), min(width, moving.shape[1])
    before[:hh, :ww] = moving[:hh, :ww]
    panels = []
    for title, other, points in (
        ("Before: fixed magenta / moving green", before, moving_points),
        ("After: fixed magenta / registered green", registered, None),
    ):
        panel = Image.fromarray(np.stack((ref, _gray(other, size), ref), axis=-1))
        draw = ImageDraw.Draw(panel)
        for index, (x, y) in enumerate(fixed_points):
            x, y = x * scale, y * scale
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="yellow", width=2)
            draw.text((x + 5, y), str(index + 1), fill="yellow")
        if points is not None:
            for x, y in points:
                x, y = x * scale, y * scale
                draw.line((x - 4, y, x + 4, y), fill="cyan", width=2)
                draw.line((x, y - 4, x, y + 4), fill="cyan", width=2)
        panels.append((title, panel))
    if fixed_mask is not None:
        a = np.asarray(Image.fromarray(np.uint8(fixed_mask > 0) * 255).resize(size, Image.Resampling.NEAREST))
        b = np.asarray(Image.fromarray(np.uint8(registered_mask > 0) * 255).resize(size, Image.Resampling.NEAREST))
        panels.append(("Masks: fixed magenta / registered green", Image.fromarray(np.stack((a, b, a), axis=-1))))
    canvas = Image.new("RGB", (size[0] * len(panels), size[1] + 28), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        left = index * size[0]
        canvas.paste(panel, (left, 28))
        draw.text((left + 5, 7), title, fill="white")
    canvas.save(path)


def run_em_registration(*, fixed_image, moving_image, landmarks_csv, output_dir,
                        fixed_section=1, moving_section=2, fixed_mask=None,
                        moving_mask=None, coordinate_origin="top-left", coordinate_base=0,
                        progress=None, cancelled=None):
    """Export a registered EM pair into a new run folder; never modify a series.

    Masks are optional as a pair and are transformed label maps, not landmarks
    or image-fitting weights. Nonlinear output must be opened as a new dataset.
    """
    check_cancelled(cancelled)
    if coordinate_origin not in ("top-left", "bottom-left") or coordinate_base not in (0, 1):
        raise ValueError("Select top-left/bottom-left origin and zero/one-based pixel coordinates.")
    fixed_section, moving_section = _section(fixed_section), _section(moving_section)
    if bool(fixed_mask) != bool(moving_mask):
        raise ValueError("Select both fixed and moving masks, or leave both blank.")
    paths = {"fixed_image": fixed_image, "moving_image": moving_image, "landmarks_csv": landmarks_csv}
    if fixed_mask:
        paths.update(fixed_mask=fixed_mask, moving_mask=moving_mask)
    for key, path in paths.items():
        if not path or not Path(path).is_file():
            raise ValueError(f"Choose an existing {key.replace('_', ' ')} file.")
    paths = {key: Path(path).resolve() for key, path in paths.items()}
    if paths["fixed_image"] == paths["moving_image"]:
        raise ValueError("Choose different fixed and moving image files.")
    if not output_dir or not Path(output_dir).is_dir():
        raise ValueError("Choose an existing output folder; a new registration run will be created inside it.")
    if progress:
        progress(2, "Reading EM images and landmarks")
    fixed = _read_plane(paths["fixed_image"], "Fixed EM")
    moving = _read_plane(paths["moving_image"], "Moving EM")
    a_mask = _read_plane(paths["fixed_mask"], "Fixed mask", True) if fixed_mask else None
    b_mask = _read_plane(paths["moving_mask"], "Moving mask", True) if moving_mask else None
    for mask, image, label in ((a_mask, fixed, "Fixed"), (b_mask, moving, "Moving")):
        if mask is not None and mask.shape != image.shape:
            raise ValueError(f"{label} mask shape {mask.shape} does not match its EM image {image.shape}.")
    landmarks = read_landmarks(paths["landmarks_csv"], fixed_section, moving_section)
    for key, shape in (("fixed", fixed.shape), ("moving", moving.shape)):
        landmarks[key] -= coordinate_base
        if coordinate_origin == "bottom-left":
            landmarks[key][:, 1] = shape[0] - 1 - landmarks[key][:, 1]
        landmarks[key] = validate_points(landmarks[key], shape, key.capitalize())
    check_cancelled(cancelled)
    inverse, forward = fit_transforms(landmarks["fixed"], landmarks["moving"])
    registered, registered_mask, coverage = warp_pair(
        moving, b_mask, fixed.shape, inverse, progress=progress, cancelled=cancelled,
    )
    predicted = forward(landmarks["moving"])
    error = np.linalg.norm(predicted - landmarks["fixed"], axis=1)
    inverse_error = np.linalg.norm(inverse(landmarks["fixed"]) - landmarks["moving"], axis=1)
    notices = [
        "TPS deforms image and mask geometry. It is not an affine PyReconstruct alignment.",
        "Landmark residuals measure fit to supplied points, not independent registration accuracy.",
        "Review areas far from landmarks; TPS extrapolation is unconstrained there.",
    ]
    if landmarks["fixed_only"] or landmarks["moving_only"]:
        notices.append("Landmark IDs present on only one selected slice were excluded.")
    if a_mask is None:
        notices.append("No masks supplied; this run registers images only.")
    mask_dice = None
    if a_mask is not None:
        foreground_a, foreground_b = a_mask > 0, registered_mask > 0
        denominator = int(foreground_a.sum()) + int(foreground_b.sum())
        mask_dice = float(2 * np.count_nonzero(foreground_a & foreground_b) / denominator) if denominator else None
    outside_fraction = float(1 - coverage.mean())
    if outside_fraction:
        notices.append(f"{outside_fraction:.1%} of the fixed canvas has no moving-image coverage (filled with 0).")
    report = {
        "status": "complete", "method": "landmark_thin_plate_spline",
        "tps_parameters": {"kernel": "thin_plate_spline", "degree": 1, "smoothing": 0},
        "scipy_version": scipy.__version__, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: str(value) for key, value in paths.items()},
        "fixed_section": fixed_section, "moving_section": moving_section,
        "input_coordinate_origin": coordinate_origin, "input_coordinate_base": coordinate_base,
        "output_coordinates": "zero-based top-left raw-image pixels on the fixed canvas",
        "fixed_shape": list(fixed.shape), "moving_shape": list(moving.shape),
        "image_dtype": str(moving.dtype),
        "mask_dtype": str(b_mask.dtype) if b_mask is not None else None,
        "image_interpolation": "bilinear", "mask_interpolation": "nearest-neighbor",
        "mask_role": "label maps warped with the image; not fitting weights",
        "point_ids": landmarks["point_ids"],
        "fixed_points": landmarks["fixed"].tolist(), "moving_points": landmarks["moving"].tolist(),
        "fixed_only_ids": landmarks["fixed_only"], "moving_only_ids": landmarks["moving_only"],
        "landmark_fit_rmse_pixels": float(np.sqrt(np.mean(error ** 2))),
        "inverse_fit_rmse_pixels": float(np.sqrt(np.mean(inverse_error ** 2))),
        "fixed_landmark_hull_area_fraction": float(ConvexHull(landmarks["fixed"]).volume / fixed.size),
        "outside_moving_fraction": outside_fraction,
        "foreground_mask_dice": mask_dice, "warnings": notices,
    }
    parent = Path(output_dir).resolve()
    destination = parent / ("em_registration_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    outputs = {
        "fixed_image": f"images/section_{fixed_section:04d}.tif",
        "registered_image": f"images/section_{moving_section:04d}.tif",
        "coverage": "coverage.tif", "preview": "preview.png",
        "landmark_report": "landmark_residuals.csv", "report": "registration.json",
    }
    if a_mask is not None:
        outputs.update(fixed_mask=f"masks/section_{fixed_section:04d}.tif",
                       registered_mask=f"masks/section_{moving_section:04d}.tif")
    report["outputs"] = outputs
    # A failed/cancelled run removes only its own temporary directory.
    with tempfile.TemporaryDirectory(prefix=".em_registration_", dir=parent) as temporary:
        staging = Path(temporary)
        (staging / "images").mkdir()
        check_cancelled(cancelled)
        if progress:
            progress(88, "Saving registered TIFFs and review preview")
        shutil.copyfile(paths["fixed_image"], staging / outputs["fixed_image"])
        imwrite(staging / outputs["registered_image"], registered, photometric="minisblack", metadata={"axes": "YX"})
        imwrite(staging / outputs["coverage"], coverage, photometric="minisblack")
        if a_mask is not None:
            (staging / "masks").mkdir()
            shutil.copyfile(paths["fixed_mask"], staging / outputs["fixed_mask"])
            imwrite(staging / outputs["registered_mask"], registered_mask, photometric="minisblack")
        _write_preview(staging / outputs["preview"], fixed, moving, registered,
                       landmarks["fixed"], landmarks["moving"], a_mask, registered_mask)
        with (staging / outputs["landmark_report"]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point", "fixed_x", "fixed_y", "moving_x", "moving_y",
                             "registered_x", "registered_y", "fit_error_pixels", "inverse_fit_error_pixels"])
            for index, name in enumerate(landmarks["point_ids"]):
                writer.writerow([name, *landmarks["fixed"][index], *landmarks["moving"][index],
                                 *predicted[index], error[index], inverse_error[index]])
        (staging / outputs["report"]).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        check_cancelled(cancelled)
        staging.rename(destination)
    if progress:
        progress(100, "Registration saved")
    return {"output_dir": str(destination), "report": report,
            "files": {key: str(destination / relative) for key, relative in outputs.items()}}
