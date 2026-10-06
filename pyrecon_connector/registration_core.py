"""Landmark-driven 2D EM registration, independent of PyReconstruct and Qt.

Like mainEMToEM_Ostroff.py, fit a thin-plate spline from fixed landmarks
to moving landmarks for inverse image resampling, and a separate forward fit
for landmark diagnostics. Coordinates are raw-image (x, y) pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates


class RegistrationCancelled(Exception):
    """The caller requested cancellation between processing blocks."""


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RegistrationCancelled("Registration cancelled.")


def validate_points(points, shape, label):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{label}: at least 3 matching landmarks are required.")
    if len(points) > 1000:
        raise ValueError("Use at most 1000 landmarks for this global TPS fit.")
    if not np.isfinite(points).all():
        raise ValueError(f"{label}: landmark coordinates must be finite.")
    height, width = shape
    if ((points < 0).any() or (points[:, 0] > width - 1).any()
            or (points[:, 1] > height - 1).any()):
        raise ValueError(
            f"{label}: landmarks fall outside the {width} × {height} image. "
            "Check slice IDs, pixel coordinates, origin, and zero/one-based indexing."
        )
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{label}: different landmark IDs have duplicate coordinates.")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered / max(np.abs(centered).max(), 1.0)) < 2:
        raise ValueError(f"{label}: landmarks must not all lie on one line.")
    return points


def fit_transforms(fixed_points, moving_points):
    """Return inverse resampling and forward landmark TPS fits."""
    try:
        inverse = RBFInterpolator(
            fixed_points, moving_points, kernel="thin_plate_spline", degree=1,
            smoothing=0,
        )
        forward = RBFInterpolator(
            moving_points, fixed_points, kernel="thin_plate_spline", degree=1,
            smoothing=0,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"Cannot fit TPS; check landmark geometry: {exc}") from exc
    return inverse, forward


def warp_pair(moving_image, moving_mask, fixed_shape, inverse, *,
              progress=None, cancelled=None, block_rows=64):
    """Warp image and labels with one shared sampling grid, in bounded blocks.

    Image: bilinear interpolation, retaining its dtype. Mask: nearest pixel
    indexing, retaining even uint64 label IDs exactly. Invalid pixels become 0.
    """
    if block_rows < 1:
        raise ValueError("block_rows must be positive.")
    height, width = fixed_shape
    registered = np.empty(fixed_shape, dtype=moving_image.dtype)
    labels = None if moving_mask is None else np.empty(fixed_shape, dtype=moving_mask.dtype)
    coverage = np.empty(fixed_shape, dtype=np.uint8)
    for first in range(0, height, block_rows):
        check_cancelled(cancelled)
        last = min(first + block_rows, height)
        yy, xx = np.mgrid[first:last, :width]
        source = inverse(np.column_stack((xx.ravel(), yy.ravel())))
        if not np.isfinite(source).all():
            raise ValueError("TPS produced non-finite coordinates; check the landmarks.")
        # Remove round-off at exact borders without clipping real displacement.
        for axis, size in ((0, moving_image.shape[1]), (1, moving_image.shape[0])):
            for edge in (0, size - 1):
                near = np.abs(source[:, axis] - edge) < 1e-7
                source[near, axis] = edge
        valid = ((source[:, 0] >= 0) & (source[:, 0] <= moving_image.shape[1] - 1)
                 & (source[:, 1] >= 0) & (source[:, 1] <= moving_image.shape[0] - 1))
        values = map_coordinates(
            moving_image, source[:, ::-1].T, order=1, mode="constant",
            cval=0, prefilter=False,
        )
        registered[first:last] = values.reshape(last - first, width)
        coverage[first:last] = valid.reshape(last - first, width)
        if labels is not None:
            sampled = np.zeros(len(source), dtype=moving_mask.dtype)
            nearest = np.floor(source[valid] + 0.5).astype(np.int64)
            sampled[valid] = moving_mask[nearest[:, 1], nearest[:, 0]]
            labels[first:last] = sampled.reshape(last - first, width)
        if progress is not None:
            progress(int(10 + 75 * last / height), "Warping EM image and mask")
    return registered, labels, coverage
