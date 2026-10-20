import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose
from tifffile import imwrite

from pyrecon_connector import (
    get_project_landmarks, get_registration_images, get_registration_image,
    load_registration_landmarks, save_registration_landmarks,
)
from pyrecon_connector.registration_landmarks import pr_to_pixels, pixels_to_pr


class LandmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = {}
        self.points = {}
        for section in (1, 2, 10):
            path = self.root / f"section.{section}.tif"
            imwrite(path, np.arange(32 * 40, dtype=np.uint16).reshape(32, 40))
            self.images[section] = path
            self.points[section] = {"point1": (3., 3.), "point2": (30., 3.), "point3": (3., 25.)}

    def test_pixel_physical_roundtrip_including_borders(self):
        pixels = np.array([[0, 0], [39, 31], [9.5, 20.25]])
        raw = pixels_to_pr(pixels, 0.0025, 32)
        assert_allclose(raw[0], [0, 0.08])
        assert_allclose(pr_to_pixels(raw, 0.0025, 32), pixels, atol=1e-12)

    def test_aligned_physical_conversion_undoes_rotation_translation_and_scale(self):
        pixels = np.array([[0., 0.], [30., 2.], [3.5, 20.]])
        transform = [0.2, -1.3, 20, 1.6, 0.1, -7]
        aligned = pixels_to_pr(pixels, 0.003, 32, transform)
        assert_allclose(pr_to_pixels(aligned, 0.003, 32, transform), pixels, atol=1e-10)
        self.assertFalse(np.allclose(pr_to_pixels(aligned, 0.003, 32), pixels))

    def test_invalid_calibration_and_singular_alignment_rejected(self):
        for size in (0, -1, np.nan):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "calibration"):
                pr_to_pixels([[1, 1]], size, 32)
        with self.assertRaisesRegex(ValueError, "singular"):
            pr_to_pixels([[1, 1]], 1, 32, [1, 2, 0, 2, 4, 0])

    def test_display_does_not_resize_original_pixels(self):
        self.assertEqual(list(get_registration_images(self.root)), [1, 2, 10])
        display = get_registration_image(self.images[1])
        self.assertEqual(display.shape, (32, 40))
        self.assertEqual(display.dtype, np.uint8)

    def test_editor_csv_roundtrip_and_duplicate_protection(self):
        path = self.root / "points.csv"
        save_registration_landmarks(path, self.points, self.images)
        loaded = load_registration_landmarks(path, self.images)
        self.assertEqual(loaded, self.points)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_registration_landmarks(path, self.points, self.images)
        self.assertEqual(original, path.read_bytes())

    def test_save_requires_shared_noncollinear_points_on_all_sections(self):
        self.points[2] = {"different1": (3, 3), "different2": (30, 3), "different3": (3, 25)}
        path = self.root / "invalid.csv"
        with self.assertRaisesRegex(ValueError, "shared landmarks"):
            save_registration_landmarks(path, self.points, self.images)
        self.assertFalse(path.exists())

    def test_out_of_bounds_or_duplicate_points_cannot_be_saved(self):
        self.points[1]["point1"] = (40, 32)
        with self.assertRaisesRegex(ValueError, "out-of-image"):
            save_registration_landmarks(self.root / "invalid.csv", self.points, self.images)
        self.points[1]["point1"] = self.points[1]["point2"]
        with self.assertRaisesRegex(ValueError, "duplicate coordinates"):
            save_registration_landmarks(self.root / "invalid.csv", self.points, self.images)

    def test_load_converts_one_based_bottom_left_indices(self):
        path = self.root / "bottom.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point", "slice", "X", "Y"])
            for section, entries in self.points.items():
                for name, (x, y) in entries.items():
                    writer.writerow([name, section, x + 1, 32 - y])
        self.assertEqual(load_registration_landmarks(
            path, self.images, coordinate_origin="bottom-left", coordinate_base=1,
        ), self.points)

    def test_pair_loader_filters_other_slices_and_keeps_extra_ids(self):
        path = self.root / "points.csv"
        self.points[1]["only1"] = (6, 6)
        save_registration_landmarks(path, self.points, self.images)
        pair_images = {1: self.images[1], 2: self.images[2]}
        loaded = load_registration_landmarks(path, pair_images, pair_sections=(1, 2))
        self.assertEqual(set(loaded), {1, 2})
        self.assertIn("only1", loaded[1])

    def fake_series(self):
        sections = {}
        for internal_number, (section_number, path) in enumerate(self.images.items()):
            mag = 0.002 * (internal_number + 1)
            contours = {}
            for name, xy in self.points[section_number].items():
                raw = tuple(pixels_to_pr([xy], mag, 32)[0])
                contours[name] = [SimpleNamespace(getCentroid=lambda raw=raw: raw)]
            sections[internal_number] = SimpleNamespace(
                src_fp=str(path), mag=mag, contours=contours,
                # Would be wrong to apply this alignment to stored raw centres.
                tform=SimpleNamespace(map=lambda *_args, **_kwargs: (999, 999)),
            )
        return SimpleNamespace(sections=sections, loadSection=sections.__getitem__), sections

    def test_project_conversion_matches_images_not_renumbered_sections(self):
        series, sections = self.fake_series()
        self.assertEqual(list(sections), [0, 1, 2])
        result = get_project_landmarks(series, self.images, prefix="point")
        self.assertEqual(list(result), [1, 2, 10])
        for section, entries in result.items():
            for name, xy in entries.items():
                assert_allclose(xy, self.points[section][name], atol=1e-10)

    def test_project_conversion_refuses_ambiguous_traces_and_wrong_sources(self):
        series, sections = self.fake_series()
        sections[0].contours["point1"] *= 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            get_project_landmarks(series, self.images, prefix="point")
        series, sections = self.fake_series()
        sections[0].src_fp = str(self.root / "other.tif")
        with self.assertRaisesRegex(ValueError, "source TIFF"):
            get_project_landmarks(series, self.images, prefix="point")


if __name__ == "__main__":
    unittest.main()
