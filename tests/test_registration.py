import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from tifffile import imread, imwrite

from pyrecon_connector import run_em_registration
from pyrecon_connector.registration import read_landmarks
from pyrecon_connector.registration_core import RegistrationCancelled, fit_transforms, warp_pair


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.points = np.array([[4., 4.], [20., 4.], [4., 20.], [20., 20.]])
        self.fixed = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        self.moving = np.zeros_like(self.fixed)
        self.moving[2:, 3:] = self.fixed[:-2, :-3]
        self.fixed_mask = np.zeros((32, 32), dtype=np.uint64)
        self.fixed_mask[8:12, 8:12] = 2 ** 60 + 7
        self.moving_mask = np.zeros_like(self.fixed_mask)
        self.moving_mask[2:, 3:] = self.fixed_mask[:-2, :-3]
        for name, array in (("fixed", self.fixed), ("moving", self.moving),
                            ("fixed_mask", self.fixed_mask), ("moving_mask", self.moving_mask)):
            imwrite(self.root / f"{name}.tif", array)
        self.write_points(self.points, self.points + [3, 2])
        self.options = {
            "fixed_image": self.root / "fixed.tif", "moving_image": self.root / "moving.tif",
            "fixed_mask": self.root / "fixed_mask.tif", "moving_mask": self.root / "moving_mask.tif",
            "landmarks_csv": self.root / "points.csv", "output_dir": self.root,
            "fixed_section": 1, "moving_section": 2,
        }

    def write_points(self, fixed, moving, wide=False):
        with (self.root / "points.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            if wide:
                writer.writerow(["point", "fixed_x", "fixed_y", "moving_x", "moving_y"])
                for index, (a, b) in enumerate(zip(fixed, moving)):
                    writer.writerow([f"point{index}", *a, *b])
            else:
                writer.writerow(["", "slice", "X", "Y"])
                for index, point in enumerate(fixed):
                    writer.writerow([f"point{index}", 1, *point])
                # Deliberately reverse the row order on the moving section.
                for index in reversed(range(len(moving))):
                    writer.writerow([f"point{index}", 2, *moving[index]])

    def test_translation_registers_image_and_preserves_large_label_ids(self):
        original = self.options["moving_image"].read_bytes()
        result = run_em_registration(**self.options)
        registered = imread(result["files"]["registered_image"])
        labels = imread(result["files"]["registered_mask"])
        assert_array_equal(registered[:30, :29], self.fixed[:30, :29])
        assert_array_equal(labels, self.fixed_mask)
        self.assertEqual(labels.dtype, np.uint64)
        self.assertEqual(registered.dtype, self.moving.dtype)
        self.assertEqual(self.options["moving_image"].read_bytes(), original)
        report = json.loads(Path(result["files"]["report"]).read_text())
        self.assertEqual(report["foreground_mask_dice"], 1.0)
        self.assertLess(report["landmark_fit_rmse_pixels"], 1e-8)
        self.assertGreater(report["outside_moving_fraction"], 0)
        self.assertTrue(Path(result["files"]["preview"]).is_file())
        self.assertEqual(len(list((Path(result["output_dir"]) / "images").glob("*.tif"))), 2)

    def test_wide_and_original_csv_match_points_by_id(self):
        long = read_landmarks(self.root / "points.csv", 1, 2)
        self.write_points(self.points, self.points + [3, 2], wide=True)
        wide = read_landmarks(self.root / "points.csv", 1, 2)
        assert_array_equal(long["fixed"], wide["fixed"])
        assert_array_equal(long["moving"], wide["moving"])

    def test_bottom_left_one_based_coordinates(self):
        fixed = self.points.copy()
        moving = self.points + [3, 2]
        fixed[:, 1] = 31 - fixed[:, 1]
        moving[:, 1] = 31 - moving[:, 1]
        self.write_points(fixed + 1, moving + 1)
        result = run_em_registration(**self.options, coordinate_origin="bottom-left", coordinate_base=1)
        assert_array_equal(imread(result["files"]["registered_mask"]), self.fixed_mask)

    def test_identity_keeps_border_pixels_and_labels(self):
        inverse, _ = fit_transforms(self.points, self.points)
        result, mask, coverage = warp_pair(self.fixed, self.fixed_mask, self.fixed.shape, inverse, block_rows=7)
        assert_array_equal(result, self.fixed)
        assert_array_equal(mask, self.fixed_mask)
        self.assertTrue(coverage.all())

    def test_non_affine_landmark_displacement_and_mask_use_same_map(self):
        fixed = np.vstack([self.points, [12., 12.]])
        moving = fixed.copy()
        moving[-1] += [2, -1]
        inverse, forward = fit_transforms(fixed, moving)
        assert_allclose(inverse(fixed), moving, atol=1e-8)
        assert_allclose(forward(moving), fixed, atol=1e-8)
        image = np.zeros((32, 32), dtype=np.uint16)
        image[11, 14] = 1000
        labels = np.zeros_like(image)
        labels[11, 14] = 17
        registered, mask, _ = warp_pair(image, labels, image.shape, inverse)
        self.assertEqual(int(registered[12, 12]), 1000)
        self.assertEqual(int(mask[12, 12]), 17)
        self.assertLessEqual(set(np.unique(mask)), {0, 17})

    def test_invalid_inputs_do_not_create_output_runs(self):
        for points, expected in (
            (np.array([[1., 1.], [2., 2.], [3., 3.]]), "one line"),
            (np.array([[1., 1.], [1., 1.], [3., 5.]]), "duplicate coordinates"),
            (np.array([[1., 1.], [2., 20.], [40., 3.]]), "outside"),
        ):
            with self.subTest(expected=expected):
                self.write_points(points, points)
                with self.assertRaisesRegex(ValueError, expected):
                    run_em_registration(**self.options)
        self.assertFalse(list(self.root.glob("em_registration_*")))

    def test_duplicate_ids_are_rejected(self):
        with (self.root / "points.csv").open("a") as stream:
            stream.write("point0,1,4,4\n")
        with self.assertRaisesRegex(ValueError, "duplicate landmark ID"):
            run_em_registration(**self.options)

    def test_missing_ids_are_reported_and_insufficient_pairs_rejected(self):
        with (self.root / "points.csv").open("a") as stream:
            stream.write("extra,1,15,15\n")
        self.assertEqual(read_landmarks(self.root / "points.csv", 1, 2)["fixed_only"], ["extra"])
        self.write_points(self.points[:2], self.points[:2])
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run_em_registration(**self.options)

    def test_mask_pair_and_shape_are_validated(self):
        with self.assertRaisesRegex(ValueError, "both fixed and moving masks"):
            run_em_registration(**dict(self.options, moving_mask=None))
        imwrite(self.root / "moving_mask.tif", np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_em_registration(**self.options)

    def test_images_only_and_repeated_runs_are_supported(self):
        results = [run_em_registration(**dict(self.options, fixed_mask=None, moving_mask=None)) for _ in range(2)]
        self.assertNotEqual(results[0]["output_dir"], results[1]["output_dir"])
        self.assertNotIn("registered_mask", results[0]["files"])

    def test_cancellation_during_save_cleans_only_its_run(self):
        cancel = [False]
        def progress(percent, message):
            if percent >= 88:
                cancel[0] = True
        with self.assertRaises(RegistrationCancelled):
            run_em_registration(**self.options, cancelled=lambda: cancel[0], progress=progress)
        self.assertFalse(list(self.root.glob("*em_registration_*")))
        self.assertTrue(self.options["fixed_image"].is_file())


if __name__ == "__main__":
    unittest.main()
