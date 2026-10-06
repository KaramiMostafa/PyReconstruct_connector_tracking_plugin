import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from tifffile import imread, imwrite

from pyrecon_connector import run_serial_registration
from pyrecon_connector.registration_core import RegistrationCancelled
from pyrecon_connector.serial_registration import discover_images


class SerialRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "input"
        self.images.mkdir()
        self.base = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        self.a = {"a": [4, 4], "b": [20, 4], "c": [4, 20]}
        self.b = {"d": [8, 8], "e": [24, 8], "f": [8, 24]}
        self.tables = {}
        for section, values in ((1, self.a), (2, dict(self.a, **self.b)), (3, self.b)):
            shift = np.array([3, 2]) * (section - 1)
            self.tables[section] = {name: np.asarray(xy) + shift for name, xy in values.items()}
            image = np.zeros_like(self.base)
            x, y = shift
            image[y:, x:] = self.base[:32-y, :32-x]
            imwrite(self.images / f"_underwood.{section}.tif", image)
        self.options = dict(image_series=self.images, landmarks_csv=self.root / "points.csv",
                            output_dir=self.root, block_rows=7)
        self.write_csv()

    def write_csv(self):
        with self.options["landmarks_csv"].open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["", "slice", "X", "Y"])
            for section in reversed(self.tables):
                for name, xy in reversed(list(self.tables[section].items())):
                    writer.writerow([name, section, *xy])

    def test_chained_landmarks_when_first_and_third_have_no_shared_ids(self):
        for method in ("tps", "affine"):
            with self.subTest(method=method):
                result = run_serial_registration(**self.options, method=method)
                self.assertEqual(result["report"]["section_order"], [1, 2, 3])
                self.assertEqual(result["report"]["pairs"][1]["previous_slice"], 2)
                for name, xy in self.b.items():
                    assert_allclose(result["registered_landmarks"][2][name], xy, atol=1e-8)
                    assert_allclose(result["registered_landmarks"][3][name], xy, atol=1e-8)
                images = Path(result["files"]["images"])
                self.assertEqual((images / "section_0001.tif").read_bytes(),
                                 (self.images / "_underwood.1.tif").read_bytes())
                warped = imread(images / "section_0003.tif")
                self.assertEqual(warped.dtype, self.base.dtype)
                self.assertEqual(warped.shape, self.base.shape)
                assert_allclose(warped[:28, :26], self.base[:28, :26], atol=1)
                self.assertTrue(Path(result["files"]["registered_landmarks"]).is_file())
                self.assertTrue(Path(result["files"]["preview"]).is_file())

    def test_numeric_filename_order_and_duplicate_slice_detection(self):
        imwrite(self.images / "_underwood.10.tif", self.base)
        self.assertEqual(list(discover_images(self.images)), [1, 2, 3, 10])
        imwrite(self.images / "other.1.tif", self.base)
        with self.assertRaisesRegex(ValueError, "Slice 1.*more than one"):
            discover_images(self.images)

    def test_unmatched_landmark_outside_reference_canvas_is_retained(self):
        self.tables[2]["outside"] = [1, 1]
        self.tables[3]["outside"] = [4, 3]
        self.write_csv()
        result = run_serial_registration(**self.options)
        assert_allclose(result["registered_landmarks"][2]["outside"], [-2, -1], atol=1e-8)
        assert_allclose(result["registered_landmarks"][3]["outside"], [-2, -1], atol=1e-8)

    def test_different_input_dimensions_use_first_canvas(self):
        imwrite(self.images / "_underwood.3.tif", np.zeros((38, 40), dtype=np.uint8))
        result = run_serial_registration(**self.options)
        plane = imread(Path(result["files"]["images"]) / "section_0003.tif")
        self.assertEqual(plane.shape, (32, 32))
        self.assertEqual(plane.dtype, np.uint8)

    def test_bottom_left_one_based_conversion(self):
        for values in self.tables.values():
            for name, xy in values.items():
                values[name] = [xy[0] + 1, 32 - xy[1]]
        self.write_csv()
        result = run_serial_registration(**self.options, coordinate_origin="bottom-left", coordinate_base=1)
        assert_allclose(result["registered_landmarks"][3]["d"], [8, 8], atol=1e-8)

    def test_missing_correspondences_stop_with_slice_number(self):
        self.tables[3].pop("f")
        self.tables[3]["unmatched"] = [10, 10]
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Slice 3 -> 2.*at least 3"):
            run_serial_registration(**self.options)
        self.assertFalse(list(self.root.glob("*serial_registration_*")))

    def test_csv_unknown_slice_and_invalid_coordinates_rejected(self):
        self.tables[99] = self.a
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "slice 99.*no corresponding image"):
            run_serial_registration(**self.options)
        del self.tables[99]
        for xy, expected in (([np.nan, 2], "finite"), ([99, 2], "outside")):
            self.tables[3]["d"] = xy
            self.write_csv()
            with self.assertRaisesRegex(ValueError, f"Slice 3.*{expected}"):
                run_serial_registration(**self.options)

    def test_failure_after_first_pair_publishes_no_partial_stack(self):
        from pyrecon_connector.serial_registration import fit_transforms
        count = [0]
        def fail_second(*args, **kwargs):
            count[0] += 1
            if count[0] == 2:
                raise ValueError("Injected fitting failure")
            return fit_transforms(*args, **kwargs)
        with patch("pyrecon_connector.serial_registration.fit_transforms", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "Slice 3 -> registered slice 2 failed"):
                run_serial_registration(**self.options)
        self.assertFalse(list(self.root.glob("*serial_registration_*")))
        self.assertEqual(len(list(self.images.glob("*.tif"))), 3)

    def test_cancellation_cleans_staging(self):
        cancel = [False]
        def progress(value, message):
            if value > 30:
                cancel[0] = True
        with self.assertRaises(RegistrationCancelled):
            run_serial_registration(**self.options, progress=progress, cancelled=lambda: cancel[0])
        self.assertFalse(list(self.root.glob("*serial_registration_*")))


if __name__ == "__main__":
    unittest.main()
