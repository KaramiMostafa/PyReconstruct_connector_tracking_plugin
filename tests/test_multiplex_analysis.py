import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyrecon_connector.multiplex_analysis import (
    _select_channel,
    measure_antibody_intensity,
    validate_mapped_rna,
)


class Trace:
    def __init__(self, name, cx, cy):
        self.name = name
        self.closed = True
        self.points = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx + 1, cy + 1), (cx - 1, cy + 1)]


class Contour:
    def __init__(self, trace):
        self.traces = [trace]


class Transform:
    def map(self, points, inverted=False):
        return list(points)


class Section:
    def __init__(self, number, traces):
        self.n = number
        self.tform = Transform()
        self.mag = 1.0
        self.contours = {trace.name: Contour(trace) for trace in traces}


class Groups:
    def getGroupObjects(self, group):
        return set()


class Series:
    def __init__(self, sections):
        self._sections = sections
        self.sections = {number: {} for number in sections}
        self.object_groups = Groups()

    def loadSection(self, number):
        return self._sections[number]


class MultiplexAnalysisTests(unittest.TestCase):
    def test_validation_reports_precision_and_recall(self):
        series = Series({
            2: Section(2, [
                Trace("mapped_rna_a", 10, 10),
                Trace("mapped_rna_b", 30, 30),
                Trace("expert_rna_a", 11, 10),
                Trace("expert_rna_c", 80, 80),
            ])
        })
        with tempfile.TemporaryDirectory() as folder:
            result = validate_mapped_rna(
                series,
                folder,
                predicted_prefix="mapped_rna_",
                predicted_group="",
                expert_prefix="expert_rna_",
                expert_group="",
                max_centroid_distance=5,
            )
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
            self.assertAlmostEqual(result["precision"], 0.5)
            self.assertAlmostEqual(result["recall"], 0.5)
            self.assertTrue(Path(result["csv"]).is_file())

    def test_channel_selection_supports_first_or_last_channel_axis(self):
        first = np.zeros((3, 10, 12))
        first[1] = 7
        last = np.zeros((10, 12, 3))
        last[:, :, 2] = 9
        np.testing.assert_array_equal(_select_channel(first, 1), np.full((10, 12), 7.0))
        np.testing.assert_array_equal(_select_channel(last, 2), np.full((10, 12), 9.0))

    def test_intensity_analysis_writes_outputs(self):
        import tifffile

        series = Series({2: Section(2, [Trace("mapped_rna_a", 5, 5)])})
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tif_folder = folder / "tiffs"
            output = folder / "output"
            tif_folder.mkdir()
            image = np.zeros((20, 20), dtype=np.uint16)
            image[13:17, 3:7] = 100
            tifffile.imwrite(tif_folder / "Section 2 antibody.tif", image)
            result = measure_antibody_intensity(
                series,
                str(tif_folder),
                str(output),
                roi_prefix="mapped_rna_",
                roi_group="",
            )
            self.assertEqual(result["measured_rois"], 1)
            self.assertEqual(result["missing_tiff_sections"], [])
            self.assertTrue(Path(result["measurements_csv"]).is_file())
            self.assertTrue(Path(result["summary_json"]).is_file())


if __name__ == "__main__":
    unittest.main()
