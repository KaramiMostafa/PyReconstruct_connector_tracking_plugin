import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from pyrecon_connector.multiplex_mapping import (
    _idw_displacement,
    _polygon_centroid,
    _roi_payloads,
    parse_mapping_windows,
    run_multiplex_rna_mapping,
)


class FakeTrace:
    def __init__(self, name, color=(0, 0, 0), closed=True):
        self.name = name
        self.color = color
        self.closed = closed
        self.points = []
        self.tags = set()
        self.fill_mode = ("none", "none")


class FakeContour:
    def __init__(self, traces):
        self.traces = list(traces)


class IdentityTransform:
    def map(self, points, inverted=False):
        return [tuple(point) for point in points]


class FakeSection:
    def __init__(self, number, traces):
        self.n = number
        self.tform = IdentityTransform()
        self.contours = {}
        for trace in traces:
            self.contours.setdefault(trace.name, FakeContour([])).traces.append(trace)

    def addTrace(self, trace, log_event=True):
        self.contours.setdefault(trace.name, FakeContour([])).traces.append(trace)

    def removeTrace(self, trace, log_event=True):
        self.contours[trace.name].traces.remove(trace)

    def save(self, update_series_data=True):
        pass


class FakeGroups:
    def __init__(self):
        self.groups = {}

    def add(self, group, name):
        self.groups.setdefault(group, set()).add(name)

    def getGroupObjects(self, group):
        return self.groups.get(group, set())


class FakeSeries:
    def __init__(self, sections):
        self._sections = sections
        self.sections = {number: {} for number in sections}
        self.object_groups = FakeGroups()

    def loadSection(self, number):
        return self._sections[number]

    def save(self):
        pass


def square(name, cx, cy, radius=2):
    trace = FakeTrace(name, closed=True)
    trace.points = [
        (cx - radius, cy - radius),
        (cx + radius, cy - radius),
        (cx + radius, cy + radius),
        (cx - radius, cy + radius),
    ]
    return trace


class MultiplexMappingTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType("PyReconstruct.modules.datatypes")
        module.Trace = FakeTrace
        self.module_patch = patch.dict(sys.modules, {
            "PyReconstruct": types.ModuleType("PyReconstruct"),
            "PyReconstruct.modules": types.ModuleType("PyReconstruct.modules"),
            "PyReconstruct.modules.datatypes": module,
        })
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()

    def test_parses_mapping_windows(self):
        self.assertEqual(
            parse_mapping_windows("1:2-4;6:7,8"),
            {1: [2, 3, 4], 6: [7, 8]},
        )
        with self.assertRaises(ValueError):
            parse_mapping_windows("1:1-3")

    def test_idw_preserves_constant_displacement(self):
        points = np.array([[2.0, 3.0], [8.0, 9.0]])
        origins = np.array([[0.0, 0.0], [10.0, 10.0]])
        displacements = np.array([[4.0, -2.0], [4.0, -2.0]])
        np.testing.assert_allclose(
            _idw_displacement(points, origins, displacements, 2),
            np.array([[4.0, -2.0], [4.0, -2.0]]),
        )

    def test_roi_discovery_skips_malformed_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            bad_zip = Path(folder) / "Section 1 bad.zip"
            bad_zip.write_text("not a zip archive", encoding="utf-8")
            self.assertEqual(list(_roi_payloads(folder)), [])

    def test_maps_rna_to_same_dapi_track_on_target(self):
        source_dapi = square("cell_00001", 10, 10, 1)
        neighbor_source = square("cell_00002", 30, 10, 1)
        rna = square("rna_anchor_001", 10, 10, 4)
        target_dapi = square("cell_00001", 17, 14, 1)
        neighbor_target = square("cell_00002", 37, 14, 1)
        series = FakeSeries({
            1: FakeSection(1, [source_dapi, neighbor_source, rna]),
            2: FakeSection(2, [target_dapi, neighbor_target]),
        })

        with tempfile.TemporaryDirectory() as output_dir:
            result = run_multiplex_rna_mapping(
                series,
                "1:2",
                dapi_prefix="cell_",
                rna_prefix="rna_",
                association_max_distance=10,
                output_dir=output_dir,
            )
            self.assertTrue(Path(result["csv"]).is_file())
            self.assertTrue((Path(result["qc_dir"]) / "section_002_mapping.png").is_file())

        self.assertEqual(result["created"], 1)
        mapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        np.testing.assert_allclose(_polygon_centroid(np.asarray(mapped.points)), [17.0, 14.0])
        self.assertIn("multiplex_mapped_rna", mapped.tags)

    def test_skips_when_tracked_dapi_is_absent_on_target(self):
        series = FakeSeries({
            1: FakeSection(1, [square("cell_00001", 10, 10, 1), square("rna_001", 10, 10, 3)]),
            2: FakeSection(2, [square("cell_other", 15, 10, 1)]),
        })

        result = run_multiplex_rna_mapping(series, "1:2", association_max_distance=10)

        self.assertEqual(result["created"], 0)
        self.assertEqual(result["skipped_missing_track"], 1)

    def test_requires_explicit_dapi_and_rna_selectors(self):
        series = FakeSeries({1: FakeSection(1, []), 2: FakeSection(2, [])})
        with self.assertRaises(ValueError):
            run_multiplex_rna_mapping(series, "1:2", dapi_prefix="", dapi_group="")
        with self.assertRaises(ValueError):
            run_multiplex_rna_mapping(series, "1:2", rna_prefix="", rna_group="")


if __name__ == "__main__":
    unittest.main()
