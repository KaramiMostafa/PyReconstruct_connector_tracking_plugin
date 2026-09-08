import sys
import tempfile
import types
import unittest
import json
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
from pyrecon_connector.feedback import (
    add_feedback_record,
    add_mapped_roi_feedback_batch,
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


class ScaleTransform:
    def __init__(self, x_scale, y_scale):
        self.scale = np.asarray([x_scale, y_scale], dtype=float)

    def map(self, points, inverted=False):
        values = np.asarray(points, dtype=float)
        mapped = values / self.scale if inverted else values * self.scale
        return [tuple(point) for point in mapped]


class FakeSection:
    def __init__(self, number, traces):
        self.n = number
        self.tform = IdentityTransform()
        self.img_dims = (100, 100)
        self.mag = 1.0
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
        self.assertEqual(
            parse_mapping_windows("6:4-12"),
            {6: [4, 5, 7, 8, 9, 10, 11, 12]},
        )
        with self.assertRaises(ValueError):
            parse_mapping_windows("1:1")

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
        np.testing.assert_allclose(
            np.asarray(mapped.points) - np.asarray(mapped.points)[0],
            np.asarray(rna.points) - np.asarray(rna.points)[0],
        )
        self.assertIn("multiplex_mapped_rna", mapped.tags)

    def test_nonuniform_neighbor_motion_does_not_deform_tracked_roi(self):
        source_dapi = square("cell_00001", 10, 10, 1)
        neighbor_source = square("cell_00002", 30, 10, 1)
        rna = square("rna_anchor_001", 10, 10, 4)
        series = FakeSeries({
            1: FakeSection(1, [source_dapi, neighbor_source, rna]),
            2: FakeSection(2, [
                square("cell_00001", 15, 12, 1),
                square("cell_00002", 70, 40, 1),
            ]),
        })

        result = run_multiplex_rna_mapping(
            series, "1:2", association_max_distance=10
        )

        mapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        np.testing.assert_allclose(_polygon_centroid(np.asarray(mapped.points)), [15.0, 12.0])
        np.testing.assert_allclose(
            np.asarray(mapped.points) - np.asarray(mapped.points)[0],
            np.asarray(rna.points) - np.asarray(rna.points)[0],
        )
        self.assertEqual(result["review"], 1)

    def test_dapi_and_rna_tags_prevent_name_collision_from_mixing_roles(self):
        source_dapi = square("shared_cell", 10, 10, 1)
        source_dapi.tags.add("multiplex_dapi")
        source_rna = square("shared_cell", 10, 10, 4)
        source_rna.tags.add("multiplex_rna_anchor")
        target_dapi = square("shared_cell", 15, 12, 1)
        target_dapi.tags.add("multiplex_dapi")
        series = FakeSeries({
            1: FakeSection(1, [source_dapi, source_rna]),
            2: FakeSection(2, [target_dapi]),
        })
        series.object_groups.add("multiplex_tracked_dapi", "shared_cell")
        series.object_groups.add("multiplex_rna_anchor", "shared_cell")

        result = run_multiplex_rna_mapping(
            series,
            "1:2",
            dapi_prefix="",
            dapi_group="multiplex_tracked_dapi",
            rna_prefix="",
            rna_group="multiplex_rna_anchor",
            association_max_distance=10,
        )

        self.assertEqual(result["created"], 1)
        self.assertEqual(result["skipped_unassociated"], 0)

    def test_existing_mappings_are_reported_when_overwrite_is_off(self):
        series = FakeSeries({
            1: FakeSection(1, [
                square("cell_00001", 10, 10, 1),
                square("rna_001", 10, 10, 3),
            ]),
            2: FakeSection(2, [square("cell_00001", 15, 12, 1)]),
        })
        first = run_multiplex_rna_mapping(series, "1:2", association_max_distance=10)
        second = run_multiplex_rna_mapping(series, "1:2", association_max_distance=10)

        self.assertEqual(first["created"], 1)
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["skipped_existing"], 1)

    def test_skips_when_tracked_dapi_is_absent_on_target(self):
        series = FakeSeries({
            1: FakeSection(1, [square("cell_00001", 10, 10, 1), square("rna_001", 10, 10, 3)]),
            2: FakeSection(2, [square("other_00001", 15, 10, 1)]),
        })

        result = run_multiplex_rna_mapping(series, "1:2", association_max_distance=10)

        self.assertEqual(result["created"], 0)
        self.assertEqual(result["skipped_missing_track"], 1)
        self.assertEqual(result["target_sections_without_dapi"], [2])

    def test_uses_review_fallback_when_associated_track_is_missing(self):
        series = FakeSeries({
            1: FakeSection(1, [
                square("cell_00001", 10, 10, 1),
                square("cell_00002", 30, 10, 1),
                square("rna_001", 10, 10, 3),
            ]),
            2: FakeSection(2, [square("cell_00002", 35, 13, 1)]),
        })

        result = run_multiplex_rna_mapping(series, "1:2", association_max_distance=10)

        self.assertEqual(result["created"], 1)
        self.assertEqual(result["review"], 1)
        mapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        np.testing.assert_allclose(_polygon_centroid(np.asarray(mapped.points)), [15.0, 13.0])
        source = series.loadSection(1).contours["rna_001"].traces[0]
        np.testing.assert_allclose(
            np.asarray(mapped.points) - np.asarray(mapped.points)[0],
            np.asarray(source.points) - np.asarray(source.points)[0],
        )

    def test_registration_scale_cannot_change_mapped_roi_shape(self):
        rna = square("rna_001", 10, 10, 4)
        source_points = np.asarray(rna.points).copy()
        source = FakeSection(1, [square("cell_00001", 10, 10, 1), rna])
        target = FakeSection(2, [square("cell_00001", 20, 20, 1)])
        source.tform = ScaleTransform(2.0, 2.0)
        target.tform = ScaleTransform(0.5, 0.5)
        series = FakeSeries({1: source, 2: target})

        result = run_multiplex_rna_mapping(
            series, "1:2", association_max_distance=10
        )

        mapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        mapped_points = np.asarray(mapped.points)
        self.assertEqual(result["created"], 1)
        np.testing.assert_allclose(_polygon_centroid(mapped_points), [20.0, 20.0])
        np.testing.assert_allclose(
            mapped_points - mapped_points[0],
            source_points - source_points[0],
        )
        np.testing.assert_allclose(np.asarray(rna.points), source_points)

    def test_outside_translation_is_skipped_without_altering_anchor(self):
        rna = square("rna_001", 95, 50, 4)
        source_points = np.asarray(rna.points).copy()
        source = FakeSection(1, [square("cell_00001", 95, 50, 1), rna])
        target = FakeSection(2, [square("cell_00001", 104, 50, 1)])
        target.img_dims = (200, 200)
        series = FakeSeries({1: source, 2: target})

        first = run_multiplex_rna_mapping(
            series, "1:2", association_max_distance=10
        )
        self.assertEqual(first["created"], 1)
        target.img_dims = (100, 100)

        result = run_multiplex_rna_mapping(
            series, "1:2", association_max_distance=10, overwrite=True
        )

        self.assertEqual(result["created"], 0)
        self.assertEqual(result["skipped_out_of_frame"], 1)
        self.assertFalse(any(
            name.startswith("mapped_rna_") and contour.traces
            for name, contour in series.loadSection(2).contours.items()
        ))
        np.testing.assert_allclose(np.asarray(rna.points), source_points)

    def test_correct_pair_feedback_forces_reviewed_dapi_identity(self):
        source_1 = square("cell_00001", 10, 10, 1)
        source_2 = square("cell_00002", 30, 10, 1)
        rna = square("rna_001", 10, 10, 3)
        series = FakeSeries({
            1: FakeSection(1, [source_1, source_2, rna]),
            2: FakeSection(2, [
                square("cell_00001", 15, 10, 1),
                square("cell_00002", 40, 10, 1),
            ]),
        })
        with tempfile.TemporaryDirectory() as folder:
            series.jser_fp = str(Path(folder) / "sample.jser")
            add_feedback_record(
                series, "rna_dapi_pair", "correct", 1, rna, secondary_trace=source_2
            )
            result = run_multiplex_rna_mapping(
                series, "1:2", association_max_distance=10, apply_feedback=True
            )

        self.assertEqual(result["feedback_applied"], 1)
        mapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        np.testing.assert_allclose(_polygon_centroid(np.asarray(mapped.points)), [20.0, 10.0])

    def test_batch_mapped_feedback_controls_rerun_status_and_color(self):
        series = FakeSeries({
            1: FakeSection(1, [
                square("cell_00001", 10, 10, 1),
                square("rna_001", 10, 10, 3),
            ]),
            2: FakeSection(2, [square("cell_00001", 15, 12, 1)]),
        })
        with tempfile.TemporaryDirectory() as folder:
            series.jser_fp = str(Path(folder) / "sample.jser")
            first = run_multiplex_rna_mapping(
                series, "1:2", association_max_distance=10
            )
            mapped = [
                trace for name, contour in series.loadSection(2).contours.items()
                if name.startswith("mapped_rna_") for trace in contour.traces
            ][0]
            add_mapped_roi_feedback_batch(
                series,
                [{
                    "section": 2,
                    "name": mapped.name,
                    "centroid": list(_polygon_centroid(np.asarray(mapped.points))),
                }],
                "incorrect",
            )
            second = run_multiplex_rna_mapping(
                series, "1:2", association_max_distance=10, overwrite=True
            )

        self.assertEqual(first["created"], 1)
        self.assertEqual(second["created"], 1)
        self.assertEqual(second["feedback_applied"], 1)
        remapped = [
            trace for name, contour in series.loadSection(2).contours.items()
            if name.startswith("mapped_rna_") for trace in contour.traces
        ][0]
        self.assertEqual(remapped.color, (230, 60, 60))
        self.assertIn("expert_rejected", remapped.tags)

    def test_missing_target_sections_are_skipped_and_receive_qc_plots(self):
        series = FakeSeries({
            1: FakeSection(1, [square("cell_00001", 10, 10, 1), square("rna_001", 10, 10, 3)]),
            2: FakeSection(2, [square("cell_00001", 15, 12, 1)]),
        })

        with tempfile.TemporaryDirectory() as output_dir:
            result = run_multiplex_rna_mapping(
                series,
                "1:2-4",
                association_max_distance=10,
                output_dir=output_dir,
            )

            self.assertEqual(result["created"], 1)
            self.assertEqual(result["missing_series_sections"], [3, 4])
            self.assertEqual(
                [item["target_section"] for item in result["skipped_target_sections"]],
                [3, 4],
            )
            for section_num in (2, 3, 4):
                self.assertTrue(
                    (Path(result["qc_dir"]) / f"section_{section_num:03d}_mapping.png").is_file()
                )

    def test_missing_anchor_still_saves_empty_results_and_diagnostic_plot(self):
        series = FakeSeries({2: FakeSection(2, [square("cell_00001", 15, 12, 1)])})

        with tempfile.TemporaryDirectory() as output_dir:
            result = run_multiplex_rna_mapping(series, "1:2", output_dir=output_dir)

            self.assertEqual(result["created"], 0)
            self.assertEqual(result["missing_series_sections"], [1])
            self.assertEqual(result["skipped_anchor_sections"], [{"section": 1, "reason": "section_not_present"}])
            self.assertTrue(Path(result["csv"]).is_file())
            self.assertTrue(Path(result["summary_json"]).is_file())
            self.assertTrue((Path(result["qc_dir"]) / "section_002_mapping.png").is_file())
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            self.assertIn("Anchor section 1 is not present and was skipped.", summary["warnings"])

    def test_requires_explicit_dapi_and_rna_selectors(self):
        series = FakeSeries({1: FakeSection(1, []), 2: FakeSection(2, [])})
        with self.assertRaises(ValueError):
            run_multiplex_rna_mapping(series, "1:2", dapi_prefix="", dapi_group="")
        with self.assertRaises(ValueError):
            run_multiplex_rna_mapping(series, "1:2", rna_prefix="", rna_group="")


if __name__ == "__main__":
    unittest.main()
