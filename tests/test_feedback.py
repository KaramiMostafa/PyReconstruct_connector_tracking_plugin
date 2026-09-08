import json
import tempfile
import unittest
from pathlib import Path

from pyrecon_connector.feedback import (
    add_dapi_link_feedback_batch,
    add_feedback_record,
    add_mapped_roi_feedback_batch,
    association_constraints,
    export_feedback_csv,
    feedback_summary,
    load_feedback,
    record_dapi_rename_feedback,
)


class FakeTrace:
    def __init__(self, name, x, y):
        self.name = name
        self.tags = {"multiplex_dapi"}
        self.points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]

    def getCentroid(self):
        return (
            sum(point[0] for point in self.points) / len(self.points),
            sum(point[1] for point in self.points) / len(self.points),
        )


class FakeSeries:
    def __init__(self, path):
        self.jser_fp = str(path)


class FakeContour:
    def __init__(self, traces):
        self.traces = traces


class FakeSection:
    def __init__(self, contours):
        self.contours = {
            name: FakeContour(traces) for name, traces in contours.items()
        }


class RenameSeries(FakeSeries):
    def __init__(self, path, sections):
        super().__init__(path)
        self._sections = sections
        self.sections = {number: str(number) for number in sections}

    def loadSection(self, number):
        return self._sections[number]


class FeedbackTests(unittest.TestCase):
    def test_feedback_is_auditable_and_replaces_same_assertion(self):
        with tempfile.TemporaryDirectory() as folder:
            series = FakeSeries(Path(folder) / "sample.jser")
            rna = FakeTrace("rna_1", 10, 10)
            dapi = FakeTrace("cell_1", 11, 10)
            add_feedback_record(series, "rna_dapi_pair", "incorrect", 1, rna, dapi)
            add_feedback_record(series, "rna_dapi_pair", "correct", 1, rna, dapi, notes="checked")

            payload = load_feedback(series)
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["verdict"], "correct")
            constraints = association_constraints(payload, 1)
            self.assertEqual(constraints["rna_1"]["correct"], {"cell_1"})
            self.assertEqual(feedback_summary(series)["total"], 1)

            csv_path = Path(folder) / "feedback.csv"
            export_feedback_csv(series, csv_path)
            self.assertTrue(csv_path.is_file())
            self.assertIn("rna_dapi_pair", csv_path.read_text(encoding="utf-8"))

    def test_unsaved_series_is_rejected(self):
        series = FakeSeries("")
        with self.assertRaises(ValueError):
            add_feedback_record(
                series, "mapped_rna", "incorrect", 2, FakeTrace("mapped_rna_1", 2, 3)
            )

    def test_reversed_dapi_pair_replaces_contradictory_assertion(self):
        with tempfile.TemporaryDirectory() as folder:
            series = FakeSeries(Path(folder) / "sample.jser")
            first = FakeTrace("cell_1", 10, 10)
            second = FakeTrace("cell_2", 11, 10)
            add_feedback_record(
                series, "dapi_track_link", "incorrect", 13, first,
                second, secondary_section=14,
            )
            add_feedback_record(
                series, "dapi_track_link", "correct", 14, second,
                first, secondary_section=13,
            )
            payload = load_feedback(series)
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["verdict"], "correct")

    def test_batch_dapi_feedback_is_saved_together(self):
        with tempfile.TemporaryDirectory() as folder:
            series = FakeSeries(Path(folder) / "sample.jser")
            endpoints = [
                {"section": section, "name": f"cell_{section}", "centroid": [section, 0]}
                for section in (13, 14, 15)
            ]
            records = add_dapi_link_feedback_batch(
                series,
                [(endpoints[0], endpoints[1]), (endpoints[1], endpoints[2])],
                "incorrect",
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(len(load_feedback(series)["records"]), 2)

    def test_batch_mapped_rna_feedback_is_saved_by_section(self):
        with tempfile.TemporaryDirectory() as folder:
            series = FakeSeries(Path(folder) / "sample.jser")
            endpoints = [
                {
                    "section": 15,
                    "name": f"mapped_rna_{index}",
                    "centroid": [float(index), 2.0],
                }
                for index in (1, 2, 3)
            ]
            records = add_mapped_roi_feedback_batch(
                series, endpoints, "incorrect", notes="checked together"
            )
            self.assertEqual(len(records), 3)
            payload = load_feedback(series)
            self.assertEqual(len(payload["records"]), 3)
            self.assertEqual(
                {(record["section"], record["primary_name"]) for record in payload["records"]},
                {(15, "mapped_rna_1"), (15, "mapped_rna_2"), (15, "mapped_rna_3")},
            )
            self.assertTrue(all(record["verdict"] == "incorrect" for record in payload["records"]))

    def test_manual_rename_creates_correct_and_incorrect_neighbor_links(self):
        with tempfile.TemporaryDirectory() as folder:
            sections = {
                13: FakeSection({
                    "cell_168": [FakeTrace("cell_168", 10, 10)],
                    "cell_177": [FakeTrace("cell_177", 20, 20)],
                }),
                14: FakeSection({}),
                15: FakeSection({
                    "cell_168": [FakeTrace("cell_168", 11, 10)],
                    "cell_177": [FakeTrace("cell_177", 21, 20)],
                }),
            }
            series = RenameSeries(Path(folder) / "sample.jser", sections)
            result = record_dapi_rename_feedback(
                series, 14, "cell_177", "cell_168", [10.5, 10]
            )
            self.assertEqual(result["correct"], 2)
            self.assertEqual(result["incorrect"], 2)
            self.assertEqual(len(load_feedback(series)["records"]), 4)


if __name__ == "__main__":
    unittest.main()
