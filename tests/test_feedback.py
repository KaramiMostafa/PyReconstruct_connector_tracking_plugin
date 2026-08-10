import json
import tempfile
import unittest
from pathlib import Path

from pyrecon_connector.feedback import (
    add_feedback_record,
    association_constraints,
    export_feedback_csv,
    feedback_summary,
    load_feedback,
)


class FakeTrace:
    def __init__(self, name, x, y):
        self.name = name
        self.points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]

    def getCentroid(self):
        return (
            sum(point[0] for point in self.points) / len(self.points),
            sum(point[1] for point in self.points) / len(self.points),
        )


class FakeSeries:
    def __init__(self, path):
        self.jser_fp = str(path)


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


if __name__ == "__main__":
    unittest.main()
