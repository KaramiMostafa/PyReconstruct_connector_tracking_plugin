import unittest

import pandas as pd

from pyrecon_connector.hungarian_inapp import _Ref, _apply_link_feedback


class Trace:
    def __init__(self, name, x):
        self.name = name
        self.x = x

    def getCentroid(self):
        return self.x, 0


class HungarianFeedbackTests(unittest.TestCase):
    def test_incorrect_link_splits_target_side(self):
        tracks = pd.DataFrame([
            {"FrameID": 0, "Label": 0, "TrackID": 1},
            {"FrameID": 1, "Label": 0, "TrackID": 1},
            {"FrameID": 2, "Label": 0, "TrackID": 1},
        ])
        refs = {
            0: [_Ref(1, Trace("cell_00001", 0))],
            1: [_Ref(2, Trace("cell_00001", 1))],
            2: [_Ref(3, Trace("cell_00001", 2))],
        }
        applied = _apply_link_feedback(tracks, refs, {1: 0, 2: 1, 3: 2}, [{
            "section": 1,
            "primary_name": "cell_00001",
            "primary_centroid": [0, 0],
            "secondary_section": 2,
            "secondary_name": "cell_00001",
            "secondary_centroid": [1, 0],
            "verdict": "incorrect",
        }])
        self.assertEqual(applied, 1)
        self.assertEqual(tracks["TrackID"].tolist()[0], 1)
        self.assertNotEqual(tracks["TrackID"].tolist()[1], 1)
        self.assertEqual(tracks["TrackID"].tolist()[1], tracks["TrackID"].tolist()[2])

    def test_correct_link_merges_target_side(self):
        tracks = pd.DataFrame([
            {"FrameID": 0, "Label": 0, "TrackID": 1},
            {"FrameID": 1, "Label": 0, "TrackID": 2},
            {"FrameID": 2, "Label": 0, "TrackID": 2},
        ])
        refs = {
            0: [_Ref(1, Trace("old_a", 0))],
            1: [_Ref(2, Trace("old_b", 1))],
            2: [_Ref(3, Trace("old_b", 2))],
        }
        applied = _apply_link_feedback(tracks, refs, {1: 0, 2: 1, 3: 2}, [{
            "section": 1,
            "primary_name": "old_a",
            "primary_centroid": [0, 0],
            "secondary_section": 2,
            "secondary_name": "old_b",
            "secondary_centroid": [1, 0],
            "verdict": "correct",
        }])
        self.assertEqual(applied, 1)
        self.assertEqual(tracks["TrackID"].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
