import unittest

import pandas as pd

from pyrecon_connector.hungarian_inapp import (
    _Ref,
    _apply_link_feedback,
    _nearest_ref_index,
    _trace_matches_source_group,
)


class Trace:
    def __init__(self, name, x, tags=()):
        self.name = name
        self.x = x
        self.tags = set(tags)

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
        stats = {}
        applied = _apply_link_feedback(tracks, refs, {1: 0, 2: 1, 3: 2}, [{
            "section": 1,
            "primary_name": "cell_00001",
            "primary_centroid": [0, 0],
            "secondary_section": 2,
            "secondary_name": "cell_00001",
            "secondary_centroid": [1, 0],
            "verdict": "incorrect",
        }], stats=stats)
        self.assertEqual(applied, 1)
        self.assertEqual(stats, {"evaluated": 1, "applied": 1})
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

    def test_correct_link_swaps_occupied_target_side_tracks(self):
        tracks = pd.DataFrame([
            {"FrameID": 0, "Label": 0, "TrackID": 1},
            {"FrameID": 0, "Label": 1, "TrackID": 2},
            {"FrameID": 1, "Label": 0, "TrackID": 1},
            {"FrameID": 1, "Label": 1, "TrackID": 2},
            {"FrameID": 2, "Label": 0, "TrackID": 1},
            {"FrameID": 2, "Label": 1, "TrackID": 2},
        ])
        refs = {
            0: [_Ref(13, Trace("cell_00168", 10)), _Ref(13, Trace("cell_00177", 20))],
            1: [_Ref(14, Trace("cell_00168", 20)), _Ref(14, Trace("cell_00177", 10))],
            2: [_Ref(15, Trace("cell_00168", 10)), _Ref(15, Trace("cell_00177", 20))],
        }
        applied = _apply_link_feedback(tracks, refs, {13: 0, 14: 1, 15: 2}, [{
            "section": 13,
            "primary_name": "cell_00168",
            "primary_centroid": [10, 0],
            "secondary_section": 14,
            "secondary_name": "cell_00177",
            "secondary_centroid": [10, 0],
            "verdict": "correct",
        }])
        self.assertEqual(applied, 1)
        self.assertEqual(tracks["TrackID"].tolist(), [1, 2, 2, 1, 2, 1])

    def test_saved_centroid_wins_when_track_name_was_reassigned(self):
        refs = [
            _Ref(14, Trace("cell_00177", 50)),
            _Ref(14, Trace("renamed_after_rerun", 10)),
        ]
        self.assertEqual(_nearest_ref_index(refs, "cell_00177", [10, 0]), 1)

    def test_tracked_dapi_source_group_excludes_rna_trace_name_collision(self):
        dapi = Trace("cell_00168", 0, {"multiplex_dapi"})
        rna = Trace("cell_00168", 0, {"multiplex_rna_anchor"})
        self.assertTrue(_trace_matches_source_group(dapi, "multiplex_tracked_dapi"))
        self.assertFalse(_trace_matches_source_group(rna, "multiplex_tracked_dapi"))


if __name__ == "__main__":
    unittest.main()
