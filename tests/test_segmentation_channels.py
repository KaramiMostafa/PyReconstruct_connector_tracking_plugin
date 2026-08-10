import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from pyrecon_connector.segmentation_inapp import (
    _normalize_cellpose_input,
    _select_cellpose_channels,
)


class CellposeChannelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.section = object()
        self.available = [np.full((3, 4), i, dtype=np.uint16) for i in range(4)]
        module = types.ModuleType("PyReconstruct.modules.backend.view.channel_utils")
        module.read_section_channels = lambda section: self.available
        self.modules = {
            "PyReconstruct": types.ModuleType("PyReconstruct"),
            "PyReconstruct.modules": types.ModuleType("PyReconstruct.modules"),
            "PyReconstruct.modules.backend": types.ModuleType("PyReconstruct.modules.backend"),
            "PyReconstruct.modules.backend.view": types.ModuleType("PyReconstruct.modules.backend.view"),
            "PyReconstruct.modules.backend.view.channel_utils": module,
        }
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()

    def test_selects_up_to_three_channels_in_requested_order(self):
        result = _select_cellpose_channels(self.section, channels=[2, 0, 3])

        self.assertEqual(result.shape, (3, 4, 3))
        np.testing.assert_array_equal(result[..., 0], self.available[2])
        np.testing.assert_array_equal(result[..., 1], self.available[0])
        np.testing.assert_array_equal(result[..., 2], self.available[3])

    def test_single_channel_api_remains_backward_compatible(self):
        result = _select_cellpose_channels(self.section, channel=1)

        np.testing.assert_array_equal(result, self.available[1])

    def test_rejects_invalid_channel_selections(self):
        for channels in ([], [0, 1, 2, 3], [4]):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                _select_cellpose_channels(self.section, channels=channels)

    def test_normalizes_each_channel_independently(self):
        image = np.stack(
            [
                np.arange(100, dtype=np.float32).reshape(10, 10),
                np.arange(100, dtype=np.float32).reshape(10, 10) * 100,
            ],
            axis=-1,
        )

        result = _normalize_cellpose_input(image)

        np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
