import unittest
from pathlib import Path

import pyrecon_connector
from pyrecon_connector.plugin_registry import (
    get_plugin_inventory,
    get_plugin_menu_spec,
)


class PluginRegistryTests(unittest.TestCase):
    def test_inventory_has_unique_ids_and_public_connector_apis(self):
        inventory = get_plugin_inventory()
        identifiers = [plugin["id"] for plugin in inventory]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for plugin in inventory:
            self.assertTrue(plugin["name"])
            self.assertTrue(plugin["repository"].startswith("https://github.com/"))
            for api_name in plugin["connector_apis"]:
                self.assertTrue(callable(getattr(pyrecon_connector, api_name, None)))

    def test_every_menu_action_references_a_registered_plugin(self):
        plugin_ids = {plugin["id"] for plugin in get_plugin_inventory()}
        action_names = []
        for group in get_plugin_menu_spec():
            self.assertTrue(group["attr_name"])
            self.assertTrue(group["text"])
            for action in group["actions"]:
                if action is None:
                    continue
                action_names.append(action["attr_name"])
                self.assertIn(action["plugin_id"], plugin_ids)
                self.assertTrue(action["handler"])
        self.assertEqual(len(action_names), len(set(action_names)))

    def test_tracking_engines_are_imported_only_by_connector_adapters(self):
        root = Path(__file__).parents[1] / "pyrecon_connector"
        hungarian = (root / "hungarian_inapp.py").read_text(encoding="utf-8")
        bayesian = (root / "bayesian_inapp.py").read_text(encoding="utf-8")
        self.assertIn("from tracking_hungarian", hungarian)
        self.assertIn("from tracking_BayesianTransformer", bayesian)


if __name__ == "__main__":
    unittest.main()
