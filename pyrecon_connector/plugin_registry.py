"""Authoritative catalog for the customized PyReconstruct plugin menu.

The host application consumes only the menu descriptors in this module. Plugin
handlers call the public :mod:`pyrecon_connector` facade, and this connector is
the only layer that imports independent algorithm packages.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module, util


PLUGIN_INVENTORY = (
    {
        "id": "expert_feedback",
        "name": "Human-in-the-loop expert feedback",
        "connector_apis": (
            "add_feedback_record",
            "add_dapi_link_feedback_batch",
            "add_mapped_roi_feedback_batch",
            "record_dapi_rename_feedback",
            "write_feedback_review",
        ),
        "engine_packages": (),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_connector_tracking_plugin",
    },
    {
        "id": "hungarian_tracking",
        "name": "Hungarian DAPI tracking",
        "connector_apis": ("run_hungarian_tracking_on_series",),
        "engine_packages": ("tracking_hungarian",),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_Tracking_PlugIn_Hungarian",
    },
    {
        "id": "bayesian_tracking",
        "name": "Bayesian Transformer DAPI tracking",
        "connector_apis": ("run_bayesian_tracking_on_series",),
        "engine_packages": ("tracking_BayesianTransformer",),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_Tracking_PlugIn_BayesianTransformer",
    },
    {
        "id": "unet_segmentation",
        "name": "Pre-trained U-Net segmentation",
        "connector_apis": ("run_unet_segmentation_on_section",),
        "engine_packages": ("torch",),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_connector_tracking_plugin",
    },
    {
        "id": "cellpose_sam",
        "name": "Cellpose-SAM segmentation",
        "connector_apis": ("run_cellpose_sam_segmentation_on_section",),
        "engine_packages": ("cellpose", "torch", "torchvision"),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_connector_tracking_plugin",
    },
    {
        "id": "multiplex_mapping",
        "name": "DAPI-guided mRNA ROI mapping",
        "connector_apis": ("import_multiplex_roi_folders", "run_multiplex_rna_mapping"),
        "engine_packages": ("numpy", "roifile"),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_connector_tracking_plugin",
    },
    {
        "id": "multiplex_analysis",
        "name": "Mapped-ROI validation and antibody analysis",
        "connector_apis": ("validate_mapped_rna", "measure_antibody_intensity"),
        "engine_packages": ("numpy", "matplotlib"),
        "repository": "https://github.com/KaramiMostafa/PyReconstruct_connector_tracking_plugin",
    },
)


def _action(attr_name: str, text: str, handler: str, plugin_id: str) -> dict:
    return {
        "attr_name": attr_name,
        "text": text,
        "handler": handler,
        "plugin_id": plugin_id,
    }


PLUGIN_MENU = (
    {
        "attr_name": "expertfeedbackpluginmenu",
        "text": "⚠ Expert feedback (HIGH RISK)",
        "actions": (
            _action("feedback_warning_act", "⚠ Read safety warning first...", "showExpertFeedbackWarning", "expert_feedback"),
            None,
            _action("feedback_dapi_review_act", "Open DAPI track-link review panel...", "openDAPITrackFeedbackPanel", "expert_feedback"),
            _action("feedback_rna_dapi_act", "Mark selected mRNA ↔ DAPI pair...", "recordRNADAPIFeedback", "expert_feedback"),
            _action("feedback_mapped_rna_act", "Mark selected mapped mRNA ROI...", "recordMappedRNAFeedback", "expert_feedback"),
            None,
            _action("feedback_report_act", "Create separate ROI/track review report...", "generateExpertFeedbackReport", "expert_feedback"),
        ),
    },
    {
        "attr_name": "trackingpluginmenu",
        "text": "Tracking",
        "actions": (
            _action("run_hungarian_tracking_act", "Hungarian tracking...", "runHungarianTracking", "hungarian_tracking"),
            _action("run_bayesian_tracking_act", "Bayesian Transformer tracking...", "runBayesianTracking", "bayesian_tracking"),
            None,
            _action("configure_linked_dapi_highlight_act", "Multicolor DAPI/mapped-RNA selection...", "configureLinkedDAPIHighlight", "expert_feedback"),
        ),
    },
    {
        "attr_name": "segmentationpluginmenu",
        "text": "Segmentation",
        "actions": (
            _action("run_unet_segmentation_act", "Pre-trained U-Net microscopy...", "runUnetSegmentation", "unet_segmentation"),
            _action("run_cellpose_sam_segmentation_act", "Cellpose-SAM...", "runCellposeSAMSegmentation", "cellpose_sam"),
        ),
    },
    {
        "attr_name": "channelpluginmenu",
        "text": "Channels / overlays",
        "actions": (
            _action("configure_image_channels_act", "Channels tool...", "configureImageChannels", "cellpose_sam"),
            _action("configure_roi_overlay_act", "ROI Manager-style layers and labels...", "configureROIOverlay", "multiplex_mapping"),
        ),
    },
    {
        "attr_name": "multiplexmappingpluginmenu",
        "text": "Multiplex mRNA mapping",
        "actions": (
            _action("import_multiplex_rois_act", "1. Import DAPI/mRNA ROI folders...", "importMultiplexROIFolders", "multiplex_mapping"),
            _action("run_multiplex_mapping_act", "2. Map mRNA through DAPI tracks...", "runMultiplexRNAMapping", "multiplex_mapping"),
            _action("review_multiplex_mapping_act", "3. Review mapped mRNA...", "reviewMultiplexMappings", "multiplex_mapping"),
            _action("validate_multiplex_mapping_act", "4. Validate against expert ROIs...", "validateMultiplexMappings", "multiplex_analysis"),
            _action("measure_multiplex_intensity_act", "5. Measure antibody intensity...", "measureMultiplexAntibodyIntensity", "multiplex_analysis"),
        ),
    },
)


def get_plugin_inventory() -> list[dict]:
    """Return a copy of the registered plugin capabilities."""
    return deepcopy(list(PLUGIN_INVENTORY))


def get_plugin_menu_spec() -> list[dict]:
    """Return menu descriptors without importing PySide or PyReconstruct."""
    return deepcopy(list(PLUGIN_MENU))


def audit_plugin_assembly() -> dict:
    """Report connector API and engine-package availability per plugin."""
    facade = import_module("pyrecon_connector")
    plugins = []
    for registered in PLUGIN_INVENTORY:
        missing_apis = [
            name for name in registered["connector_apis"]
            if not callable(getattr(facade, name, None))
        ]
        missing_packages = [
            name for name in registered["engine_packages"]
            if util.find_spec(name) is None
        ]
        plugins.append({
            **deepcopy(registered),
            "missing_connector_apis": missing_apis,
            "missing_engine_packages": missing_packages,
            "available": not missing_apis and not missing_packages,
        })
    return {
        "ok": all(plugin["available"] for plugin in plugins),
        "plugins": plugins,
    }
