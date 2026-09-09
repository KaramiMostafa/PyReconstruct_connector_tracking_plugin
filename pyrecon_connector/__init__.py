def run_hungarian_tracking_on_series(*args, **kwargs):
    from .hungarian_inapp import run_hungarian_tracking_on_series as _run
    return _run(*args, **kwargs)


def run_bayesian_tracking_on_series(*args, **kwargs):
    from .bayesian_inapp import run_bayesian_tracking_on_series as _run
    return _run(*args, **kwargs)


def run_unet_segmentation_on_section(*args, **kwargs):
    from .segmentation_inapp import run_unet_segmentation_on_section as _run
    return _run(*args, **kwargs)


def run_cellpose_sam_segmentation_on_section(*args, **kwargs):
    from .segmentation_inapp import run_cellpose_sam_segmentation_on_section as _run
    return _run(*args, **kwargs)


def run_multiplex_rna_mapping(*args, **kwargs):
    from .multiplex_mapping import run_multiplex_rna_mapping as _run
    return _run(*args, **kwargs)


def import_multiplex_roi_folders(*args, **kwargs):
    from .multiplex_mapping import import_multiplex_roi_folders as _run
    return _run(*args, **kwargs)


def add_feedback_record(*args, **kwargs):
    from .feedback import add_feedback_record as _run
    return _run(*args, **kwargs)


def add_dapi_link_feedback_batch(*args, **kwargs):
    from .feedback import add_dapi_link_feedback_batch as _run
    return _run(*args, **kwargs)


def add_mapped_roi_feedback_batch(*args, **kwargs):
    from .feedback import add_mapped_roi_feedback_batch as _run
    return _run(*args, **kwargs)


def record_dapi_rename_feedback(*args, **kwargs):
    from .feedback import record_dapi_rename_feedback as _run
    return _run(*args, **kwargs)


def feedback_summary(*args, **kwargs):
    from .feedback import feedback_summary as _run
    return _run(*args, **kwargs)


def export_feedback_csv(*args, **kwargs):
    from .feedback import export_feedback_csv as _run
    return _run(*args, **kwargs)


def write_feedback_review(*args, **kwargs):
    from .feedback import write_feedback_review as _run
    return _run(*args, **kwargs)


def validate_mapped_rna(*args, **kwargs):
    from .multiplex_analysis import validate_mapped_rna as _run
    return _run(*args, **kwargs)


def measure_antibody_intensity(*args, **kwargs):
    from .multiplex_analysis import measure_antibody_intensity as _run
    return _run(*args, **kwargs)


def get_plugin_inventory(*args, **kwargs):
    from .plugin_registry import get_plugin_inventory as _run
    return _run(*args, **kwargs)


def get_plugin_menu_spec(*args, **kwargs):
    from .plugin_registry import get_plugin_menu_spec as _run
    return _run(*args, **kwargs)


def audit_plugin_assembly(*args, **kwargs):
    from .plugin_registry import audit_plugin_assembly as _run
    return _run(*args, **kwargs)


__all__ = [
    "run_hungarian_tracking_on_series",
    "run_bayesian_tracking_on_series",
    "run_unet_segmentation_on_section",
    "run_cellpose_sam_segmentation_on_section",
    "run_multiplex_rna_mapping",
    "import_multiplex_roi_folders",
    "add_feedback_record",
    "add_dapi_link_feedback_batch",
    "add_mapped_roi_feedback_batch",
    "record_dapi_rename_feedback",
    "feedback_summary",
    "export_feedback_csv",
    "write_feedback_review",
    "validate_mapped_rna",
    "measure_antibody_intensity",
    "get_plugin_inventory",
    "get_plugin_menu_spec",
    "audit_plugin_assembly",
]
