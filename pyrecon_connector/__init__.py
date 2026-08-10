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


__all__ = [
    "run_hungarian_tracking_on_series",
    "run_bayesian_tracking_on_series",
    "run_unet_segmentation_on_section",
    "run_cellpose_sam_segmentation_on_section",
]
