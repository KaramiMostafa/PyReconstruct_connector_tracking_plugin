# Connector architecture

This repository is the adapter boundary between PyReconstruct and optional
microscopy capabilities.

```text
Customized PyReconstruct host
        │ public pyrecon_connector API only
        ▼
Connector registry and adapters
        │ ROI tables / model inputs
        ├── Hungarian tracking core
        ├── Bayesian Transformer tracking core
        ├── Cellpose and PyTorch
        └── connector-native EM registration, mapping, feedback, and analysis
```

`pyrecon_connector/plugin_registry.py` is the authoritative plugin list. It
provides menu descriptors to the customized host and records the public
connector APIs, engine import names, and repositories for each capability.

The separation has three purposes:

1. PyReconstruct-specific trace, section, dialog, and write-back behavior stays
   in the host/connector boundary.
2. Reusable tracking engines accept ordinary ROI tables and contain no
   PyReconstruct GUI dependency.
3. Adding or removing a plugin does not add an algorithm import to the
   upstream application core.

Run the installed-assembly audit with:

```bash
python -c "from pyrecon_connector import audit_plugin_assembly; import pprint; pprint.pp(audit_plugin_assembly())"
```

An unavailable engine is reported against its plugin without preventing the
base PyReconstruct application from starting.

EM registration uses `registration.py` for file validation, CSV adaptation, and
review exports, with the GUI-independent numerical TPS/resampling functions in
`registration_core.py`. The host owns only the file-selection/review dialog and
background worker and calls the public `run_em_registration` facade. It does
not modify open-series affine transforms to represent a nonlinear warp.
