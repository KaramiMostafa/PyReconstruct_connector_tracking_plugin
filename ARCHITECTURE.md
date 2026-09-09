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
        └── connector-native mapping, feedback, and analysis
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
