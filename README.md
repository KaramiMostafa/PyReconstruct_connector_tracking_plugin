# PyReconstruct microscopy connector

This repository provides the tracking, segmentation, multiplex mRNA mapping,
expert-feedback, validation, and intensity-analysis backend used by the
customized [PyReconstruct](https://github.com/KaramiMostafa/PyReconstruct/tree/cellpose-custom-model)
branch.

The connector owns the machine-readable plugin registry consumed by the
customized host menu. See
[the connector architecture](ARCHITECTURE.md) and
[the host assembly contract](https://github.com/KaramiMostafa/PyReconstruct/blob/cellpose-custom-model/PLUGINS.md).

Most users should follow the complete macOS, Linux, or Windows instructions in
the [customized PyReconstruct installation guide](https://github.com/KaramiMostafa/PyReconstruct/blob/cellpose-custom-model/readme.md).

## Requirements

- Python 3.11
- The customized host, connector, and desired tracking-engine repositories
  cloned beside one another
- `cellpose-custom-model` for the host/connector and `main` for the engines
- PyTorch and torchvision compatible with the selected operating system

The Hungarian and Bayesian engines remain separate installable packages. This
connector imports them and adapts their ROI-table interfaces to PyReconstruct;
the PyReconstruct host never imports them directly.

## Developer installation

From the parent directory containing all four repositories:

```bash
conda create -n pyreconstruct-custom python=3.11 pip setuptools wheel -y
conda activate pyreconstruct-custom
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./PyReconstruct
python -m pip install -e ./PyReconstruct_Tracking_PlugIn_Hungarian
python -m pip install -e ./PyReconstruct_Tracking_PlugIn_BayesianTransformer
python -m pip install -e ./PyReconstruct_connector_tracking_plugin
```

On Windows PowerShell, the final paths may be written as
`.\PyReconstruct` and `.\PyReconstruct_connector_tracking_plugin`.

Verify the editable import:

```bash
python -c "import pyrecon_connector; print(pyrecon_connector.__file__)"
python -c "from pyrecon_connector import audit_plugin_assembly; import pprint; pprint.pp(audit_plugin_assembly())"
```

The path must point into this clone.

## Backend modules

| Module | Responsibility |
|---|---|
| `plugin_registry.py` | Authoritative plugin inventory and host-menu descriptors |
| `segmentation_inapp.py` | U-Net and Cellpose-SAM segmentation from selected image channels |
| `hungarian_inapp.py` | Centroid/area-based one-to-one DAPI tracking |
| `bayesian_inapp.py` | Bayesian Transformer DAPI tracking |
| `multiplex_mapping.py` | Fiji ROI import and DAPI-guided mRNA propagation |
| `feedback.py` | Auditable expert corrections and review reports |
| `multiplex_analysis.py` | Expert-ROI validation and antibody-intensity analysis |
| `connector.py` | Legacy command-line `.jser` tracking connector |

## In-app workflow

1. Import or segment DAPI and anchor-mRNA ROIs with distinct prefixes/groups.
2. Track DAPI nuclei with the Hungarian or Bayesian command, using the
   `multiplex_dapi` source group so anchor mRNA ROIs cannot enter tracking.
3. Map anchor mRNA ROIs through the DAPI tracks using
   `multiplex_tracked_dapi` for DAPI and `multiplex_rna_anchor` for mRNA.
4. Review DAPI and mRNA results separately. Tracked DAPI and mapped RNA both
   support persistent multicolor selection across sections.
5. Record expert feedback only after checking every selected identity and
   section. One verdict can be applied to one or more selected mapped RNA ROIs.
6. Re-run tracking or mapping with **Apply saved expert feedback** enabled.
7. Optionally validate mapped ROIs and measure antibody intensity.

Missing image sections, anchors, or target DAPI ROIs are reported instead of
terminating the entire analysis. If the associated DAPI identity is missing
but common neighboring DAPI tracks remain, the mapper produces a
low-confidence local-translation fallback for mandatory review.

Mapping never edits an anchor mRNA ROI and never warps its contour. Every
generated ROI is an exact shape-preserving copy translated by one DAPI-derived
vector. If that translation would cross an image edge, the unsafe mapped copy
is skipped and reported instead of being clipped, warped, or repositioned.
Re-run with **Replace previously generated mapped ROIs** enabled to replace
older generated results; anchor mRNA ROIs remain untouched.

Mapping ranges may include their anchor section for convenience. For example,
`6:4-12` maps to sections 4, 5, and 7–12; section 6 is automatically retained
only as the anchor.

Expert feedback is stored beside the `.jser` as
`<series>.multiplex_feedback.json`. It applies explicit link, unlink, force,
and exclude constraints; it does not silently retrain a neural network from a
single click.

## Tests

With the host, connector, and tracking engines installed:

```bash
python -m unittest discover -v tests
```

The suite covers multichannel segmentation inputs, missing-section handling,
feedback persistence, DAPI link corrections, mRNA association constraints,
displacement fallback, expert validation, and antibody-intensity outputs.

## Conventional Commits

Every commit unique to `cellpose-custom-model` follows
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
type(scope): description
```

Examples:

```text
feat(mapping): add local-translation fallback
fix(tracking): preserve reviewed DAPI links
test(feedback): cover forced mRNA associations
docs(install): link platform setup guide
```

Enable the local validator once per clone:

```bash
git config core.hooksPath .githooks
```

GitHub Actions runs the same validation for the customized branch. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-3.0. See [LICENSE](LICENSE).
