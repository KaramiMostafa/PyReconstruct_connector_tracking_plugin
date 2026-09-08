# PyReconstruct microscopy connector

This repository provides the tracking, segmentation, multiplex mRNA mapping,
expert-feedback, validation, and intensity-analysis backend used by the
customized [PyReconstruct](https://github.com/KaramiMostafa/PyReconstruct/tree/cellpose-custom-model)
branch.

Most users should follow the complete macOS, Linux, or Windows instructions in
the [customized PyReconstruct installation guide](https://github.com/KaramiMostafa/PyReconstruct/blob/cellpose-custom-model/readme.md).

## Requirements

- Python 3.11
- The customized PyReconstruct and connector repositories cloned beside one
  another
- The `cellpose-custom-model` branch in both repositories
- PyTorch and torchvision compatible with the selected operating system

The package metadata installs NumPy, pandas, SciPy, scikit-image, Cellpose,
PyTorch, torchvision, ROI/TIFF readers, Matplotlib, and the tracking modules
contained in this repository.

## Developer installation

From the parent directory containing both repositories:

```bash
conda create -n pyreconstruct-custom python=3.11 pip setuptools wheel -y
conda activate pyreconstruct-custom
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./PyReconstruct
python -m pip install -e ./PyReconstruct_connector_tracking_plugin
```

On Windows PowerShell, the final paths may be written as
`.\PyReconstruct` and `.\PyReconstruct_connector_tracking_plugin`.

Verify the editable import:

```bash
python -c "import pyrecon_connector; print(pyrecon_connector.__file__)"
```

The path must point into this clone.

## Backend modules

| Module | Responsibility |
|---|---|
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
4. Review DAPI and mRNA results separately.
5. Record expert feedback only after checking both identities and sections.
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

With both editable repositories installed:

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
feat(mapping): add local-field fallback
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
