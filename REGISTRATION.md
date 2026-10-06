# EM registration

Open **Plug-In → Registration → EM images + masks + landmarks…** in customized
PyReconstruct. Select a fixed/reference EM TIFF, a moving EM TIFF, their two
mask TIFFs, a landmark CSV, the fixed/moving slice numbers, and an output parent
folder. Both masks can be omitted for an images-only run. Inputs are single 2D
grayscale TIFFs; stacks and RGB images must first be exported as grayscale planes.

This adapts the landmark thin-plate-spline (TPS) method in
`mainEMToEM_Ostroff.py` to a selectable **pair** of sections. It does not run
the original ten-section serial loop automatically. SciPy supplies TPS fitting
so the existing PyReconstruct scikit-image pin does not need to change.

## Landmarks

Use the original long table (an unnamed first column is also accepted):

```csv
point,slice,X,Y
p1,1,10,10
p1,2,12,11
p2,1,80,10
p2,2,82,11
p3,1,10,80
p3,2,12,81
```

Alternatively use paired columns:

```csv
point,fixed_x,fixed_y,moving_x,moving_y
p1,10,10,12,11
p2,80,10,82,11
p3,10,80,12,81
```

Landmarks are matched by ID, never by row order. At least three distinct,
non-collinear shared points are required on both images; more points spread
across the field provide local deformation information. Three points alone
only determine an affine mapping. Unmatched IDs are reported and excluded.
Duplicate IDs/positions, invalid numbers, and out-of-image points are rejected.

Coordinates refer to **raw TIFF pixels**: X is the column and Y is the row.
The default is zero-based, top-left origin, as in the supplied script.
Bottom-left and one-based pixel coordinates can be selected explicitly.
PyReconstruct's aligned coordinates in micrometres are not accepted directly.
The plugin does not guess coordinate conventions or apply the open series'
alignment to these file inputs.

## Images and masks

The fixed image defines the output canvas. A TPS fit from fixed to moving
landmarks supplies the inverse sampling map for both the moving image and its
mask. Images use bilinear interpolation; masks use nearest-neighbor sampling,
preserving original label IDs and dtype. Masks must have the same dimensions
as their corresponding image, with nonnegative integer labels and 0 for
background. They are carried through registration, **not used as fitting
weights**. The fixed image and fixed mask are copied unchanged.

TPS intentionally deforms image/mask contours. This is different from the
shape-preserving RNA mapping plugin. Areas outside the moving image are filled
with zero and recorded in `coverage.tif`; extrapolation far from landmarks
requires visual review. Images may have different dimensions. Output pixel
spacing is the fixed image's spacing: enter that value when creating a new
PyReconstruct series. The plugin does not infer physical calibration.

## Results and review

Every run creates a unique `em_registration_…` folder containing:

- `images/`: the reference TIFF and registered moving TIFF, named by slice.
- `masks/`: corresponding fixed and registered masks, when supplied.
- `preview.png`: fixed/moving overlays before and after, plus a mask overlay.
  Fixed is magenta, moving is green. Yellow circles are fixed landmarks;
  cyan crosses in the before panel are the moving landmarks. These preview
  images are contrast-scaled for display only.
- `registration.json`: input paths, coordinate convention, actual matched
  coordinates, fitting parameters, coverage, and review notices.
- `landmark_residuals.csv`: forward and inverse landmark fitting residuals.
- `coverage.tif`: 1 where the moving image supplies output pixels; 0 outside.

Residuals use the fitting landmarks and are **not independent accuracy
measurements**. Foreground-mask Dice is a descriptive overlap statistic; label
IDs are not assumed to correspond between sections.

Review the preview and full-resolution results, then use PyReconstruct's
**File → New → From images…** workflow with the result's `images/` folder.
The nonlinear transform cannot be stored in PyReconstruct's affine alignment
field. Existing project images, alignments, and vector ROIs are not replaced
or transformed. Masks remain TIFFs in their own folder for subsequent use.

Registration runs in a background worker. Cancellation is checked between
processing blocks and before publishing outputs. Failed/cancelled runs clean
up only their own temporary directory; existing runs and source files remain.

Python API (no GUI dependency):

```python
from pyrecon_connector import run_em_registration

result = run_em_registration(
    fixed_image="section1.tif", fixed_mask="mask1.tif",
    moving_image="section2.tif", moving_mask="mask2.tif",
    landmarks_csv="points.csv", fixed_section=1, moving_section=2,
    output_dir="results",  # existing parent folder
)
print(result["output_dir"])
```
