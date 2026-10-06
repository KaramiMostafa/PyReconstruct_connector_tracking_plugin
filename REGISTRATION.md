# Serial-section and paired EM registration

## Serial stack (default)

Open **Plug-In → Registration → Serial sections / EM landmarks…** and select
**Serial section stack (chained registration)**. Choose the folder containing
numbered TIFF planes, the landmark CSV, an output parent folder, and TPS (the
default) or affine registration. No fixed/moving pair selection is needed.

For the supplied dataset use `Mostafa/EM_10` and
`correspondence points underwood 1 to 10.csv`. Filenames such as
`_underwood.1.tif` through `_underwood.10.tif` are sorted **numerically**.
The lowest-numbered section is copied unchanged as the reference. Any number
of sections is supported. The predecessor is the preceding available image
in numeric order; section IDs do not have to start at 1 or be consecutive.
Every CSV slice must have an image, and every image must have landmarks.

The serial CSV is `point,slice,x,y`; `X,Y` and the prototype's unnamed point
column are accepted. IDs, never CSV row order, establish correspondence.
For each section:

1. Match its IDs to the preceding section's **registered** landmarks.
2. Fit an inverse map from those registered points to current original points.
3. Warp the current **original** image once onto the first image's canvas.
4. Fit a forward map from current original points to the registered reference.
5. Transform **all** current landmarks, including IDs absent from the preceding
   section, and use them as the next section's reference.

Thus Slice 3 uses transformed Slice 2 coordinates, not original Slice 2
coordinates and not a direct match against Slice 1. Missing IDs are excluded
only from the current fit. Raw landmarks must be inside their input image;
transformed landmarks outside the reference canvas are retained, not clipped.
Each adjacent pair needs at least three distinct non-collinear points. Global
TPS currently permits up to 1000 landmarks per section.

TPS uses SciPy's thin-plate-spline RBF with degree 1 and zero smoothing, avoiding
a change to the host's scikit-image version. Separate forward/inverse fits
follow `mainEMToEM_Ostroff.py`; a forward TPS fit is not assumed to be the exact
inverse away from fitting landmarks. Serial image resampling uses floating-point
bilinear interpolation, zero padding (including interpolation at the boundary),
then conversion to the original image dtype, as in the prototype. Affine mode
fits a forward least-squares map and uses its matrix inverse for resampling.

Each successful run creates `serial_registration_<timestamp>_<id>/`:

- `images/section_0001.tif`, etc.: complete registered stack on the first canvas.
- `registered_landmarks.csv`: all propagated coordinates, plus original points
  converted to zero-based top-left pixels, for every section.
- `registration.json`: section order, method, original matched points, registered
  reference points, unmatched IDs, fitting residuals, and canvas coverage.
- `previews/`: before/after overlays against each preceding registered section.
- `coverage/`: per-section masks identifying output pixels within the input image.

The API also returns all registered landmarks. Images may have different
dimensions; output dimensions always match the first image. Original image
dtypes are preserved. There are no masks in serial mode; optional label-mask
warping remains available in pair mode. Failed slices are identified by number,
processing stops, and no partial stack is published. Cancellation removes only
the current temporary run. Inputs and previous results remain unchanged.

```python
from pyrecon_connector import run_serial_registration

result = run_serial_registration(
    image_series="EM_10",
    landmarks_csv="EM_10/correspondence points underwood 1 to 10.csv",
    output_dir="results",  # existing parent folder
    method="tps",         # or "affine"
)
print(result["files"]["images"])
```

For non-numbered filenames, the API accepts an explicit mapping such as
`image_series={1: "first.tif", 2: "second.tif"}`. Review the stack before use:
sequential registration can accumulate drift, and small fitting residuals alone
do not establish accuracy. Import the output `images/` folder as a new series,
using the first section's pixel size.

## Optional single-pair mode

In the same dialog, select **Single fixed/moving pair (optional masks)**.
Select a fixed/reference EM TIFF, a moving EM TIFF, their two
mask TIFFs, a landmark CSV, the fixed/moving slice numbers, and an output parent
folder. Both masks can be omitted for an images-only run. Inputs are single 2D
grayscale TIFFs; stacks and RGB images must first be exported as grayscale planes.

This mode handles one selected pair using TPS, including optional label masks.

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
