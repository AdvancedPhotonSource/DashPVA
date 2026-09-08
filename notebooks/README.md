# DashPVA Notebooks

Runnable documentation for the DashPVA Python API. Every code cell has its own
markdown cell explaining what it does and why, so these read as documentation and
execute as demos.

| Notebook | Covers |
|---|---|
| [DashAnalysis_Quickstart.ipynb](DashAnalysis_Quickstart.ipynb) | Slicing, line cuts and volume rendering with `DashAnalysis` |
| [RSM_Gridder.ipynb](RSM_Gridder.ipynb) | Building gridded HKL volumes, offline and live |
| [DashPVA_Tools_Tour.ipynb](DashPVA_Tools_Tour.ipynb) | File I/O, masking, reciprocal-space conversion, settings |

## Setup

1. Install DashPVA (if not already done):
   ```bash
   cd DashPVA
   bash install.sh
   ```

2. Register the DashPVA environment as a Jupyter kernel:
   ```bash
   source .venv/bin/activate
   pip install ipykernel ipympl
   python -m ipykernel install --user --name DashPVA --display-name "DashPVA"
   ```

3. Open a notebook and select the **DashPVA** kernel:
   ```bash
   jupyter notebook notebooks/DashAnalysis_Quickstart.ipynb
   ```

All dependencies (numpy, h5py, matplotlib, pyvista, etc.) are installed by DashPVA.
`ipympl` is needed only for the interactive line cut, which uses `%matplotlib widget`.

## Usage

In Python and notebooks, import the package as lowercase `dashpva`:

```python
from dashpva.utils import DashAnalysis

da = DashAnalysis()
data = da.load_data("your_data.h5")
```

> **Note:** The CLI command is `DashPVA` (mixed case), but the Python import is
> `dashpva` (lowercase).

Each notebook has a filename constant near the top — set it to one of your own
scans. Cells that need a real file or live hardware report and continue rather than
raising, so a notebook runs top to bottom before you have data in place.

## What each notebook covers

### DashAnalysis_Quickstart.ipynb

- Loading HDF5 data and inspecting metadata
- 3D point cloud visualization
- 2D slicing — canonical planes, custom normals, explicit HKL axes, thick slabs
- Line cuts — presets, custom endpoints, width averaging, interactive mode
- Volume creation and display, and the image-based workflow

### RSM_Gridder.ipynb

- Memory budgeting before a build (`estimate_grid_memory`, `ensure_memory_available`)
- Bounds discovery versus fixing bounds for comparable volumes
- `build_volume` with detector masking and monitor normalisation
- Merging several scans, and the energy/UB consistency check
- The live grid: why bounds are fixed before the first frame, frame-by-frame
  accumulation, previews, and the geometry fingerprint that stops a run

### DashPVA_Tools_Tour.ipynb

- `dashpva.settings` — which configuration is live, switching profile
- `HDF5Loader` — inspecting, loading and writing files
- `MaskManager` — hot and dead pixel detection, combining and persisting masks
- `RSMConverter` — reading a file's geometry, converting angles to Q
- A combined offline pass feeding into `DashAnalysis`

## A note on gridding

Both `DashAnalysis.create_vol` and `rsm_gridder.build_volume` produce volumes, but
they are not the same tool. `create_vol` interpolates a point cloud you already
have in memory — quick, good for a look. `build_volume` is the full offline
pipeline: it reads scan files, discovers bounds, applies masks and monitor
normalisation, budgets memory, and can merge several scans into one grid. The RSM
Volume Builder runs the latter.

Voxels that nothing scattered into are left `NaN`, not zero — unmeasured is not the
same as measured zero. Use `finite_intensity_range()` for colour limits, or the
NaNs will flatten the scale.
