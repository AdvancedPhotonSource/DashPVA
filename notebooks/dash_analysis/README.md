# DashAnalysis Quickstart

Interactive notebook for exploring HKL point cloud data using the DashAnalysis API from DashPVA.

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

3. Open the notebook and select the **DashPVA** kernel:
   ```bash
   jupyter notebook notebooks/dash_analysis/DashAnalysis_Quickstart.ipynb
   ```

All dependencies (numpy, h5py, matplotlib, pyvista, etc.) are already installed by DashPVA.

## Usage

In Python/notebooks, import the package as lowercase `dashpva`:

```python
from dashpva.utils import DashAnalysis

da = DashAnalysis()
data = da.load_data("your_data.h5")
```

> **Note:** The CLI command is `DashPVA` (mixed case), but the Python import is `dashpva` (lowercase).

## What's covered

- Loading HDF5 data and inspecting metadata
- 3D point cloud visualization
- 2D slicing (canonical planes, custom normals, custom HKL axes)
- Line cuts (presets, custom endpoints, interactive mode)
- Volume creation and visualization
