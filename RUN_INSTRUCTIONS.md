# Run Instructions for DashPVA

This guide provides step-by-step instructions to set up, configure, and run the DashPVA application.

---

## Setup

### Install Dependencies
Using the [environment.yml](environment.yml) file, you can install the environment using the conda command:
```bash
conda env create -f environment.yml
```

Instead of using the `environment.yml` file, follow these manual instructions to set up the environment:

1. Create a new Conda environment:
   ```bash
   conda create -n DashPVA python=3.11 numpy pyqt pyqtgraph xrayutilities h5py toml
   ```
2. Activate the environment:
   ```bash
   conda activate DashPVA
   ```
3. Install additional dependencies:
   ```bash
   conda install -c apsu pvapy
   pip install pyepics
   ```

### Verify Installation
Ensure all dependencies are installed correctly:
```bash
conda list
```

---

## Running the Application

### 1. Configuration GUI (ConfigDialog)
The configuration GUI is used to set up detector prefixes, collector addresses, and PV configurations.

**Run Command**:
```bash
python area_det_viewer.py
```

**Key Features**:
- Set PVA prefix and collector address.
- Load, edit, or create PV configuration files.
- Input caching frequency for live view.

### 2. Live Viewer GUI (ImageWindow)
The live image visualization GUI allows users to:
- Stream live images from a PVA source.
- View and manipulate regions of interest (ROIs).
- Monitor statistics for live analysis.

**Run Command**:
```bash
python area_det_viewer.py
```

**GUI Features**:
- **Start/Stop Live View**: Begin or end live image streaming.
- **ROI Tools**: Add, view, and manipulate ROIs on the displayed image.
- **Statistical Monitoring**: View and log key metrics from the live feed.
- **Frame-by-Frame Processing**: Supports both predetermined and spontaneous scan modes.

---

## Configuration Files

### Location
All configuration files are stored in the `pv_configs/` directory.

### Example File
Below is an example configuration file (`example_config.toml`):
```toml
# Required Setup
CONSUMER_TYPE = "spontaneous"

# Section used specifically for Metadata Pvs
[METADATA]

    [METADATA.CA]
    x = "x"
    y = "y"

    [METADATA.PVA]

# Section specifically for ROI PVs
[ROI]

    [ROI.ROI1]
    MIN_X = "dp-ADSim:ROI1:MinX"
    MIN_Y = "dp-ADSim:ROI1:MinY"
    SIZE_X = "dp-ADSim:ROI1:SizeX"
    SIZEY = "dp-ADSim:ROI1:SizeY"

    [ROI.ROI2]
    MIN_X = "dp-ADSim:ROI2:MinX"
    MIN_Y = "dp-ADSim:ROI2:MinY"
    SIZE_X = "dp-ADSim:ROI2:SizeX"
    SIZE_Y = "dp-ADSim:ROI2:SizeY"

    [ROI.ROI3]
    MIN_X = "dp-ADSim:ROI3:MinX"
    MIN_Y = "dp-ADSim:ROI3:MinY"
    SIZE_X = "dp-ADSim:ROI3:SizeX"
    SIZE_Y = "dp-ADSim:ROI3:SizeY"

    [ROI.ROI4]
    MIN_X = "dp-ADSim:ROI4:MinX"
    MIN_Y = "dp-ADSim:ROI4:MinY"
    SIZE_X = "dp-ADSim:ROI4:SizeX"
    SIZE_Y = "dp-ADSim:ROI4:SizeY"

[STATS]

[STATS.STATS1]
    TOTAL = "dp-ADSim:Stats1:Total_RBV"
    MIN = "dp-ADSim:Stats1:MinValue_RBV"
    MAX = "dp-ADSim:Stats1:MaxValue_RBV"
    SIGMA = "dp-ADSim:Stats1:Sigma_RBV"
    MEAN = "dp-ADSim:Stats1:MeanValue_RBV"

    [STATS.STATS4]
    TOTAL = "dp-ADSim:Stats4:Total_RBV"
    MIN = "dp-ADSim:Stats4:MinValue_RBV"
    MAX = "dp-ADSim:Stats4:MaxValue_RBV"
    SIGMA = "dp-ADSim:Stats4:Sigma_RBV"
    MEAN = "dp-ADSim:Stats4:MeanValue_RBV"

# For use in the analysis server, not on the client side.
[ANALYSIS]
    # substitute with real PVs that are also in Metadata
    AXIS1 = "x" 
    AXIS2 = "y"
```

To use a custom configuration, load the file through the ConfigDialog GUI or place it in the `pv_configs/` folder.

---

## Troubleshooting

### Environment Issues
- Ensure the Conda environment is activated:
  ```bash
  conda activate DashPVA
  ```

### Missing Dependencies
- Reinstall the necessary packages using the commands listed above.

### GUI Crashes
- Verify `.ui` files (e.g., `imageshow.ui`) exist in the `gui/` folder.
- Ensure correct paths for configuration files.

---

## Need Help?
Refer to the [README.md](README.md) for an overview of the project or contact the repository maintainer for assistance.

