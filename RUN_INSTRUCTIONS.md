# Run Instructions for DashPVA

This guide provides step-by-step instructions to set up, configure, and run the DashPVA application.

---

## Setup

### Install Dependencies
Instead of using the `environment.yml` file, follow these manual instructions to set up the environment:

1. Create a new Conda environment:
   ```bash
   conda create -n DashPVA python=3.8 numpy=1.24.4
   ```
2. Activate the environment:
   ```bash
   conda activate DashPVA
   ```
3. Install essential libraries:
   ```bash
   conda install -c conda-forge pyqt=5.15 pyqtgraph=0.12
   conda install -c apsu pvapy=5.3
   conda install matplotlib=3.7
   conda install scipy pandas h5py
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
Below is an example configuration file (`example_config.json`):
```json
{
  "PVA_PREFIX": "dp-ADSim",
  "COLLECTOR_ADDRESS": "collector:1:output",
  "CACHE_FREQUENCY": 10
}
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

