# DashPVA: Distributed Analysis and Streaming Hub with Process Variable Access

DashPVA is a modular analysis and visualization platform for X-ray experiments at synchrotron beamlines. It connects to area detectors and EPICS process variables through PvaPy, then distributes data through a pipeline of computation nodes — each node can perform live viewing, real-time analysis, or data reduction depending on the experiment's needs.

## Key Capabilities

- **Live Area Detector Streaming** — real-time 2D image display with ROI monitoring and statistics
- **HKL 3D Reciprocal Space Mapping** — live reciprocal-space visualization from diffractometer motor positions
- **pyFAI 1D Azimuthal Integration** — live reduction of 2D diffraction images to 1D patterns
- **XRD Phase Fitting** — fit crystal phases to 1D diffraction patterns (file or live mode via pyFAI output)
- **Bayesian 2D Scan** — Gaussian-process-guided adaptive scanning to efficiently locate features of interest in samples
- **Post-Analysis Workbench** — HDF5 data exploration with 1D/2D/3D views, ROI tools, and slicing
- **Scan Monitor** — track scan progress in real time

## Architecture

DashPVA spawns analysis tools as independent processes, each acting as a node in a data processing pipeline:

```
Detector → Metadata Associator → Collector → Analysis Consumer(s) → Viewer(s)
```

Each node can view data, perform analysis, or pass processed results downstream. This lets you compose different analysis chains — for example, a pyFAI integration node feeding a live phase fitter, or an RSM consumer feeding the HKL 3D viewer.

## Installation

### Requirements

- Python >= 3.11
- Git

### Quick Start

```bash
git clone https://github.com/AdvancedPhotonSource/DashPVA.git
cd DashPVA
bash install.sh
```

The installer will ask which edition to set up:

| Edition | What's included | Use case |
| --- | --- | --- |
| **Full** | All tools — live streaming, pvaccess/EPICS, Bayesian | Beamline deployment |
| **Standalone** | Post-analysis tools — Workbench, File Convert, Phase Fitter | Offline analysis |

You can skip the prompt with a flag:

```bash
bash install.sh --full        # Full edition (live + analysis)
bash install.sh --standalone  # Standalone edition (analysis only)
```

### Updating

```bash
bash install.sh --update
```

Or use the **Updates** button inside the launcher to pull the latest release.

## Launching

```bash
DashPVA run
```

This opens the launcher menu with access to all tools.

### Available Commands

```bash
DashPVA run           # Open the launcher menu
DashPVA detector      # Launch Area Detector Viewer
DashPVA hkl3d         # Launch HKL 3D Viewer
DashPVA setup         # Run PVA workflow setup
DashPVA setup --ioc   # Run simulator setup
DashPVA workbench     # Launch Workbench data analysis tool
DashPVA bayesian      # Launch Bayesian 2D Scan Viewer
DashPVA phasefitter   # Launch XRD Phase Fitter
DashPVA monitor scan  # Open scan monitor

DashPVA --help        # Show all available commands
```

## Project Structure

```
DashPVA/
├── src/dashpva/           # Main package
│   ├── cli.py             # CLI entry point
│   ├── settings.py        # Application settings
│   ├── consumers/         # PVA data consumers (metadata, HPC, IOC)
│   ├── database/          # SQLite profile/settings storage
│   ├── gui/               # Qt .ui files
│   ├── hdf_viewer/        # HDF5 interactive viewer
│   ├── utils/             # Shared utilities (HDF5, masks, RSM, etc.)
│   ├── viewer/            # All viewer GUIs
│   │   ├── bayesian/      # Bayesian adaptive scanning
│   │   ├── hkl3d/         # HKL 3D reciprocal space viewer
│   │   ├── launcher/      # Main launcher and process registry
│   │   ├── workbench/     # Post-analysis workbench
│   │   └── ...            # Area detector, scan, pyFAI, phase fitter
│   └── workflow/          # PVA workflow setup
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── test_data/         # Calibration and test files
├── pv_configs/            # PV configuration TOML files
├── notebooks/             # Analysis notebooks
├── install.sh             # Installer script
├── pyproject.toml         # Build config and dependencies
├── uv.lock                # Locked dependency versions
└── README.md
```

## HKL Live Streaming Setup

For HKL (reciprocal space) live streaming and analysis, DashPVA uses a multi-stage pipeline that processes detector images through several consumers before displaying HKL coordinates in real-time.

### Data Flow Pipeline

```
Detector → Metadata Associator → Collector → RSM Consumer → HKL Viewer
```

Each stage adds or processes data:
- **Detector**: Raw image data from area detector
- **Metadata Associator**: Attaches motor positions and metadata to images
- **Collector**: Collects and buffers images with metadata
- **RSM Consumer**: Calculates HKL coordinates from motor positions using xrayutilities
- **HKL Viewer**: Displays 3D reciprocal-space visualization

### Configuration

Before starting HKL streaming, configure the TOML file with your beamline-specific PVs.

#### Edit `pv_configs/metadata_pvs.toml`

**A. Set Detector Prefix:**
```toml
DETECTOR_PREFIX = 'your_beamline:detector_prefix'
```

**B. Configure Metadata PVs:**
```toml
[METADATA]
    [METADATA.CA]
    x = 'your_beamline:x_motor_RBV'
    y = 'your_beamline:y_motor_RBV'

    [METADATA.PVA]
```

**C. Configure HKL Section:**
```toml
[HKL]
    [HKL.SAMPLE_CIRCLE_AXIS_1]
    AXIS_NUMBER = 'your_beamline:motor1_RBV:AxisNumber'
    DIRECTION_AXIS = 'your_beamline:motor1_RBV:DirectionAxis'
    POSITION = 'your_beamline:motor1_RBV:Position'
    # Repeat for SAMPLE_CIRCLE_AXIS_2, 3, 4
    # And DETECTOR_CIRCLE_AXIS_1, 2

    [HKL.SPEC]
    ENERGY_VALUE = 'your_beamline:spec:Energy:Value'
    UB_MATRIX_VALUE = 'your_beamline:spec:UB_matrix:Value'

    [HKL.DETECTOR_SETUP]
    CENTER_CHANNEL_PIXEL = 'your_beamline:DetectorSetup:CenterChannelPixel'
    DISTANCE = 'your_beamline:DetectorSetup:Distance'
    PIXEL_DIRECTION_1 = 'your_beamline:DetectorSetup:PixelDirection1'
    PIXEL_DIRECTION_2 = 'your_beamline:DetectorSetup:PixelDirection2'
    SIZE = 'your_beamline:DetectorSetup:Size'
    UNITS = 'your_beamline:DetectorSetup:Units'
```

For different beamlines, create a beamline-specific config file:
```bash
cp pv_configs/metadata_pvs.toml pv_configs/metadata_pvs_YOUR_BEAMLINE.toml
```

### Startup Sequence

Follow these steps in order to start the complete HKL streaming pipeline:

**Terminal 1 — Area Detector Viewer:**
```bash
DashPVA detector
```
Enter your PVA channel name and click "Start Live View". Keep this running.

**Terminal 2 — PVA Workflow Setup:**
```bash
DashPVA setup
```
Configure each tab in order:
1. **Config Upload** — Browse and select your `metadata_pvs.toml`
2. **Metadata Associator** — Set input/output channels, click "Run Associator Consumers"
3. **Collector** — Set input/output channels, click "Run Collector"
4. **Analysis Consumer** — Set input/output channels, click "Run Analysis Consumer"

**Terminal 3 — HKL 3D Viewer:**
```bash
DashPVA hkl3d
```
Enter the RSM Consumer output channel, browse your config file, and click "Start Live View".

### Important Notes

1. **Channel names must match** — the output channel of one component must match the input channel of the next
2. **TOML file is critical** — all motor PVs must be correctly specified in the `[HKL]` section
3. **Startup order matters** — Detector Viewer → Setup consumers (Associator → Collector → RSM) → HKL Viewer

## Configuration Files

All configuration files are stored in the `pv_configs/` directory.

Example configuration (`pv_configs/example_config.toml`):
```toml
CONSUMER_TYPE = "spontaneous"

[METADATA]
    [METADATA.CA]
    x = "x"
    y = "y"
    [METADATA.PVA]

[ROI]
    [ROI.ROI1]
    MIN_X = "dp-ADSim:ROI1:MinX"
    MIN_Y = "dp-ADSim:ROI1:MinY"
    SIZE_X = "dp-ADSim:ROI1:SizeX"
    SIZE_Y = "dp-ADSim:ROI1:SizeY"

[STATS]
    [STATS.STATS1]
    TOTAL = "dp-ADSim:Stats1:Total_RBV"
    MIN = "dp-ADSim:Stats1:MinValue_RBV"
    MAX = "dp-ADSim:Stats1:MaxValue_RBV"
    SIGMA = "dp-ADSim:Stats1:Sigma_RBV"
    MEAN = "dp-ADSim:Stats1:MeanValue_RBV"

[ANALYSIS]
    AXIS1 = "x"
    AXIS2 = "y"
```

## Troubleshooting

### EPICS Database Definition (DBD) Directory Not Found

If you see:
```
Cannot find dbd directory, please set EPICS_DB_INCLUDE_PATH environment variable
```

**Solution:** Set the environment variable:
```bash
export EPICS_DB_INCLUDE_PATH=/APSshare/epics/base-7.0.8/dbd
```

Add to `~/.bashrc` or `~/.zshrc` to make it persistent.

### GUI Issues

- Verify `.ui` files exist in `src/dashpva/gui/`
- Ensure correct paths for PV configuration files in `pv_configs/`

## What's New

Check out [CHANGELOG.md](CHANGELOG.md) for the latest features, improvements, and changes.

## FAQ

**What is DashPVA?**
DashPVA is a modular platform for real-time X-ray data acquisition, visualization, and analysis at synchrotron beamlines. It connects to area detectors via PvaPy/EPICS and distributes data through a pipeline of computation nodes — each node can perform live viewing, data reduction, or analysis depending on the experiment's needs.

**What analysis tools are available?**
DashPVA provides several analysis modules, each targeting a different experimental need:
- **HKL 3D** — live reciprocal-space mapping for diffractometer experiments
- **pyFAI integration** — live 1D azimuthal reduction of 2D diffraction images
- **Phase Fitter** — XRD crystal phase fitting against 1D patterns (file or live via pyFAI)
- **Bayesian 2D Scan** — Gaussian-process-guided adaptive scanning to efficiently locate features in a sample
- **Workbench** — post-experiment HDF5 exploration with 1D/2D/3D views and ROI tools

**How does the pipeline architecture work?**
DashPVA spawns each analysis tool as an independent process (node). Data flows from the detector through metadata association, collection, and analysis stages. Each node can be a viewer, a data reducer, or an analysis engine. You compose different chains depending on what you need — for example, detector → pyFAI integration → live phase fitting, or detector → RSM calculation → HKL 3D viewer.

**What is the difference between Full and Standalone editions?**
Full includes everything — live streaming via pvaccess/EPICS, all real-time analysis tools, and Bayesian scanning. Standalone is a lighter install for offline post-analysis: Workbench, File Convert, Metadata Converter, and Phase Fitter (file mode only). Use `bash install.sh --full` or `bash install.sh --standalone` to choose.
