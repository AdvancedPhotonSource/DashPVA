# Run Instructions for DashPVA

This guide provides step-by-step instructions to set up, configure, and run the DashPVA application.

---

## Setup

### Install Dependencies

You can install DashPVA dependencies using either **Conda** (recommended for full compatibility) or **UV** (faster installation). Choose the method that best fits your needs.

#### Option 1: Using UV (Fast Installation)

[UV](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver written in Rust. It provides much faster dependency resolution and installation compared to traditional pip.

**Prerequisites:**
- Python 3.11 installed on your system

**Installation Steps:**

1. **Install UV** (if not already installed):

   **Linux/macOS:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   After installation, add UV to your PATH:
   ```bash
   # For bash/zsh (Linux/macOS)
   source $HOME/.local/bin/env
   
   # Or add permanently to ~/.bashrc or ~/.zshrc:
   export PATH="$HOME/.local/bin:$PATH"
   ```
   
   **Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   
   After installation, restart your terminal or add to PATH:
   ```powershell
   $env:PATH += ";$env:USERPROFILE\.cargo\bin"
   ```
   
   **Alternative (all platforms):**
   ```bash
   pip install uv
   ```

2. **Install dependencies** (UV will automatically create a virtual environment):
   ```bash
   uv sync
   ```
   
   This single command will:
   - Create a virtual environment (`.venv/`) automatically
   - Install all dependencies from `pyproject.toml`
   - Use locked versions from `uv.lock` for reproducible installs

3. **Activate the environment and run the application:**
   
   **Option A: Activate the virtual environment (traditional way):**
   ```bash
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   .venv\Scripts\activate.bat
   ```
   
   Then run your commands normally:
   ```bash
   python dashpva.py setup
   ```
   
   **Option B: Use UV to run commands directly (no activation needed):**
   ```bash
   uv run python dashpva.py setup
   uv run python dashpva.py detector
   ```

**Note:** All dependencies including `pvapy` (required for PVAccess) are automatically installed via `uv sync`. No conda installation is needed!

**Quick Start Summary:**
```bash
# 1. Install UV (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# OR: pip install uv  # All platforms

# 2. Install all dependencies (creates .venv automatically)
uv sync

# 3. Activate and run
source .venv/bin/activate  # Linux/macOS
# OR: uv run python dashpva.py setup  # No activation needed
```

**Verify Installation:**
```bash
uv --version
uv pip list
```

**Updating Dependencies:**
```bash
# Update dependencies and regenerate lock file
uv lock --upgrade

# Sync with updated dependencies
uv sync
```

#### Option 2: Using Conda (Full Compatibility)

Using the [environment.yml](environment.yml) file, you can install the environment using the conda command:
```bash
conda env create -f environment.yml
```

Instead of using the `environment.yml` file, you can follow these manual instructions to set up the environment:

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

**Verify Conda Installation:**
Ensure all dependencies are installed correctly:
```bash
conda list
```

---

## Running the Application

DashPVA now uses a command-line interface (CLI) for launching different components. All commands use the main `dashpva.py` script.

### Available Commands

#### 1. Setup and Configuration
Set up PVA workflow and configure the system:

**Run Command**:
```bash
python dashpva.py setup
```

**With Simulator**:
```bash
python dashpva.py setup --sim
```

**Key Features**:
- Set PVA prefix and collector address.
- Load, edit, or create PV configuration files.
- Input caching frequency for live view.

#### 2. Area Detector Viewer
Launch the live image visualization GUI:

**Run Command**:
```bash
python dashpva.py detector
```

**GUI Features**:
- **Start/Stop Live View**: Begin or end live image streaming.
- **ROI Tools**: Add, view, and manipulate ROIs on the displayed image.
- **Statistical Monitoring**: View and log key metrics from the live feed.
- **Frame-by-Frame Processing**: Supports both predetermined and spontaneous scan modes.

#### 3. HKL 3D Viewer
Launch the interactive 3D visualization tool:

**Run Command**:
```bash
python dashpva.py hkl3d
```

**Features**:
- Interactive 3D point cloud visualization
- Real-time data streaming and analysis
- Integration with PVA data sources

#### 4. HKL 3D Slicer (Standalone)
Launch the standalone 3D slicer for offline data analysis:

**Run Command**:
```bash
python dashpva.py slice3d
```

**Features**:
- Interactive 3D visualization with real-time slicing
- HDF5 data loading capabilities
- Slice extraction and analysis tools
- Loading indicators for large datasets
- Configurable reduction factors for performance optimization

### Quick Reference
```bash
# Run the launcher
python dashpva.py run 

# Setup the system
python dashpva.py setup

# Launch area detector viewer
python dashpva.py detector

# Launch 3D visualization tools
python dashpva.py hkl3d
python dashpva.py slice3d

# Get help on available commands
python dashpva.py --help
```

---

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
- **RSM Consumer**: Calculates HKL coordinates from motor positions
- **HKL Viewer**: Displays 3D HKL visualization

### Configuration Requirements

Before starting HKL streaming, you must configure the TOML file with your beamline-specific PVs.

#### 1. Edit `pv_configs/metadata_pvs.toml`

**A. Set Detector Prefix (Line 2):**
```toml
DETECTOR_PREFIX = 'your_beamline:detector_prefix'
# Example: '11idb:AD1' or '8idb:detector'
```

**B. Configure Metadata PVs (Lines 24-30):**
```toml
[METADATA]
    [METADATA.CA]
    # Add your Channel Access PVs (motor positions, etc.)
    x = 'your_beamline:x_motor_RBV'
    y = 'your_beamline:y_motor_RBV'
    # Add any other metadata PVs needed
    
    [METADATA.PVA]
    # Add your PVAccess PVs here if any
```

**C. Configure HKL Section (Lines 83-154):**
This section is critical for HKL calculations. Update all motor PVs, spec PVs, and detector setup:

```toml
[HKL]
    # Sample Circle Motors (typically 4 axes)
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

**Note:** For different beamlines, create a beamline-specific config file:
```bash
cp pv_configs/metadata_pvs.toml pv_configs/metadata_pvs_YOUR_BEAMLINE.toml
```

### Startup Sequence for HKL Live Streaming

Follow these steps in order to start the complete HKL streaming pipeline:

#### **Terminal 1: Area Detector Viewer (Live View)**
```bash
python dashpva.py detector
```

1. Enter your PVA channel name (e.g., `'11idb:detector:Image'`)
2. Click "Start Live View"
3. **Keep this terminal running** - This shows live detector images

**Purpose:** Verify detector is streaming correctly before starting the processing pipeline.

---

#### **Terminal 2: PVA Workflow Setup**
```bash
python dashpva.py setup
```

This opens the PVA Setup Dialog with multiple tabs. Configure each component:

##### **Tab 1: Config Upload**
1. Click "Browse" and select your `metadata_pvs.toml` file (or beamline-specific version)
2. The "Current Mode" label will show the caching mode from your config
3. **This config file will be used by all consumers**

##### **Tab 2: Metadata Associator**
This consumer attaches metadata (motor positions, etc.) to detector images.

**Configuration:**
- **Input Channel**: Your detector PVA channel (e.g., `'11idb:detector:Image'`)
- **Output Channel**: Where associator sends data (e.g., `'processor:associator:output'`)
- **Control Channel**: `'processor:*:control'` (default)
- **Status Channel**: `'processor:*:status'` (default)
- **Processor File**: `consumers/hpc_metadata_consumer.py`
- **Processor Class**: `HpcAdMetadataProcessor`
- **Report Period**: `5` (seconds, default)
- **Server Queue Size**: `100` (default)
- **N Consumers**: `1` (default)
- **Distributor Updates**: `10` (default)

**Action:** Click **"Run Associator Consumers"**

**What it does:** Reads PVs from `[METADATA]` and `[HKL]` sections of your TOML file and attaches their values to each detector image frame.

---

##### **Tab 3: Collector**
This consumer collects and buffers images with attached metadata.

**Configuration:**
- **Collector ID**: `1` (default)
- **Producer ID List**: `1` (default, comma-separated if multiple)
- **Input Channel**: Same as Associator Output Channel (e.g., `'processor:associator:output'`)
- **Output Channel**: Where collector sends data (e.g., `'processor:collector:output'`)
- **Control Channel**: `'processor:*:control'` (default)
- **Status Channel**: `'processor:*:status'` (default)
- **Processor File**: `consumers/hpc_passthrough_consumer.py`
- **Processor Class**: `HpcPassthroughProcessor`
- **Report Period**: `5` (seconds, default)
- **Server Queue Size**: `100` (default)
- **Collector Cache Size**: `1000` (default)

**Action:** Click **"Run Collector"**

**What it does:** Collects images with metadata from the associator and forwards them to the next stage.

---

##### **Tab 4: Analysis Consumer (RSM Consumer)**
This consumer calculates HKL coordinates from motor positions.

**Configuration:**
- **Input Channel**: Same as Collector Output Channel (e.g., `'processor:collector:output'`)
- **Output Channel**: Where RSM data goes (e.g., `'processor:rsm:output'`)
- **Control Channel**: `'processor:*:control'` (default)
- **Status Channel**: `'processor:*:status'` (default)
- **Processor File**: `consumers/hpc_rsm_consumer.py`
- **Processor Class**: `HpcRsmProcessor`
- **Report Period**: `5` (seconds, default)
- **Server Queue Size**: `100` (default)
- **N Consumers**: `1` (default)
- **Distributor Updates**: `10` (default)

**Action:** Click **"Run Analysis Consumer"**

**What it does:** 
- Reads motor positions from the `[HKL]` section of your TOML file
- Calculates reciprocal space (HKL) coordinates using xrayutilities
- Outputs HKL data (qx, qy, qz) for visualization

---

#### **Terminal 3: HKL 3D Viewer**
```bash
python dashpva.py hkl3d
```

1. **Input Channel**: Enter the RSM Consumer Output Channel (e.g., `'processor:rsm:output'`)
2. **Config File**: Browse and select your `metadata_pvs.toml` file
3. Click **"Start Live View"**

**What it does:** Displays real-time 3D HKL visualization with point cloud data streaming from the RSM consumer.

---

### Important Notes

1. **Channel Names Must Match**: The output channel of one component must match the input channel of the next:
   - Associator Output → Collector Input
   - Collector Output → RSM Consumer Input
   - RSM Consumer Output → HKL Viewer Input

2. **TOML File is Critical**: 
   - The Metadata Associator reads PVs from `[METADATA]` and `[HKL]` sections
   - The RSM Consumer uses `[HKL]` section PVs to calculate HKL coordinates
   - All motor PVs must be correctly specified in the `[HKL]` section

3. **Startup Order Matters**: 
   - Start Detector Viewer first (to verify detector is working)
   - Then start PVA Setup and launch consumers in order: Associator → Collector → RSM Consumer
   - Finally, start HKL Viewer

4. **For Different Beamlines**: 
   - Create a beamline-specific TOML config file
   - Update all PV names to match your beamline's EPICS PVs
   - The same startup sequence applies, just use your beamline's config file

### Quick Checklist

Before starting HKL streaming, ensure:

- [ ] TOML config file has correct `DETECTOR_PREFIX`
- [ ] `[METADATA]` section has your metadata PVs
- [ ] `[HKL]` section has all motor PVs (sample circle, detector circle)
- [ ] `[HKL.SPEC]` section has energy and UB matrix PVs
- [ ] `[HKL.DETECTOR_SETUP]` section has detector geometry PVs
- [ ] PVA channel names are consistent across all components
- [ ] All consumers are started in the correct order

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
    SIZE_Y = "dp-ADSim:ROI1:SizeY"

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

### EPICS Database Definition (DBD) Directory Not Found

If you encounter the error:
```
Cannot find dbd directory, please set EPICS_DB_INCLUDE_PATH environment variable to use CA metadata PVs.
```

This occurs when using CA (Channel Access) metadata PVs in the collector testing script. The script needs to find EPICS database definition files.

**Solution 1: Set EPICS_DB_INCLUDE_PATH manually**

Find your EPICS base installation and set the environment variable:

```bash
# For APS systems, EPICS base is typically at:
export EPICS_DB_INCLUDE_PATH=/APSshare/epics/base-7.0.8/dbd

# Or if using conda-installed pvapy:
# Find where pvapy is installed, then look for dbd directory
# Usually: $CONDA_PREFIX/share/epics/dbd or similar

# To find it automatically:
python -c "import pvaccess as pva; import os; print(os.path.dirname(pva.__file__))"
# Then navigate to the dbd directory relative to that location
```

**Solution 2: Add to your shell configuration**

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export EPICS_DB_INCLUDE_PATH=/APSshare/epics/base-7.0.8/dbd
```

**Solution 3: Use PVA metadata instead of CA**

If you don't need CA metadata, use PVA metadata instead:
```bash
# Instead of: -mpv ca://x,ca://y
# Use: -mpv pva://x,pva://y
```

**Note:** The script will attempt to auto-detect the dbd directory, but if `pvData` library cannot be found, you must set `EPICS_DB_INCLUDE_PATH` manually.

---

## Need Help?
Refer to the [README.md](README.md) for an overview of the project or contact the repository maintainer for assistance.
