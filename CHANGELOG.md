# What's New in DashPVA

This file tracks the latest changes, features, and improvements in DashPVA.

---

## Latest Changes (March 2026)
### New Features
- Scan Monitor gives you an option to save or write temp
- Universal log system, all outputs go to logs/general.log
- ROI Plot dock: added Single Frame mode with axis projection (Proj X / Proj Y) to sum ROI counts along a chosen axis for the current frame
- Database to replace Toml configuration for PV's

### Fix
- Profile import issue


## Latest Changes (February 2026)
### Added
  - ROI calculated for math between specific ROI's
    - Can then be exported to an h5 file
    - Creates an image or a plot of the calculated ROI's
  - Workbench supports all image file formats (.jpeg, .png, .bmp, .tif, .tiff)
  - Conda supports hdf5plugin

### Change
  - All placeholder methods raise NotImplementedError

### Fixed
  - Vmin/Vmax now syncs with histogram in 2d workbench


## Latest Changes (January 2026)
- Workbench now supports loading compressed datasets for smoother analysis and smaller storage footprints. For legacy files, a converter will be provided to update the file structure so compression loads seamlessly.

- Added data file tree to view files from folder from atree or a singular file

# Latest Changes (December 2025)
Merry Christmas

The Post-Analysis Workbench is a unified workspace that turns raw HKL data into an interactive 3D volume. It keeps 1D line profiles, 2D detector slices, and 3D voxel views in sync—select a point in any view and the others update instantly. This makes it easy to confirm peaks, filter noise in real time, and explore crystal symmetry from every angle.

## Latest Changes (November 2025)

### New Features
- Selective compression support for large HDF5 datasets using Blosc (LZ4), applied only where it provides clear benefits.
- Compatible file structure for compressed data, ensuring seamless loading via `utils.hdf5_loader.HDF5Loader`.

### Improvements
- Faster load times and reduced disk footprint for large arrays written to `/entry/data/data`.
- Graceful handling when compression plugins are unavailable (falls back to uncompressed writes).
- Clearer loading messages and error reporting during data import.

### Usage
- Files produced by `compress.py` can be loaded with:
  - `HDF5Loader.load_h5_to_3d(path)` for points + intensities
  - `HDF5Loader.load_h5_volume_3d(path)` for volume workflows


## Latest Changes (October 2025)

### New Features
- Launcher: Force-shutdown dialog now lists all running modules with their process IDs (PIDs) for full visibility before termination.
- HKL Slice 3D Tool: Added a 2D viewer under Tools to quickly inspect any 3D slice in 2D.

### Bug Fixes and Improvements
- HKL Range Handling: Adjusted HKL index/range bounds and validation in HKL 3D Viewer and HKL 3D Slicer to ensure accurate limits and improved user feedback.
- Launcher: Enhanced process tracking with clearer status text and contextual enablement of “Shutdown All”.

## Latest Changes (September 2025)

### New Features

#### Command Line Interface (CLI)
- **NEW**: Introduced unified CLI interface via `dashpva.py`
  - `python dashpva.py run` - Launch dashpva
  - `python dashpva.py hkl3d` - Launch HKL 3D Viewer
  - `python dashpva.py slice3d` - Launch HKL 3D Slicer (standalone mode)
  - `python dashpva.py detector` - Launch Area Detector Viewer
  - `python dashpva.py setup` - Run PVA workflow setup (with optional `--sim` flag)

#### 3D Visualization Enhancements
- **NEW**: HKL 3D Slice Window for interactive 3D point cloud visualization
- **NEW**: Standalone HKL 3D Slicer mode for offline data analysis
- **ENHANCED**: Loading indicators for 3D parent slice window data loading
- **ENHANCED**: Configurable reduction factor before loading data
- **ENHANCED**: Window disabling during data loading operations

#### Data Loading and Processing
- **NEW**: HDF5 data loading capabilities
- **NEW**: Slice extraction and analysis tools
- **NEW**: Interactive 3D visualization with real-time slicing
- **ENHANCED**: Improved data caching and processing workflows

#### User Interface Improvements
- **NEW**: SizeManager for automatic and clean window scaling on resize
- **ENHANCED**: Area detector viewer now includes SizeManager
- **IMPROVED**: Replaced old font scaling with SizeManager system

#### 2D Slice viewing in 3d viewer
View the 3D sliced data in 2d

### Bug Fixes and Improvements

#### Configuration and Setup
- **FIXED**: PV simulator server size can now be changed through GUI
- **UPDATED**: CLI setup command changed from `sim` to `setup` for setup dialog
- **IMPROVED**: Path management for parent directory navigation in area detector

#### Performance and Stability
- **OPTIMIZED**: Performance improvements for large dataset handling
- **ENHANCED**: Better memory management for 3D visualization
- **IMPROVED**: More responsive UI during data loading operations

#### Code Quality
- **CLEANUP**: Commented out unused LoadDataHandler and PerformanceDialog utilities
- **REFACTORED**: Improved code organization and structure
- **ADDED**: Comprehensive .gitignore file for better repository management

---

## How to Use This File

This changelog follows these conventions:
- New Features: Major new functionality and capabilities
- Bug Fixes and Improvements: Fixes, optimizations, and enhancements
- Documentation: Updates to documentation and guides
- Breaking Changes: Changes that may affect existing workflows

---

## Getting Started with New Features

### Using the New CLI
```bash
# Get help on all available commands
python dashpva.py --help

# Launch different components
python dashpva.py setup      # Configure the system
python dashpva.py detector   # Area detector viewer
python dashpva.py hkl3d      # 3D visualization
python dashpva.py slice3d    # Standalone 3D slicer
```

### 3D Visualization
The new 3D visualization tools support:
- Interactive point cloud visualization
- Real-time slicing and analysis
- HDF5 file loading
- Performance optimization for large datasets

For detailed instructions, see [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md).

---

## Previous Versions

*This is the initial version of the What's New file. Future releases will be documented here.*
