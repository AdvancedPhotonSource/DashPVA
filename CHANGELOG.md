# What's New in DashPVA

This file tracks the latest changes, features, and improvements in DashPVA.

---

## Latest Changes (September 2025)

### 🚀 New Features

#### Command Line Interface (CLI)
- **NEW**: Introduced unified CLI interface via `dashpva.py`
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

### 🔧 Bug Fixes and Improvements

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
- **🚀 New Features**: Major new functionality and capabilities
- **🔧 Bug Fixes and Improvements**: Fixes, optimizations, and enhancements
- **📚 Documentation**: Updates to documentation and guides
- **⚠️ Breaking Changes**: Changes that may affect existing workflows

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
