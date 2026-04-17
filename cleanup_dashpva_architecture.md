# DashPVA Architecture Cleanup Plan

**Authors:** P. Myint, Claude (AI Architect)
**Date:** 2026-04-15
**Status:** Draft — awaiting team review

---

## 1. Problem Statement

DashPVA has a "branch-per-beamline" anti-pattern. What started as lightweight customization
has grown into 10+ divergent branches (150-330 commits ahead of `main`), each with hardcoded
PV defaults, UI themes, and analysis logic. Meanwhile, `main` has received **zero commits**
since these branches forked — it is an abandoned baseline, not a living trunk.

The result:
- Bug fixes and features don't propagate between beamlines
- New beamlines copy-paste an existing branch and diverge further
- Three major development branches (`dev`, `dev-osayi`, `dev_SSRL17-2`) have each evolved
  independently with ~300 commits of unique work
- The workbench and HDFviewer exist as isolated modules with no launcher integration (HDFviewer)
  or incomplete integration (workbench)

This plan establishes a single trunk, a config-driven deployment model, and a formalized
subprocess-based module system — preserving DashPVA's distributed architecture while
eliminating the branch sprawl.

---

## 2. Current State Audit

### 2.1 Branch Inventory

| Branch | Commits ahead of main | Files changed | Key content |
|--------|----------------------|---------------|-------------|
| `dev` | 328 | 134 | Database layer, workflow, consumers, launcher registry |
| `dev-osayi` | 324 | 134 | Config cleanup, LogMixin (nearly identical to `dev`) |
| `dev_SSRL17-2` | 307 | 139 | Advanced viewer: masking, live plotting, stats, scan view |
| `dash-ai` | 255 | 118 | VIT stitching, workbench restructure, ROI calc, SAM |
| `dev_26ID` | 248 | 102 | VIT 5-panel layout (VIT stitch already on `dev`) |
| `dev_8ID` | 232 | 79 | Bad pixel mask for Lambda2m/Eiger4m |
| `dev-jrodriguez` | 153 | 38 | Merged dev-osayi + slice/config fixes |
| `dev_19ID` | 151 | 47 | 19ID PV config, sim data |
| `dev_4ID` | 149 | 47 | Eiger detector TOML configs |
| `dev_9ID` | 6 | 7 | Pilatus config, original VIT stitch location |
| `dev_11ID` | 10 (53 behind) | 15+ | Collector infrastructure, pyFAI, pyproject.toml |
| `dev_12ID` | 7 (53 behind) | 15+ | pyFAI analysis + calibration files |
| `workbench` | 17 | 60 | Workbench + recent SSRL17-2 mask work |

### 2.2 Open Pull Requests

**PR #65** (dev -> main): "feat: SQLite database layer for PV config profiles and settings"
- Author: Osayi-ANL
- Size: +6,693 / -2,757 across 53 files
- Adds: `ProfileManager`, `SettingsManager`, `DatabaseInterface` facade
- Adds: New launcher UI, log viewer, workflow rework
- Removes: `pva_setup/pva_workflow_setup_dialog.py`, `viewer/roi_cropping.py`
- **Impact on this plan:** Introduces the database-backed settings system that should
  be the foundation for config-driven UI. Must be reviewed and resolved before proceeding.

### 2.3 What Works Well (Keep)

- **Subprocess isolation**: Every module (viewer, pyFAI, workbench, bayesian) launches as
  its own process. Crashes are contained. Modules can run on different machines. This IS the
  "distributed" in DashPVA.
- **PVA pub/sub**: Modules communicate via EPICS PVA channels, not shared memory. This is
  the right inter-process communication pattern for beamline software.
- **TOML for PV configs**: The schema (`pv_configs/sample_config.toml`) cleanly separates
  PV address mapping from application logic.
- **Launcher registry** (`viewer/launcher/registry.py`): Simple list-of-dicts pattern for
  declaring available modules. Easy to extend.

### 2.4 What Needs Fixing

- **Hardcoded PV defaults**: `s6lambda1:Pva1:Image` in `pva_reader.py:22` and
  `area_det_viewer.py:100`. New beamlines hit this on first launch.
- **Hardcoded filesystem paths**: `viewer/bayesian/bluesky_compat.py` has absolute paths
  to 6-ID-B conda environments. Non-portable.
- **No UI/preference configuration**: Colormaps, log scale defaults, transpose settings,
  and feature toggles are either hardcoded or set at runtime with no persistence across
  sessions.
- **HDFviewer is orphaned**: Lives in `HDFviewer/` with no launcher entry and duplicates
  some workbench functionality.
- **pyFAI calibration files scattered**: `pyFAI/` directory with .poni and .edf files at
  project root, only on some branches.
- **No "analysis module" formalization**: pyFAI is launched ad-hoc from the viewer. Other
  analysis types (VIT stitching, Bayesian) are similarly ad-hoc.

---

## 3. Target Architecture

### 3.1 Core Principles

1. **One trunk, many configs**: A single `main` branch serves all beamlines. Beamline
   identity is defined by a TOML config file, not a Git branch.

2. **Distributed by design**: Every module is a subprocess. The launcher is the orchestrator.
   Modules communicate via PVA channels and file paths. No in-process plugin framework.

3. **Registry-driven**: Available modules (viewers, analyses, tools) are declared in a
   registry. The TOML config specifies which registry entries are relevant for a given
   beamline deployment.

4. **Config layered**: TOML for deployment-time settings (PV addresses, enabled modules,
   calibration paths). SQLite (PR #65) for runtime user preferences (colormap, window
   positions, recent files). Neither replaces the other.

### 3.2 Configuration Schema

Extend the existing TOML schema with new sections. No new config format — just new keys.

```toml
# ── Existing sections (unchanged) ──
DETECTOR_PREFIX = 'BL172:eiger4M'
OUTPUT_FILE_LOCATION = 'OUTPUT.h5'
CONSUMER_MODE = 'continuous'

[CACHE_OPTIONS]
# ... (unchanged)

[METADATA]
# ... (unchanged)

[ROI]
# ... (unchanged)

[STATS]
# ... (unchanged)

[ANALYSIS]
# ... (unchanged)

[HKL]
# ... (unchanged)

# ── New sections ──

[BEAMLINE]
NAME = 'SSRL 17-2'                          # Human-readable name
DEFAULT_PV_CHANNEL = 'BL172:eiger4M:Pva1:Image'  # Replaces hardcoded default
FACILITY = 'SLAC'                            # For display/logging

[VIEWER]
DEFAULT_COLORMAP = 'viridis'                 # Colormap on first launch
DEFAULT_LOG_SCALE = true                     # Log scale on by default
DEFAULT_TRANSPOSE = false                    # Image transpose
DEFAULT_ROTATION = 0                         # 0, 1, 2, 3 (90-degree increments)

[MASK]
DEFAULT_MASK_FILE = 'masks/active_mask.npy'  # Auto-load on startup
PONI_FILE = 'calibration/2022-3_calib.poni'  # For pyFAI integration

[MODULES]
# Which analysis/tool modules appear in the launcher for this beamline.
# These reference keys in the module registry.
ENABLED = [
    'area_det',
    'pyFAI',
    'workbench',
    'hdfviewer',
    'metadata_converter',
    'file_convert',
]
```

### 3.3 Module Registry (Extended)

Extend `viewer/launcher/registry.py` to include analysis modules, not just top-level views.
Each entry declares its subprocess command, required config sections, and category.

```python
VIEWS = [
    # ── Streaming ──
    {
        'key': 'area_det',
        'label': 'Area Detector 2D',
        'section': 'Streaming',
        'cmd': [sys.executable, 'dashpva.py', 'detector'],
        'config_requires': [],  # Always available
    },
    # ── Analysis (launched from viewer or standalone) ──
    {
        'key': 'pyFAI',
        'label': 'pyFAI Integration',
        'section': 'Analysis',
        'cmd': [sys.executable, 'viewer/pyFAI_analysis.py'],
        'config_requires': ['MASK'],  # Needs MASK.PONI_FILE
    },
    # ── Post Analysis ──
    {
        'key': 'workbench',
        'label': 'Workbench',
        'section': 'Post Analysis',
        'cmd': [sys.executable, 'dashpva.py', 'workbench'],
        'config_requires': [],
    },
    {
        'key': 'hdfviewer',
        'label': 'HDF5 Viewer',
        'section': 'Post Analysis',
        'cmd': [sys.executable, 'dashpva.py', 'hdfviewer'],
        'config_requires': [],
    },
    # ...
]
```

The launcher reads `[MODULES].ENABLED` from the active TOML config and only shows
matching registry entries. Beamlines that don't use pyFAI never see it in the menu.

### 3.4 Analysis Module Interface

Analysis modules are NOT in-process plugins. They are standalone scripts that follow
a contract:

```
# Launch contract:
python <module.py> --pv-address <PV> [--config <toml>] [--mask-file <path>] [module-specific args]

# Communication:
- Input:  Subscribe to PVA channel for live data, or accept HDF5 file path
- Output: Publish results on PVA channel (e.g., <pv>:pyFAI) and/or write to file
- Status: Exit code 0 = success, nonzero = error
```

The viewer's "Launch Analysis" button reads the registry, builds the CLI command with
the current PV address and config paths, and spawns the subprocess. This is exactly
what `open_analysis_window_clicked()` already does for pyFAI — we just formalize it.

### 3.5 Workbench & HDFviewer Integration

**Workbench** (already in launcher): No structural changes needed. It stays as a separate
subprocess. The viewer can offer "Open in Workbench" for the current HDF5 output file.

**HDFviewer** (currently orphaned):
1. Add a `dashpva.py hdfviewer` CLI command
2. Add it to the launcher registry
3. Move `HDFviewer/interactive.py` to `viewer/hdfviewer/hdfviewer.py` for consistency
4. The viewer can offer "Open in HDF Viewer" for the current output file

Both receive the file path as a CLI argument. No tight coupling needed.

**Shared context** between live viewer and post-analysis tools:
- File path: passed via CLI arg (`--file <path>`)
- Current ROI: passed via CLI arg or written to a shared JSON sidecar
- No shared memory, no IPC beyond PVA channels

---

## 4. Branch Consolidation Strategy

### 4.1 Choosing the New Trunk

`main` is dead (zero commits since all branches forked). The new trunk must be built from
the most mature branches. The merge order matters.

**Merge sequence:**

```
main  (current dead baseline)
  |
  +-- Merge PR #65 (dev -> main)          # Database layer, launcher, workflow
  |     Review first. Request changes to align with this plan.
  |
  +-- Merge dev_SSRL17-2 features         # Advanced viewer, masking, stats
  |     Cherry-pick or rebase the viewer/ and utils/ changes.
  |     Resolve conflicts with PR #65's viewer changes.
  |
  +-- Integrate dash-ai unique features   # ROI calc, config source, log manager
  |     Cherry-pick: utils/roi_ops.py, viewer/workbench/docks/roi_calc.py,
  |     utils/log_manager.py, utils/config/source.py
  |
  +-- Extract beamline configs            # TOML files from beamline branches
  |     See Section 4.2
  |
  = New unified main
```

### 4.2 Branch-by-Branch Extraction & Archival

| Branch | Extract | Then |
|--------|---------|------|
| `dev_4ID` | `pv_configs/metadata_pvs_eiger.toml`, `pv_configs/metadata_pvs_eiger_no_scan.toml` | Archive |
| `dev_8ID` | Bad pixel mask logic from area_det_viewer (3 commits: `26b67b0`, `52f42e5`, `468315d`) — adapt to new MaskManager | Archive |
| `dev_9ID` | `pv_configs/metadata_pvs_pilatus.toml` | Archive |
| `dev_11ID` | `collectors/` directory (hpc_s6lambda, hpc_dpADSim), collector setup dialogs, `pyproject.toml`/`uv.lock` if not already on dev | Archive |
| `dev_12ID` | `viewer/pyFAI_analysis.py`, `viewer/pyFAI_analysis_matplot.py`, `pyFAI/` calibration files (if not already from dev_11ID) | Archive |
| `dev_19ID` | `pv_configs/19ID_pvs.toml`, `consumers/19id_sim_rsm_data.py` | Archive |
| `dev_26ID` | Nothing unique (VIT stitch already on `dev`) | Archive |
| `dev-osayi` | Nothing unique (subset of `dev`) | Delete |
| `dev-jrodriguez` | Nothing unique (merged dev-osayi + fixes) | Archive |
| `dash-ai` | `utils/roi_ops.py`, `viewer/workbench/docks/roi_calc.py`, `utils/log_manager.py`, `utils/config/source.py`, `viewer/tools/file_convert.py` | Archive |

**Archive procedure:**
```bash
# Rename branch to archive namespace
git branch -m origin/dev_4ID origin/archive/dev_4ID
# Or use tags:
git tag archive/dev_4ID origin/dev_4ID
git push origin archive/dev_4ID
git push origin --delete dev_4ID
```

### 4.3 Extracted TOML Config Collection

All beamline-specific TOML files go into `pv_configs/` with a naming convention:

```
pv_configs/
    sample_config.toml          # Template with placeholder PVs
    ssrl_17-2.toml              # SSRL beamline 17-2
    aps_4id_eiger.toml          # APS 4-ID (Eiger detector)
    aps_4id_eiger_no_scan.toml  # APS 4-ID (Eiger, no scan mode)
    aps_8id.toml                # APS 8-ID
    aps_9id_pilatus.toml        # APS 9-ID (Pilatus detector)
    aps_11id.toml               # APS 11-ID
    aps_19id.toml               # APS 19-ID
```

Each file follows the extended schema from Section 3.2, including `[BEAMLINE]`,
`[VIEWER]`, `[MASK]`, and `[MODULES]` sections.

---

## 5. Hardcoded Default Elimination

### 5.1 Default PV Channel

**Current** (hardcoded in 3 files):
```python
# utils/pva_reader.py:22
input_channel='s6lambda1:Pva1:Image'

# viewer/area_det_viewer.py:100
input_channel='s6lambda1:Pva1:Image'

# viewer/hkl_3d_viewer.py:98
input_channel='s6lambda1:Pva1:Image'
```

**Target**: Read from `[BEAMLINE].DEFAULT_PV_CHANNEL` in the active TOML config.
If no config is loaded, show a setup dialog instead of silently connecting to a
nonexistent PV.

### 5.2 Bayesian Module Paths

**Current** (hardcoded in `viewer/bayesian/bluesky_compat.py`):
```python
CONDA_ENV_ROOT = "/home/beams/USER6IDB/.conda/envs/6idb-bits"
BITS_SRC_DIR = "/home/beams/USER6IDB/6idb-bits/src"
```

**Target**: Move to TOML config under a `[BAYESIAN]` section, or make the Bayesian
module detect its environment automatically.

### 5.3 pyFAI Calibration Paths

**Current**: Hardcoded fallback to `pyFAI/2022-3_calib.poni`.

**Target**: Read from `[MASK].PONI_FILE` in TOML config. No hardcoded fallback — if
no PONI file is configured, the pyFAI module prompts the user.

---

## 6. Execution Roadmap

### Phase 0: PR #65 Resolution (Week 1)

**Goal:** Decide on the database layer before building on top of it.

- [ ] Review PR #65 in detail (see Section 7 for review notes)
- [ ] Request changes to align with this architecture plan
- [ ] Merge PR #65 into main (or reject and document why)
- [ ] Verify main builds and launches after merge

### Phase 1: Trunk Establishment (Weeks 2-3)

**Goal:** A single `main` branch that contains the best of all development lines.

- [ ] Create `cleanup/consolidation` branch from main (post-PR #65)
- [ ] Merge dev_SSRL17-2 viewer/masking features (resolve conflicts)
- [ ] Cherry-pick dash-ai unique features (roi_ops, log_manager, config source)
- [ ] Extract beamline TOML configs from branch audit (Section 4.2)
- [ ] Add HDFviewer to launcher registry and dashpva.py CLI
- [ ] Move `HDFviewer/interactive.py` to `viewer/hdfviewer/`
- [ ] Run full test suite, manual smoke test of each module
- [ ] Merge `cleanup/consolidation` into `main`
- [ ] Archive old branches (Section 4.2 procedure)

### Phase 2: Config-Driven Deployment (Weeks 3-4)

**Goal:** Beamline identity comes from config, not branch.

- [ ] Extend TOML schema with `[BEAMLINE]`, `[VIEWER]`, `[MASK]`, `[MODULES]` sections
- [ ] Implement config loading in `dashpva.py` (global `--config <toml>` flag)
- [ ] Replace hardcoded PV defaults with config lookup + fallback dialog
- [ ] Launcher reads `[MODULES].ENABLED` to filter registry entries
- [ ] Viewer reads `[VIEWER].*` for default colormap, log scale, transpose
- [ ] Viewer reads `[MASK].*` for default mask and PONI file paths
- [ ] Create `pv_configs/` collection from all extracted beamline configs
- [ ] Update `sample_config.toml` template with new sections
- [ ] Test: launch with each beamline config, verify correct behavior

### Phase 3: Analysis Module Formalization (Weeks 4-5)

**Goal:** Analysis modules follow a documented subprocess contract.

- [ ] Document the analysis module interface (Section 3.4)
- [ ] Refactor pyFAI launch to use registry + TOML config for mask/PONI paths
- [ ] Add VIT stitching as a registered analysis module (if needed for 26ID)
- [ ] Viewer gets generic "Launch Analysis" menu populated from registry
- [ ] Fix Bayesian module's hardcoded paths (move to TOML or auto-detect)
- [ ] Test: each analysis module launches correctly from viewer and standalone

### Phase 4: Cleanup & Documentation (Week 5-6)

**Goal:** Remove dead code, document the architecture for future contributors.

- [ ] Remove hardcoded beamline defaults from all Python files
- [ ] Delete archived branch references from any CI/CD configs
- [ ] Write CONTRIBUTING.md: "How to add a new beamline" (copy sample_config.toml,
  fill in PV addresses, done)
- [ ] Write CONTRIBUTING.md: "How to add a new analysis module" (create script,
  add to registry, declare config requirements)
- [ ] Update README.md with new architecture overview
- [ ] Final review: `grep -r` for any remaining hardcoded beamline strings

---

## 7. PR #65 Review Notes

PR #65 introduces significant infrastructure that overlaps with this plan. Key questions
for review:

### Align with plan
- The `SettingsManager` / `ProfileManager` pattern is the right place for runtime UI
  preferences (colormap, window positions). Confirm it can coexist with TOML for
  deployment-time config.
- The launcher rework should use the registry pattern. Verify it does.
- The `DatabaseInterface` facade should not tightly couple modules. Verify it's
  optional — modules that don't need the DB (like pyFAI) should work without it.

### Potential conflicts
- PR #65 removes `pva_setup/pva_workflow_setup_dialog.py` and `viewer/roi_cropping.py`.
  Verify nothing on dev_SSRL17-2 or dash-ai depends on these.
- PR #65 modifies `viewer/workbench/` files that dash-ai also restructured. These will
  conflict during Phase 1 merge.
- PR #65 removes `pv_configs/metadata_pvs.toml`. This may conflict with the config
  collection strategy. We may need to re-add beamline-specific configs after merge.

### Requested changes for alignment
- Add `--config <toml>` support to the launcher if not already present
- Ensure `SettingsManager` has a clear boundary: it owns runtime preferences,
  TOML owns deployment config. No overlap.
- Ensure all new modules follow the subprocess launch pattern (no in-process imports
  from the viewer)

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Merge conflicts between dev, dev_SSRL17-2, dash-ai | High | Merge in sequence (Section 4.1). Resolve viewer/ conflicts manually. |
| PR #65 database layer adds complexity without value | Medium | Review PR #65 first. Accept or reject before Phase 1. |
| Beamline scientists on old branches resist migration | Medium | Provide per-beamline TOML configs that replicate their current behavior exactly. |
| HDFviewer and workbench have overlapping features | Low | Audit overlap. If significant, deprecate HDFviewer and port unique features to workbench. |
| TOML config schema becomes too complex | Low | Keep TOML flat. Complex state goes in SQLite (PR #65's domain). |
| Breaking changes during consolidation | High | Tag current branch tips before archiving. Keep `archive/*` tags indefinitely. |

---

## 9. Success Criteria

The cleanup is complete when:

1. **One branch**: All active development happens on `main` (or short-lived feature branches
   merged back to `main`).
2. **Config-driven**: A new beamline deployment requires only copying `sample_config.toml`,
   filling in PV addresses, and optionally enabling analysis modules. Zero code changes.
3. **No hardcoded beamline strings**: `grep -ri "s6lambda1\|6idb\|8idLambda\|USER6IDB"
   --include="*.py"` returns zero results (excluding test/simulation files).
4. **All modules launchable**: Every entry in the launcher registry starts successfully
   as a subprocess.
5. **Archived branches tagged**: All old beamline branches have `archive/*` tags and are
   deleted from remote.
6. **Documentation exists**: CONTRIBUTING.md covers "add a beamline" and "add an analysis
   module" workflows.
