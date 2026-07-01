# DashPVA — Agent Guide

DashPVA is a modular analysis and visualization platform for X-ray experiments at synchrotron beamlines. It connects to area detectors and EPICS process variables through PvaPy and distributes data through a pipeline of computation and display nodes.

## Project layout

```
src/dashpva/
  cli.py                  # Click CLI entry point (command: DashPVA)
  settings.py             # Single source of truth for ALL runtime config
  gui/
    __init__.py           # configure_app() loads theme.qss
    theme.qss             # ALL widget styles live here
    theme_colors.py
  viewer/
    area_det/             # Area detector live viewer
    scan_view.py          # Scan monitor
    hkl_3d_slice_window.py
    phase_fitter.py
    pyFAI_analysis.py
    ...
  workflow/
    workflow.py           # PVA Workflow UI + Metadata Associator
  utils/
    hdf5_writer.py        # All h5py write logic
    hdf5_handler.py       # Delegator — no h5py write logic of its own
    pva_reader.py
    ca_monitor_registry.py
    ...
  consumers/              # pvapy HPC consumer processes
  database/               # SQLAlchemy models + profile DB
  scripts/                # One-off seed scripts
```

## Configuration — always use `settings.py`

`src/dashpva/settings.py` is the single source of truth. It resolves constants from the active configuration source, selected by the Workflow dialog (`settings.set_locator()` + `settings.reload()`).

**The database (DB profile) is the primary config source. TOML is secondary** — supported for legacy configs and quick one-off use, but new features and defaults should be built around the DB profile model. When both sources can satisfy a need, prefer the DB path.

**Rules:**
- Read config as `settings.HKL`, `settings.ROI`, `settings.METADATA_CA`, `settings.TOML_FILE`, etc.
- Use `settings.ensure_path()` when a file path is needed (handles TOML and DB → temp file).
- If config is missing, show a `QMessageBox.warning` directing the user to the Workflow dialog — never open a `QFileDialog` as a config fallback.
- Build file paths from `settings.PROJECT_ROOT`; do not recompute the root with `Path(__file__).parent...` chains in modules.
- `HDF5_STRUCTURE` (the NeXus schema constant) lives in `settings.py`, not in any handler.

```python
import dashpva.settings as app_settings

path = app_settings.PROJECT_ROOT / "outputs" / "result.h5"
toml  = app_settings.ensure_path()
channels = app_settings.METADATA_CA
```

## Styles — always use `theme.qss`

All widget styling goes in `src/dashpva/gui/theme.qss`. Never call `widget.setStyleSheet(...)` in Python for cosmetic styles.

To target a widget, give it an object name or dynamic property:

```python
widget.setObjectName("scanStatus")          # → QLabel#scanStatus { ... }
widget.setProperty("highlight", True)       # → QLabel[highlight="true"] { ... }
```

The one documented exception is `bayesian_viewer.py`, which applies its own Catppuccin palette intentionally via inline setStyleSheet.

## HDF5 writing

All actual `h5py` write logic lives in `utils/hdf5_writer.py`. `utils/hdf5_handler.py` is a pure delegator — add no write logic there. Use `settings.HDF5_STRUCTURE` for the NeXus NX_class schema.

Motor positions are written with resolved axis labels (ETA, MU, CHI, PHI) only. Raw PV strings are never stored; PVs without a TOML axis-label mapping are skipped.

## UI persistence — QSettings

Window geometry and dock state are persisted across sessions using `QSettings("DashPVA", "<ViewerName>")`. Use a module-level factory to get the handle:

```python
from PyQt5.QtCore import QSettings

def _settings() -> QSettings:
    return QSettings("DashPVA", "Viewer")
```

Read in `__init__` (after `uic.loadUi`) and write in `closeEvent`:

```python
# restore
s = _settings()
if geom := s.value("window_geom"):
    self.restoreGeometry(geom)

# save
def closeEvent(self, event):
    s = _settings()
    s.setValue("window_geom", self.saveGeometry())
    super().closeEvent(event)
```

For dock layout, bump a module-level `_DOCK_STATE_VERSION` integer whenever the dock set changes — `restoreState` silently rejects mismatched versions.

**Seed scripts** — `QSettings` stores UI state only. Application-level defaults (paths, profile JSON blobs, DB sections) are seeded into `dashpva.db` via the scripts in `src/dashpva/scripts/`. If you need to add or change a default value in the database, edit the appropriate seed script there:

- `seed_settings_defaults_sql.py` — app-level settings tree (paths, consumer names, APP_DATA sections)
- `seed_profile_defaults_sql.py` — per-profile JSON blob (`__data__`)

Both scripts are idempotent and safe to re-run.

## Adding a view

All standalone viewer windows must inherit from `BaseWindow` (`src/dashpva/viewer/core/base_window.py`), not raw `QMainWindow`. `BaseWindow` provides logging, the status bar, File/Windows/Documentation menus, dock-toggle wiring, and `update_status(message, level)`.

1. **Module** — create `src/dashpva/viewer/<name>/<name>.py` (subpackage) or `src/dashpva/viewer/<name>.py` for simpler views.

2. **Class** — inherit from `BaseWindow` and pass the UI file name and a human-readable viewer name:

   ```python
   from dashpva.viewer.core.base_window import BaseWindow

   class MyView(BaseWindow):
       def __init__(self):
           super().__init__(ui_file_name="my_view.ui", viewer_name="My View")
           # logger is already available as self.logger via BaseWindow
           self._setup_docks()
           self._connect_signals()
   ```

   Use `QDialog` (not `BaseWindow`) only for secondary modal/modeless panels that are not top-level application windows.

3. **UI file** — place the Qt Designer `.ui` file at `src/dashpva/gui/<name>.ui`. `BaseWindow` loads it automatically when `ui_file_name` is passed.

4. **`__main__` block** — every viewer module must be runnable directly:

   ```python
   if __name__ == "__main__":
       import sys
       from PyQt5.QtWidgets import QApplication
       from dashpva.gui import configure_app
       app = QApplication(sys.argv)
       configure_app(app)
       window = MyView()
       window.show()
       sys.exit(app.exec_())
   ```

5. **CLI command** — register a `@cli.command()` in `src/dashpva/cli.py` that launches the module as a subprocess:

   ```python
   @cli.command()
   def myview():
       """Launch My View — one-line description."""
       click.echo("Running My View")
       exit_code = subprocess.run(
           [sys.executable, "-m", "dashpva.viewer.<name>.<name>"]
       ).returncode
       sys.exit(exit_code)
   ```

6. **Settings** — import `dashpva.settings as app_settings` for any config. Never hardcode paths or PV names in the viewer module.

7. **Documentation** — `BaseWindow` already provides a **Documentation** entry in the menu bar that opens `DocumentationDialog` (`viewer/documentation/dialog.py`). Do not add your own documentation action, menu, or dialog. To add docs for a viewer, drop the file at:

   ```
   src/dashpva/viewer/<name>/doc/README.md      # Markdown (converted to HTML)
   src/dashpva/viewer/<name>/doc/index.html     # preferred when present
   ```

   The dialog auto-discovers either file next to the viewer's module. For a non-standard location, set `self.doc_path = "<absolute or relative path>"` on the viewer instance after `super().__init__(...)`.

## Architecture — components own their domain (all viewers)

This applies to **every** viewer (Workbench, Area Detector, HKL, Scan, …), not just the Workbench. Each viewer window is a **thin coordinator**, not a god object: it instantiates components, wires cross-cutting signals, and holds only genuinely shared state. Every dock and workspace owns the functionality specific to *its* concern; domain logic must live in the component, never as a blob inside the window (or the workspace).

**Rules (follow these in all viewers):**
- **Docks own their panel.** A dock computes and formats its own content. `Info2DDock` owns 2D-info presentation (dims, current-frame details, pixel readout), `Info3DDock` owns 3D info (HKL/volume/slice), `AnnotationDock` owns per-frame notes, `ROICalcDock` owns ROI math, the Area Detector's waterfall/stats docks own their own analysis, etc. A change to "how 2D info is shown" touches `info_2d_dock.py` alone.
- **Workspaces own their viewer, not their docks.** `Workspace2D` owns the 2D image view, controls, playback, levels, hover; `Workspace3D` owns the 3D view. A workspace may *notify* a dock that data changed (coordination), but must not contain that dock's presentation/analysis logic. Keep dock-specific behavior out of the workspace.
- **Components reach shared state via `self.main_window.<x>`** (e.g. `main_window.roi_manager`, `main_window.get_current_frame_data()`) — they do not duplicate it or reach into another component's internals.
- **The window exposes thin forwards for its own data**, so components need no edits when logic is relocated. Example: `WorkbenchWindow` provides read-only `@property` forwards (`image_view`, `plot_item`, `frame_spinbox`, `current_2d_data`) and passthrough methods (`display_2d_data`, `clear_2d_plot`, `get_current_frame_data`, `start_playback`, `open_roi_2d_plot_dock`) that delegate to `self.tab_2d`. This is the template: when a component moves out of the window, add a guarded forward rather than editing every caller.

When extending any viewer, put the code in the owning component. If it needs window-owned data, add/extend a forward — don't grow the window or push dock logic into a workspace.

## Adding a dock

All dockable panels must inherit from `BaseDock` (`src/dashpva/viewer/core/docks/base_dock.py`), not raw `QDockWidget`. `BaseDock.setup()` handles `addDockWidget`, sets a namespaced `objectName` for `saveState`/`restoreState`, and registers a toggle action under the Windows menu automatically.

**Subclass pattern:**

```python
from PyQt5.QtCore import Qt
from dashpva.viewer.core.docks.base_dock import BaseDock

class MyDock(BaseDock):
    def __init__(self, main_window, show: bool = True):
        super().__init__(
            title="My Panel",
            main_window=main_window,
            segment_name="analysis",   # groups this dock under Windows > Analysis
            dock_area=Qt.RightDockWidgetArea,
            show=show,
        )
        self._build_content()          # build widgets after super().__init__

    def _build_content(self):
        ...
        self.setWidget(my_widget)
```

**Instantiate and arrange in the viewer's `__init__`:**

```python
self.info_dock = InfoDock(main_window=self)
self.plot_dock = PlotDock(main_window=self)
self.detail_dock = DetailDock(main_window=self, show=False)  # hidden by default

# layout: split vertically, then tab
self.splitDockWidget(self.info_dock, self.plot_dock, Qt.Vertical)
self.tabifyDockWidget(self.info_dock, self.detail_dock)
self.info_dock.raise_()   # bring the default tab forward
```

**Key rules:**
- A dock owns the logic for its own concern (see *Workbench architecture* above). Compute/format the dock's content inside the dock, pulling shared data via `main_window` forwards — do not put dock-specific logic in the viewer window.
- `segment_name` groups related docks under a Windows submenu (e.g. `"2d"`, `"3d"`, `"analysis"`). Use a consistent name across docks that belong together.
- Bump the module-level `_DOCK_STATE_VERSION` integer in the viewer whenever the dock set changes — `restoreState` silently rejects mismatched versions.
- Use `splitDockWidget` / `tabifyDockWidget` / `resizeDocks` after all docks are created to establish the initial layout.
- Save and restore geometry/state via `QSettings` in `closeEvent` / `__init__` (see UI persistence section).

## Linting

This project uses `ruff`. Before producing a commit message, run:

```bash
ruff check src/dashpva/
```

Fix any issues first, then write the message.

## Commit style

One-line, prefix-tagged, lowercase. No body, no period, no AI-generated footer.

**Format:** `<prefix>: <short description>`

**Prefixes:** `fix:`, `add:`, `feat:`, `remove:`, `rename:`, `change:`, `revert:`, `Refactor` (capitalized, for large restructurings), `ruff check:` (lint-only commits).

Multiple unrelated changes → separate each prefix with a **blank line**:

```
remove: Fusion style from area_det main — was breaking macOS dark-mode palette

fix: theme.qss dock-header comment after Fusion drop

add: roi overlay color selector
```

`remove:` lines must include a brief why (`— replaced by X`, `— superseded by Y`). Other prefixes describe the change itself and need no extra justification.

Scope the message to staged changes only (`git diff --cached`).

## Releasing

Canonical release procedure — run every version bump in this order:

1. **`pyproject.toml`** → bump `version = "X.Y.Z"`.
2. **`uv lock`** → regenerates the `uv.lock` root entry to `X.Y.Z`. Commit `pyproject.toml` + `uv.lock` together.
3. **PR** → merge to `main`.
4. **`uv sync`** (or `install.sh --update`) so `__version__` reflects it locally.
5. **`git tag vX.Y.Z <main-commit> && git push origin vX.Y.Z`**.
6. **`gh release create vX.Y.Z --title vX.Y.Z --notes "…"`** (latest, non-draft) — this is what triggers the in-app "update available."
7. **Verify:** a `DashPVA` run shows `vX.Y.Z` + "Up to date"; `gh release list` shows it as Latest.

## Coding conventions

- **Change only what was asked.** Do not bundle adjacent cleanups, refactors, or style fixes into a scoped request. Mention adjacent issues in chat as follow-up suggestions instead.
- **No leftover scraps.** After any change, remove every import, helper, constant, comment, dead branch, or orphaned file the change made obsolete. `grep -rn` the old symbol to confirm zero leftover references.
- **Minimal comments.** Prefer no comment. If a comment is genuinely needed (hidden constraint, upstream bug workaround, non-obvious invariant), keep it to 1–2 lines maximum. Never write multi-line explanatory blocks alongside a fix.
- **No inline config.** All configuration constants belong in `settings.py`; never hardcode paths, PV names, or thresholds in viewer or utility modules.
- **Commit messages in chat.** When asked for a commit message, output it as text in the response. Do not run `git commit` unless the user explicitly asks to commit. If you are working in your own fork, you may commit and push freely — this rule applies only when working directly in the upstream repo.
