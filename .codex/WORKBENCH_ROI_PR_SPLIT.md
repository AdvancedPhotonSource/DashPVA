# Workbench ROI / Notebook PR Split Notes

This file is a local Codex handoff. Do not include it in a product PR unless requested.

## Current Branch

- Repository: `/home/beams18/USER6IDB/DashPVA`
- Branch: `dev-osayi`
- Remote tip: `5c17b6e1`
- Do not push unless the user explicitly says to push.
- Current working tree has uncommitted batch-ROI changes.

## Concern 1: Notebook Updates

- Already inherited from `main` through PR #159.
- Relevant commits include:
  - `5bf9640d` add notebooks and notebook index
  - `0197fca1` add RSM grid 3D rendering
  - `29ec9575` use a temporary mask directory
  - `359ce41a` guard Quickstart cells without data
  - `9d0d0cbf` remove stale outputs/execution counts
- Keep notebook-only work separate from Workbench ROI code.

## Concern 2: Saved ROI Reappears

- Commit on `dev-osayi`: `f2483f63`
- Purpose: retain source dataset linkage so saved Workbench ROIs render after reload.
- Expected test: load a file with an ROI, save, reload, and confirm the ROI is still present.
- Primary file: `src/dashpva/viewer/workbench/managers/roi_manager.py`

## Concern 3: ROI Geometry Fields

- Commit on `dev-osayi`: `38b9d88c`
- Purpose: X, Y, width, and height edits remain independent.
- Test file: `tests/unit/test_workbench_roi_geometry_fields.py`

## Concern 4: Independent ROI Save To

- Commits on `dev-osayi`:
  - `41223c26` initial multi-file save
  - `7021ccdf` file/folder chooser and destination details
  - `5c17b6e1` collision-safe names and editable default name
- `Save ROI` saves to the current file.
- `Save To…` saves independent copies to selected files or a folder.
- The current/original file is included.
- The confirmation dialog's **Show Details…** lists destination paths.
- Name collisions append `(N)` rather than overwrite.

## Concern 5: Tracked Batch ROIs

- Status: local, uncommitted, and unpushed.
- Current modified files:
  - `src/dashpva/viewer/workbench/managers/roi_manager.py`
  - `tests/unit/test_workbench_roi_batch_save.py`
- Intended behavior:
  - `Save as Batch…` creates linked ROI copies in chosen files/folder.
  - Batch identity, file manifest, name, and color persist as HDF5 attributes.
  - Moving/resizing a batch member does not write any file.
  - `Save ROI` is disabled for batch members; explicit `Save Batch` updates all remaining linked copies without another confirmation.
  - BaseWindow always installs `Ctrl+S` and routes it through overridable `handle_save_shortcut()`, even when a viewer has no File → Save action.
  - A viewer can override `handle_save_shortcut()`; the BaseWindow default calls `save_file()`.
  - In Workbench, `Ctrl+S` saves the active regular ROI or explicitly saves its batch.
  - Batch members use the same persisted color.
  - `Detach from Batch` keeps the current file's ROI but removes its batch metadata.
  - Deleting a batch member deletes only the current file's ROI and removes that file from the batch manifest; other copies remain.
  - `Save To…` remains independent and does not create a tracked batch.
- Latest validation before the newest requests: 21 focused tests passed and Ruff passed.

### Still Requested for Batch ROIs

- Add batch representation to the existing ROI Manager dock/list.
- Put batch actions in the ROI dock context menu as well as the canvas ROI context menu.

## Concern 6: 2D Information / Eta Motor Update

- Status: focused portion applied locally and uncommitted.
- "Eta changing" refers to the sample diffraction motor changing by frame, not an estimated-time field.
- Applied only:
  - Frame-dependent sample/detector motor readout in the 2D Info dock.
  - `RSMConverter.get_motor_positions()` helper.
- Modified files for this concern:
  - `src/dashpva/viewer/workbench/docks/info_2d_dock.py`
  - `src/dashpva/utils/rsm_converter.py`
  - `tests/unit/test_rsm_converter_geometry_refactor.py`
- Likely source is `stash@{0}` (`user work before ROI branch testing`), but it is a very large mixed stash (about 70 files) and must not be applied wholesale.
- Candidate 2D-information files in that stash:
  - `src/dashpva/gui/workbench/docks/information_dock.ui`
  - `src/dashpva/gui/workbench/workspace/workspace_2d.ui`
  - `src/dashpva/viewer/workbench/docks/info_2d_dock.py`
  - `src/dashpva/viewer/workbench/docks/information_dock_base.py`
  - `src/dashpva/viewer/workbench/workspace/workspace_2d.py`
- If more 2D-info behavior is requested, extract only reviewed hunks into a separate branch/commit.
- Do not combine the broad Workbench/HKL/UI changes from the stash without review.

## Recommended Branch / PR Order

1. `fix/workbench-saved-roi-reappears` — saved ROI source linkage only.
2. `fix/workbench-roi-geometry-fields` — X/Y/W/H editing only.
3. `add/workbench-roi-save-to` — independent file/folder save and collision naming.
4. `add/workbench-batch-rois` — tracked batches, update, detach, delete membership, shared color, dock representation, copy/convert.
5. `fix/workbench-2d-info-eta` — narrowly extracted ETA/2D-info stash changes.
6. Notebook changes remain their own docs/notebooks history and should not be mixed with ROI PRs.

## Validation Commands

```bash
PYTHONPATH=src QT_QPA_PLATFORM=offscreen \
  /home/beams/USER6IDB/.conda/envs/bluesky_2025_1/bin/python -m pytest -q \
  tests/unit/test_workbench_roi_batch_save.py \
  tests/unit/test_workbench_roi_geometry_fields.py \
  tests/unit/test_window_layout_persistence.py

ruff check \
  src/dashpva/viewer/workbench/managers/roi_manager.py \
  tests/unit/test_workbench_roi_batch_save.py

git diff --check
```

## Safety Notes

- Do not drop or apply stashes wholesale.
- Preserve `backup/dev-osayi-before-cleanup`.
- Before splitting commits, make a temporary backup branch or stash the local batch work.
- Never push until the user explicitly requests it.
