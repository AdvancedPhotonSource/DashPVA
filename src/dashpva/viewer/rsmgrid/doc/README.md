# RSM Volume Builder

Merges one or more **completed** scan HDF5 files into a single gridded 3D
reciprocal-space volume, using `xrayutilities.Gridder3D`. This is the capability
rsMap3D provides that DashPVA previously lacked: the live area-detector viewer
converts angles to Q per frame, but can only show a bounded point cloud — never a
merged, interpolated volume.

Launch with `DashPVA rsmgrid`, or from the **Post Analysis** launcher tile.

Output uses the existing HDF5 volume format (`HDF5Loader.save_vol_to_h5`), so results
open directly in the Workbench 3D viewer with no extra steps.

---

## Using it

1. **Add Files…** — one or more scan HDF5 files. Each must contain
   `entry/data/data` with shape `(frames, direction1, direction2)` plus the
   `entry/data/metadata/HKL/...` geometry group.
2. **Grid Resolution** — `nx`, `ny`, `nz`, each at least 2. The estimated peak RAM
   updates as you change them.
3. **Apply active mask** — optional; see *Masked pixels* below.
4. **Normalize by** — optional monitor (I0) channel; see *Monitor normalization*.
5. **Output** — destination `.h5`.
6. **Start.**

---

## Behavior that is not obvious from the UI

### Axes are H, K, L — not Q

Each file's own UB matrix is passed to `Ang2Q.area`, which returns **hkl**, not Q in
Å⁻¹. The saved volume is labelled `H`/`K`/`L` and `coordinate_system` is `HKL`.

### Two passes over the data are required

This is a correctness requirement, not an optimization. `Gridder3D` latches its
`fixed_range` flag on the **first** call when `KeepData` is set. After that, any point
outside the range established by that first batch is dropped silently — no error, no
warning. The global bounds across every file must therefore be known and installed via
`dataRange(..., fixed=True)` *before* any gridding happens.

So pass 1 computes coordinates and accumulates bounds; pass 2 recomputes coordinates
and bins intensity. Intensity is only read in pass 2.

### Masked pixels are excluded, never zeroed

`Gridder3D.data` is a **per-bin mean**, not a sum. A masked pixel written as `0` would
count as a genuine `intensity = 0` measurement and drag its voxel's mean down. Masked
pixels are therefore removed from the coordinate and intensity arrays entirely.

The mask is read from the active `MaskManager`, which stores it in the **viewer's
display orientation**. That orientation is not persisted anywhere, so the dialog asks
rather than guessing — tick *"Mask was made with the viewer transposed"* if it was.

A mask captured at one detector binning or ROI cannot be applied to a scan taken at
another; the shapes will not match and the build is rejected.

### Non-finite points are dropped and reported

Points with non-finite coordinates or intensity are excluded from binning. The count is
reported in the log and stored as `num_points_excluded_nonfinite`. Bounds are computed
from finite coordinates only, so the binned set is always a subset of the bounded set —
no point can silently fall outside the latched range.

### Energy and UB differences warn, they do not block

Each file's own UB is applied, so the merge already lands in a common crystal-fixed HKL
frame; HKL is also geometrically energy-independent. rsMap3D applies UB per scan and
does not block either. Every file's energy and UB are recorded individually in the
output metadata (`source_energies_eV`, `source_ub_matrices`) so the merge stays
auditable.

### Monitor normalization

Optional, from `entry/data/metadata/ca/<name>`, matching rsMap3D's monitor division.
Without it, scans taken at different exposure time or attenuation **will not match in
intensity where they overlap**, and the per-bin mean across the overlap will be
meaningless. Monitor values must be finite and strictly positive.

### Grid origin sits half a voxel below the first bin

`Gridder3D`'s axes are bin **centers**, but the PyVista viewer treats `grid_origin` as
a cell **corner**. The saved origin is therefore shifted down by half a voxel so the
first cell is centered on the first bin.

---

## Memory

The dense result is unavoidably `nx · ny · nz` voxels; `Gridder3D` holds two float64
arrays and `.data` materializes a third, so budget roughly **24 bytes per voxel** for
the grid alone, plus one processing batch.

Coordinate conversion is batched against a **byte** budget rather than a fixed frame
count — a batch size that is fine on a 512² detector needs many GB on a 2048² one.

The dialog estimates the conservative peak and refuses to start a build that would
exceed a safe fraction of currently available RAM. If a grid is rejected, reduce `nx`,
`ny`, or `nz`. All the thresholds live in `settings.py`
(`RSM_GRID_BATCH_MEMORY_BYTES`, `RSM_GRID_MAX_MEMORY_FRACTION`,
`RSM_GRID_WORKING_BYTES_PER_PIXEL`).

Scripted and HPC callers can bypass the guard with
`build_volume(..., memory_limit_fraction=None)` when memory is managed externally.

---

## Supported geometry

Any number of sample and detector circles is resolved from numerically ordered
`SAMPLE_CIRCLE_AXIS_N` / `DETECTOR_CIRCLE_AXIS_N` groups. When a role has no
numbered groups, the legacy `MU`/`ETA`/`CHI`/`PHI` and `NU`/`DELTA` names remain
supported.

Metadata that the conversion treats as static (photon energy, detector center, size,
distance, beam and sample reference directions) is validated for consistency across
frames. Small EPICS readback jitter is tolerated; a genuinely swept value is rejected,
since a single static geometry cannot represent it.

---

## Known gaps vs rsMap3D

- No flat-field correction.
- No user-specified grid crop.
- No detector tilt — a pre-existing `rsm_converter` limitation.
- No out-of-core gridding: the volume is held entirely in RAM, so a very large grid
  (~1000³, roughly 24 GB) is rejected by the memory guard rather than degrading to
  chunked accumulation.
