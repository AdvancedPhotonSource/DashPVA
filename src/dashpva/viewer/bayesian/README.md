# Bayesian Optimization (blop) — DashPVA

DashPVA's Bayesian optimizer is powered by **[blop](https://github.com/bluesky/blop)**
(`blop.ax.Agent`), Bluesky's Ax/BoTorch-backed optimization library. It drives a
set of motors (degrees of freedom) to maximize or minimize a detector signal —
e.g. aligning to peak intensity.

## Architecture

| File | Role |
|------|------|
| `blop_adapter.py` | All blop contact lives here. Config dataclasses (`DOFSpec`, `ObjectiveSpec`, `OptimizerConfig`), device resolution, `build_agent()`, and the Bluesky plan `blop_optimize_plan()`. |
| `bayesian_viewer.py` | PyQt5 GUI: scalable DOF/objective tables (GUI-editable limits, many motors), live pyqtgraph plots, a background `ScanWorker` that owns the `RunEngine`. |
| `bluesky_compat.py` | Injects a beamline conda env's `site-packages` so `bluesky`/`ophyd`/`blop` import when DashPVA runs from its own venv. |

We use blop through its **public optimizer API** — `agent.suggest()` /
`agent.ingest()` — inside our own plan, reusing the proven
`move → trigger_and_read → extract_scalar` loop. This needs **no Tiled/Databroker**
(blop's default acquisition path reads acquired data back from Tiled; we read the
detector value directly from the `trigger_and_read` reading instead). The Agent is
still the optimizer brain (`suggest`/`ingest`); its underlying Ax model
(`agent.ax_client`) powers the model-surface views.

Heavy imports (`blop`, `ophyd`, `bluesky`, `torch`) are **lazy** — importing the
adapter or viewer module does not pull them, so the lean `area-det` install stays
torch-free.

## Installing

blop and its Ax/torch stack are in the `bayesian` (and `full`) extras, not the
base/`area-det` tier:

```bash
uv pip install -e '.[bayesian]'     # optimizer only
# or
uv pip install -e '.[full]'         # everything
```

> **blop version note.** The `blop.ax.Agent` API exists only in **blop 1.0.0b1+**
> (the legacy `blop.Agent` / `agent.learn()` shown in older docs is gone). On the
> released `1.0.0b1` the `[ax]` extra name does not exist yet (ax-platform/botorch
> are base deps there), so `blop[ax]` prints a harmless `uv`/`pip` warning but
> installs correctly. Newer versions split it into the `[ax]` extra, so we keep it.
>
> **Ax + SQLAlchemy.** Ax wants `sqlalchemy<2.0` while DashPVA needs `>=2.0`. Ax
> simply disables its own (unused-by-us) SQL storage and prints a warning. Harmless.

## Running the viewer

```bash
DashPVA bayesian          # or: python -m dashpva.viewer.bayesian.bayesian_viewer
```

> **Make sure `DashPVA` points at *this* checkout.** If you have more than one
> DashPVA clone, the `DashPVA` on your `PATH` may be a symlink into a different
> one (check with `which DashPVA` and `readlink -f $(which DashPVA)`). To run the
> code in this checkout, either call its venv binary directly —
> `/path/to/this/checkout/.venv/bin/DashPVA bayesian` — or repoint the symlink.

### Quick simulation test (no EPICS, no IOC)
The fastest way to try the optimizer is the built-in **Simulate (offline, no EPICS)**
checkbox in *Run controls*:

1. Add the DOFs you want (names + limits). In simulate mode the **motor/detector
   PV fields are ignored**, so you can leave them blank.
2. Tick **Simulate (offline, no EPICS)**.
3. Click **Start**.

This builds `ophyd.sim` motors and a synthetic detector (a Gaussian peaked at the
midpoint of each DOF's range) entirely in-process — no IOC or PVs required — so a
*maximize* run visibly climbs to the centre of your ranges and the live plots update.

### Testing against a real simulation IOC
To exercise the live-EPICS path instead, point the DOF rows at real motor PVs and
the detector at a scalar PV served by a soft IOC (e.g. an `epics` motorsim IOC plus
an areaDetector sim, or `DashPVA sim` for the detector image). Leave **Simulate**
unticked. The viewer connects via ophyd exactly as it will at the beamline.

In the GUI:

1. **Degrees of freedom** — add a row per motor (PV/name), set the search **Low/High
   limits** (editable spinboxes), and the type (`float`/`int`). Add as many motors
   as needed.
2. **Detector PV / Name** — the device whose signal you optimize.
3. **Objectives** — what to optimize, read from the detector (see
   [Adding objectives](#adding-objectives) below).
4. **Run controls** — how long to run (see
   [Run controls](#run-controls-iterations--points-per-iteration) below).
5. **Simulate (offline)** — tick to run without EPICS using `ophyd.sim` devices
   (handy for trying the UI).
6. **Start** — live plots show the objective vs. evaluation (with best-so-far) and a
   2-D projection scatter for a selectable DOF pair.

The configuration is remembered between launches (via `QSettings`).

### Adding objectives
An **objective** is a scalar value pulled from the detector reading that the
optimizer drives up or down. Edit objectives in the **Objectives** table:

| Column | Meaning |
|--------|---------|
| **Name** | A unique label for the objective (e.g. `intensity`). This is what the live plot and the "Best so far" readout report. |
| **Signal key** | The detector field to read (e.g. `stats1_total`). **Leave blank to auto-detect** — it tries common AreaDetector ROI-stat suffixes (`stats1_total`, `stats1_net`, `stats1_mean_value`, `total`, `net`, `mean_value`, `intensity`, `value`) and otherwise falls back to the first numeric field in the reading. |
| **Direction** | `maximize` (e.g. peak intensity / flux / alignment) or `minimize` (e.g. beam width / background). |

Steps:
1. Click **＋ Add objective** to add a row; **－ Remove selected** deletes the
   highlighted row (at least one objective is always kept).
2. Fill in **Name**, optionally a **Signal key** (blank = auto-detect), and pick a
   **Direction**.
3. Click **Start**.

Notes:
- **Single objective** (the common case): one row — the optimizer maximizes or
  minimizes it, and the live plots track it.
- **Names must be unique.** Each new row is auto-named uniquely (`intensity`, then
  `objective2`, …); duplicate names are rejected at Start. So you can't have two
  objectives both literally named `intensity`.
- **Two of the *same* signal makes no sense** — optimizing the identical detector
  value twice adds nothing. Multiple objectives are for trading off *different*
  quantities, e.g. **maximize** `stats1_total` (flux) **and minimize** a width/FWHM
  signal. Give each a distinct *Name* and *Signal key*.
- **Multiple objectives:** add more rows. blop optimizes them jointly
  (multi-objective / Pareto); the **first** row drives the live
  convergence/best-so-far plot.
- All objective signals are read from the **same Detector PV** at each point, so
  each *Signal key* must exist in that detector's reading.

### Run controls: iterations & points per iteration
The optimizer runs in **rounds (iterations)**. Each round it (1) suggests one or
more candidate points from its current model, (2) moves the motors there and reads
the detector for each, then (3) updates the model with the result(s).

- **Iterations (rounds)** — how many such rounds to run.
- **Points / iteration (batch)** — how many points are suggested *and measured*
  within a round, **before** the model updates:
  - **1** (default, recommended) — classic *sequential* Bayesian optimization:
    measure one point, learn from it, then pick the next. Most sample-efficient.
  - **> 1** — *batch* optimization: the model proposes several points at once from
    the same model state, you measure them all, then it updates once. Fewer model
    updates per point (handy if you'd rather refit less often), but it usually
    needs more total points to converge than sequential.
- **Total evaluations** ≈ iterations × points/iteration — an **upper bound**, since
  the initial exploration rounds can return slightly fewer points.

Ax automatically explores with quasi-random (Sobol) points for the first trials,
then switches to the Gaussian-process model — there's no separate "random init"
setting to configure.

### Stop, Resume, Reset
- **Stop** halts the run but keeps the model and history; the button becomes
  **Resume**.
- **Resume** (Start after a stop *or* after a finish) continues the **same** model
  from where it left off — bump *Iterations* first to run that many **more** points.
  While a model is loaded the run-control labels switch to **"+ Iterations"** /
  **"+ Total evaluations"** to signal the counts are *additions* to the run.
- **Reset** clears everything (model, plots, surface) back to scratch: the labels
  revert and the button returns to **Start**, so the next Start builds a fresh model.

### The 2-D projection views
The lower plot projects the search space onto a **pair of DOFs** you pick with the
**X axis / Y axis** dropdowns (they list *your* DOF names). The **View** dropdown
chooses what to draw:

- **Measured points** — where you've sampled, each point colored by its objective
  value. (Live; updates as the scan runs.)
- **Predicted surface** — the model's predicted objective across the two DOFs: the
  landscape it has learned.
- **Uncertainty (σ)** — the model's standard error: bright = where it's unsure.
- **Acquisition (UCB)** — an optimistic estimate (`mean + κ·σ`, or `mean − κ·σ`
  when minimizing) — roughly where the optimizer is drawn to sample next.

The three model views are computed **on demand**: pick a View and click
**Update surface** (enabled after a run). Other DOFs are held at the current best
point, and the measured points are overlaid so you can see coverage. Computing a
surface evaluates the model on a grid, so it runs only when you ask and only while
a scan isn't active.

### Beamline vs. local
At the beamline, `bluesky`/`ophyd`/`blop` and the device definitions normally live
in a conda env — set its path in **Bluesky Conda Env** (or `DASHPVA_BLUESKY_ROOT`).
Locally, whatever is importable on `sys.path` (the uv venv) is used first.

## Tests

```bash
# Fast unit tests (config validation, scalar extraction, serialization).
# The single in-file blop integration test is skipped if blop is absent.
pytest tests/unit/test_blop_adapter.py -v

# Thorough end-to-end simulation suite (real GP optimization through the full
# DashPVA path). Skipped automatically if blop/bluesky/ophyd are not installed.
pytest tests/integration/test_bayesian_simulation.py -v
```

The simulation suite (`tests/integration/test_bayesian_simulation.py`) runs the
real optimizer against simulated devices whose detector computes a known analytic
function from the motor positions, and asserts convergence toward the known
optimum:

- **Himmelblau (2-D, minimize)** — the same benchmark as blop's own
  `simple-experiment` tutorial; must converge to one of the four global minima.
- **Gaussian peak (2-D, maximize)** — must climb to the peak.
- **Sphere (4-D, minimize)** — proves the adapter/GUI scale past two motors.
- **Batch acquisition** — `n_points > 1` per iteration.
- **Direction** — maximize vs. minimize on the same parabola land on opposite ends.

These are slow (each is a real GP loop); expect a few minutes.

## Extending

- **More motors / objectives:** purely a config change — add `DOFSpec` /
  `ObjectiveSpec` entries (the GUI does this via the tables). Multi-objective uses
  blop's `Objective` list; `ScalarizedObjective` / `OutcomeConstraint` are available
  in `blop.ax` if needed.
- **Level-set / boundary finding:** the old home-grown engine's "straddle"
  acquisition was intentionally dropped (blop does maximize/minimize). If you ever
  need boundary tracing, add a custom Ax acquisition inside `blop_adapter.py`.
- **Model contour in-GUI:** `agent.plot_objective(x_dof, y_dof, obj)` returns an Ax
  analysis (Plotly); it can be popped out or embedded as a future enhancement.
