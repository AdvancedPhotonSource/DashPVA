"""
viewer.bayesian
===============
Bayesian optimization for DashPVA, powered by Bluesky's **blop**
(``blop.ax.Agent`` — an Ax/BoTorch Bayesian optimizer).

Modules
-------
blop_adapter   – DashPVA-owned adapter around ``blop.ax.Agent``: config
                 dataclasses (DOFSpec/ObjectiveSpec/OptimizerConfig),
                 device resolution, agent construction, and the Bluesky
                 suggest/move/read/ingest optimization plan.
bayesian_viewer – PyQt5 GUI: scalable DOF/objective tables (with GUI-editable
                 limits) and live optimization plots.
bluesky_compat  – Compatibility layer for importing bluesky/ophyd/blop from a
                 beamline conda environment.
"""
