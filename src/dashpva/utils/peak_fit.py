"""Fast 1D peak fitting for live beam-profile metrology.

Pure numpy/scipy (no ssrl-xrd-tools), so this works in every install tier
including the lean ``area-det`` edition. Used by the area-detector Beam Profiler
dock to fit a summed ROI intensity profile with one of three peak shapes and
report FWHM, amplitude, center offset, and a Poisson-weighted reduced
("regularized") chi-square.

The module is coordinate-agnostic: the caller passes an ``x`` axis already
centered on the ROI (0 == ROI center), so ``PeakFit.center`` is directly the
beam offset from the ROI center in whatever units ``x`` is in (ROI-local pixels
in the dock).

Models (``A`` = peak height above background ``c``):
  * gaussian    A*exp(-(x-x0)^2 / (2*sigma^2)) + c      FWHM = 2*sqrt(2*ln2)*sigma
  * lorentzian  A*gamma^2 / ((x-x0)^2 + gamma^2) + c    FWHM = 2*gamma
  * laplacian   A*exp(-|x-x0| / b) + c                  FWHM = 2*ln2*b
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

# Sigma -> FWHM for a Gaussian (matches fast_phase_fit._GAUSS_FWHM_FACTOR usage).
_GAUSS_FWHM_FACTOR = np.sqrt(2.0 * np.log(2.0))
_FWHM_GAUSS = 2.0 * _GAUSS_FWHM_FACTOR   # ~2.354820
_FWHM_LAPLACE = 2.0 * np.log(2.0)        # ~1.386294

# Minimum finite samples needed to fit 4 parameters and still have >=1 dof.
_MIN_POINTS = 5

MODELS = ("gaussian", "lorentzian", "laplacian")


@dataclass
class PeakFit:
    """Result of fitting one 1D profile. ``success=False`` means the numbers
    other than ``center``/``amplitude`` (best-effort seeds) are not meaningful."""

    amplitude: float = float("nan")   # A, height above background
    center: float = float("nan")      # x0, offset from ROI center (x units)
    fwhm: float = float("nan")        # model FWHM from the width parameter
    width: float = float("nan")       # native width param: sigma | gamma | b
    background: float = float("nan")  # c, constant offset
    redchi: float = float("nan")      # reduced chi-square (Poisson weights)
    r_squared: float = float("nan")   # coefficient of determination (1 = perfect)
    success: bool = False
    model: str = "gaussian"
    peak_x: float = float("nan")      # x of the maximum (== center for these models)
    y_fit: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


# --- model functions (module level so curve_fit can call them plainly) --------
def gaussian(x, A, x0, sigma, c):
    sigma = max(float(sigma), 1e-12) if np.isscalar(sigma) else np.maximum(sigma, 1e-12)
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


def lorentzian(x, A, x0, gamma, c):
    gamma = max(float(gamma), 1e-12) if np.isscalar(gamma) else np.maximum(gamma, 1e-12)
    return A * (gamma * gamma) / ((x - x0) ** 2 + gamma * gamma) + c


def laplacian(x, A, x0, b, c):
    b = max(float(b), 1e-12) if np.isscalar(b) else np.maximum(b, 1e-12)
    return A * np.exp(-np.abs(x - x0) / b) + c


# model name -> (function, sigma->native width seed, native width->FWHM)
_MODEL_CONFIG = {
    "gaussian": (gaussian, lambda s: s, lambda w: _FWHM_GAUSS * w),
    "lorentzian": (lorentzian, lambda s: s, lambda w: 2.0 * w),
    "laplacian": (laplacian, lambda s: s / np.sqrt(2.0), lambda w: _FWHM_LAPLACE * w),
}


def _initial_guess(x, y):
    """Moment-based seed: (A0, x0, sigma0, c0) plus the x-span and sample step."""
    c0 = float(np.min(y))
    a0 = float(np.max(y) - c0)
    span = float(x[-1] - x[0]) if x[-1] != x[0] else float(np.ptp(x))
    dx = span / max(len(x) - 1, 1)

    yb = np.clip(y - c0, 0.0, None)
    ssum = float(yb.sum())
    if ssum > 0.0:
        x0 = float((x * yb).sum() / ssum)
        var = float((yb * (x - x0) ** 2).sum() / ssum)
    else:  # flat/degenerate profile -> fall back to argmax and a quarter-span
        x0 = float(x[int(np.argmax(y))])
        var = (span / 4.0) ** 2

    sigma0 = np.sqrt(max(var, dx * dx))
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = max(span / 4.0, dx)
    return a0, x0, float(sigma0), c0, span


def fit_profile(x, y, model="gaussian", *, weights="poisson", maxfev=2000) -> PeakFit:
    """Fit a 1D profile ``y(x)`` with ``model`` and return a :class:`PeakFit`.

    ``weights="poisson"`` uses sigma = sqrt(max(y, 1)), appropriate for *summed
    detector counts* (shot-noise dominated), so ``redchi ~ 1`` means the fit is
    consistent with Poisson noise. Use ``weights="none"`` (ordinary least
    squares) for background-subtracted, flat-fielded, or frame-*averaged* data,
    where the Poisson assumption is invalid and ``redchi`` is only a relative
    goodness indicator.

    Never raises: returns ``PeakFit(success=False, ...)`` on non-convergence or
    degenerate input (flat, all-NaN, fewer than 5 finite points).
    """
    model = model.lower()
    if model not in _MODEL_CONFIG:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    func, sigma_to_width, width_to_fwhm = _MODEL_CONFIG[model]

    x_full = np.asarray(x, dtype=float)
    y_full = np.asarray(y, dtype=float)
    fail = PeakFit(model=model)

    m = np.isfinite(x_full) & np.isfinite(y_full)
    xf, yf = x_full[m], y_full[m]
    if xf.size < _MIN_POINTS:
        return fail

    a0, x0, sigma0, c0, span = _initial_guess(xf, yf)
    # Flat profile: nothing to fit. Seed center/amplitude for a best-effort read.
    if not np.isfinite(a0) or a0 <= 0.0 or span <= 0.0:
        fail.center = fail.peak_x = x0 if np.isfinite(x0) else float("nan")
        fail.amplitude = a0 if np.isfinite(a0) else float("nan")
        return fail

    width0 = float(np.clip(sigma_to_width(sigma0), 1e-9, span))
    xmin, xmax = float(xf.min()), float(xf.max())
    lo = [0.0, xmin, 1e-9, c0 - abs(a0) - 1.0]
    hi = [np.inf, xmax, span, float(yf.max())]
    p0 = [
        a0,
        float(np.clip(x0, xmin, xmax)),
        width0,
        float(np.clip(c0, lo[3], hi[3])),
    ]

    if weights == "poisson":
        sigma = np.sqrt(np.maximum(yf, 1.0))
    else:
        sigma = None

    try:
        popt, _ = curve_fit(
            func, xf, yf, p0=p0, bounds=(lo, hi),
            sigma=sigma, absolute_sigma=(sigma is not None), maxfev=maxfev,
        )
    except (RuntimeError, ValueError, TypeError):
        fail.center = fail.peak_x = x0
        fail.amplitude = a0
        return fail

    amp, cen, width, bg = (float(v) for v in popt)
    fwhm = float(width_to_fwhm(width))

    model_y = func(xf, *popt)
    resid_sigma = sigma if sigma is not None else 1.0
    resid = (yf - model_y) / resid_sigma
    redchi = float(np.sum(resid ** 2) / max(xf.size - 4, 1))

    # R^2 (coefficient of determination): 1 = perfect, robust to the absolute
    # noise scale, so it reads ~1 for a good fit even when ROI interpolation
    # smooths the pixel noise below the Poisson level assumed by redchi.
    ss_res = float(np.sum((yf - model_y) ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Reject physically implausible fits (runaway width, non-finite goodness).
    ok = np.isfinite(fwhm) and 0.0 < fwhm <= 4.0 * span and np.isfinite(redchi)
    return PeakFit(
        amplitude=amp, center=cen, fwhm=fwhm, width=width, background=bg,
        redchi=redchi, r_squared=r2, success=bool(ok), model=model, peak_x=cen,
        y_fit=func(x_full, *popt),
    )
