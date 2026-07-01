"""Unit tests for dashpva.utils.peak_fit (edition-agnostic numpy/scipy fitter)."""

import numpy as np
import pytest

from dashpva.utils.peak_fit import (
    MODELS,
    PeakFit,
    fit_profile,
    gaussian,
    laplacian,
    lorentzian,
)

# model name -> (generator func, width param, true FWHM from that width)
_CASES = {
    "gaussian": (gaussian, 10.0, 2.0 * np.sqrt(2.0 * np.log(2.0)) * 10.0),
    "lorentzian": (lorentzian, 8.0, 2.0 * 8.0),
    "laplacian": (laplacian, 10.0, 2.0 * np.log(2.0) * 10.0),
}


def _synth(model, x0=0.0, amp=2000.0, bg=20.0, seed=0):
    """Poisson-noised counts for `model` on a centered axis."""
    func, width, true_fwhm = _CASES[model]
    x = np.arange(-60.0, 60.0 + 1.0)  # 121 points, spacing 1
    clean = func(x, amp, x0, width, bg)
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.clip(clean, 0, None)).astype(float)
    return x, y, true_fwhm


@pytest.mark.parametrize("model", MODELS)
def test_recovers_fwhm_and_center(model):
    x, y, true_fwhm = _synth(model, x0=5.0)
    fit = fit_profile(x, y, model)
    assert fit.success
    assert fit.center == pytest.approx(5.0, abs=1.5)
    assert fit.fwhm == pytest.approx(true_fwhm, rel=0.10)
    assert fit.amplitude > 0
    assert fit.r_squared > 0.9
    assert fit.y_fit.shape == x.shape


@pytest.mark.parametrize("model", MODELS)
def test_redchi_near_one_for_poisson_data(model):
    # Matching model + Poisson weights => reduced chi-square near 1.
    x, y, _ = _synth(model, seed=3)
    fit = fit_profile(x, y, model)
    assert fit.success
    assert 0.3 < fit.redchi < 3.0


def test_center_offset_is_relative_to_axis_zero():
    x, y, _ = _synth("gaussian", x0=-12.0)
    fit = fit_profile(x, y, "gaussian")
    assert fit.success
    assert fit.center == pytest.approx(-12.0, abs=1.5)


def test_flat_profile_fails_gracefully():
    x = np.arange(-30.0, 30.0)
    y = np.full_like(x, 100.0)
    fit = fit_profile(x, y, "gaussian")
    assert isinstance(fit, PeakFit)
    assert not fit.success


def test_all_nan_fails_gracefully():
    x = np.arange(-30.0, 30.0)
    y = np.full_like(x, np.nan)
    fit = fit_profile(x, y, "gaussian")
    assert not fit.success


def test_too_few_points_fails_gracefully():
    fit = fit_profile(np.array([0.0, 1.0, 2.0]), np.array([1.0, 9.0, 1.0]), "gaussian")
    assert not fit.success


def test_zero_amplitude_fails_gracefully():
    x = np.arange(-30.0, 30.0)
    y = np.zeros_like(x)
    fit = fit_profile(x, y, "gaussian")
    assert not fit.success


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        fit_profile(np.arange(10.0), np.ones(10), "voigt")


def test_weights_none_still_fits():
    x, y, true_fwhm = _synth("gaussian")
    fit = fit_profile(x, y, "gaussian", weights="none")
    assert fit.success
    assert fit.fwhm == pytest.approx(true_fwhm, rel=0.10)
