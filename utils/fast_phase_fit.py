"""
Fast multi-phase XRD fitting that bypasses lmfit's inner-loop overhead.

Uses PhaseFitter from ssrl_xrd_tools for setup (build_model, build_parameters)
but replaces the lmfit Model.fit() call with a direct scipy.optimize.leastsq
call, eliminating ~44% overhead from parameter bookkeeping (make_funcargs,
update_constraints, _strip_prefix).

Supports texture modes: 'none', 'march_dollase', and 'free'.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)

_GAUSS_FWHM_FACTOR = np.sqrt(2.0 * np.log(2.0))


def _metric_tensor_fast(a, b, c, alpha_d, beta_d, gamma_d):
    ar, br, gr = np.radians(alpha_d), np.radians(beta_d), np.radians(gamma_d)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)
    vol = a * b * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)
    a_star = b * c * sa / vol
    b_star = a * c * sb / vol
    c_star = a * b * sg / vol
    cos_alpha_star = (cb*cg - ca) / (sb*sg)
    cos_beta_star  = (ca*cg - cb) / (sa*sg)
    cos_gamma_star = (ca*cb - cg) / (sa*sb)
    return np.array([
        [a_star**2, a_star*b_star*cos_gamma_star, a_star*c_star*cos_beta_star],
        [a_star*b_star*cos_gamma_star, b_star**2, b_star*c_star*cos_alpha_star],
        [a_star*c_star*cos_beta_star, b_star*c_star*cos_alpha_star, c_star**2],
    ])


def _q_from_hkl_fast(hkl, G_star):
    inv_d2 = np.einsum("ij,jk,ik->i", hkl, G_star, hkl)
    inv_d2 = np.clip(inv_d2, 1e-30, None)
    return 2.0 * np.pi * np.sqrt(inv_d2)


def _march_dollase_fast(hkl, G_star, march_axis, march_r):
    """March-Dollase preferred-orientation correction."""
    h0 = march_axis
    num = np.einsum("ij,jk,k->i", hkl, G_star, h0) ** 2
    denom_hkl = np.einsum("ij,jk,ik->i", hkl, G_star, hkl)
    denom_h0 = float(h0 @ G_star @ h0)
    cos2a = np.clip(num / (denom_hkl * denom_h0 + 1e-30), 0.0, 1.0)
    sin2a = 1.0 - cos2a
    r = max(float(march_r), 1e-10)
    return (r**2 * cos2a + sin2a / r) ** (-1.5)


def _eval_phase_dense(x, centers, amplitudes, sigmas, fraction):
    if len(centers) == 0:
        return np.zeros_like(x)
    dx = x[:, None] - centers[None, :]
    sig = np.clip(sigmas, 1e-12, None)[None, :]
    amp = amplitudes[None, :]
    sig_g = sig / _GAUSS_FWHM_FACTOR
    gauss = (amp / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dx / sig_g)**2)
    lorentz = (amp / np.pi) * (sig / (dx*dx + sig*sig))
    return ((1.0 - fraction) * gauss + fraction * lorentz).sum(axis=1)


def _caglioti_sigma(q, U, V, W):
    sigma2 = U * q**2 + V * np.abs(q) + W
    return np.sqrt(np.clip(sigma2, 1e-10, None))


def _to_internal(val, lb, ub):
    """Map bounded value to unbounded internal space (lmfit-compatible)."""
    has_lb = np.isfinite(lb)
    has_ub = np.isfinite(ub)
    if has_lb and has_ub:
        return np.arcsin(2.0 * (val - lb) / (ub - lb) - 1.0)
    elif has_lb:
        return np.sqrt((val - lb + 1.0)**2 - 1.0)
    elif has_ub:
        return np.sqrt((ub - val + 1.0)**2 - 1.0)
    return val


def _to_external(internal, lb, ub):
    """Map unbounded internal value back to bounded external space."""
    has_lb = np.isfinite(lb)
    has_ub = np.isfinite(ub)
    if has_lb and has_ub:
        return lb + (ub - lb) * (np.sin(internal) + 1.0) / 2.0
    elif has_lb:
        return lb - 1.0 + np.sqrt(internal**2 + 1.0)
    elif has_ub:
        return ub + 1.0 - np.sqrt(internal**2 + 1.0)
    return internal


def _to_internal_array(vals, lbs, ubs):
    out = vals.copy()
    both = np.isfinite(lbs) & np.isfinite(ubs)
    lb_only = np.isfinite(lbs) & ~np.isfinite(ubs)
    ub_only = ~np.isfinite(lbs) & np.isfinite(ubs)
    if both.any():
        out[both] = np.arcsin(2.0 * (vals[both] - lbs[both]) / (ubs[both] - lbs[both]) - 1.0)
    if lb_only.any():
        out[lb_only] = np.sqrt((vals[lb_only] - lbs[lb_only] + 1.0)**2 - 1.0)
    if ub_only.any():
        out[ub_only] = np.sqrt((ubs[ub_only] - vals[ub_only] + 1.0)**2 - 1.0)
    return out


def _to_external_array(internals, lbs, ubs):
    out = internals.copy()
    both = np.isfinite(lbs) & np.isfinite(ubs)
    lb_only = np.isfinite(lbs) & ~np.isfinite(ubs)
    ub_only = ~np.isfinite(lbs) & np.isfinite(ubs)
    if both.any():
        out[both] = lbs[both] + (ubs[both] - lbs[both]) * (np.sin(internals[both]) + 1.0) / 2.0
    if lb_only.any():
        out[lb_only] = lbs[lb_only] - 1.0 + np.sqrt(internals[lb_only]**2 + 1.0)
    if ub_only.any():
        out[ub_only] = ubs[ub_only] + 1.0 - np.sqrt(internals[ub_only]**2 + 1.0)
    return out


class FastFitProblem:
    """Precompiled fitting problem for direct scipy minimisation."""

    def __init__(self, fitter, params):
        self.fitter = fitter
        self.original_params = deepcopy(params)

        # Data
        self.x = fitter.x
        self.y_fit = fitter.y_fit
        self.sigma = fitter.sigma if fitter.sigma is not None else None
        self.mask = getattr(fitter, 'fit_mask', np.ones(len(self.x), dtype=bool))
        self.x_fit = self.x[self.mask]
        self.y_data = self.y_fit[self.mask]
        self.weights = None
        if self.sigma is not None:
            sigma_slice = self.sigma[self.mask]
            self.weights = np.where(sigma_slice > 0, 1.0 / sigma_slice, 1.0)

        # Phase info
        self.n_phases = len(fitter._phase_models)
        self.phase_hkl = []
        self.phase_template_amp = []
        self.phase_alpha = []
        self.phase_beta = []
        self.phase_gamma = []
        self.phase_n_peaks = []
        self.phase_texture = []
        self.phase_march_axis = []

        for pm in fitter._phase_models:
            self.phase_hkl.append(pm.hkl.astype(float))
            self.phase_template_amp.append(pm.template_amp.copy())
            pre = pm.prefix
            self.phase_alpha.append(params[f"{pre}alpha"].value)
            self.phase_beta.append(params[f"{pre}beta"].value)
            self.phase_gamma.append(params[f"{pre}gamma"].value)
            self.phase_n_peaks.append(pm.hkl.shape[0])
            self.phase_texture.append(pm.texture)
            self.phase_march_axis.append(
                np.asarray(pm.march_axis, dtype=float) if pm.march_axis else np.array([0., 0., 1.])
            )

        # Background template
        self._has_bg_template = False
        self._bg_template_on_grid = None
        self._has_chebyshev = False
        self._cheb_degree = 0
        self._cheb_x_min = 0.0
        self._cheb_span = 1.0

        if fitter._bg_model is not None:
            for comp in getattr(fitter._bg_model, 'components', [fitter._bg_model]):
                if hasattr(comp, 'degree') and hasattr(comp, 'x_min'):
                    self._has_chebyshev = True
                    self._cheb_degree = comp.degree
                    self._cheb_x_min = comp.x_min
                    self._cheb_span = comp.x_max - comp.x_min
                    self._cheb_prefix = comp.prefix
                elif f'{comp.prefix}A' in params:
                    self._has_bg_template = True
                    x_ref, y_ref = fitter._fit_background_template
                    order = np.argsort(x_ref)
                    self._bg_template_on_grid = np.interp(
                        self.x_fit, x_ref[order], y_ref[order]
                    )

        self._has_amorphous = fitter._amorphous_model is not None
        self._build_param_map(params)

    def _build_param_map(self, params):
        self.vary_names = []
        self.vary_values = []
        self.vary_lower = []
        self.vary_upper = []

        for name, p in params.items():
            if p.vary:
                self.vary_names.append(name)
                self.vary_values.append(p.value)
                self.vary_lower.append(p.min if np.isfinite(p.min) else -np.inf)
                self.vary_upper.append(p.max if np.isfinite(p.max) else np.inf)

        self.n_vary = len(self.vary_names)
        self.x0 = np.array(self.vary_values)
        self.lower_bounds = np.array(self.vary_lower)
        self.upper_bounds = np.array(self.vary_upper)
        self._vary_idx = {name: i for i, name in enumerate(self.vary_names)}

        # Precompute bound masks for vectorized transforms
        self._both_bounded = np.isfinite(self.lower_bounds) & np.isfinite(self.upper_bounds)
        self._lb_only = np.isfinite(self.lower_bounds) & ~np.isfinite(self.upper_bounds)
        self._ub_only = ~np.isfinite(self.lower_bounds) & np.isfinite(self.upper_bounds)

        self._constraints = {}
        for name, p in params.items():
            if p.expr:
                self._constraints[name] = p.expr

        # Per-phase parameter info
        self._phase_param_info = []
        for i in range(self.n_phases):
            pre = f"p{i}_"
            info = {
                'prefix': pre,
                'scale_idx': self._vary_idx.get(f'{pre}scale'),
                'U_idx': self._vary_idx.get(f'{pre}U'),
                'V_idx': self._vary_idx.get(f'{pre}V'),
                'W_idx': self._vary_idx.get(f'{pre}W'),
                'fraction_idx': self._vary_idx.get(f'{pre}fraction'),
            }

            # Texture-specific params
            texture = self.phase_texture[i]
            if texture == 'free':
                info['pk_indices'] = [
                    self._vary_idx.get(f'{pre}pk{j}')
                    for j in range(self.phase_n_peaks[i])
                ]
            elif texture == 'march_dollase':
                info['march_r_idx'] = self._vary_idx.get(f'{pre}march_r')
            # texture='none' needs no extra params

            # Precompile lattice parameter resolution
            for key in ('a', 'b', 'c'):
                pname = f'{pre}{key}'
                if pname in self._vary_idx:
                    info[f'{key}_op'] = ('idx', self._vary_idx[pname])
                else:
                    info[f'{key}_op'] = self._compile_expr(pname)

            self._phase_param_info.append(info)

        # Background params
        self._bg_A_idx = self._vary_idx.get('bg_A')
        self._cheb_c_indices = []
        if self._has_chebyshev:
            cheb_pre = getattr(self, '_cheb_prefix', 'bg_')
            for j in range(self._cheb_degree + 1):
                self._cheb_c_indices.append(self._vary_idx.get(f'{cheb_pre}c{j}'))

        # Amorphous peak
        self._am_amp_idx = self._vary_idx.get('am_amplitude')
        self._am_center_idx = self._vary_idx.get('am_center')
        self._am_sigma_idx = self._vary_idx.get('am_sigma')
        self._q_shift_idx = self._vary_idx.get('q_shift')

        # Pre-compute Chebyshev mapped x (constant across evals)
        if self._has_chebyshev:
            self._cheb_t = 2.0 * (self.x_fit - self._cheb_x_min) / self._cheb_span - 1.0

        # Precompile constraint ops for fast resolution
        # Only compile simple constraints (add/sub/alias). Skip complex
        # expressions like am_fwhm/am_height that aren't used in the residual.
        self._constraint_ops = {}
        for name, expr in self._constraints.items():
            try:
                self._constraint_ops[name] = self._compile_expr(name)
            except (KeyError, ValueError):
                pass

    def _compile_expr(self, name):
        """Precompile a parameter's constraint expression into an opcode tuple."""
        if name in self._vary_idx:
            return ('idx', self._vary_idx[name])

        expr = self._constraints.get(name)
        if expr is None:
            return ('const', self.original_params[name].value)

        expr = expr.strip()
        if ' - ' in expr:
            parts = expr.split(' - ')
            if len(parts) == 2:
                a = self._compile_expr(parts[0].strip())
                b = self._compile_expr(parts[1].strip())
                return ('sub', a, b)
        if ' + ' in expr:
            parts = expr.split(' + ')
            if len(parts) == 2:
                a = self._compile_expr(parts[0].strip())
                b = self._compile_expr(parts[1].strip())
                return ('add', a, b)
        return self._compile_expr(expr)

    def _exec_op(self, theta, op):
        """Execute a precompiled opcode tuple."""
        tag = op[0]
        if tag == 'idx':
            return theta[op[1]]
        if tag == 'const':
            return op[1]
        if tag == 'add':
            return self._exec_op(theta, op[1]) + self._exec_op(theta, op[2])
        if tag == 'sub':
            return self._exec_op(theta, op[1]) - self._exec_op(theta, op[2])

    def _resolve_lattice(self, theta, phase_idx):
        info = self._phase_param_info[phase_idx]
        a = self._exec_op(theta, info['a_op'])
        b = self._exec_op(theta, info['b_op'])
        c = self._exec_op(theta, info['c_op'])
        return a, b, c

    def _eval_expr(self, theta, expr):
        expr = expr.strip()
        if ' - ' in expr:
            parts = expr.split(' - ')
            if len(parts) == 2:
                return self._get_param_value(theta, parts[0].strip()) - \
                       self._get_param_value(theta, parts[1].strip())
        if ' + ' in expr:
            parts = expr.split(' + ')
            if len(parts) == 2:
                return self._get_param_value(theta, parts[0].strip()) + \
                       self._get_param_value(theta, parts[1].strip())
        return self._get_param_value(theta, expr)

    def _get_param_value(self, theta, name):
        if name in self._vary_idx:
            return theta[self._vary_idx[name]]
        if name in self._constraints:
            return self._eval_expr(theta, self._constraints[name])
        return self.original_params[name].value

    def _eval_phase(self, theta, i, x_shifted):
        """Evaluate one phase's contribution."""
        info = self._phase_param_info[i]
        n_peaks = self.phase_n_peaks[i]
        if n_peaks == 0:
            return np.zeros_like(x_shifted)

        a, b, c = self._resolve_lattice(theta, i)
        G = _metric_tensor_fast(a, b, c, self.phase_alpha[i],
                                self.phase_beta[i], self.phase_gamma[i])
        centers = _q_from_hkl_fast(self.phase_hkl[i], G)

        U, V, W = theta[info['U_idx']], theta[info['V_idx']], theta[info['W_idx']]
        sigmas = _caglioti_sigma(centers, U, V, W)

        scale = theta[info['scale_idx']]
        texture = self.phase_texture[i]

        if texture == 'free':
            pk_mults = np.array([theta[idx] for idx in info['pk_indices']])
            amps = self.phase_template_amp[i] * scale * pk_mults
        elif texture == 'march_dollase':
            march_r = theta[info['march_r_idx']]
            md = _march_dollase_fast(self.phase_hkl[i], G,
                                     self.phase_march_axis[i], march_r)
            amps = self.phase_template_amp[i] * scale * md
        else:  # 'none'
            amps = self.phase_template_amp[i] * scale

        fraction = theta[info['fraction_idx']]
        return _eval_phase_dense(x_shifted, centers, amps, sigmas, fraction)

    def residual(self, theta):
        x = self.x_fit
        q_shift = theta[self._q_shift_idx] if self._q_shift_idx is not None else 0.0
        x_shifted = x - q_shift
        y_model = np.zeros_like(x)

        if self._has_bg_template and self._bg_A_idx is not None:
            y_model += theta[self._bg_A_idx] * self._bg_template_on_grid

        if self._has_chebyshev:
            coeffs = np.array([theta[idx] if idx is not None else 0.0
                               for idx in self._cheb_c_indices])
            y_model += np.polynomial.chebyshev.chebval(self._cheb_t, coeffs)

        if self._has_amorphous:
            am_amp = theta[self._am_amp_idx] if self._am_amp_idx is not None else 0.0
            am_cen = theta[self._am_center_idx] if self._am_center_idx is not None else 1.5
            am_sig = max(theta[self._am_sigma_idx] if self._am_sigma_idx is not None else 0.3, 1e-15)
            dx = x - am_cen
            y_model += (am_amp / (am_sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (dx / am_sig)**2)

        for i in range(self.n_phases):
            y_model += self._eval_phase(theta, i, x_shifted)

        resid = self.y_data - y_model
        if self.weights is not None:
            resid *= self.weights
        return resid


def fast_fit(fitter, params=None, max_nfev=3000, q_range=None, **fit_kwargs):
    """Run a fast multi-phase fit using direct scipy.optimize.leastsq.

    Completely bypasses lmfit during optimization — no per-eval overhead from
    make_funcargs, update_constraints, or _strip_prefix (~46% of baseline).
    Uses the same arcsin/sqrt parameter transforms as lmfit for bounded params,
    so the optimization landscape is identical.

    lmfit is only used for setup (build_parameters) and result wrapping
    (MultiPhaseResult compatibility).

    Parameters
    ----------
    fitter : PhaseFitter
        Fully configured fitter with phases added.
    params : lmfit.Parameters, optional
        Pre-built parameters. None → fitter.build_parameters(**fit_kwargs).
    max_nfev : int
        Max function evaluations.
    q_range : tuple or None
        (qmin, qmax) to restrict fit domain.
    **fit_kwargs
        Passed to fitter.build_parameters() if params is None.

    Returns
    -------
    MultiPhaseResult
    """
    from ssrl_xrd_tools.analysis.fitting.phase_fitting import MultiPhaseResult
    from scipy.optimize import leastsq

    build_kw_keys = {'phase_profile', 'texture', 'lattice_pct', 'q_shift_bound',
                     'lock_cross_phase', 'lock_lattice_order', 'pk_scale_range',
                     'width_model', 'caglioti', 'width_max', 'width_min',
                     'march_axis'}

    if params is None:
        build_kw = {k: v for k, v in fit_kwargs.items() if k in build_kw_keys}
        params = fitter.build_parameters(**build_kw)
    elif fitter.composite is None:
        build_kw = {k: v for k, v in fit_kwargs.items() if k in build_kw_keys}
        fitter.build_model(
            phase_profile=build_kw.get('phase_profile', 'pseudovoigt'),
            texture=build_kw.get('texture', 'none'),
        )

    if q_range is not None:
        qmin, qmax = float(q_range[0]), float(q_range[1])
        if qmin > qmax:
            qmin, qmax = qmax, qmin
        mask = (fitter.x >= qmin) & (fitter.x <= qmax)
    else:
        mask = np.ones_like(fitter.x, dtype=bool)
    fitter.fit_mask = mask

    problem = FastFitProblem(fitter, params)

    # Transform initial values to internal (unbounded) space
    x0_internal = _to_internal_array(
        problem.x0, problem.lower_bounds, problem.upper_bounds
    )

    lbs = problem.lower_bounds
    ubs = problem.upper_bounds

    def objective(x_internal):
        theta = _to_external_array(x_internal, lbs, ubs)
        return problem.residual(theta)

    result = leastsq(
        objective, x0_internal,
        maxfev=max_nfev,
        full_output=True,
    )
    x_opt_internal, cov_x, infodict, mesg, ier = result
    nfev = infodict['nfev']
    success = ier in (1, 2, 3, 4)

    # Transform optimised values back to external space
    theta_opt = _to_external_array(x_opt_internal, lbs, ubs)

    # Write optimised values back into lmfit params for MultiPhaseResult
    for i, name in enumerate(problem.vary_names):
        params[name].set(value=float(theta_opt[i]))

    # Resolve constraints so dependent params are updated
    for name, expr in problem._constraints.items():
        try:
            op = problem._constraint_ops[name]
            params[name].set(value=float(problem._exec_op(theta_opt, op)))
        except (KeyError, TypeError):
            pass

    # Compute fit statistics
    residual_final = problem.residual(theta_opt)
    ndata = len(residual_final)
    nvarys = problem.n_vary
    chisqr = float(np.sum(residual_final**2))
    redchi = chisqr / max(ndata - nvarys, 1)

    # Create a lightweight result object compatible with MultiPhaseResult
    fit_result = _FastResult(params, chisqr, redchi, ndata, nvarys,
                             nfev, success, mesg, ier)

    return MultiPhaseResult(fit_result, fitter)


class _FastResult:
    """Minimal lmfit.ModelResult-compatible object for MultiPhaseResult."""

    def __init__(self, params, chisqr, redchi, ndata, nvarys,
                 nfev, success, message, ier):
        self.params = params
        self.chisqr = chisqr
        self.redchi = redchi
        self.ndata = ndata
        self.nvarys = nvarys
        self.nfev = nfev
        self.success = success
        self.message = message
        self.ier = ier


def fast_fit_sequence(patterns, phases, config, *, sequential=False,
                      labels=None, fit_background_template=None,
                      progress_callback=None):
    """Fast version of fit_sequence using direct scipy solver.

    Drop-in replacement for ssrl_xrd_tools.analysis.fitting.fit_sequence
    but ~2x faster per pattern.
    """
    import time
    from ssrl_xrd_tools.analysis.fitting import PhaseFitter, FitResultStore

    store = FitResultStore()
    n = len(patterns)
    if labels is None:
        labels = [str(i) for i in range(n)]

    selected_phases = [
        p for p in phases if getattr(p, 'name', None) in config.phase_names
    ]

    prev_params = None

    for i, pat in enumerate(patterns):
        q, y = pat[0], pat[1]
        sigma = pat[2] if len(pat) > 2 else None

        init_kw = dict(config.init_kw)
        if fit_background_template is not None:
            init_kw.setdefault("fit_background", "template")
            init_kw["fit_background_template"] = fit_background_template

        if sigma is not None:
            fitter = PhaseFitter(q, y, sigma=sigma, **init_kw)
        else:
            fitter = PhaseFitter(q, y, **init_kw)

        for ph in selected_phases:
            fitter.add_phase(ph, min_intensity=config.min_intensity)

        fit_kw = dict(config.fit_kw)
        if sequential and prev_params is not None:
            fit_kw['params'] = prev_params

        t0 = time.perf_counter()
        try:
            result = fast_fit(fitter, **fit_kw)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            logger.warning("Pattern %s (%s) failed: %s", i, labels[i], exc)
            elapsed = time.perf_counter() - t0
            if progress_callback is not None:
                progress_callback(i, n, None)
            continue

        store.append(result, index=i, label=labels[i], elapsed=elapsed)

        if sequential:
            prev_params = deepcopy(result.params)

        if progress_callback is not None:
            progress_callback(i, n, result)

    return store
