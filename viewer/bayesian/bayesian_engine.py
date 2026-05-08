"""
bayesian_engine.py
==================
Production Bayesian Optimization Engine for X-ray Beamline 6-ID (APS).

This module provides a self-contained, stateful BayesianOptimizer class that
wraps GPyTorch's Exact GP framework.  It is designed to operate entirely on
real experimental data – there is NO simulator or ground-truth logic here.

Mathematical overview
---------------------
The GP models a latent function  f: R^2 → R  that maps (x, y) sample
coordinates to a measured scalar (e.g., an areaDet ROI intensity).

Prior:   f(·) ~ GP(μ(·), k(·,·))
    μ(·) – constant mean (learned)
    k(·,·) – ARD Matérn-5/2 kernel (learned length-scales per axis)

Likelihood:  y_i = f(x_i) + ε_i,   ε_i ~ N(0, σ_n²)

Posterior (Exact GP, given training data D = {X, y}):
    p(f* | X*, D) = N(μ_post(X*), Σ_post(X*, X*))

The kernel hyperparameters (length-scales, output scale, noise) are optimised
by maximising the exact marginal log-likelihood (MLL) using Adam, run for a
fixed number of steps after each call to .tell().

Acquisition – Straddle Rule for Level-Set Estimation
------------------------------------------------------
We want to map a contour f = threshold (e.g., 0.5 for a binary domain wall).
The Straddle acquisition function balances:
    1. Uncertainty reduction  :  large predictive σ*(x)
    2. Proximity to boundary  :  small |μ*(x) – threshold|

Formally:
    α(x) = β · σ*(x)  –  |μ*(x) – threshold|

where β controls the exploration–exploitation trade-off (default β = 1.96,
corresponding to a 95 % credible interval half-width).

Points that are simultaneously uncertain AND close to the estimated level-set
boundary receive the highest acquisition values and are chosen next.

During an early exploration phase (iteration < explore_until), we drop the
distance penalty and use α(x) = σ*(x) (pure uncertainty sampling) to avoid
fixating on a possibly wrong boundary estimate before we have enough data.

Usage (standalone, for testing without Bluesky)
-----------------------------------------------
    from bayesian_engine import BayesianOptimizer
    import numpy as np

    opt = BayesianOptimizer(x_bounds=(0, 100), y_bounds=(0, 100),
                            grid_nx=64, grid_ny=64)

    # Seed with a few random measurements
    for _ in range(5):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        value = my_measurement_function(x, y)   # real or mocked
        opt.tell(x, y, value)

    # Ask for the next best location
    next_x, next_y = opt.ask()

    # Fetch grids for plotting
    mean_grid, std_grid, xi, yi = opt.get_prediction_grid()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import gpytorch
from torch.optim import Adam
import linear_operator  # noqa: F401 – used for CG solver settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GP regression model (kernel definition)
# ---------------------------------------------------------------------------

class _GPRegressionModel(gpytorch.models.ExactGP):
    """
    Internal ExactGP model.

    Kernel choice:  ARD Matérn-5/2
        – 5/2 is twice-differentiable, giving smooth but not hyper-smooth
          predictions; a good default for physical signals that may have
          moderate local roughness (domain walls, intensity gradients).
        – ARD (Automatic Relevance Determination): each input dimension
          gets its own characteristic length-scale, allowing the GP to
          discover anisotropy in the (x, y) space automatically.

    The model is deliberately kept kernel-agnostic via the `kernel` parameter
    so callers can swap it if needed (e.g., RBF for very smooth signals).

    Parameters
    ----------
    train_x : torch.Tensor, shape (N, 2)
        Normalised input coordinates in [0, 1]².
    train_y : torch.Tensor, shape (N,)
        Scalar observations.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Noise model.
    kernel : str
        'matern52' (default) or 'rbf'.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: str = "matern52",
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        # Constant mean – a simple but effective prior mean for bounded signals
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel selection
        n_dims = train_x.shape[1]  # should be 2 for (x, y)
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
        else:  # default: matern52
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=n_dims
            )

        # ScaleKernel wraps the base kernel with an output-scale parameter σ_f²,
        # allowing the amplitude of the GP to be learned from data.
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute prior distribution p(f | x) = N(μ(x), k(x, x))."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------------------------------------------------------------------------
# Public API: BayesianOptimizer
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """
    Stateful 2-D Bayesian optimizer backed by a GPyTorch Exact GP.

    All coordinate inputs/outputs use the *physical* coordinate system
    (i.e., real motor positions in mm or µm as defined by x_bounds /
    y_bounds).  Internally, coordinates are normalised to [0, 1]² before
    being passed to the GP, which improves numerical stability of the
    kernel hyperparameter optimisation.

    Parameters
    ----------
    x_bounds : tuple (x_lo, x_hi)
        Physical range of the X motor.
    y_bounds : tuple (y_lo, y_hi)
        Physical range of the Y motor.
    grid_nx : int
        Number of grid points along X for the prediction map (default 64).
    grid_ny : int
        Number of grid points along Y for the prediction map (default 64).
    level_set_threshold : float
        Value at which the level-set boundary is defined (default 0.5).
        The Straddle acquisition function focuses sampling near this level.
    beta : float
        Exploration weight in the Straddle acquisition (default 1.96,
        corresponding to the 95 % credible interval half-width).
    explore_until : int
        Number of data points below which the optimizer uses pure
        uncertainty sampling instead of the full Straddle rule (default 15).
    train_iters : int
        Number of Adam steps used to optimise the MLL after each tell()
        call (default 50).  Increase for final convergence; decrease for
        faster per-step latency during a scan.
    kernel : str
        Kernel type: 'matern52' (default) or 'rbf'.
    n_acquisition_candidates : int
        Size of the random candidate pool evaluated by the acquisition
        function on each ask() call (default 5000).
    device : str
        PyTorch device, e.g., 'cpu' (default) or 'cuda'.
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
        grid_nx: int = 64,
        grid_ny: int = 64,
        level_set_threshold: float = 0.5,
        beta: float = 1.96,
        explore_until: int = 15,
        train_iters: int = 50,
        kernel: str = "matern52",
        n_acquisition_candidates: int = 5000,
        device: str = "cpu",
    ) -> None:
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.threshold = level_set_threshold
        self.beta = beta
        self.explore_until = explore_until
        self.train_iters = train_iters
        self.kernel = kernel
        self.n_candidates = n_acquisition_candidates
        self.device = torch.device(device)

        # ----------------------------------------------------------------
        # Training data tensors – grown incrementally by tell()
        # ----------------------------------------------------------------
        # train_x_norm: (N, 2) float32 – normalised coords in [0,1]²
        # train_y:      (N,)   float32 – raw measured scalars
        self._train_x_norm: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None

        # Parallel lists of *physical* coordinates, kept for external use
        self._x_physical: list[float] = []
        self._y_physical: list[float] = []

        # GP model and likelihood; created lazily on first tell()
        self._model: Optional[_GPRegressionModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None

        # Precompute a fixed evaluation grid (physical coords + normalised)
        self._build_eval_grid()

        # Tune linear_operator CG solver for larger datasets
        linear_operator.settings.max_cg_iterations._set_value(2000)
        linear_operator.settings.cg_tolerance._set_value(0.005)

        logger.info(
            "BayesianOptimizer initialised: x=[%.3g, %.3g], y=[%.3g, %.3g], "
            "grid=(%d×%d), kernel=%s",
            *x_bounds, *y_bounds, grid_nx, grid_ny, kernel,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tell(self, x: float, y: float, value: float) -> None:
        """
        Ingest a new experimental measurement and update the GP model.

        Mathematical steps
        ------------------
        1. Normalise the physical position (x, y) to the unit square:
               x_n = (x – x_lo) / (x_hi – x_lo)
               y_n = (y – y_lo) / (y_hi – y_lo)

        2. Append x_norm = [x_n, y_n] to the training tensor X (N×2)
           and the scalar `value` to the target tensor y (N,).

        3. If a GP model already exists, update its training data via
           model.set_train_data() (O(1)) rather than recreating the model
           from scratch.  The kernel hyperparameters are re-optimised by
           running `train_iters` steps of Adam on the negative MLL.

        4. The updated model is left in eval mode so get_prediction_grid()
           and ask() can call it immediately without a mode switch.

        Parameters
        ----------
        x : float
            Physical X coordinate (motor position).
        y : float
            Physical Y coordinate (motor position).
        value : float
            Measured scalar value (e.g., detector ROI total intensity).
        """
        # --- 1. Normalise coordinates to [0, 1]² ---
        x_n = self._normalise_x(x)
        y_n = self._normalise_y(y)
        new_point = torch.tensor([[x_n, y_n]], dtype=torch.float32, device=self.device)
        new_target = torch.tensor([value], dtype=torch.float32, device=self.device)

        # --- 2. Accumulate training data ---
        if self._train_x_norm is None:
            self._train_x_norm = new_point
            self._train_y = new_target
        else:
            self._train_x_norm = torch.cat([self._train_x_norm, new_point], dim=0)
            self._train_y = torch.cat([self._train_y, new_target], dim=0)

        self._x_physical.append(float(x))
        self._y_physical.append(float(y))

        n_points = self._train_x_norm.shape[0]
        logger.debug("tell(): n_data=%d, x=%.4g, y=%.4g, value=%.4g", n_points, x, y, value)

        # --- 3. Build or update GP model ---
        if self._model is None:
            # First data point: create the model from scratch
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self._model = _GPRegressionModel(
                self._train_x_norm,
                self._train_y,
                self._likelihood,
                kernel=self.kernel,
            ).to(self.device)
        else:
            # Subsequent points: inject new data without re-creating the model.
            # strict=False allows the training inputs to change size.
            self._model.set_train_data(
                inputs=self._train_x_norm,
                targets=self._train_y,
                strict=False,
            )

        # --- 4. (Re-)optimise hyperparameters via MLL ---
        # We need at least 2 data points for MLL to be non-trivial
        if n_points >= 2:
            self._train_model()

        # Leave model in eval mode for immediate prediction
        self._model.eval()
        self._likelihood.eval()

    def ask(self) -> Tuple[float, float]:
        """
        Compute the next recommended measurement location using the
        Straddle acquisition function.

        Algorithm
        ---------
        1. Draw `n_candidates` points uniformly at random in [0, 1]²
           (the normalised domain).

        2. Compute the GP posterior predictive mean μ* and standard
           deviation σ* at all candidates via a single batched forward
           pass (torch.no_grad + fast_pred_var for O(N) cost).

        3. Evaluate the Straddle acquisition:
               α(x) = β · σ*(x) – |μ*(x) – threshold|

           During early exploration (n_data < explore_until) use:
               α(x) = σ*(x)    (pure uncertainty sampling)

        4. Return the physical coordinates of the candidate with the
           highest α value.

        Returns
        -------
        x_next : float  –  physical X coordinate
        y_next : float  –  physical Y coordinate

        Raises
        ------
        RuntimeError
            If called before any data has been provided via tell().
        """
        if self._model is None or self._train_x_norm is None:
            raise RuntimeError(
                "BayesianOptimizer.ask() called before any data. "
                "Call tell() with at least one measurement first."
            )

        n_data = self._train_x_norm.shape[0]

        # --- 1. Random candidate pool in normalised [0,1]² ---
        candidates = torch.rand(
            self.n_candidates, 2, dtype=torch.float32, device=self.device
        )

        # --- 2. Posterior predictions ---
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self._likelihood(self._model(candidates))
            mu = preds.mean           # shape (n_candidates,)
            sigma = preds.variance.sqrt()  # shape (n_candidates,)

        # --- 3. Acquisition function ---
        if n_data < self.explore_until:
            # Pure exploration: prioritise high-uncertainty regions
            acquisition = sigma
            logger.debug(
                "ask(): exploration phase (n=%d < %d), using pure σ",
                n_data, self.explore_until,
            )
        else:
            # Straddle rule: balance uncertainty and proximity to level-set
            distance_from_boundary = torch.abs(mu - self.threshold)
            acquisition = self.beta * sigma - distance_from_boundary
            logger.debug(
                "ask(): straddle phase (n=%d), β=%.2f, threshold=%.2f",
                n_data, self.beta, self.threshold,
            )

        # --- 4. Select best candidate and denormalise ---
        best_idx = torch.argmax(acquisition).item()
        best_norm = candidates[best_idx]

        x_next = self._denormalise_x(best_norm[0].item())
        y_next = self._denormalise_y(best_norm[1].item())

        logger.info("ask(): next point → x=%.4g, y=%.4g", x_next, y_next)
        return x_next, y_next

    def get_prediction_grid(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the GP predictive mean and standard deviation on the
        pre-built evaluation grid.

        This method is intended to be called by the GUI after each tell()
        to refresh the 2-D heatmap.

        Returns
        -------
        mean_grid : np.ndarray, shape (grid_nx, grid_ny)
            Predictive mean μ* at each grid point.
        std_grid  : np.ndarray, shape (grid_nx, grid_ny)
            Predictive standard deviation σ* at each grid point.
        xi : np.ndarray, shape (grid_nx,)
            Physical X coordinates of grid columns.
        yi : np.ndarray, shape (grid_ny,)
            Physical Y coordinates of grid rows.

        Raises
        ------
        RuntimeError
            If called before any data has been provided via tell().
        """
        if self._model is None:
            raise RuntimeError(
                "No GP model available yet. Call tell() with at least one "
                "measurement before requesting predictions."
            )

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # _eval_grid_norm: (grid_nx * grid_ny, 2) tensor
            preds = self._likelihood(self._model(self._eval_grid_norm))
            mean_flat = preds.mean.cpu().numpy()
            std_flat = preds.variance.sqrt().cpu().numpy()

        mean_grid = mean_flat.reshape(self.grid_nx, self.grid_ny)
        std_grid = std_flat.reshape(self.grid_nx, self.grid_ny)

        return mean_grid, std_grid, self._xi, self._yi

    @property
    def n_observations(self) -> int:
        """Number of measurements ingested so far."""
        if self._train_x_norm is None:
            return 0
        return self._train_x_norm.shape[0]

    @property
    def observed_x(self) -> list[float]:
        """List of physical X coordinates of all observations (for scatter plot)."""
        return list(self._x_physical)

    @property
    def observed_y(self) -> list[float]:
        """List of physical Y coordinates of all observations (for scatter plot)."""
        return list(self._y_physical)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_eval_grid(self) -> None:
        """
        Pre-compute the fixed (grid_nx × grid_ny) evaluation grid used by
        get_prediction_grid().  This avoids rebuilding tensors on every GUI
        refresh call.
        """
        xi = np.linspace(self.x_bounds[0], self.x_bounds[1], self.grid_nx, dtype=np.float32)
        yi = np.linspace(self.y_bounds[0], self.y_bounds[1], self.grid_ny, dtype=np.float32)
        self._xi = xi  # shape (grid_nx,)
        self._yi = yi  # shape (grid_ny,)

        # Meshgrid → flattened normalised tensor
        gx, gy = np.meshgrid(xi, yi, indexing="ij")  # (grid_nx, grid_ny) each
        gx_norm = (gx - self.x_bounds[0]) / max(self.x_bounds[1] - self.x_bounds[0], 1e-9)
        gy_norm = (gy - self.y_bounds[0]) / max(self.y_bounds[1] - self.y_bounds[0], 1e-9)

        flat_x = gx_norm.ravel().astype(np.float32)
        flat_y = gy_norm.ravel().astype(np.float32)

        self._eval_grid_norm = torch.from_numpy(
            np.stack([flat_x, flat_y], axis=-1)
        ).to(self.device)  # shape (grid_nx * grid_ny, 2)

    def _normalise_x(self, x: float) -> float:
        lo, hi = self.x_bounds
        return float((x - lo) / max(hi - lo, 1e-9))

    def _normalise_y(self, y: float) -> float:
        lo, hi = self.y_bounds
        return float((y - lo) / max(hi - lo, 1e-9))

    def _denormalise_x(self, x_n: float) -> float:
        lo, hi = self.x_bounds
        return float(x_n * (hi - lo) + lo)

    def _denormalise_y(self, y_n: float) -> float:
        lo, hi = self.y_bounds
        return float(y_n * (hi - lo) + lo)

    def _train_model(self) -> None:
        """
        Optimise GP hyperparameters by maximising the exact marginal
        log-likelihood (MLL) for `self.train_iters` Adam steps.

        The MLL is:
            log p(y | X, θ) = –½ yᵀ(K + σ_n²I)⁻¹y  –  ½ log|K + σ_n²I|  –  N/2 log(2π)

        where θ = {length-scales, output-scale, noise variance}.

        Supress GPyTorch's numerical warnings during backprop – these are
        expected during early optimisation with very few data points and are
        not indicative of a bug.
        """
        assert self._model is not None and self._likelihood is not None

        self._model.train()
        self._likelihood.train()

        optimizer = Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=0.1,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.train_iters):
                optimizer.zero_grad()
                output = self._model(self._train_x_norm)
                loss = -mll(output, self._train_y)
                loss.backward()
                optimizer.step()

        logger.debug("_train_model(): final MLL loss = %.5f", loss.item())
