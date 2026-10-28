"""Exact GP of the ESBO-GP port, min-value entropy search and truncated-normal sampling.

The GP follows the GPML layout of the original MATLAB ESBO code (thesis appendix, our port of
ESBO-GP): constant mean, ARD squared-exponential kernel ``sf^2 exp(-|x - x'|^2_ell / 2)`` and
homoscedastic Gaussian noise ``sn^2``, all scales stored as natural logs of standard deviations
(``log ell_1 .. log ell_d, log sf, log sn``). Inputs are mapped to the unit cube of the search box
and outputs are standardized before the fit. Hyperparameters maximize the log marginal likelihood
(plus GPML's smooth-box length-scale prior when no per-point noise is known) with Adam, and are
clamped into their bounds after every step. The fit uses no randomness.

``cold_refit_growth`` (a documented deviation, see ``docs/DEVIATIONS.md``) rations full fits: a GP
instance is one chain of fits on a growing training set. A fit is cold (fixed initialization, all
``iters`` steps) unless the previous fit exists and the training set has grown by less than
``cold_refit_growth`` relative to the size at the last cold fit, in which case it starts from the
previous hyperparameters and runs ``round(0.15 * iters)`` steps. ``0.0`` makes every fit cold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import truncnorm

DTYPE = torch.float64

#: Distance factor at which the squared-exponential correlation drops to 0.1.
R_AT_0_1 = math.sqrt(2.0 * math.log(10.0))
#: Length scales such that a correlation of 0.1 is reached between 1 % and 50 % of the unit cube.
DEFAULT_LENGTHSCALE_BOUNDS = (0.01 / R_AT_0_1, 0.50 / R_AT_0_1)
DEFAULT_OUTPUTSCALE_BOUNDS = (math.exp(-4.0), math.exp(1.0))
DEFAULT_NOISE_BOUNDS = (math.exp(-10.0), math.exp(3.0))
DEFAULT_ITERS = 100
DEFAULT_LR = 0.1
DEFAULT_SMOOTH_BOX_ETA = 150000.0
DEFAULT_GRAD_CLIP = 100.0
#: Initial noise standard deviation (standardized units) of a cold fit.
INIT_NOISE_STD = 0.1
#: Fraction of ``iters`` that a warm fit runs.
WARM_ITER_FRACTION = 0.15
#: Floor of the output standard deviation used for standardization.
Y_STD_FLOOR = 1e-10

_MIN_VAR = 1e-12
_JITTERS = (1e-10, 1e-8, 1e-6, 1e-4)


def _cholesky(K: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(K.shape[0], dtype=K.dtype)
    error: Exception | None = None
    for jitter in _JITTERS:
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except RuntimeError as exc:  # torch.linalg.LinAlgError subclasses RuntimeError
            error = exc
    raise RuntimeError("kernel matrix is not positive definite") from error


@dataclass
class _Params:
    """Hyperparameters of one fit, natural-log standard deviations, standardized units."""

    log_lengthscale: torch.Tensor
    log_outputscale: torch.Tensor
    log_noise: torch.Tensor
    mean: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        return [self.log_lengthscale, self.log_outputscale, self.log_noise, self.mean]

    def detached(self) -> _Params:
        return _Params(*(t.detach().clone() for t in self.tensors()))


class ExactGP:
    """Exact GP regression in the GPML hyperparameter layout.

    ``lo`` and ``hi`` define the box that is mapped to the unit cube (``None`` means the inputs are
    used as given). One instance is one chain of fits, which is what ``cold_refit_growth`` acts on.
    """

    def __init__(
        self,
        lo: np.ndarray | None = None,
        hi: np.ndarray | None = None,
        *,
        iters: int = DEFAULT_ITERS,
        lr: float = DEFAULT_LR,
        lengthscale_bounds: tuple[float, float] = DEFAULT_LENGTHSCALE_BOUNDS,
        outputscale_bounds: tuple[float, float] = DEFAULT_OUTPUTSCALE_BOUNDS,
        noise_bounds: tuple[float, float] = DEFAULT_NOISE_BOUNDS,
        smooth_box_eta: float = DEFAULT_SMOOTH_BOX_ETA,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        learn_mean: bool = True,
        map_prior: bool | None = None,
        cold_refit_growth: float = 0.0,
    ) -> None:
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float64)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float64)
        self.iters = int(iters)
        self.lr = float(lr)
        for name, b in (("lengthscale", lengthscale_bounds), ("outputscale", outputscale_bounds),
                        ("noise", noise_bounds)):
            if not (0.0 < float(b[0]) < float(b[1]) < math.inf):
                raise ValueError(f"{name}_bounds must be finite positive with lower < upper")
        self.lengthscale_bounds = (float(lengthscale_bounds[0]), float(lengthscale_bounds[1]))
        self.outputscale_bounds = (float(outputscale_bounds[0]), float(outputscale_bounds[1]))
        self.noise_bounds = (float(noise_bounds[0]), float(noise_bounds[1]))
        self.smooth_box_eta = float(smooth_box_eta)
        self.grad_clip = float(grad_clip)
        self.learn_mean = bool(learn_mean)
        self.map_prior = map_prior
        if cold_refit_growth < 0:
            raise ValueError("cold_refit_growth must be >= 0")
        self.cold_refit_growth = float(cold_refit_growth)
        # chain state
        self.n_cold_fits = 0
        self.n_warm_fits = 0
        self.last_fit_warm = False
        self._last_cold_n: int | None = None
        self._params: _Params | None = None
        # posterior state
        self._X: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._fixed: torch.Tensor | None = None
        self._known: torch.Tensor | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # -- transforms ---------------------------------------------------------------------------
    def _unit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :] if self.lo is not None and X.shape[0] == self.lo.shape[0] else X[:, None]
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D design matrix, got shape {X.shape}")
        if self.lo is None:
            return X
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return (X - self.lo) / span

    # -- hyperparameters ----------------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self._params is not None and self._X is not None

    @property
    def log_lengthscale(self) -> np.ndarray:
        return self._require().log_lengthscale.detach().numpy().copy()

    @property
    def log_outputscale(self) -> float:
        return float(self._require().log_outputscale)

    @property
    def log_noise(self) -> float:
        return float(self._require().log_noise)

    @property
    def noise_variance(self) -> float:
        """Learned homoscedastic noise variance in raw output units."""
        return math.exp(2.0 * self.log_noise) * self.y_std**2

    def _require(self) -> _Params:
        if self._params is None:
            raise RuntimeError("the GP has not been fitted")
        return self._params

    def _cold_params(self, d: int) -> _Params:
        lo, hi = self.lengthscale_bounds
        log_ls = 0.5 * (math.log(lo) + math.log(hi))
        return _Params(
            log_lengthscale=torch.full((d,), log_ls, dtype=DTYPE),
            log_outputscale=torch.tensor(0.0, dtype=DTYPE),
            log_noise=torch.tensor(math.log(INIT_NOISE_STD), dtype=DTYPE),
            mean=torch.tensor(0.0, dtype=DTYPE),
        )

    def _clamp(self, p: _Params) -> None:
        with torch.no_grad():
            p.log_lengthscale.clamp_(*(math.log(b) for b in self.lengthscale_bounds))
            p.log_outputscale.clamp_(*(math.log(b) for b in self.outputscale_bounds))
            p.log_noise.clamp_(*(math.log(b) for b in self.noise_bounds))

    # -- linear algebra -----------------------------------------------------------------------
    @staticmethod
    def _kernel(p: _Params, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ls = torch.exp(p.log_lengthscale)
        sq = torch.cdist(A / ls, B / ls, p=2.0).pow(2)
        return torch.exp(2.0 * p.log_outputscale) * torch.exp(-0.5 * sq)

    def _noise_diag(self, p: _Params, n: int) -> torch.Tensor:
        learned = torch.clamp(torch.exp(2.0 * p.log_noise), min=_MIN_VAR).expand(n)
        if self._fixed is None:
            return learned
        return torch.where(self._known, self._fixed, learned)

    def _posterior_terms(self, p: _Params) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y = self._X, self._y
        K = self._kernel(p, X, X) + torch.diag(self._noise_diag(p, X.shape[0]))
        L = _cholesky(K)
        centered = y - p.mean
        alpha = torch.cholesky_solve(centered[:, None], L).squeeze(-1)
        return L, alpha, centered

    def _neg_log_ml(self, p: _Params) -> torch.Tensor:
        L, alpha, centered = self._posterior_terms(p)
        n = centered.shape[0]
        return (0.5 * torch.dot(centered, alpha) + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * math.log(2.0 * math.pi))

    def _smooth_box(self, p: _Params) -> torch.Tensor:
        """GPML ``priorSmoothBox2`` on the length scales, constants dropped."""
        lo, hi = self.lengthscale_bounds
        sab = (hi - lo) / (self.smooth_box_eta * math.sqrt(2.0 * math.pi))
        ls = torch.exp(p.log_lengthscale)
        out = torch.clamp(lo - ls, min=0.0) + torch.clamp(ls - hi, min=0.0)
        return 0.5 * ((out / sab) ** 2).sum()

    # -- fit / predict ------------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise: np.ndarray | None = None,
        warm: ExactGP | None = None,
    ) -> ExactGP:
        """Fit hyperparameters and condition on ``(X, y)``.

        ``noise`` holds known per-point noise variances in raw output units. Positive finite
        entries are used as given, the others use the learned noise. The smooth-box prior is used
        when ``map_prior`` is ``None`` and no noise is known (as in ESBO, where known-noise fits
        are plain maximum likelihood). ``warm`` continues the chain of another fitted GP instead
        of this instance's own chain.
        """
        Xu = self._unit(X)
        y_raw = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = Xu.shape
        if n == 0:
            raise ValueError("cannot fit a GP without data")
        if y_raw.shape[0] != n:
            raise ValueError(f"y has length {y_raw.shape[0]}, expected {n}")
        if not (np.all(np.isfinite(Xu)) and np.all(np.isfinite(y_raw))):
            raise ValueError("X and y must be finite")

        self.y_mean = float(np.mean(y_raw))
        self.y_std = max(float(np.std(y_raw)), Y_STD_FLOOR)
        self._X = torch.as_tensor(Xu, dtype=DTYPE)
        self._y = torch.as_tensor((y_raw - self.y_mean) / self.y_std, dtype=DTYPE)
        self._fixed, self._known = None, None
        if noise is not None:
            raw = np.asarray(noise, dtype=np.float64).reshape(-1)
            if raw.shape[0] != n:
                raise ValueError(f"noise has length {raw.shape[0]}, expected {n}")
            known = np.isfinite(raw) & (raw > 0.0)
            if known.any():
                scaled = np.full(n, np.nan)
                scaled[known] = np.maximum(raw[known] / self.y_std**2, _MIN_VAR)
                self._fixed = torch.as_tensor(scaled, dtype=DTYPE)
                self._known = torch.as_tensor(known)
        map_prior = (self._fixed is None) if self.map_prior is None else bool(self.map_prior)

        source = warm if warm is not None else self
        previous = source._params if source._params is not None else None
        last_cold_n = source._last_cold_n
        if previous is not None and previous.log_lengthscale.shape[0] != d:
            previous = None
        is_warm = (
            self.cold_refit_growth > 0.0
            and previous is not None
            and last_cold_n is not None
            and n < last_cold_n * (1.0 + self.cold_refit_growth)
        )
        params = previous.detached() if is_warm else self._cold_params(d)
        self._clamp(params)
        trainable = params.tensors() if self.learn_mean else params.tensors()[:3]
        for t in trainable:
            t.requires_grad_(True)
        if not self.learn_mean:
            params.mean = torch.tensor(0.0, dtype=DTYPE)

        steps = max(1, round(WARM_ITER_FRACTION * self.iters)) if is_warm else self.iters
        opt = torch.optim.Adam(trainable, lr=self.lr)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                loss = self._neg_log_ml(params)
                if map_prior:
                    loss = loss + self._smooth_box(params)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=self.grad_clip)
                opt.step()
                self._clamp(params)

        self._params = params.detached()
        self.last_fit_warm = is_warm
        if is_warm:
            self.n_warm_fits += 1
            self._last_cold_n = last_cold_n
        else:
            self.n_cold_fits += 1
            self._last_cold_n = n
        return self

    def predict(self, Xq: np.ndarray, with_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance in raw output units at ``Xq`` (variance clipped at 0).

        ``with_noise`` adds the learned observation noise variance.
        """
        p = self._require()
        Xn = torch.as_tensor(self._unit(Xq), dtype=DTYPE)
        with torch.no_grad():
            L, alpha, _ = self._posterior_terms(p)
            Kxs = self._kernel(p, self._X, Xn)
            mean = p.mean + Kxs.T.matmul(alpha)
            v = torch.linalg.solve_triangular(L, Kxs, upper=False)
            var = torch.exp(2.0 * p.log_outputscale) - v.pow(2).sum(dim=0)
            if with_noise:
                var = var + torch.exp(2.0 * p.log_noise)
        mean_np = mean.numpy() * self.y_std + self.y_mean
        var_np = var.numpy() * self.y_std**2
        return mean_np, np.maximum(var_np, 0.0)


# ------------------------------------------------------------------------------ acquisition
def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def sample_min_values(
    mu: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 20,
    grid: int = 512,
    incumbent: float = math.inf,
) -> np.ndarray:
    """Samples of the minimum ``y*`` over the candidates by inverting
    ``F(y) = 1 - prod_x P(Y_x >= y)`` on a grid, capped at ``incumbent``."""
    lo = float((mu - 5.0 * sigma).min())
    hi = float((mu + 5.0 * sigma).min())
    if hi <= lo:
        hi = lo + 1e-6
    ygrid = np.linspace(lo, hi, int(grid))
    z = (ygrid[:, None] - mu[None, :]) / sigma[None, :]
    log_sf = np.log(np.clip(1.0 - ndtr(z), 1e-300, 1.0))
    F = np.clip(1.0 - np.exp(log_sf.sum(axis=1)), 0.0, 1.0)
    ystars = np.interp(rng.random(int(n_samples)), F, ygrid)
    return np.minimum(ystars, incumbent)


def min_value_entropy(
    mu: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 20,
    grid: int = 512,
    incumbent: float = math.inf,
) -> np.ndarray:
    """Max-value entropy search (Wang and Jegelka 2017) for minimization, higher is better.

    With ``gamma = (mu - y*) / sigma`` the score is the mean over the sampled ``y*`` of
    ``gamma phi(gamma) / (2 Phi(gamma)) - log Phi(gamma)``.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-9)
    ystars = sample_min_values(mu, sigma, rng, n_samples, grid, incumbent)
    gamma = (mu[:, None] - ystars[None, :]) / sigma[:, None]
    Phi = np.clip(ndtr(gamma), 1e-10, 1.0 - 1e-12)
    return (gamma * _normal_pdf(gamma) / (2.0 * Phi) - np.log(Phi)).mean(axis=1)


def mes_acquisition(
    gp: ExactGP,
    candidates: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 20,
    grid: int = 512,
    incumbent: float = math.inf,
) -> np.ndarray:
    """MES scores of ``candidates`` (raw controller parameters) under the fitted ``gp``.

    ``incumbent`` is the best observed total cost ``J^-_n``, which caps the sampled minima.
    """
    mu, var = gp.predict(candidates)
    return min_value_entropy(mu, np.sqrt(var), rng, n_samples, grid, incumbent)


def sample_positive(
    mean: float | np.ndarray,
    std: float | np.ndarray,
    lower: float | np.ndarray,
    rng: np.random.Generator,
) -> float | np.ndarray:
    """Draw from ``N(mean, std^2)`` truncated to ``[lower, inf)`` (one draw per element).

    A non-positive ``std`` returns ``max(mean, lower)``.
    """
    m, s, lo = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (mean, std, lower)))
    shape = m.shape
    m, s, lo = (np.atleast_1d(v).ravel() for v in (m, s, lo))
    u = rng.random(m.shape)
    out = np.array(np.maximum(m, lo), dtype=np.float64)
    ok = s > 0
    if np.any(ok):
        alpha = (lo[ok] - m[ok]) / s[ok]
        draw = truncnorm.ppf(u[ok], alpha, np.inf, loc=m[ok], scale=s[ok])
        out[ok] = np.maximum(draw, lo[ok])
    return float(out[0]) if shape == () else out.reshape(shape)
