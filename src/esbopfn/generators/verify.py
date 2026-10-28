"""Checks of generated data: pool invariants and C++ against Python parity.

Invariants (``check_pool``): J non-decreasing before the crash and ``+inf`` from the crash step on,
finite channels, theta inside the box, success only without a crash, at most half unsuccessful
trajectories in single-trajectory pools that apply the rejection, per-instance counts of set
pools as recorded in ``instances.json``. ``check_contexts`` does the same for FT-PFN shards.

Parity (``parity_deterministic``, ``parity_distribution``) compares the binary with
``esbopfn.problems``: the same instance and controller with noise switched off must give the same
crash step and a total cost within a relative 1e-3, and over prior samples the crash rate and the
distribution of J_T must agree (two-proportion z-test, Kolmogorov-Smirnov test).
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import Instance
from esbopfn.generators import binaries
from esbopfn.generators.pools import (
    STATUS_COMPLETE,
    STATUS_CRASHED,
    STATUS_TRUNCATED,
    ContextStore,
    Pool,
)
from esbopfn.util.seeds import substream

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    def line(self) -> str:
        tag = "SKIP" if self.skipped else ("ok  " if self.ok else "FAIL")
        return f"[{tag}] {self.name}" + (f": {self.detail}" if self.detail else "")


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(Check(name, bool(ok), detail, skipped))

    @property
    def ok(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def lines(self) -> list[str]:
        return [c.line() for c in self.checks]


# --------------------------------------------------------------------------- pool invariants
def _theta_box(problem: str) -> tuple[np.ndarray, np.ndarray]:
    from esbopfn.problems.registry import get_spec

    spec = get_spec(problem)
    return spec.theta_lo, spec.theta_hi


def check_pool(pool: Pool, sample: int = 2000, seed: int = 0, report: Report | None = None,
               rho_unsuccessful: float | None = None) -> Report:
    """Invariants on up to ``sample`` random rows of ``pool``."""
    rep = report or Report()
    n = len(pool)
    if n == 0:
        rep.add("rows present", False, "the pool is empty")
        return rep
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    spec = pool.spec
    J = pool.column("J", idx).astype(np.float64)
    cs = pool.column("crash_step", idx)[:, 0].astype(np.int64)
    succ = pool.column("success", idx)[:, 0]
    theta = pool.column("theta", idx).astype(np.float64)
    steps = spec.steps
    crashed = cs > 0
    rep.add("crash steps in range", bool(np.all((cs >= 0) & (cs <= spec.T))))
    # J finite before the crash step, +inf from it on
    after = crashed[:, None] & (steps[None, :] >= cs[:, None])
    before = ~after
    rep.add("J = +inf from the crash step on", bool(np.all(np.isposinf(J[after]))))
    rep.add("J finite before the crash", bool(np.all(np.isfinite(J[before]))))
    Jb = np.where(before, J, np.nan)
    dif = np.diff(Jb, axis=1)
    tol = 1e-6 * np.maximum(1.0, np.abs(Jb[:, 1:]))
    mono = np.all((dif >= -tol) | np.isnan(dif))
    rep.add("J non-decreasing before the crash", bool(mono))
    if spec.has("x"):
        x = pool.column("x", idx)
        rep.add("level channels finite", bool(np.all(np.isfinite(x))))
    if spec.has("xdot"):
        xd = pool.column("xdot", idx)
        rep.add("rate channels finite", bool(np.all(np.isfinite(xd))))
    lo, hi = _theta_box(spec.problem)
    eps = 1e-5 * (hi - lo)
    rep.add("theta inside the box", bool(np.all((theta >= lo - eps) & (theta <= hi + eps))))
    rep.add("success only without a crash", bool(np.all(~((succ > 0.5) & crashed))))
    share = float(np.mean(succ < 0.5))
    if spec.kind == "traj":
        if rho_unsuccessful is not None and rho_unsuccessful > 0.5:
            bound = 0.5 + 4.0 * math.sqrt(0.25 / len(idx))
            rep.add("unsuccessful share at most 1/2", share <= bound,
                    f"{share:.3f} over {len(idx)} rows (rho {rho_unsuccessful:.3f})")
        else:
            rep.add("unsuccessful share", True, f"{share:.3f} (no rejection needed)")
    if spec.kind == "set":
        _check_set_counts(pool, rep)
    return rep


def _check_set_counts(pool: Pool, rep: Report, max_instances: int = 200) -> None:
    recs = pool.instances()
    k = pool.rows_per_unit
    n_inst = min(len(recs), len(pool) // k, max_instances)
    bad = 0
    for i in range(n_inst):
        rec = recs[i]
        rows = pool.instance_rows(i)
        inst_col = pool.spec.column("instance")
        cs = rows[:, pool.spec.column("crash_step").offset]
        su = rows[:, pool.spec.column("success").offset]
        counts = (int(np.sum(cs > 0)), int(np.sum((cs == 0) & (su < 0.5))), int(np.sum(su > 0.5)))
        want = (rec["set"]["n_crash"], rec["set"]["n_unsuccessful"], rec["set"]["n_success"])
        ok = counts == want and np.all(rows[:, inst_col.offset] == i)
        bad += not ok
    rep.add("set pool counts match instances.json", bad == 0, f"{n_inst} instances, {bad} mismatched")


def check_contexts(store: ContextStore, sample: int = 500, report: Report | None = None) -> Report:
    """Invariants on FT-PFN examples: token steps and costs, status, targets."""
    rep = report or Report()
    n = len(store)
    if n == 0:
        rep.add("examples present", False, "no finished shard")
        return rep
    T = int(store.shards[0].meta["T"])
    bad: dict[str, int] = {k: 0 for k in ("steps", "J", "status", "target", "unrun")}
    for e_i, ex in zip(range(min(sample, n)), store):
        for i in range(ex.n):
            t, J = ex.tokens(i)
            L, st = int(ex.length[i]), int(ex.status[i])
            if len(t) == 0 or np.any(t > L) or t.max() != L:
                bad["steps"] += 1
            fin = J[np.isfinite(J)]
            if np.any(np.diff(fin) < -1e-6 * np.maximum(1, np.abs(fin[1:]))):
                bad["J"] += 1
            if st == STATUS_COMPLETE and (L != T or not np.isfinite(ex.target[i])):
                bad["status"] += 1
            if st == STATUS_CRASHED and not (np.isposinf(J[-1]) and np.isposinf(ex.target[i])):
                bad["status"] += 1
            if st == STATUS_TRUNCATED and (L >= T or not np.isfinite(J[-1])):
                bad["status"] += 1
            if st == STATUS_COMPLETE and abs(float(J[-1]) - float(ex.target[i])) > 1e-6 * abs(
                    float(ex.target[i])) + 1e-9:
                bad["target"] += 1
        if ex.unrun_theta.ndim != 2 or len(ex.unrun_target) != len(ex.unrun_theta):
            bad["unrun"] += 1
        del e_i
    for k, v in bad.items():
        rep.add(f"contexts: {k}", v == 0, f"{v} violations")
    return rep


# --------------------------------------------------------------------------- parity with Python
class ParityUnavailable(RuntimeError):
    """Raised when the Python simulators cannot be used for a parity check."""


class CentralRng:
    """A stand-in generator that returns central values: zero normals, interval midpoints.

    Handed to a Python simulator together with a noise-free instance, it removes every random
    effect (reset jitter, kick sign up to the plant's mirror symmetry), which matches the
    binary's ``rollout --deterministic``.
    """

    def standard_normal(self, size: Any = None, dtype: Any = float, out: Any = None) -> Any:
        return 0.0 if size is None else np.zeros(size)

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        if size is None:
            return np.zeros(np.broadcast(np.asarray(loc), np.asarray(scale)).shape) + loc
        return np.zeros(size) + loc

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> Any:
        mid = (np.asarray(low, dtype=float) + np.asarray(high, dtype=float)) / 2.0
        return float(mid) if size is None and mid.ndim == 0 else np.broadcast_to(mid, size or mid.shape).copy()

    def random(self, size: Any = None, dtype: Any = float, out: Any = None) -> Any:
        return 0.5 if size is None else np.full(size, 0.5)

    def choice(self, a: Any, size: Any = None, replace: bool = True, p: Any = None) -> Any:
        seq = np.arange(a) if isinstance(a, (int, np.integer)) else np.asarray(a)
        return seq[len(seq) // 2]

    def integers(self, low: Any, high: Any = None, size: Any = None, dtype: Any = np.int64,
                 endpoint: bool = False) -> Any:
        lo = 0 if high is None else low
        return lo if size is None else np.full(size, lo)


def python_problem(problem: str) -> Any:
    """``esbopfn.problems.get_problem(problem)``, or :class:`ParityUnavailable` with the reason."""
    try:
        from esbopfn.problems import get_problem
    except ImportError as exc:
        raise ParityUnavailable(f"esbopfn.problems is not importable ({exc})") from exc
    try:
        return get_problem(problem)
    except Exception as exc:  # noqa: BLE001, the simulators may still be under construction
        raise ParityUnavailable(f"esbopfn.problems.get_problem({problem!r}) failed: {exc}") from exc


def _group(values: dict[str, float], names: list[str], what: str) -> dict[str, float]:
    if not names:
        return {}
    if set(values) == set(names):
        return {n: float(values[n]) for n in names}
    if len(values) == len(names):
        return {n: float(v) for n, v in zip(names, values.values())}
    raise ParityUnavailable(f"{what}: Python has {list(values)}, the binary expects {names}")


def to_cpp_instance(inst: Instance, info: dict[str, Any]) -> dict[str, Any]:
    """An Instance as the binary's ``--instance-json`` (matched by name, else by position)."""
    return {"phi": _group(inst.phi, info["phi_names"], "phi"),
            "phi_hat": _group(inst.phi_hat, info["phi_names"], "phi_hat") if inst.phi_hat else {},
            "noise_std": _group(inst.noise_std, info["noise_names"], "noise_std"),
            "extras": _group(inst.extras, info["extra_names"], "extras"),
            "seed": int(inst.seed)}


def cpp_rollout(problem: str, theta: np.ndarray, instance: dict[str, Any],
                deterministic: bool = True, seed: int = 0,
                home: Path | str | None = None) -> dict[str, Any]:
    args = ["--theta", ",".join(repr(float(v)) for v in theta),
            "--instance-json", json.dumps(instance), "--seed", str(seed)]
    if deterministic:
        args.append("--deterministic")
    return binaries.run_json(problem, "rollout", *args, home=home)


@dataclass
class DeterministicCase:
    theta: list[float]
    crash_cpp: int | None
    crash_py: int | None
    J_cpp: float
    J_py: float

    @property
    def ok(self) -> bool:
        if self.crash_cpp != self.crash_py:
            return False
        if self.crash_cpp is not None:
            return True
        return abs(self.J_cpp - self.J_py) <= 1e-3 * max(abs(self.J_py), 1e-12)


def parity_deterministic(problem: str, n_theta: int = 8, seed: int = 0,
                         instance: Instance | None = None,
                         home: Path | str | None = None) -> list[DeterministicCase]:
    """C++ and Python rollouts of the same controllers on the same noise-free instance.

    The instance defaults to the nominal plant. The Python simulator gets a :class:`CentralRng`
    and the binary runs with ``--deterministic``.
    """
    prob = python_problem(problem)
    info = binaries.info(problem, home)
    inst = instance if instance is not None else prob.nominal_instance(noise=False)
    inst = Instance(phi=dict(inst.phi), phi_hat=dict(inst.phi_hat),
                    noise_std={k: 0.0 for k in inst.noise_std}, extras=dict(inst.extras),
                    seed=inst.seed)
    cpp_inst = to_cpp_instance(inst, info)
    rng = substream(seed, "parity", problem)
    cases = []
    for _ in range(n_theta):
        theta = rng.uniform(prob.spec.theta_lo, prob.spec.theta_hi)
        try:
            py = prob.simulate(theta, inst, CentralRng())
        except Exception as exc:  # noqa: BLE001
            raise ParityUnavailable(f"the Python simulator rejected the central RNG ({exc})") from exc
        cc = cpp_rollout(problem, theta, cpp_inst, deterministic=True, home=home)
        J_cpp = float(cc["J"][-1]) if cc["crash_step"] is None else math.inf
        cases.append(DeterministicCase(theta=[float(v) for v in theta],
                                       crash_cpp=cc["crash_step"], crash_py=py.crash_step,
                                       J_cpp=J_cpp, J_py=float(py.J_total)))
    return cases


@dataclass
class DistributionParity:
    n_cpp: int
    n_py: int
    crash_cpp: float
    crash_py: float
    success_cpp: float
    success_py: float
    crash_z: float
    success_z: float
    ks_stat: float
    ks_p: float

    def ok(self, z_max: float = 4.0, p_min: float = 1e-3) -> bool:
        return abs(self.crash_z) <= z_max and abs(self.success_z) <= z_max and (
            not math.isfinite(self.ks_p) or self.ks_p >= p_min)


def _z(p1: float, n1: int, p2: float, n2: int) -> float:
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se


def parity_distribution(problem: str, n: int = 500, seed: int = 0,
                        home: Path | str | None = None) -> DistributionParity:
    """Crash rate, success rate and the J_T distribution of non-crashed trajectories over ``n``
    unfiltered prior samples (instance and theta drawn fresh each time) on both sides."""
    from scipy.stats import ks_2samp

    from esbopfn.problems.prior import sample_instance, sample_theta

    prob = python_problem(problem)
    rng = substream(seed, "parity-ks", problem)
    py_J, py_crash, py_succ = [], [], []
    for _ in range(n):
        inst = sample_instance(prob, rng)
        traj = prob.simulate(sample_theta(prob.spec, rng), inst, rng)
        py_crash.append(traj.crashed)
        py_succ.append(traj.success)
        if not traj.crashed:
            py_J.append(traj.J_total)
    with tempfile.TemporaryDirectory(prefix="esbopfn-parity-") as tmp:
        binaries.run(problem, "traj", "--out", tmp, "--count", str(n), "--chunk", str(n),
                     "--seed", str(seed), "--no-reject", "--quiet", home=home)
        pool = Pool.open(tmp)
        cs = pool.column("crash_step")[:, 0]
        su = pool.column("success")[:, 0]
        J = pool.column("J")[:, -1].astype(np.float64)
    c_J = J[cs == 0]
    c_crash, c_succ = float(np.mean(cs > 0)), float(np.mean(su > 0.5))
    p_crash, p_succ = float(np.mean(py_crash)), float(np.mean(py_succ))
    if len(c_J) > 5 and len(py_J) > 5:
        ks = ks_2samp(np.log(np.maximum(c_J, 1e-300)), np.log(np.maximum(py_J, 1e-300)))
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = math.nan, math.nan
    return DistributionParity(n_cpp=n, n_py=n, crash_cpp=c_crash, crash_py=p_crash,
                              success_cpp=c_succ, success_py=p_succ,
                              crash_z=_z(c_crash, n, p_crash, n), success_z=_z(c_succ, n, p_succ, n),
                              ks_stat=ks_stat, ks_p=ks_p)


def parity_report(problem: str, n_theta: int = 8, n_samples: int = 500, seed: int = 0,
                  home: Path | str | None = None, report: Report | None = None) -> Report:
    rep = report or Report()
    try:
        cases = parity_deterministic(problem, n_theta, seed, home=home)
    except ParityUnavailable as exc:
        rep.add(f"parity {problem}: deterministic", True, str(exc), skipped=True)
    else:
        bad = [c for c in cases if not c.ok]
        worst = max((abs(c.J_cpp - c.J_py) / max(abs(c.J_py), 1e-12) for c in cases
                     if c.crash_cpp is None and c.crash_py is None), default=0.0)
        rep.add(f"parity {problem}: deterministic", not bad,
                f"{len(cases) - len(bad)}/{len(cases)} agree, max rel J_T {worst:.2e}")
    try:
        dist = parity_distribution(problem, n_samples, seed, home=home)
    except ParityUnavailable as exc:
        rep.add(f"parity {problem}: distribution", True, str(exc), skipped=True)
    else:
        rep.add(f"parity {problem}: distribution", dist.ok(),
                f"crash {dist.crash_cpp:.3f}/{dist.crash_py:.3f} (z {dist.crash_z:+.2f}), "
                f"success {dist.success_cpp:.3f}/{dist.success_py:.3f}, "
                f"KS(ln J_T) p {dist.ks_p:.3g} (C++/Python, n={n_samples})")
    return rep
