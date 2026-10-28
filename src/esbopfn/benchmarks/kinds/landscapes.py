"""Kind ``landscapes``: true cost landscapes on a slice through Theta and the prior-sample curves
(thesis fig:app:prior-landscapes and fig:setup:prior-samples).

``unit: "slice"``: per (problem, instance, slice seed) a slice through the center of Theta. With
``d = 1`` it is the whole interval on an 80-point grid, with ``d = 2`` the axis-aligned plane, with
``d > 2`` a random oblique plane spanned by two orthonormal Gaussian directions (the widest of up
to 200 draws, stopping at the first whose extent reaches 5 % of the smallest half-width), on a
24 x 24 grid of ``center + s u + t v`` with ``s, t in [-E, E]`` and ``E`` the largest extent that
keeps the square inside Theta. Every grid point is simulated ``repeats = 6`` times.
Blob ``landscape.npz``: ``coords (G,)``, ``thetas (G, d)`` or ``(G, G, d)``, ``p_crash``,
``cost_mean`` (mean ``J_T`` over successful repeats, NaN if none), ``cost_mean_nocrash`` (over
non-crashed repeats), ``center (d,)``, ``u (d,)``, ``v (d,)``, ``extent ()``. The figure colors
the rank of ``cost_mean`` (:func:`esbopfn.benchmarks.metrics.cost_rank`) and ``p_crash``.

``unit: "prior_samples"``: 10 controllers ``theta ~ U(Theta)`` on screened instance 0, where an
unsuccessful trajectory is kept only while fewer than 5 of the kept ones are unsuccessful.
Blob ``samples.npz``: ``theta (n, d)``, ``J (n, T)``, ``success (n,)``, ``crash_step (n,)``.

Keys ``{unit: "slice", problem, instance, slice_seed, grid, repeats, instances_seed}`` and
``{unit: "prior_samples", problem, instance, n, max_unsuccessful, instances_seed}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks.instances import BENCH_SEED, load_instances
from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.util.seeds import substream


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    from esbopfn.benchmarks.groups import instances_seed, problems, section
    from esbopfn.problems.base import cached_spec

    sec = section(cfg, "landscapes")
    keys: list[Key] = []
    for p in problems(cfg, sec):
        d = cached_spec(p).d
        for i in sec.get("instances", [0, 1, 2]):
            for s in sec.get("slice_seeds", [1]):
                keys.append({"unit": "slice", "problem": p, "instance": int(i),
                             "slice_seed": int(s),
                             "grid": int(sec.get("grid_1d", 80) if d == 1
                                         else sec.get("grid_2d", 24)),
                             "repeats": int(sec.get("repeats", 6)),
                             "instances_seed": instances_seed(cfg)})
        ps = sec.get("prior_samples")
        if ps:
            keys.append({"unit": "prior_samples", "problem": p,
                         "instance": int(ps.get("instance", 0)), "n": int(ps.get("n", 10)),
                         "max_unsuccessful": int(ps.get("max_unsuccessful", 5)),
                         "instances_seed": instances_seed(cfg)})
    return keys


def max_extent(center: np.ndarray, lo: np.ndarray, hi: np.ndarray, u: np.ndarray,
               v: np.ndarray) -> float:
    """Largest E with every corner of ``center + [-E, E] u + [-E, E] v`` inside [lo, hi]."""
    room = np.minimum(center - lo, hi - center)
    denom = np.abs(u) + np.abs(v)
    mask = denom > 1e-12
    return float(np.min(room[mask] / denom[mask])) if mask.any() else 0.0


def make_slice(lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(center, u, v, extent)`` of a 2-D slice (axis-aligned for d = 2, oblique for d > 2)."""
    d = lo.shape[0]
    center = 0.5 * (lo + hi)
    if d == 2:
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        return center, u, v, max_extent(center, lo, hi, u, v)
    min_extent = 0.05 * float(np.min(0.5 * (hi - lo)))
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for _ in range(200):
        g1 = rng.standard_normal(d)
        u = g1 / np.linalg.norm(g1)
        g2 = rng.standard_normal(d)
        g2 -= np.dot(g2, u) * u
        v = g2 / np.linalg.norm(g2)
        e = max_extent(center, lo, hi, u, v)
        if best is None or e > best[3]:
            best = (center, u, v, e)
        if e >= min_extent:
            break
    assert best is not None
    return best


def _evaluate(problem: Any, instance: Any, theta: np.ndarray, repeats: int,
              rng: np.random.Generator) -> tuple[float, float, float]:
    crashes, ok, nocrash = 0, [], []
    for _ in range(repeats):
        traj = problem.simulate(theta, instance, rng)
        if traj.crashed:
            crashes += 1
        else:
            nocrash.append(float(traj.J[-1]))
        if traj.success:
            ok.append(float(traj.J[-1]))
    return (crashes / repeats, float(np.mean(ok)) if ok else float("nan"),
            float(np.mean(nocrash)) if nocrash else float("nan"))


def run_slice(key: Key, blob_dir: Path) -> None:
    from esbopfn.problems import get_problem

    problem = get_problem(key["problem"])
    spec = problem.spec
    instance = load_instances(key["problem"], key.get("instances_seed", BENCH_SEED))[
        int(key["instance"])]
    labels = ("landscape", key["problem"], int(key["instance"]), int(key["slice_seed"]))
    G, R = int(key["grid"]), int(key["repeats"])
    lo, hi = np.asarray(spec.theta_lo, float), np.asarray(spec.theta_hi, float)
    if spec.d == 1:
        coords = np.linspace(lo[0], hi[0], G)
        thetas = coords[:, None]
        center, u, v, extent = 0.5 * (lo + hi), np.ones(1), np.zeros(1), 0.5 * float(hi[0] - lo[0])
    else:
        center, u, v, extent = make_slice(lo, hi, substream(BENCH_SEED, *labels, "slice"))
        coords = np.linspace(-extent, extent, G)
        tt, ss = np.meshgrid(coords, coords, indexing="ij")     # [t index, s index]
        thetas = center + ss[..., None] * u + tt[..., None] * v
    flat = thetas.reshape(-1, spec.d)
    out = np.array([_evaluate(problem, instance, th, R, substream(BENCH_SEED, *labels, "eval", k))
                    for k, th in enumerate(flat)])
    shape = thetas.shape[:-1]
    np.savez_compressed(Path(blob_dir) / "landscape.npz", coords=coords, thetas=thetas,
                        p_crash=out[:, 0].reshape(shape), cost_mean=out[:, 1].reshape(shape),
                        cost_mean_nocrash=out[:, 2].reshape(shape), center=center, u=u, v=v,
                        extent=np.asarray(extent))


def prior_samples(problem: Any, instance: Any, n: int, max_unsuccessful: int,
                  rng: np.random.Generator, max_draws: int = 10000) -> list[Any]:
    from esbopfn.problems.prior import sample_theta

    kept: list[Any] = []
    bad = 0
    for _ in range(max_draws):
        traj = problem.simulate(sample_theta(problem.spec, rng), instance, rng)
        if not traj.success:
            if bad >= max_unsuccessful:
                continue
            bad += 1
        kept.append(traj)
        if len(kept) == n:
            return kept
    raise RuntimeError(f"{problem.name}: no {n} prior samples in {max_draws} draws")


def run_prior_samples(key: Key, blob_dir: Path) -> None:
    from esbopfn.problems import get_problem

    problem = get_problem(key["problem"])
    instance = load_instances(key["problem"], key.get("instances_seed", BENCH_SEED))[
        int(key["instance"])]
    rng = substream(BENCH_SEED, "prior_samples", key["problem"], int(key["instance"]))
    trajs = prior_samples(problem, instance, int(key["n"]), int(key["max_unsuccessful"]), rng)
    np.savez_compressed(Path(blob_dir) / "samples.npz",
                        theta=np.stack([t.theta for t in trajs]),
                        J=np.stack([np.asarray(t.J, np.float32) for t in trajs]),
                        success=np.array([t.success for t in trajs]),
                        crash_step=np.array([t.crash_step or 0 for t in trajs], np.int32))


def run(key: Key, blob_dir: Path) -> None:
    if key["unit"] == "slice":
        run_slice(key, blob_dir)
    elif key["unit"] == "prior_samples":
        run_prior_samples(key, blob_dir)
    else:
        raise ValueError(f"unknown landscapes unit {key['unit']!r}")


KIND = BenchmarkKind(name="landscapes", enumerate=enumerate_units, run=run)
