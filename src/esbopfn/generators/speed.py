"""Generation-speed benchmark (thesis Appendix table tab:app:prior-cost), measured on one core."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from esbopfn.generators import binaries

#: Display names of the problems in tables.
DISPLAY = {"boiler": "Boiler", "threetank_pi": "Three-Tank", "pt2_deadtime": "PT2 dead-time",
           "cartpole": "Cartpole", "furuta": "Furuta"}

#: tab:app:prior-cost as printed in the thesis (one core of an Intel Xeon Platinum 8468): seconds
#: per 10,000 single trajectories, trajectory acceptance, system acceptance, seconds per 10,000
#: FT-PFN sets and per 10,000 XDecPFN sets.
THESIS_PRIOR_COST: dict[str, tuple[float, float, float, float, float]] = {
    "boiler": (0.27, 1.00, 1.00, 60.0, 342.0),
    "threetank_pi": (0.59, 1.00, 1.00, 228.0, 588.0),
    "pt2_deadtime": (0.57, 0.88, 0.88, 294.0, 1218.0),
    "cartpole": (0.78, 0.65, 0.64, 1260.0, 1500.0),
    "furuta": (45.0, 0.0136, 0.015, 20520.0, 21600.0),
}


@dataclass
class SpeedResult:
    """One row of the generation-speed table."""

    problem: str
    traj_s_per_10k: float
    traj_acceptance: float
    system_acceptance: float
    ftpfn_s_per_10k: float
    xdecpfn_s_per_10k: float
    single_trajectories: int = 0
    xdecpfn_sets: int = 0
    ftpfn_sets: int = 0
    system_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_duration(seconds: float) -> str:
    """Seconds as the thesis prints them: s below a minute, min below an hour, h above."""
    if not math.isfinite(seconds):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def format_acceptance(a: float) -> str:
    if not math.isfinite(a):
        return "n/a"
    return f"{a:.2f}" if a >= 0.1 else f"{a:.4f}"


def row(r: SpeedResult) -> str:
    """The table row ``Problem | time | acceptance | acceptance | FT-PFN | XDecPFN``."""
    return " | ".join([DISPLAY.get(r.problem, r.problem), format_duration(r.traj_s_per_10k),
                       format_acceptance(r.traj_acceptance), format_acceptance(r.system_acceptance),
                       format_duration(r.ftpfn_s_per_10k), format_duration(r.xdecpfn_s_per_10k)])


def thesis_row(problem: str) -> str:
    t = THESIS_PRIOR_COST[problem]
    return row(SpeedResult(problem, t[0], t[1], t[2], t[3], t[4]))


HEADER = "Problem | Traj. time | Traj. acceptance | System acceptance | FT-PFN sets | XDecPFN sets"


def measure(problem: str, count: int = 2000, sets: int = 10, budget_s: float = 120.0,
            seed: int = 0, home: Path | str | None = None) -> SpeedResult:
    """Run the binary's ``speed`` subcommand (single core) and parse its result.

    ``count`` single trajectories and ``sets`` trajectory sets of each kind are timed, each phase
    stopping early after ``budget_s`` seconds. Times are scaled to 10,000 samples.
    """
    res = binaries.run_json(problem, "speed", "--count", str(count), "--sets", str(sets),
                            "--budget", str(budget_s), "--seed", str(seed), home=home)
    return SpeedResult(problem=problem, traj_s_per_10k=float(res["traj_s_per_10k"]),
                       traj_acceptance=float(res["traj_acceptance"]),
                       system_acceptance=float(res["system_acceptance"]),
                       ftpfn_s_per_10k=float(res["ftpfn_s_per_10k"]),
                       xdecpfn_s_per_10k=float(res["xdecpfn_s_per_10k"]),
                       single_trajectories=int(res["single_trajectories"]),
                       xdecpfn_sets=int(res["xdecpfn_sets"]), ftpfn_sets=int(res["ftpfn_sets"]),
                       system_candidates=int(res["system_candidates"]))
