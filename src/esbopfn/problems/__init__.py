"""Control problems: specs and priors from ``configs/problems.yaml``, simulators per subpackage.

``get_problem(name)`` returns a :class:`Problem` bundling the spec, the universal prior and the
simulator of one of ``boiler``, ``threetank_pi``, ``pt2_deadtime``, ``cartpole`` and ``furuta``.
"""

from __future__ import annotations

import importlib

from esbopfn.problems.base import Problem, cached_prior, cached_spec
from esbopfn.problems.registry import PROBLEM_NAMES

__all__ = ["PROBLEM_NAMES", "Problem", "get_problem"]


def get_problem(name: str, dead_time: bool = True) -> Problem:
    """Return the problem ``name``.

    ``dead_time=False`` (Furuta only) fixes the actuation dead time at zero, the plant of the
    hardware prediction ablation without dead time.
    """
    if name not in PROBLEM_NAMES:
        raise KeyError(f"unknown problem {name!r}, known: {', '.join(PROBLEM_NAMES)}")
    module = importlib.import_module(f"esbopfn.problems.{name}.sim")
    prior = cached_prior(name)
    if not dead_time:
        if name != "furuta":
            raise ValueError("dead_time=False only applies to furuta")
        prior = prior.with_extra("dead_time", 0.0)
    return Problem(spec=cached_spec(name), prior=prior, simulator=module.simulate)
