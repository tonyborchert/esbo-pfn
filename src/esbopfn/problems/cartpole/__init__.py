"""Cartpole: Gymnasium cart-pole under two parallel PD loops with a bang-bang force.

``SPEC`` and ``PRIOR`` are loaded from ``configs/problems.yaml`` on first access.
"""

from __future__ import annotations

from typing import Any

from esbopfn.problems.base import cached_prior, cached_spec
from esbopfn.problems.cartpole.sim import NAME, simulate

__all__ = ["PRIOR", "SPEC", "simulate"]


def __getattr__(name: str) -> Any:
    if name == "SPEC":
        return cached_spec(NAME)
    if name == "PRIOR":
        return cached_prior(NAME)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
