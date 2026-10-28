"""Round trip against a real ``qube`` rig server (opt-in: ``--run-hardware``).

Start the server on the workstation first:
``python -m esbopfn.hardware.server --plant qube --port 8765``. The test moves the rig: it runs
one reset (brake, coast, centering, settle, kick) and a short trial with an early stop.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.hardware.client import DEFAULT_URL, RigClient
from esbopfn.hardware.system import RigSystem

pytestmark = pytest.mark.hardware


def test_qube_round_trip():
    client = RigClient(DEFAULT_URL)
    try:
        info = client.hello()
        assert info["plant"] == "qube" and info["frequency"] == 120.0
        state = client.reset_down()
        assert state.shape == (4,) and abs(abs(state[1]) - math.pi) < math.radians(10)
        system = RigSystem(client)
        system.resetter.opened = True
        theta = np.array([65.0, 0.95, 12.0, 5.0, 1.0, 1.0, 1.0, 2.5])
        res = system.run_trial(theta, np.random.default_rng(0), stop=lambda k, tr: k >= 60)
        assert res.reset.diagnostics["settled"]
        assert res.t_obs <= 60 and len(res.J) == res.t_obs
    finally:
        client.close()
