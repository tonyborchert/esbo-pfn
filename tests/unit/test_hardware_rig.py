"""Client, simulated server, reset, rig trials and the rig field runner, all in process."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.hardware import records
from esbopfn.hardware.client import RigClient, RigError, RigTimeout
from esbopfn.hardware.reset import RigReset, wrap
from esbopfn.hardware.runner import KIND, FieldPlan, run_field
from esbopfn.hardware.server import serve
from esbopfn.hardware.system import RigSystem, stage_cost
from esbopfn.problems import get_problem

SPEC = get_problem("furuta").spec


@pytest.fixture(scope="module")
def server():
    s = serve("sim", port=0).start()
    yield s
    s.shutdown()


@pytest.fixture
def client(server):
    c = RigClient(server.url)
    yield c
    c.close()


def _theta(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return SPEC.theta_lo + rng.random(SPEC.d) * SPEC.extent


def test_hello_reset_down_step(client):
    info = client.hello()
    assert info == {"plant": "sim", "frequency": 120.0, "protocol": 1, "reseed_state": True}
    s = client.reset_down()
    assert s.shape == (4,) and abs(abs(s[1]) - math.pi) < 1e-9
    s2 = client.step(0.5)
    assert s2.shape == (4,) and s2[2] != 0.0


def test_server_errors_reach_the_client(server):
    c = RigClient(server.url)
    try:
        with pytest.raises(RigError, match="board is not open"):
            c.step(0.1)
    finally:
        c.close()


def test_reply_timeout_discards_and_reconnects(server):
    c = RigClient(server.url, reply_timeouts={"reset_down": 1e-6})
    try:
        with pytest.raises(RigTimeout):
            c.reset_down()
        assert not c.connected
        c.reply_timeouts["reset_down"] = 30.0
        assert c.reset_down().shape == (4,)
        assert c.reconnects == 1
    finally:
        c.close()


def test_connection_reset(client):
    client.hello()
    client.reset_down()
    for _ in range(20):
        client.step(1.0)
    n = client.connections
    s = client.connection_reset()
    assert client.connections == n + 1
    assert s[0] == 0.0 and abs(abs(s[1]) - math.pi) < 1e-9   # board reopened, encoders zeroed


def test_reset_completes_with_diagnostics(client):
    client.hello()
    reset = RigReset(client)
    client.begin_trial(5)
    res = reset.run(np.random.default_rng(1))
    d = res.diagnostics
    assert d["kind"] == "initial" and d["settled"] is True and d["frozen"] is False
    assert res.kick_sign in (-1.0, 1.0) and d["kick_voltage"] == 0.6
    for phase in ("brake", "coast", "center_drive", "center_coast", "settle_brake",
                  "settle_wait", "kick"):
        assert phase in d["phase_steps"]
    assert d["phase_steps"]["coast"] == 181          # 1.5 s at 120 Hz plus the first read
    assert d["phase_steps"]["kick"] == 5             # 4 kick periods and the 0 V step
    assert d["phase_steps"]["settle_wait"] >= 240    # 2 s continuous rest
    assert res.state[0] == pytest.approx(wrap(res.raw_state[0] - res.zero_raw), abs=1e-12)
    # the kick sign comes from the trial generator
    again = RigReset(client)
    again.opened = True
    client.begin_trial(5)
    assert again.run(np.random.default_rng(1)).kick_sign == res.kick_sign


def test_reset_with_connection_reset(client):
    client.hello()
    reset = RigReset(client)
    reset.open()
    res = reset.run(np.random.default_rng(0), reopen=True)
    assert res.diagnostics["kind"] == "connection_reset"
    assert "reopen_settled" in res.diagnostics


def test_one_trial_record(client):
    system = RigSystem(client)
    system.open_session()
    res = system.run_trial(_theta(4), np.random.default_rng(0), plant_seed=1)
    assert 1 <= res.t_obs <= 600 and len(res.J) == res.t_obs == len(res.x)
    assert np.all(np.diff(res.J) >= 0)
    np.testing.assert_allclose(res.J, np.cumsum(stage_cost(res.x[:, 1])), rtol=1e-5)
    if res.crash_step is not None:
        assert res.crash_step == res.t_obs and not res.success and res.max_arm_deg > 90
    else:
        assert res.t_obs == 600
    assert res.control_s > 0 and res.reset.seconds > 0


def test_full_trial_runs_600_steps(client):
    """A controller that swings up and holds on the simulated rig (theta draw 22): 600 logged
    steps, success by the 2 s hold test. The simulated rig restarts every trial from a seeded
    rest state, so the outcome does not depend on earlier tests."""
    system = RigSystem(client)
    system.open_session()
    res = system.run_trial(_theta(22), np.random.default_rng(22), plant_seed=22)
    assert res.crash_step is None and not res.stopped
    assert res.t_obs == 600 and res.J.shape == (600,) and res.x.shape == (600, 2)
    assert res.success and res.mean_pend_final_deg < 5.0
    assert abs(res.x[-1, 1]) < math.radians(5)


def test_early_stop_callback(client):
    system = RigSystem(client)
    system.open_session()
    seen = []

    def stop(k, running):
        seen.append((k, running.t_obs, float(running.J_obs)))
        return k == 7

    res = system.run_trial(_theta(1), np.random.default_rng(0), stop=stop, plant_seed=2)
    if res.crash_step is not None and res.crash_step < 7:
        pytest.skip("crashed before the stop step")
    assert res.stopped and res.t_obs == 7 and not res.success
    assert [k for k, _, _ in seen] == list(range(1, 8))
    assert seen[-1][2] == pytest.approx(float(res.J[-1]))


def _deterministic(path):
    return [t.deterministic_view() for t in records.read_trials(path)]


def test_field_runs_and_resumes_identically(server, pfn_home, monkeypatch):
    plan = FieldPlan("without_weight", ["esrs", "bo"], [0], n_max=4, t_budget=3000)
    db = Db()
    c = RigClient(server.url)
    try:
        results = run_field(plan, c, db)
    finally:
        c.close()
    assert [r.n_trials for r in results] == [4, 4] and all(r.finished for r in results)
    reference = {r.key["method"]: _deterministic(r.blob_dir) for r in results}
    for r in results:
        assert db.has(KIND, r.key)
        recs = records.read_trials(r.blob_dir)
        assert [t.index for t in recs] == [0, 1, 2, 3]
        assert all(t.source == "init" for t in recs)
        run = records.read_run(r.blob_dir)
        assert run["status"] == "done" and run["n_trials"] == 4
    # esrs and bo share the initial design of seed 0
    assert [t["theta"] for t in reference["esrs"]] == [t["theta"] for t in reference["bo"]]

    # a second field in a fresh home, killed during its third trial, then rerun
    monkeypatch.setenv("PFN_HOME", str(pfn_home.parent / "second"))
    db2 = Db()
    calls = {"n": 0}
    real = RigSystem.run_trial

    def dying(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(self, *a, **kw)

    monkeypatch.setattr(RigSystem, "run_trial", dying)
    c = RigClient(server.url)
    with pytest.raises(KeyboardInterrupt):
        run_field(plan, c, db2)
    c.close()
    key = plan.keys()[0]
    assert len(records.read_trials(db2.blob_dir(KIND, key))) == 2
    assert not db2.has(KIND, key)
    monkeypatch.setattr(RigSystem, "run_trial", real)
    c = RigClient(server.url)
    try:
        resumed = run_field(plan, c, db2)
    finally:
        c.close()
    assert [r.new_trials for r in resumed] == [2, 4]
    for r in resumed:
        assert _deterministic(r.blob_dir) == reference[r.key["method"]]


def test_pause_flag_stops_the_field(server, pfn_home):
    from esbopfn.hardware.runner import pause_flag

    db = Db()
    flag = pause_flag(db)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    c = RigClient(server.url)
    try:
        assert run_field(FieldPlan("with_weight", ["esrs"], [0], n_max=2), c, db) == []
    finally:
        c.close()
