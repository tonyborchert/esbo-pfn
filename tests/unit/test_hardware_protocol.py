"""Rig protocol messages and the rig registry."""

from __future__ import annotations

import json

import numpy as np
import pytest

from esbopfn.hardware import protocol
from esbopfn.hardware.rigs import field_defaults, get_rig, load_rigs


def test_request_round_trip():
    text = protocol.encode_request("step", voltage=0.6)
    msg = protocol.decode_request(text)
    assert msg == {"cmd": "step", "voltage": 0.6}
    assert protocol.decode_request(protocol.encode_request("hello")) == {"cmd": "hello"}
    c = protocol.decode_request('{"cmd": "center"}')
    assert c["u_max"] == 1.0 and c["max_seconds"] == 15.0 and c["tol"] == 0.02
    assert protocol.decode_request(protocol.encode_request("begin_trial", seed=3))["seed"] == 3


@pytest.mark.parametrize("text", ['{"cmd": "fly"}', '{"voltage": 1}', "not json",
                                  '{"cmd": "step"}', '{"cmd": "step", "voltage": "x"}',
                                  '{"cmd": "step", "voltage": NaN}',
                                  '{"cmd": "begin_trial", "seed": 1.5}'])
def test_bad_requests(text):
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_request(text)


def test_replies():
    msg = protocol.decode_reply(protocol.state_reply([0.0, np.pi, 0.1, -0.2]))
    assert msg["type"] == "state" and msg["state"].shape == (4,)
    assert msg["state"][1] == pytest.approx(np.pi)
    err = protocol.decode_reply(protocol.reply("error", message="boom"))
    assert err == {"type": "error", "message": "boom"}
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_reply(json.dumps({"type": "state", "state": [1, 2]}))
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_reply('{"type": "weird"}')


def test_rigs_registry():
    rigs = load_rigs()
    assert {"without_weight", "with_weight"} <= set(rigs)
    for rig in rigs.values():
        assert rig.description and rig.reset_reference_method == "decpfn-h"
        assert rig.j_best == "best_on_rig"
    assert get_rig("with_weight").thesis_tree.endswith("different_plant")
    with pytest.raises(KeyError):
        get_rig("no_such_rig")
    fd = field_defaults()
    assert fd.n_max == 180 and fd.t_budget == 36000 and list(fd.seeds) == [0, 1, 2, 3, 4]
    assert fd.methods == ("esrs", "bo", "esbo-gp", "decpfn", "decpfn-h", "xdecpfn-pi")
    assert fd.method_params["decpfn"] == {"tau": 0.9, "model": "decpfn/furuta/binned/1M"}
    assert fd.method_params["xdecpfn-pi"]["tau"] == 0.99


def test_new_rig_is_a_new_entry(tmp_path):
    path = tmp_path / "rigs.yaml"
    path.write_text("rigs:\n  long_link:\n    description: longer pendulum link\n")
    assert load_rigs(path)["long_link"].reset_reference_method == "decpfn-h"
    path.write_text("rigs:\n  bad:\n    mass_note: x\n")
    with pytest.raises(ValueError):
        load_rigs(path)
