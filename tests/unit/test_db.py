import json

import numpy as np
import pytest

from esbopfn.benchmarks.db import DONE, FAILED, Db, canonical_json, key_hash


def test_key_hash_canonical():
    a = {"b": 1, "a": [1, 2], "c": {"y": 1.5, "x": "s"}}
    b = {"c": {"x": "s", "y": 1.5}, "a": [1, 2], "b": 1}
    assert key_hash(a) == key_hash(b)
    assert canonical_json({"n": np.int64(3), "f": np.float32(0.5)}) == '{"f":0.5,"n":3}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_run_idempotent(tmp_path):
    db = Db(tmp_path / "db.sqlite")
    calls = []

    def fn(key, blob_dir):
        calls.append(key)
        (blob_dir / "result.json").write_text(json.dumps({"v": key["seed"] * 2}))

    key = {"problem": "cartpole", "seed": 3}
    assert not db.has("toy", key)
    out = db.run("toy", key, fn)
    assert db.has("toy", key) and json.loads((out / "result.json").read_text())["v"] == 6
    assert db.run("toy", key, fn) == out
    assert len(calls) == 1
    unit = db.get("toy", key)
    assert unit.status == DONE and unit.duration_s >= 0 and unit.key == key
    assert db.counts() == {("toy", DONE): 1}


def test_failure_marks_failed(tmp_path):
    db = Db(tmp_path / "db.sqlite")

    def boom(key, blob_dir):
        (blob_dir / "partial").write_text("x")
        raise RuntimeError("fail")

    key = {"seed": 1}
    with pytest.raises(RuntimeError):
        db.run("toy", key, boom)
    assert not db.has("toy", key) and db.get("toy", key).status == FAILED
    assert not db.blob_dir("toy", key).exists()
    db.run("toy", key, lambda k, d: (d / "ok").write_text("1"))
    assert db.has("toy", key)


def test_query_filters(tmp_path):
    db = Db(tmp_path / "db.sqlite")
    for p in ("boiler", "cartpole"):
        for s in range(3):
            db.run("toy", {"problem": p, "seed": s}, lambda k, d: None)
    assert len(db.query("toy")) == 6
    assert len(db.query("toy", problem="cartpole")) == 3
    assert [u.key["seed"] for u in db.query("toy", problem="boiler", seed=np.int64(2))] == [2]
    assert db.query("other") == []
