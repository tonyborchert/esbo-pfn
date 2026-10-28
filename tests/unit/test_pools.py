"""Pool format, chunk bookkeeping, dataset catalog and chunk seeds (no generator binaries needed)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from esbopfn.core import Instance, Trajectory
from esbopfn.generators import (
    chunk_seed,
    chunks_of_shard,
    format_chunk_list,
    open_dataset,
    parse_shard,
    status,
)
from esbopfn.generators.datasets import UnknownDataset, dataset_dir, load_catalog
from esbopfn.generators.pools import (
    STATUS_COMPLETE,
    STATUS_CRASHED,
    STATUS_TRUNCATED,
    ContextExample,
    ContextStore,
    Pool,
    PoolSpec,
    append_chunk_log,
    chunk_units,
    make_spec,
    n_chunks,
    pack_trajectory,
    read_chunk_log,
    write_context_shard,
)
from esbopfn.generators.verify import check_contexts, check_pool
from esbopfn.grids import decoder_grid
from esbopfn.problems.registry import get_spec
from esbopfn.util.seeds import derive_seed

FIELDS = {"phi": ["m", "l"], "noise_std": ["x", "v"], "extras": ["kick"]}


def _spec(T: int = 12) -> PoolSpec:
    return make_spec(kind="traj", problem="cartpole", d=4, T=T, channels=["x_cart", "phi_pole"],
                     rate_channels=["xdot_cart", "phidot_pole"],
                     theta_names=["a", "b", "c", "d"], instance_fields=FIELDS)


def _traj(rng: np.random.Generator, T: int = 12, crash: int | None = None) -> Trajectory:
    J = np.cumsum(rng.uniform(0.1, 1.0, T)).astype(np.float32)
    if crash is not None:
        J[crash - 1:] = np.inf
    return Trajectory(theta=rng.uniform(size=4), J=J, x=rng.normal(size=(T, 2)).astype(np.float32),
                      xdot=rng.normal(size=(T, 2)).astype(np.float32), crash_step=crash,
                      success=crash is None)


def _instance(i: int) -> Instance:
    return Instance(phi={"m": 1.0 + i, "l": 0.5}, phi_hat={}, noise_std={"x": 0.1, "v": 0.2},
                    extras={"kick": 0.3}, seed=i)


# --------------------------------------------------------------------------- format
def test_spec_round_trip_and_tiling():
    spec = _spec()
    again = PoolSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again.to_dict() == spec.to_dict()
    names = [c.name for c in spec.columns]
    assert names == ["theta", "phi", "noise_std", "extras", "crash_step", "success", "J", "x", "xdot"]
    assert spec.row_floats == 4 + 2 + 2 + 1 + 1 + 1 + 12 + 24 + 24
    assert spec.column("x").shape == (12, 2) and spec.xdot_mode == "per_step"
    bad = spec.to_dict()
    bad["row_floats"] += 1
    with pytest.raises(ValueError):
        PoolSpec.from_dict(bad)


def test_grid_and_label_layouts():
    spec = get_spec("pt2_deadtime")
    steps = list(decoder_grid(spec)) + [spec.T]
    g = make_spec(kind="set", problem=spec.name, d=spec.d, T=spec.T, channels=spec.channels,
                  resolution="grid", steps=steps, rows_per_unit=135)
    assert [c.name for c in g.columns][:2] == ["instance", "theta"]
    assert g.xdot_mode == "interval_sum" and g.column("xdot").shape == (len(steps), 1)
    assert g.extra["rows_per_unit"] == 135
    lab = make_spec(kind="labels", problem=spec.name, d=spec.d, T=spec.T, channels=spec.channels)
    assert lab.column("J").shape == (1,) and int(lab.steps[0]) == spec.T and not lab.has("x")


def test_pool_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    spec = _spec()
    pool = Pool.create(tmp_path / "p", spec, meta={"chunk": 4})
    trajs = [_traj(rng, crash=(5 if k % 3 == 0 else None)) for k in range(8)]
    for chunk in (1, 0):  # out of order on purpose
        rows = np.stack([pack_trajectory(spec, trajs[k], _instance(k))
                         for k in range(4 * chunk, 4 * chunk + 4)])
        pool.append_chunk(chunk, rows)
    pool = Pool.open(tmp_path / "p")
    assert len(pool) == 8 and pool.done_chunks() == {1: 4, 0: 4}
    for k, tr in enumerate(trajs):
        got = pool.read(k)
        np.testing.assert_array_equal(got.J, tr.J)
        np.testing.assert_allclose(got.theta, tr.theta, rtol=1e-6)
        np.testing.assert_array_equal(got.x, tr.x)
        np.testing.assert_array_equal(got.xdot, tr.xdot)
        assert got.crash_step == tr.crash_step and got.success == tr.success
        assert pool.instance(k).phi == {"m": 1.0 + k, "l": 0.5}
    assert pool.slice(2, 5).shape == (3, spec.row_floats)
    assert pool.column("J", np.arange(3)).shape == (3, 12)
    assert math.isinf(pool.read(0).J_total) and pool.read(0).cut(10).t_obs == 5
    limited = Pool.open(tmp_path / "p", limit=3)
    assert len(limited) == 3 and limited.rows.shape[0] == 3


def test_pool_create_refuses_other_layout(tmp_path):
    Pool.create(tmp_path / "p", _spec(12))
    with pytest.raises(ValueError):
        Pool.create(tmp_path / "p", _spec(13))


def test_check_pool_on_synthetic_rows(tmp_path, monkeypatch):
    from esbopfn.generators import verify

    rng = np.random.default_rng(1)
    spec = _spec()
    pool = Pool.create(tmp_path / "p", spec, meta={"chunk": 6})
    rows = np.stack([pack_trajectory(spec, _traj(rng, crash=3 if k == 0 else None), _instance(k))
                     for k in range(6)])
    pool.append_chunk(0, rows)
    monkeypatch.setattr(verify, "_theta_box", lambda p: (np.zeros(4), np.ones(4)))
    rep = check_pool(Pool.open(tmp_path / "p"))
    assert rep.ok, rep.lines()
    # a finite J after the crash step must be caught
    raw = np.memmap(tmp_path / "p" / "rows.f32", dtype="<f4", mode="r+",
                    shape=(6, spec.row_floats))
    raw[0, spec.column("J").offset + 8] = 1.0
    raw.flush()
    assert not check_pool(Pool.open(tmp_path / "p")).ok


# --------------------------------------------------------------------------- chunks and resume
def test_chunk_arithmetic_and_log(tmp_path):
    assert n_chunks(1_000_000, 10_000) == 100 and n_chunks(1024, 1000) == 2
    assert chunk_units(1024, 1000, 1) == 24 and chunk_units(1024, 1000, 2) == 0
    log = tmp_path / "chunks.log"
    assert read_chunk_log(log) == {}
    append_chunk_log(log, 3, 10)
    append_chunk_log(log, 0, 10)
    assert read_chunk_log(log) == {3: 10, 0: 10}


def test_shards_and_chunk_lists():
    assert parse_shard("1/4") == (1, 4) and parse_shard(None) is None
    with pytest.raises(ValueError):
        parse_shard("4/4")
    assert chunks_of_shard(10, (1, 4)) == [1, 5, 9]
    parts = [chunks_of_shard(10, (i, 3)) for i in range(3)]
    assert sorted(sum(parts, [])) == list(range(10))
    assert format_chunk_list([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert format_chunk_list([4]) == "4"


def test_chunk_seed_contract():
    # chunk k of a dataset with seed S uses derive_seed(S, k); values pinned from numpy's
    # SeedSequence (the binaries reproduce them, see tests/cpp)
    assert chunk_seed(0, 0) == derive_seed(0, 0) == 8668861027912758289
    assert chunk_seed(12345, 7) == 4008720786225166622
    assert chunk_seed(1099511627781, 8589934593) == 7021997142557742453
    assert len({chunk_seed(1001, k) for k in range(100)}) == 100
    assert all(0 <= chunk_seed(5, k) < 2**63 for k in range(10))


# --------------------------------------------------------------------------- catalog
def test_shipped_catalog():
    cat = load_catalog()
    assert cat.problems == ("boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta")
    for p in cat.problems:
        one, small = cat.resolve(f"traj/{p}/1M"), cat.resolve(f"traj/{p}/100k")
        assert one.count == 1_000_000 and small.count == 100_000
        assert small.prefix_of == one.id and cat.storage(small.id) == one
        assert small.seed == one.seed and small.chunk == one.chunk
        assert one.count % one.chunk == 0 and small.count % one.chunk == 0
        val, val_lc = cat.resolve(f"traj/{p}/val"), cat.resolve(f"traj/{p}/val_lcpfn")
        assert val.count == 3000 and val_lc.count == 1024 and val_lc.prefix_of == val.id
        assert val.seed != one.seed
        assert cat.resolve(f"sets/{p}/xdecpfn").count == 132_400
        assert cat.resolve(f"sets/{p}/xdecpfn_heldout").count == 256
        assert cat.resolve(f"contexts/{p}/ftpfn").count == 3_500_000
        assert cat.resolve(f"contexts/{p}/ftpfn_heldout").count == 2000
        assert cat.resolve(f"fields/{p}/oracle65").placeholder
        assert cat.resolve(f"fields/{p}/oracle181").placeholder
    assert cat.resolve("traj/furuta/1M_nodeadtime").options == {"dead_time": False}
    with pytest.raises(UnknownDataset):
        cat.resolve("traj/cartpole/1M_nodeadtime")
    assert cat.for_model("decpfn", "pt2_deadtime") == ["traj/pt2_deadtime/1M", "traj/pt2_deadtime/val"]
    assert cat.for_model("ftpfn", "furuta")[0] == "contexts/furuta/ftpfn"
    seeds = {(d.kind, d.seed) for d in cat.datasets.values() if not d.prefix_of and not d.placeholder}
    assert len({s for _, s in seeds}) == len(seeds)


def test_dataset_dirs(tmp_path):
    cat = load_catalog()
    one = dataset_dir("traj/cartpole/1M", tmp_path, cat)
    assert one == tmp_path.resolve() / "datasets" / "cartpole" / "traj" / "1M"
    assert dataset_dir("traj/cartpole/100k", tmp_path, cat) == one


CATALOG = """
problems: [cartpole]
datasets:
  traj/{p}/big: {kind: traj, count: 10, seed: 7, chunk: 2}
  traj/{p}/small: {prefix_of: "traj/{p}/big", count: 4}
  contexts/{p}/c: {kind: contexts, count: 5, seed: 8, chunk: 2}
  fields/{p}/f: {kind: fields, placeholder: true}
models:
  m: ["traj/{p}/big"]
"""


def test_status_resume_and_prefix(tmp_path):
    (tmp_path / "datasets.yaml").write_text(CATALOG)
    cat = load_catalog(tmp_path / "datasets.yaml")
    home = tmp_path / "home"
    st = status("traj/cartpole/big", home, cat)
    assert (st.chunks_total, st.chunks_done, st.finished, st.exists) == (5, 0, False, False)
    spec = _spec()
    directory = dataset_dir("traj/cartpole/big", home, cat)
    pool = Pool.create(directory, spec, meta={"chunk": 2})
    rng = np.random.default_rng(2)
    for chunk in (0, 1, 3):
        rows = np.stack([pack_trajectory(spec, _traj(rng), _instance(k)) for k in range(2)])
        pool.append_chunk(chunk, rows)
    append_chunk_log(directory / "chunks.log", 4, 1)  # wrong unit count: not done
    big = status("traj/cartpole/big", home, cat)
    assert (big.chunks_done, big.units_done, big.finished) == (3, 6, False)
    small = status("traj/cartpole/small", home, cat)
    assert (small.chunks_total, small.chunks_done, small.units_done, small.finished) == (2, 2, 4, True)
    view = open_dataset("traj/cartpole/small", home, cat)
    assert isinstance(view, Pool) and len(view) == 4
    assert len(open_dataset("traj/cartpole/big", home, cat)) == 8
    f = status("fields/cartpole/f", home, cat)
    assert f.placeholder and not f.finished


def test_generate_refuses_placeholders(tmp_path):
    from esbopfn.generators import NotGeneratable, generate

    (tmp_path / "datasets.yaml").write_text(CATALOG)
    cat = load_catalog(tmp_path / "datasets.yaml")
    with pytest.raises(NotGeneratable):
        generate("fields/cartpole/f", home=tmp_path, catalog=cat)


# --------------------------------------------------------------------------- set pools
def test_set_pool_instances(tmp_path):
    spec = make_spec(kind="set", problem="cartpole", d=4, T=12, channels=["x", "p"],
                     rate_channels=["v", "w"], rows_per_unit=3)
    pool = Pool.create(tmp_path / "s", spec, meta={"chunk": 2})
    rng = np.random.default_rng(3)
    rows = np.stack([pack_trajectory(spec, _traj(rng), instance_index=k // 3) for k in range(6)])
    pool.append_chunk(0, rows)
    (tmp_path / "s" / "instances.d").mkdir()
    recs = [{"index": i, "instance": {"phi": {"m": 1.0}, "phi_hat": {}, "noise_std": {"x": 0.1},
                                      "extras": {}, "seed": 10 + i}} for i in range(2)]
    (tmp_path / "s" / "instances.d" / "chunk_000000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    pool = Pool.open(tmp_path / "s")
    assert pool.rows_per_unit == 3 and pool.done_chunks() == {0: 2}
    assert pool.instance(1).seed == 11
    np.testing.assert_array_equal(pool.instance_rows(1)[:, spec.column("instance").offset], 1.0)
    assert pool.read_grid(4).instance == 1


# --------------------------------------------------------------------------- context shards
def _example(rng: np.random.Generator, n: int, T: int = 40) -> ContextExample:
    grid = [5 * (k + 1) for k in range(8)]
    tok_t = np.zeros((n, 9), np.int16)
    tok_J = np.zeros((n, 9), np.float32)
    length = np.zeros(n, np.int32)
    status_ = np.zeros(n, np.int8)
    target = np.zeros(n, np.float32)
    for i in range(n):
        L = int(rng.integers(1, T + 1))
        g = [t for t in grid if t <= L] + ([L] if L < T else [])
        tok_t[i, :len(g)] = g
        tok_J[i, :len(g)] = np.arange(1, len(g) + 1, dtype=np.float32)
        length[i] = L
        status_[i] = STATUS_COMPLETE if L == T else STATUS_TRUNCATED
        target[i] = 100.0 if L < T else tok_J[i, len(g) - 1]
    if n > 1:  # one crashed trial
        status_[1], tok_J[1, (tok_t[1] > 0).sum() - 1], target[1] = STATUS_CRASHED, np.inf, np.inf
        if length[1] == T:
            length[1] = T - 1
            tok_t[1, (tok_t[1] > 0).sum() - 1] = T - 1
    return ContextExample(theta=rng.uniform(size=(n, 3)).astype(np.float32), length=length,
                          status=status_, target=target, tok_t=tok_t, tok_J=tok_J,
                          unrun_theta=rng.uniform(size=(16, 3)).astype(np.float32),
                          unrun_target=rng.uniform(size=16).astype(np.float32), p_cut=0.3)


def test_context_shard_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    ex = [_example(rng, n) for n in (1, 5, 3)]
    write_context_shard(tmp_path / "shards" / "chunk_000000", ex[:2], problem="pt2_deadtime",
                        T=40, d=3, chunk=0, first_example=0)
    write_context_shard(tmp_path / "shards" / "chunk_000001", ex[2:], problem="pt2_deadtime",
                        T=40, d=3, chunk=1, first_example=2)
    store = ContextStore(tmp_path)
    assert len(store) == 3
    for i, e in enumerate(ex):
        got = store.example(i)
        assert got.n == e.n
        np.testing.assert_array_equal(got.tok_t, e.tok_t)
        np.testing.assert_array_equal(got.tok_J, e.tok_J)
        np.testing.assert_array_equal(got.status, e.status)
        np.testing.assert_array_equal(got.unrun_target, e.unrun_target)
        t, J = got.tokens(0)
        assert t.max() == got.length[0] and len(t) == len(J)
    with pytest.raises(IndexError):
        store.example(3)
    rep = check_contexts(store)
    assert rep.ok, rep.lines()
