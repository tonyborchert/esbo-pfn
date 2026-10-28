from esbopfn.benchmarks.kinds import discover_kinds
from esbopfn.cli import build_parser, discover_commands, main
from esbopfn.plots import discover_groups


def test_commands_discovered():
    names = {n.rsplit(".", 1)[1] for n in discover_commands()}
    assert {"status", "doctor", "submit", "config", "bench", "plot"} <= names
    build_parser()


def test_empty_registries(pfn_home, capsys):
    assert isinstance(discover_kinds(), dict) and isinstance(discover_groups(), dict)
    for argv in (["bench", "plan"], ["bench", "run", "--shard", "0/2"], ["bench", "kinds"],
                 ["plot", "--format", "png", "--theme", "black"], ["status"], ["doctor"]):
        assert main(argv) == 0
    out = capsys.readouterr().out
    assert "PFN_HOME" in out


def test_plan_and_run_with_toy_group(tmp_path):
    from esbopfn.benchmarks.db import Db
    from esbopfn.benchmarks.kinds import BenchmarkKind
    from esbopfn.benchmarks.runner import Shard, plan, run_units
    from esbopfn.plots import PlotGroup

    kind = BenchmarkKind(
        name="toy",
        enumerate=lambda spec: [{"seed": s} for s in range(spec["n"])],
        run=lambda key, d: (d / "v.txt").write_text(str(key["seed"])),
    )
    group = PlotGroup(
        name="g",
        needs=lambda cfg, db: [("toy", k) for k in kind.enumerate({"n": 5})] * 2,
        render=lambda cfg, db, out, fmt, theme: None,
    )
    db = Db(tmp_path / "db.sqlite")
    shards = [plan({"g": group}, db, Shard(i, 2)) for i in range(2)]
    assert sorted(len(s) for s in shards) == [2, 3]
    assert run_units(shards[0], db, {"toy": kind}) == (len(shards[0]), 0)
    assert run_units(plan({"g": group}, db), db, {"toy": kind}) == (len(shards[1]), len(shards[0]))
    assert len(db.query("toy")) == 5
