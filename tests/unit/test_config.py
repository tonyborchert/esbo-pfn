from pathlib import Path

from esbopfn.config import SUBDIRS, paths, pfn_home, profile_name


def test_env_home(pfn_home):
    assert pfn_home.resolve() == paths().root
    assert not pfn_home.exists()


def test_default_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PFN_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert pfn_home() == (tmp_path / "out").resolve()
    assert pfn_home({"PFN_HOME": ""}) == (tmp_path / "out").resolve()


def test_lazy_creation(tmp_path):
    tree = paths(tmp_path / "h")
    assert not (tmp_path / "h").exists()
    assert set(tree.as_dict()) == set(SUBDIRS)
    for name in ("datasets", "models", "hpo", "checkpoints", "benchmarks", "figures", "jobs",
                 "logs"):
        p = getattr(tree, name)
        assert isinstance(p, Path) and p.is_dir() and p.name == name
    assert tree.db_file == tree.root / "benchmarks" / "db.sqlite"


def test_profile_name():
    assert profile_name({}) == "local"
    assert profile_name({"PFN_PROFILE": "rwth_claix"}) == "rwth_claix"
