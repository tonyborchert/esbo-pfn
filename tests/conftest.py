"""Marker handling: tests marked cpp, gpu, slurm or hardware are skipped unless the matching
``--run-<marker>`` option is given or the resource probe finds the resource. Unmarked tests get
the ``unit`` marker."""

from __future__ import annotations

import functools
import shutil

import matplotlib
import pytest

RESOURCE_MARKERS = ("cpp", "gpu", "slurm", "hardware")


@functools.cache
def _probe(marker: str) -> bool:
    if marker == "cpp":
        from esbopfn.commands.doctor import find_cpp_binaries

        return all(p is not None for p in find_cpp_binaries().values())
    if marker == "gpu":
        return False  # gpu trainings are slow, run them with --run-gpu
    if marker == "slurm":
        return shutil.which("sbatch") is not None
    return False  # hardware needs an explicit --run-hardware


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker in RESOURCE_MARKERS:
        parser.addoption(
            f"--run-{marker}", action="store_true", default=False,
            help=f"run tests marked {marker} even if the resource probe fails",
        )


def pytest_configure(config: pytest.Config) -> None:
    matplotlib.use("Agg")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        needs = [m for m in RESOURCE_MARKERS if m in item.keywords]
        if not needs:
            item.add_marker(pytest.mark.unit)
            continue
        for marker in needs:
            if not (config.getoption(f"--run-{marker}") or _probe(marker)):
                item.add_marker(pytest.mark.skip(reason=f"needs {marker} (--run-{marker})"))
                break


@pytest.fixture
def pfn_home(tmp_path, monkeypatch):
    """An isolated PFN_HOME under tmp_path."""
    home = tmp_path / "pfn_home"
    monkeypatch.setenv("PFN_HOME", str(home))
    monkeypatch.delenv("PFN_PROFILE", raising=False)
    return home
