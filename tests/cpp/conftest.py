"""Generator binaries for the ``cpp`` tests: the installed or checkout builds if present, else a
fresh cmake build into a temporary directory (only when ``--run-cpp`` forces the tests)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from esbopfn.generators import binaries


@pytest.fixture(scope="session")
def gen_home(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """A PFN_HOME whose bin/ holds the binaries, or None when they are already found."""
    try:
        for p in binaries.BINARY_OF:
            binaries.find_binary(p)
        return None
    except binaries.BinaryMissing:
        pass
    if shutil.which("cmake") is None:
        pytest.skip("generator binaries missing and cmake not available")
    home = tmp_path_factory.mktemp("pfn_home")
    binaries.build(home=home, build_dir=tmp_path_factory.mktemp("cpp_build"))
    return home


@pytest.fixture
def run(gen_home):
    """Call a binary subcommand: run(problem, sub, *args) -> stdout."""

    def _run(problem: str, sub: str, *args: str) -> str:
        return binaries.run(problem, sub, *args, home=gen_home)

    return _run
