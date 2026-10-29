import subprocess
import sys

import pytest

from esbopfn.cli import main
from esbopfn.cluster.profile import available_profiles, load_profile
from esbopfn.cluster.submit import parse_array


@pytest.fixture(autouse=True)
def _accounts(monkeypatch):
    monkeypatch.setenv("PFN_SLURM_ACCOUNT_CPU", "acct_cpu")
    monkeypatch.setenv("PFN_SLURM_ACCOUNT_GPU", "acct_gpu")


def dry_run(capsys, *args):
    code = main(["submit", "--dry-run", *args])
    assert code == 0
    return capsys.readouterr().out


def test_profiles_load():
    assert {"local", "rwth_claix"} <= set(available_profiles())
    claix = load_profile("rwth_claix")
    assert claix.scheduler == "slurm" and claix.resources("train").gpus == 1
    assert load_profile("local").scheduler == "none"


def test_dry_run_claix_gpu(pfn_home, capsys):
    out = dry_run(capsys, "--profile", "rwth_claix", "--kind", "train", "--time", "01:00:00",
                  "--name", "tr", "--", "pfn", "train", "decpfn", "--problem", "cartpole")
    assert out.startswith("#!/bin/bash\n#SBATCH --job-name=tr\n")
    assert "#SBATCH --account=acct_gpu" in out and "#SBATCH --partition=c23g" in out
    assert "#SBATCH --gres=gpu:1" in out and "#SBATCH --time=01:00:00" in out
    assert "#SBATCH --cpus-per-task=8" in out and "#SBATCH --mem=64G" in out
    assert f"#SBATCH --output={pfn_home.resolve()}/logs/%x_%j.out" in out
    assert "module load GCCcore/13.3.0 Python/3.12.3" in out
    assert "PYTORCH_CUDA_ALLOC_CONF" in out
    assert f"export PFN_HOME={pfn_home.resolve()}" in out
    assert "export PFN_PROFILE=rwth_claix" in out
    assert out.rstrip().endswith("uv run --no-sync pfn train decpfn --problem cartpole")
    assert not (pfn_home / "jobs").exists()


def test_account_placeholders(monkeypatch):
    claix = load_profile("rwth_claix")
    monkeypatch.delenv("PFN_SLURM_ACCOUNT_CPU", raising=False)
    monkeypatch.delenv("PFN_SLURM_ACCOUNT_GPU", raising=False)
    monkeypatch.setenv("PROJECT", "proj_from_cluster")
    assert claix.resolved_account("gpu") == "proj_from_cluster"
    with pytest.raises(RuntimeError, match="PFN_SLURM_ACCOUNT_CPU"):
        claix.resolved_account("cpu")
    monkeypatch.setenv("PFN_SLURM_ACCOUNT_CPU", "x")
    assert claix.resolved_account("cpu") == "x"


def test_dry_run_claix_array_cpu(pfn_home, capsys):
    out = dry_run(capsys, "--profile", "rwth_claix", "--kind", "bench", "--array", "5-9%2",
                  "--after", "123", "--", "pfn", "bench", "run", "--group", "g")
    assert "#SBATCH --account=acct_cpu" in out and "#SBATCH --partition=c23ms" in out
    assert "--gres" not in out and "PYTORCH_CUDA_ALLOC_CONF" not in out
    assert "#SBATCH --array=5-9%2" in out and "#SBATCH --dependency=afterok:123" in out
    assert "%x_%A_%a.out" in out
    assert 'SHARD="$((SLURM_ARRAY_TASK_ID - 5))/5"' in out
    assert out.rstrip().endswith('pfn bench run --group g --shard "$SHARD"')
    assert "#SBATCH --job-name=bench-run" in out


def test_dry_run_gen_loads_cpp_modules(pfn_home, capsys):
    out = dry_run(capsys, "--profile", "rwth_claix", "--kind", "gen", "--", "pfn", "data", "gen")
    assert "imkl/2024.2.0" in out and "#SBATCH --cpus-per-task=24" in out


def test_dry_run_local(pfn_home, capsys):
    out = dry_run(capsys, "--profile", "local", "--", "pfn", "config")
    assert "#SBATCH" not in out and "module" not in out
    assert "export PFN_PROFILE=local" in out


def test_local_runs_pfn_config(pfn_home, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "esbopfn", "submit", "--local", "--", "pfn", "config"],
        capture_output=True, text=True, check=True,
    )
    assert f"PFN_HOME     {pfn_home.resolve()}" in proc.stdout


def test_local_array_appends_shards(pfn_home, capsys):
    code = main(["submit", "--local", "--array", "0-1", "--", sys.executable, "-c",
                 "import sys; assert sys.argv[1:3] in (['--shard','0/2'], ['--shard','1/2'])"])
    assert code == 0


def test_parse_array_errors():
    assert parse_array("0-19").n == 20
    for bad in ("3", "5-2", "a-b"):
        with pytest.raises(ValueError):
            parse_array(bad)


def test_expand_env_nested_defaults():
    from esbopfn.cluster.profile import expand_env

    env = {"A": "a", "P": "p", "EMPTY": ""}
    assert expand_env("${A:-${P}}", env) == "a"
    assert expand_env("${X:-${P}}", env) == "p"
    assert expand_env("${EMPTY:-${X:-d}}/x", env) == "d/x"
    assert expand_env("pre-${A}-${P}", env) == "pre-a-p"
    assert expand_env("${X:-${Y}}", env) is None
    assert expand_env("${A", env) is None


def test_dry_run_account_line(pfn_home, capsys):
    out = dry_run(capsys, "--profile", "rwth_claix", "--kind", "train", "--", "pfn", "status")
    assert "#SBATCH --account=acct_gpu\n" in out


def test_dry_run_copies_profile_env(pfn_home, capsys, monkeypatch):
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/work/u/venv dir")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    out = dry_run(capsys, "--profile", "rwth_claix", "--kind", "gen", "--", "pfn", "config")
    assert "export UV_PROJECT_ENVIRONMENT='/work/u/venv dir'\n" in out
    assert "UV_CACHE_DIR" not in out
    assert out.index("export UV_PROJECT_ENVIRONMENT") < out.index("uv sync --frozen")
