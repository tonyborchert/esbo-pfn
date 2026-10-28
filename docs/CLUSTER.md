# Running on a cluster

Every `pfn` command runs the same way on a laptop, a workstation or a Slurm cluster. Everything
that is specific to a machine lives in one profile file, and `pfn submit` turns any command into a
job for that machine.

## Output root and profile

Two environment variables exist.

- `PFN_HOME` is the output root (datasets, models, benchmark database, figures, job scripts, logs).
  Default `./out` in the current directory.
- `PFN_PROFILE` names the cluster profile. Default `local`.

`pfn config` prints the resolved paths and the active profile. `pfn doctor` checks Python, the
torch device, the compilers, BLAS, the TabPFN checkpoint and the C++ binaries. `pfn status` lists
datasets, models, benchmark units per kind and status, and your Slurm queue if `squeue` exists.

## Profiles

Profiles are YAML files in `src/esbopfn/configs/cluster/`. Two ship with the repository.

- `local.yaml`: no scheduler, commands run in place.
- `rwth_claix.yaml`: Slurm on RWTH CLAIX-2023.

Fields:

| field | meaning |
|---|---|
| `name` | profile name |
| `root` | `PFN_HOME` for jobs when `PFN_HOME` is not set at submit time, `${VAR}` is expanded |
| `scheduler` | `slurm` or `none` |
| `accounts`, `partitions` | per resource class `cpu` and `gpu`. Accounts are placeholders such as `${PFN_SLURM_ACCOUNT_CPU}` or `${PFN_SLURM_ACCOUNT_GPU:-${PROJECT}}`, expanded from the environment at submit time, so no account name is ever committed. On RWTH CLAIX put `export PFN_SLURM_ACCOUNT_CPU=<your cpu project>` and `export PFN_SLURM_ACCOUNT_GPU=<your gpu project>` into your shell profile |
| `modules` | modules loaded by every job (after `module purge`) |
| `modules_cpp` | extra modules loaded by `--kind gen` jobs (compiler and BLAS runtime) |
| `pre`, `pre_gpu` | shell lines run before the command, `pre_gpu` only for jobs with GPUs |
| `launcher` | prefix for `pfn` commands inside the job, e.g. `uv run --no-sync` |
| `defaults` | resources per job kind `gen`, `train`, `bench`, `bench_gpu` (`cpus`, `mem`, `time`, `gpus`) |

A job uses the `gpu` account and partition whenever it requests at least one GPU.

## Submitting

```
pfn submit [--profile NAME] [--kind gen|train|bench|bench_gpu]
           [--time HH:MM:SS] [--cpus N] [--mem 16G] [--gpus N]
           [--array a-b[%k]] [--after JOBID] [--name NAME]
           [--dry-run] [--local] -- <command ...>
```

- The kind selects the default resources of the profile, explicit flags override them.
- The job script is rendered from `src/esbopfn/cluster/templates/job.sh.tmpl` into
  `$PFN_HOME/jobs/<timestamp>_<name>.sh` and submitted with `sbatch`. The job id is printed.
  Job output goes to `$PFN_HOME/logs/`.
- The script exports `PFN_HOME` and `PFN_PROFILE`, changes into the directory `pfn submit` was
  called from and runs the command. A leading `pfn` is prefixed with the profile's `launcher`.
- `--array a-b` submits an array of `n = b - a + 1` tasks and appends `--shard i/n` to the command,
  with `i = SLURM_ARRAY_TASK_ID - a`. Every `pfn bench run` shard then takes every `n`-th unit.
- `--after JOBID` adds an `afterok` dependency.
- `--dry-run` prints the script and submits nothing.
- `--local`, or a profile with `scheduler: none`, runs the command in place (each array shard in
  turn) with the same `PFN_HOME`.

All commands are resumable, so a job that hits its time limit is simply submitted again.

Example, all benchmark units of a plot group in 20 shards:

```
pfn submit --profile rwth_claix --kind bench --array 0-19 -- pfn bench run --group <group>
```

## Adding a cluster

1. Copy `src/esbopfn/configs/cluster/rwth_claix.yaml` to `<name>.yaml`.
2. Edit the accounts, partitions, modules, `pre` lines and `root`.
3. Check the rendered script with `pfn submit --profile <name> --dry-run -- pfn config`.
4. Use it with `--profile <name>` or by exporting `PFN_PROFILE=<name>`.

A cluster without Slurm takes `scheduler: none`, which runs every submitted command in place.
