# Training data

All training and validation data of the PFN models come from the priors of the thesis
(Section "Priors" of the method chapter), sampled by three C++ binaries and stored as pools under
`$PFN_HOME/datasets/`. This page describes the binaries, how to build them, the dataset ids, the
on-disk formats and the seeding contract.

## Binaries

| Binary | Problems | Needs LAPACK |
|---|---|---|
| `furuta_gen` | `furuta` | yes (the LQR design solves the discrete Riccati equation like scipy) |
| `cartpole_gen` | `cartpole` | no |
| `plants_gen` | `boiler`, `threetank_pi`, `pt2_deadtime` (select with `--problem`) | no |

Every binary has the same subcommands:

| Subcommand | Output |
|---|---|
| `traj` | single-trajectory prior, a row pool |
| `sets` | trajectory-set prior for XDecPFN, a set pool with label trials and `instances.json` |
| `contexts` | FT-PFN examples, one context shard per chunk |
| `screen` | screened instances only, `instances.json` |
| `screen-one` | the degeneracy screen of one given instance (`--nominal` or `--instance-json`), as JSON |
| `pilot` | the unsuccessful-trajectory rate of the unscreened prior, as JSON |
| `rollout` | one rollout (`--theta`, `--instance-json` or `--nominal`, `--seed`, `--deterministic`), as JSON |
| `speed` | the generation-speed benchmark on one core, as JSON |
| `info` | names, box, grids and the nominal instance of the problem, as JSON |
| `version`, `derive-seed`, `sobol` | version string and two helpers for tests |

The chunked subcommands (`traj`, `sets`, `contexts`, `screen`) take
`--out DIR --count N --seed S --workers W --chunk K [--chunks LIST]`. `--count` counts units:
rows for `traj`, screened instances for `sets` and `screen`, examples for `contexts`. A chunk is
`K` consecutive units and the unit of parallelism, resumption and sharding. `--chunks 0-9,20`
restricts a run to some chunks. Finished chunks are recorded in `DIR/chunks.log` (one line
`<chunk> <units>`) or, for contexts, by their shard directory, and a rerun skips them. A
directory remembers its seed, chunk size and prior options in `identity.json`, and a run with
different ones is refused. `--no-dead-time` (Furuta only) selects the prior without actuation
dead time of the hardware ablation.

Normally the binaries are driven by `pfn data`:

```
pfn data build [--blas accelerate|openblas|mkl] [--native]
pfn data status [--problem P]
pfn data gen <dataset_id> | --for train <model> --problem P | --for all [--problem P]
             [--workers N] [--seed S] [--shard i/n] [--no-resume]
pfn data verify <dataset_id> [--sample N] [--no-parity] [--n-ks 500]
pfn data speed --problem P [--count 2000] [--sets 10] [--budget 120]
```

`pfn data gen` writes `meta.json`, computes the unsuccessful rate for `traj` pools once (stored in
`meta.json` and passed to every later run) and calls the binary for the chunks that are not done.
With `--shard i/n` it takes every n-th chunk starting at chunk i, so a Slurm array of n tasks
(`pfn submit --array 0-<n-1> -- pfn data gen ...`) writes the dataset in parallel into one
directory.

## Building

All binaries are built on the machine that runs them. `pfn data build` configures, builds and
installs into `$PFN_HOME/bin` (build directory `cpp/build-<blas>`), which is where `pfn` looks for
them (then in `cpp/build*`). The flags are C++17, `-O3` and `-ffp-contract=off` (so GCC and clang
produce the same trajectories). `--native` adds `-march=native`.

**macOS (Apple clang, Accelerate).** Nothing to install beyond cmake:

```
pfn data build                       # same as --blas accelerate
```

**Linux (GCC, OpenBLAS).** Install OpenBLAS with its LAPACK part (Debian/Ubuntu
`libopenblas-dev`), then

```
pfn data build --blas openblas
```

If no `libopenblas` is found, cmake falls back to the system LAPACK (`find_package(LAPACK)`).

**RWTH cluster (GCC, MKL).** Load the toolchain and MKL, then build on a login node:

```
module load GCC/13.3.0 imkl/2024.2.0
pfn data build --blas mkl
```

MKL is linked sequentially (`mkl_intel_lp64`, `mkl_sequential`, `mkl_core`) with an rpath to
`$MKLROOT/lib`, because the parallelism comes from one thread per worker. `pfn submit --kind gen`
loads the same modules for the generation jobs.

The equivalent manual build is

```
cmake -S cpp -B cpp/build -DBLAS=<accelerate|openblas|mkl> [-DNATIVE=ON]
cmake --build cpp/build -j
cmake --install cpp/build --prefix "$PFN_HOME"
```

## Dataset ids

Ids have the form `<kind>/<problem>/<name>` and are defined in `src/esbopfn/configs/datasets.yaml`.
A dataset lives in `$PFN_HOME/datasets/<problem>/<kind>/<name>/`.

| Id | Units | Seed | Chunk | Content |
|---|---|---|---|---|
| `traj/<p>/1M` | 1,000,000 rows | 1001 | 10,000 | single-trajectory prior, training pool of the 1M LC-PFN and DecPFN |
| `traj/<p>/100k` | 100,000 rows | | | the first 100,000 rows of `traj/<p>/1M` |
| `traj/<p>/val` | 3,000 rows | 1002 | 1,000 | DecPFN validation set, own seed stream |
| `traj/<p>/val_lcpfn` | 1,024 rows | | | the first 1,024 rows of `traj/<p>/val` (LC-PFN validation) |
| `traj/furuta/1M_nodeadtime` | 1,000,000 rows | 1003 | 10,000 | Furuta prior without dead time (hardware ablation) |
| `traj/furuta/val_nodeadtime` | 3,000 rows | 1004 | 1,000 | its validation set |
| `sets/<p>/xdecpfn` | 132,400 instances | 2001 | 100 | XDecPFN pools, N_max trajectories per screened instance |
| `sets/<p>/xdecpfn_heldout` | 256 instances | 2002 | 32 | held-out XDecPFN pools |
| `contexts/<p>/ftpfn` | 3,500,000 examples | 3001 | 2,000 | FT-PFN examples, a fresh trajectory set each |
| `contexts/<p>/ftpfn_heldout` | 2,000 examples | 3002 | 500 | held-out FT-PFN examples |
| `fields/<p>/oracle65` | | | | ESBO-Oracle runs for the prediction benchmark, written by the benchmark phase |
| `fields/<p>/oracle181` | | | | ESBO-Oracle runs for the context ladder, written by the benchmark phase |

A prefix dataset (`100k`, `val_lcpfn`) owns no directory. Generating it generates the first chunks
of its parent, and `open_dataset` returns the parent pool limited to the prefix. The chunk sizes
divide 100,000 and 1,000,000, so `traj/<p>/100k` is exactly chunks 0 to 9 of `traj/<p>/1M`.

`pfn data gen --for train <model> --problem P` generates what a model needs: `lcpfn` and
`lcpfn-last` take `traj/P/1M` and `traj/P/val_lcpfn`, `decpfn` and `decpfn-nostates` take
`traj/P/1M` and `traj/P/val`, `decpfn-nodeadtime` the two no-dead-time pools, `xdecpfn` the two
set pools and `ftpfn` the two context datasets.

## Seeds

Chunk `k` of a dataset with seed `S` is generated from the 63-bit seed
`esbopfn.util.seeds.derive_seed(S, k)`. The binaries compute it with a port of numpy's
SeedSequence (`cpp/include/esbo/rng.hpp`), so `derive_seed(S, k)` in Python and
`<binary> derive-seed --seed S --label k` agree bit for bit (tested in `tests/cpp`). Inside a
chunk, `traj` draws its rows from one `std::mt19937_64` stream seeded with that value (passed
through splitmix64). `sets`, `screen` and `contexts` give unit `j` of chunk `k` its own stream
seeded with `splitmix64(chunk_seed ^ splitmix64(j + c))` (`sub_seed` in `rng.hpp`), stored as the
instance `seed`. A dataset is therefore a pure function of `(S, chunk size, prior options)`. The
number of workers, processes or shards never changes it. Changing the chunk size does.

The C++ and Python generators share the distributions, not the random streams.

## Pool format

A pool directory holds

| File | Content |
|---|---|
| `rows.f32` | little-endian float32, `N` rows of `row_floats` values, read as an `(N, R)` memmap |
| `spec.json` | column layout and conventions (below) |
| `meta.json` | dataset id, seed, count, chunk size, options, generator name and version, problem spec hash, unsuccessful rate |
| `identity.json` | what the binary checks before resuming (seed, chunk, options, rate) |
| `chunks.log` | finished chunks |
| `pool.finished` | present once every chunk is done |

`spec.json` lists the columns in row order, each with `name`, `offset`, `length`, `shape` and,
where it applies, the entry `names`. For a `traj` pool of Cartpole:

| Column | Shape | Content |
|---|---|---|
| `theta` | `(d,)` | controller parameters |
| `phi` | `(n_p,)` | true physical parameters |
| `phi_hat` | `(n_p,)` | the estimate (Furuta only) |
| `noise_std` | `(n_noise,)` | sensor and actuation noise levels of the instance |
| `extras` | `(n_extra,)` | Furuta dead time [s] and kick amplitude [V] |
| `crash_step` | `(1,)` | 1-based step of the crash, 0 if none |
| `success` | `(1,)` | 1 if not crashed and the success criterion holds |
| `J` | `(S,)` | cumulative cost at the stored steps, `+inf` from the crash step on |
| `x` | `(S, C)` | level channels as the predictors see them (measured, noisy) |
| `xdot` | `(S, C_rate)` | measured rates, only for problems that have them |

`steps` is `"all"` (every step 1 to T, `resolution: full`) or the list of stored steps
(`resolution: grid`). Channel rows after the crash step repeat the last simulated row. The level
channels per problem are those of `configs/problems.yaml`: Boiler `[temperature]`, Three-Tank
`[V2, V3, V4]`, PT2 dead-time `[y]`, Cartpole `[x_cart, phi_pole]` with rates
`[xdot_cart, phidot_pole]`, Furuta `[phi_arm, phi_pend]` with rates `[phidot_arm, phidot_pend]`.
Instance entries carry the names of the Python prior (`esbopfn.problems.prior`), so
`Pool.instance(i)` is a `core.Instance` the Python simulators accept.

**Set pools** (`kind: set`) hold one screened instance per unit and `rows_per_unit = N_max` rows
per instance (45, 90, 135, 180 and 180 trajectories for Boiler, Three-Tank, PT2 dead-time,
Cartpole and Furuta), each tagged with its `instance` index in the first column and run to the
horizon or to its crash. Instead of the instance columns they have `instances.json`, a list with
one record per instance: `index`, `instance` (`phi`, `phi_hat`, `noise_std`, `extras`, `seed`),
`screen` (passes, candidates tried, rollouts, crash and unsuccessful rates of the first repeats)
and `set` (the drawn rates and exact counts of crashed, unsuccessful and successful trajectories).
The rows of one instance are in random order. `labels/` is a pool of `kind: labels` with 128 unrun
controllers per instance (`theta`, `crash_step`, `success`, `J` at the horizon), the targets of
the unrun queries.

The XDecPFN set pools are stored at `resolution: grid`: the decoder token steps
`l * N_stride` plus the horizon. Their `xdot` block holds the rates summed over each interval
`(t_{l-1}, t_l]` (`xdot_mode: interval_sum`, level differences for problems without measured
rates), so each row already contains the DecPFN covariates `[J_l - J_{l-1}, x_l, xdot_l]`. At
full resolution the 132,400 pools would take up to about 480 GB per problem (Cartpole), on the
grid about 33 GB. `--resolution full` stores full trajectories anyway.

**Python access.** `Pool.open(path)`, `len(pool)`, `pool.read(i) -> Trajectory` (full
resolution), `pool.read_grid(i) -> GridTrajectory` (any resolution), `pool.slice(a, b)`,
`pool.column(name, rows)`, `pool.instance(i)`, `pool.instances()`, `pool.instance_rows(i)`,
`pool.labels()`, and `Pool.create` / `pool.append_chunk(k, rows)` for writers in Python.
`esbopfn.generators.open_dataset(id)` resolves an id, including prefixes.

## Context shards (FT-PFN)

A `contexts` dataset holds `shards/chunk_<k>/`, one directory per chunk, moved into place when
complete. Every array is raw little-endian binary with the shape in `shard.json`
(`examples` E, `trials` N, `d`, `n_tokens` = 9, `n_unrun` = 16, `grid`, `T`):

| File | dtype | Shape | Content |
|---|---|---|---|
| `example_ptr.i64` | int64 | (E+1,) | trials of example e are `ptr[e]` to `ptr[e+1]` |
| `example_n.i32` | int32 | (E,) | number of trials n ~ U{1..N_max} |
| `example_meta.f32` | float32 | (E, 3) | cut probability, drawn crash rate, drawn unsuccessful rate |
| `example_seed.i64` | int64 | (E,) | seed of the example's instance stream |
| `trial_theta.f32` | float32 | (N, d) | controller of each trial |
| `trial_len.i32` | int32 | (N,) | observed length T_i |
| `trial_status.i8` | int8 | (N,) | 0 complete, 1 truncated (queried at the horizon), 2 crashed |
| `trial_target.f32` | float32 | (N,) | total cost of the full trajectory, `+inf` if it crashes |
| `trial_tok_t.i16` | int16 | (N, 9) | token steps, 0 is padding |
| `trial_tok_J.f32` | float32 | (N, 9) | cumulative cost at the token steps, `+inf` marks the crash |
| `unrun_theta.f32` | float32 | (E, 16, d) | unrun query controllers |
| `unrun_target.f32` | float32 | (E, 16) | their total costs, `+inf` for a crash |

The tokens of trial i are the grid steps `t_l = floor(l T / 8) <= T_i` followed by `T_i` itself
if `T_i < T` (the context of the FT-PFN section). `ContextStore(path)` and `ContextShard(path)`
read them, `write_context_shard` writes the same layout from Python.

## The priors as implemented

- **Instances.** Scale, Bounded and Tied entries, sensor noise `base * S * eta` with one `S` per
  instance and one `eta` per entry of the noise table (Furuta: angles, angular velocities, motor
  voltage), the estimate `max(0, N(p, (p/10)^2))` for Furuta, and the Furuta dead time
  U[8.33, 25] ms and kick amplitude U[0, 0.6] V per instance with a random kick sign per trial.
- **Degeneracy screen.** 100 controllers from a Sobol' design over Theta (Joe-Kuo direction
  numbers, a random digital shift per candidate instance), each run up to 10 times, the instance
  is kept if at least 25 controllers succeed in all their runs. A controller stops at its first
  unsuccessful run and the screen stops once 25 passes are out of reach. Problems that can
  neither crash nor fail skip it.
- **Single-trajectory prior.** No screen. An unsuccessful trajectory is rejected with probability
  `max(0, 1 - (1 - rho)/rho)`, where `rho` is the unsuccessful rate of 50,000 unscreened draws
  (`pilot`, fixed seed 20261028, so every dataset of a problem uses the same rate).
- **Trajectory-set prior.** The crash rate and the unsuccessful-given-not-crashed rate of a
  screened instance come from the first runs of its 100 screen controllers. Each example draws
  `rho'_c ~ U(0, rho_c)`, `rho'_s|c ~ U(0, rho_s|c)`, the binomial counts and rejection-samples
  controllers until the counts are met (after `1000 + 1000 n` draws the remaining slots take any
  trajectory, recorded as `forced` in `instances.json`). XDecPFN pools use `n = N_max` once per
  instance. FT-PFN examples draw `n ~ U{1..N_max}`, one cut probability `~U(0, 1)` and a cutoff
  `~U{1..T_i}` for each cut trial.

## Verification

`pfn data verify <id>` checks on a sample of rows: crash steps in range, J non-decreasing before
the crash and `+inf` from it on, finite channels, theta inside the box, success only without a
crash, at most half unsuccessful trajectories where the rejection applies, and for set pools the
per-instance counts against `instances.json`. Context datasets get the analogous token and status
checks. Unless `--no-parity` is given it then compares the binary with `esbopfn.problems`:

- deterministic: the nominal instance without noise, 8 controllers, the Python simulator driven by
  a central random generator (zero normals, interval midpoints) and the binary with
  `rollout --deterministic`. Crash steps must agree and J_T within a relative 1e-3.
- distribution: `--n-ks` unfiltered prior samples on each side, crash and success rates by a
  two-proportion z-test (`|z| <= 4`) and ln J_T of the non-crashed trajectories by a
  Kolmogorov-Smirnov test (`p >= 1e-3`).

`tests/cpp` runs the same checks on small outputs (`uv run pytest --run-cpp`).

## Generation speed

`pfn data speed --problem P` measures on one core what the prior-cost table of the appendix
reports: the time for 10,000 single trajectories, the trajectory acceptance of the rejection, the
system acceptance of the screen, and the time for 10,000 FT-PFN examples and 10,000 XDecPFN set
pools (screened instance, N_max trajectories and the 128 label trials). It times `--count` single
trajectories and `--sets` sets of each kind, each phase stopping after `--budget` seconds, and
prints its row next to the row printed in the thesis (measured on one core of an Intel Xeon
Platinum 8468). `esbopfn.generators.speed.measure` returns the same numbers for use from Python
(the `generation_speed` benchmark kind).
