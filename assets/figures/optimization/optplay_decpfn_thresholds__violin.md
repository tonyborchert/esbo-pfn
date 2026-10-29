# optplay_decpfn_thresholds__violin

Thesis asset: `writing/thesis/assets_generated/simulation/optimization/decpfn_100_binned/optplay_decpfn_thresholds__violin.pdf` (fig:app:optplay-decpfn-thresholds-violin)

Files: `optplay_decpfn_thresholds__violin.png`

## What is shown
- one violin per method and budget fraction (100 % of the simulated time budget 15 T min(d, 4)), body = mirrored histogram drawn smooth, line = median
- methods: DecPFN tau=0.9, DecPFN tau=0.99, DecPFN-H tau=0.9, DecPFN-H tau=0.99
- cells: DecPFN tau=0.9 = `decpfn` {'crash_threshold': 0.9} (thesis series `decpfn`); DecPFN tau=0.99 = `decpfn` {'crash_threshold': 0.9, 'tau': 0.99} (thesis series `decpfn@tau=0.99`); DecPFN-H tau=0.9 = `decpfn-h` {'crash_threshold': 0.9} (thesis series `decpfn-h`); DecPFN-H tau=0.99 = `decpfn-h` {'crash_threshold': 0.9, 'tau': 0.99} (thesis series `decpfn-h@tau=0.99`)

## Metric
- scaled regret (J - J_best) / (J_ESRS - J_best) per run, log10 with display floor -4 (values below are drawn at the floor), lower is better
- J_best: best final cost of the reference strategies on the instance, J_ESRS: the final incumbent of ESRS

## Key numbers
- all problems: median log10 scaled regret: DecPFN tau=0.9 100% -0.253 (n=100, 6 at floor); DecPFN tau=0.99 100% -0.276 (n=100, 4 at floor); DecPFN-H tau=0.9 100% -0.347 (n=100, 2 at floor); DecPFN-H tau=0.99 100% -0.325 (n=100, 4 at floor)
- 50 % budget not shown: the per-run incumbent traces (native `optimization` units, `pfn bench run optimization`) are not in the database, the thesis import holds final incumbents only

## Data
- thesis import (`optplay_finals`, final incumbents at the time budget) on boiler, threetank_pi, pt2_deadtime, cartpole, furuta. `J_best` = `optplay_reference` of the import (best successful final cost over every thesis cell on disk, the thesis definition)

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 165 units (thesis), hashes 07b1755f3a, 0d30333e94, 109851d786, 174e983a4d, 3d3b157215, 3e140b898d, ...
