# optplay_qx_variants__violin

Thesis asset: `writing/thesis/assets_generated/simulation/optimization/ftpfn_xdecpfn/optplay_qx_variants__violin.pdf` (fig:app:optplay-qx-variants-violin)

Files: `optplay_qx_variants__violin.png`

## What is shown
- one violin per method and budget fraction (100 % of the simulated time budget 15 T min(d, 4)), body = mirrored histogram drawn smooth, line = median
- methods: XDecPFN-EI tau=0.9, XDecPFN-EI tau=0.99, XDecPFN-PI tau=0.9, XDecPFN-PI tau=0.99, FT-PFN-EI tau=0.9, FT-PFN-EI tau=0.99, FT-PFN-PI tau=0.9, FT-PFN-PI tau=0.99
- cells: XDecPFN-EI tau=0.9 = `xdecpfn-ei` {'tau': 0.9} (thesis series `xdecpfn-ei@tau=0.9`); XDecPFN-EI tau=0.99 = `xdecpfn-ei`  (thesis series `xdecpfn-ei`); XDecPFN-PI tau=0.9 = `xdecpfn-pi` {'tau': 0.9} (thesis series `xdecpfn-pi@tau=0.9`); XDecPFN-PI tau=0.99 = `xdecpfn-pi`  (thesis series `xdecpfn-pi`); FT-PFN-EI tau=0.9 = `ftpfn-ei` {'tau': 0.9} (thesis series `ftpfn-ei@tau=0.9`); FT-PFN-EI tau=0.99 = `ftpfn-ei`  (thesis series `ftpfn-ei`); FT-PFN-PI tau=0.9 = `ftpfn-pi` {'tau': 0.9} (thesis series `ftpfn-pi@tau=0.9`); FT-PFN-PI tau=0.99 = `ftpfn-pi`  (thesis series `ftpfn-pi`)

## Metric
- scaled regret (J - J_best) / (J_ESRS - J_best) per run, log10 with display floor -4 (values below are drawn at the floor), lower is better
- J_best: best final cost of the reference strategies on the instance, J_ESRS: the final incumbent of ESRS

## Key numbers
- all problems: median log10 scaled regret: XDecPFN-EI tau=0.9 100% -0.656 (n=100, 9 at floor); XDecPFN-EI tau=0.99 100% -0.726 (n=100, 8 at floor); XDecPFN-PI tau=0.9 100% -0.736 (n=100, 12 at floor); XDecPFN-PI tau=0.99 100% -0.824 (n=100, 14 at floor); FT-PFN-EI tau=0.9 100% -0.585 (n=100, 7 at floor); FT-PFN-EI tau=0.99 100% -0.758 (n=100, 12 at floor); FT-PFN-PI tau=0.9 100% -0.526 (n=100, 5 at floor); FT-PFN-PI tau=0.99 100% -0.585 (n=100, 7 at floor)
- 50 % budget not shown: the per-run incumbent traces (native `optimization` units, `pfn bench run optimization`) are not in the database, the thesis import holds final incumbents only

## Data
- thesis import (`optplay_finals`, final incumbents at the time budget) on boiler, threetank_pi, pt2_deadtime, cartpole, furuta. `J_best` = `optplay_reference` of the import (best successful final cost over every thesis cell on disk, the thesis definition)

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 165 units (thesis), hashes 07b1755f3a, 0d30333e94, 109851d786, 174e983a4d, 3d3b157215, 3e140b898d, ...
