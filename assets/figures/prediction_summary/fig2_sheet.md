# fig2_sheet

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/at_the_end/fig2_sheet.pdf` (fig:sim:fig2-sheet)

Files: `fig2_sheet.png`

## What is shown
- panels: PT2 dead-time, cartpole, furuta
- XDecPFN and FT-PFN at n = 0, n = 64 (n = 0 faint), DecPFN, DecPFN-no-St (dashed) and LC-PFN pooled over n
- models: `decpfn/{p}/binned/1M`, `decpfn-nostates/{p}/binned/1M`, `lcpfn/{p}/binned/1M`, `xdecpfn/{p}/default/default`, `ftpfn/{p}/default/default`

## Metric
- x: share of non-crashed queries the rule 'stop at the first step with p_crash >= tau' stops, tau swept over every emitted probability, window [0, 0.1]
- y: mean steps saved on crashed queries (crash step minus stop step, 0 when never stopped), each panel its own range (higher is better)

## Key numbers
- PT2 dead-time, DecPFN: 62.9 steps saved at false-stop share <= 0.1
- PT2 dead-time, LC-PFN: 55.0 steps saved at false-stop share <= 0.1
- PT2 dead-time, XDecPFN N=0: 82.8 steps saved at false-stop share <= 0.1
- PT2 dead-time, XDecPFN N=64: 70.5 steps saved at false-stop share <= 0.1
- PT2 dead-time, FT-PFN N=0: 60.1 steps saved at false-stop share <= 0.1
- PT2 dead-time, FT-PFN N=64: 41.1 steps saved at false-stop share <= 0.1
- cartpole, DecPFN: 146.9 steps saved at false-stop share <= 0.1
- cartpole, LC-PFN: 49.5 steps saved at false-stop share <= 0.1
- cartpole, XDecPFN N=0: 274.7 steps saved at false-stop share <= 0.1
- cartpole, XDecPFN N=64: 396.3 steps saved at false-stop share <= 0.1
- cartpole, FT-PFN N=0: 226.6 steps saved at false-stop share <= 0.1
- cartpole, FT-PFN N=64: 314.8 steps saved at false-stop share <= 0.1
- furuta, DecPFN: 65.4 steps saved at false-stop share <= 0.1
- furuta, LC-PFN: 19.8 steps saved at false-stop share <= 0.1
- furuta, XDecPFN N=0: 76.8 steps saved at false-stop share <= 0.1
- furuta, XDecPFN N=64: 204.2 steps saved at false-stop share <= 0.1
- furuta, FT-PFN N=0: 19.3 steps saved at false-stop share <= 0.1
- furuta, FT-PFN N=64: 120.7 steps saved at false-stop share <= 0.1

## Data
- source: thesis
- not in the thesis import, not drawn: DecPFN-no-St (`decpfn-nostates/{p}/binned/1M`) (the thesis curve came from cluster prediction files)

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 12 units (thesis), hashes 1ad9037ecf, 3c9d4dd44e, 8008bb0f5a, 87538ce42b, 8e8797c3f6, 96a9260a20, ...
