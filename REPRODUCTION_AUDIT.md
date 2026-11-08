# Reproduction audit map

This file maps every quantitative element of the paper to the script that produces it and the file it is
read from, so a reviewer can check each claim without reading the manuscript sources. All numbers in the
paper are read programmatically from the files below; none is typed by hand.

## One-command reproduction

```bash
pip install -r requirements.txt
bash reproduce.sh            # fetch + hash-check data, 55 runs, analysis, figures, supplement (~1 h, 2 cores)
```

To check a fresh run against the released results:

```bash
python scripts/run_all.py --out results/runs_check
python scripts/compare_runs.py results/runs results/runs_check     # expect: runs identical: 55  different: 0
RUN_DIR=results/runs_check python scripts/analyze.py               # regenerates results/analysis/
```

## Paper element → source

| Paper element | Content | Produced by | File |
|---|---|---|---|
| Table III | corpora, rows, frauds, periods | `scripts/prepare_data.py` | `results/dataset_summary.json`, `data/manifest.json` |
| Table IV | RQ1 recall@budget, stationary test, CIs, Holm *p* | `scripts/analyze.py` | `results/analysis/main_base.csv` |
| Fig. 2 | recall vs budget multiplier *c* | `scripts/make_figures.py` | `main_base.csv` |
| Table V | RQ2 held-out mechanisms, pooled novel recall, Holm *p* | `scripts/analyze.py` | `results/analysis/emerging_pooled.csv` |
| Fig. 3 | protection–cost frontier | `scripts/make_figures.py` | `frontier_points.csv` |
| RQ3 text | equal-cost comparison (Δ and CI at 5% / 10% cost) | `scripts/analyze.py` | `frontier_summary.csv` |
| Table VI | ablations, feedback (v1.1) and randomised-audit (v1.2) variants | `scripts/analyze.py` | `ablation.csv`, `emerging_pooled.csv` |
| Mean ρ values | DRT-ADAPT and variants | `scripts/analyze.py` | `summary.json` (`mean_rho_adaptive`, `mean_rho_feedback_variants`) |
| Assumption (A2) | precision by SUP rank band | `scripts/analyze.py` | `assumption_a2.csv` |
| Cost prediction | calibration vs test premium | `scripts/analyze.py` | `premium.csv` |
| Discussion (per group) | detector advantage vs DRT gain | `scripts/analyze.py` | `group_signal.csv`, `emerging_groups.csv` |
| Signal quality | AP / AUC, novel AP | `scripts/analyze.py` | `signals.csv` |
| Scenario provenance | candidate groups from pre-test frauds only | `scripts/check_scenario_provenance.py` | `scenario_provenance.json`, `SUPPLEMENT.md` S10b |
| Supplement S0–S10b | full hashes, per-group and per-budget tables | `scripts/make_supplement.py` | `results/SUPPLEMENT.md` |

## Re-run records

| Record | Result |
|---|---|
| v1.1 re-run vs v1.0 (`results/reproduction_check.txt`) | 45 runs identical, 0 different; 10 ULB runs skipped (scenarios redefined by v1.1) |
| v1.2 re-run vs v1.1 (`results/reproduction_check_v1_2.txt`) | 55 runs identical, 0 different; 448,400 policy rows (new DRT-ADAPT-AUDIT rows excluded) |

Archived outputs: `results/runs_v1_0/` (protocol v1.0), `results/runs_v1_1/` (amendment v1.1).
`results/runs/` holds the v1.2 outputs used for every statistic.
