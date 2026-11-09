# Reproduction audit map

This file maps every quantitative element of the paper to the script that produces it and the file it is
read from, so a reviewer can check each claim without reading the manuscript sources. All numbers in the
paper are read programmatically from the files below; none is typed by hand.

## One-command reproduction

```bash
pip install -r requirements.txt
bash reproduce.sh            # fetch + hash-check data, 55 runs, replication, analysis, figures, supplement (~2.5 h, 2 cores)
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
| Fig. 1 | method architecture (no data) | `scripts/make_architecture.py` | `figures/fig_architecture.*` |
| Table II | corpora, rows, frauds, periods | `scripts/prepare_data.py` | `results/dataset_summary.json`, `data/manifest.json` |
| Table III | RQ1 recall@budget, stationary test, CIs, Holm *p* | `scripts/analyze.py` | `results/analysis/main_base.csv` |
| Table IV | RQ2 held-out groups, pooled recall, Holm *p*; DRT-AMT and DRT-IDF rows | `scripts/analyze.py` | `results/analysis/emerging_pooled.csv` |
| Fig. 2 | protection–cost frontier | `scripts/make_figures.py` | `frontier_points.csv` |
| RQ3 text | equal-cost comparison (Δ and CI at 5% / 10% cost) | `scripts/analyze.py` | `frontier_summary.csv` |
| Table VI | ablations and exploratory variants: channel variants, adaptive ρ, DRT-AMT, DRT-IDF (DRT-LD, SUP-NOID in S5) | `scripts/analyze.py` | `ablation.csv`, `emerging_pooled.csv` |
| Mean ρ values | DRT-ADAPT and variants | `scripts/analyze.py` | `summary.json` (`mean_rho_adaptive`, `mean_rho_feedback_variants`) |
| Assumption (A2) | precision by SUP rank band | `scripts/analyze.py` | `assumption_a2.csv` |
| Cost prediction | calibration vs test premium | `scripts/analyze.py` | `premium.csv` |
| Discussion (per group) | detector advantage vs DRT gain | `scripts/analyze.py` | `group_signal.csv`, `emerging_groups.csv` |
| Signal quality | AP / AUC, novel AP | `scripts/analyze.py` | `signals.csv` |
| Scenario provenance | candidate groups from pre-test frauds only | `scripts/check_scenario_provenance.py` | `scenario_provenance.json`, `SUPPLEMENT.md` S10b |
| Inference under temporal dependence | autocorrelation, block sign-flip *p* and block-bootstrap CIs (blocks of 1/2/5/10 windows) | `scripts/check_dependence.py` | `dependence.csv`, `SUPPLEMENT.md` S11 |
| Capacity and time to first detection | held-out budgets vs stationary; first retrieved case per group | `scripts/operational_metrics.py` | `capacity_heldout.csv`, `time_to_detection.csv`, S12–S13 |
| Amount as a signal | AP of raw amount per held-out group | `scripts/amount_signal.py` | `amount_signal.csv`, S14 |
| Family-wise correction | every held-out comparison with SUP ever reported (22 policies × 3 corpora × 4 budgets), one Holm family | `scripts/multiplicity_all.py` | `multiplicity_all.csv`, `multiplicity_all.json`, S16 |
| Table V, Section VI-E, abstract (confirmation) | pre-registered replications of DRT-IDF: v1.6 H1–H5 (new BankSim segments; first TabFormer window, untestable) | `scripts/run_all.py --datasets banksim_u tabformer_p5`, `scripts/confirm_v1_6.py` | `results/replication_v1_6/confirmatory.json`, `descriptive_*.csv`, `time_to_detection.csv`, S17 |
| Table V, Section VI-E, abstract (confirmation) | v1.7 H6–H8 (unused TabFormer window 2011-01..2014-04) | `scripts/run_all.py --datasets tabformer_e14`, `scripts/confirm_v1_7.py` | `results/replication_v1_7/confirmatory.json`, `descriptive_*.csv`, `time_to_detection.csv`, S18 |
| Registration evidence | what was inspected before v1.7 | — | `results/feasibility/tabformer_monthly_counts_2004_2014.csv` |
| Supplement S0–S18 | full hashes, per-group and per-budget tables | `scripts/make_supplement.py` | `results/SUPPLEMENT.md` |

## What an independent check should cover beyond matching outputs

Matching outputs show computational reproducibility only. A reviewer should also inspect: the chronological
splits and leakage-safe features (`src/dstriage/data.py`, `tests/test_core.py`, `tests/test_experiment.py`),
the held-out scenario rule and its provenance (`scripts/run_all.py::holdout_groups`, supplement S10/S10b),
the policies (`src/dstriage/policies.py`), and the statistics (`src/dstriage/stats.py`, `scripts/analyze.py`,
supplement S11).

## Pre-registered confirmation (amendments v1.6 and v1.7)

| Registration | Commit, pushed to `main` | Runs (fresh clone of that commit) | Analysis |
|---|---|---|---|
| v1.6: TabFormer 2011-01..2014-10 (`tabformer_p5`) and eight new BankSim held-out categories (`banksim_u`) | `996081e`, 2026-11-09 12:39 UTC | `results/replication_v1_6/runs`, `run.log` | `scripts/confirm_v1_6.py` → `confirmatory.json` |
| v1.7: TabFormer 2011-01..2014-04 (`tabformer_e14`), correcting v1.6's infeasible window | `c6c9488`, 2026-11-09 13:15 UTC | `results/replication_v1_7/runs`, `run.log` | `scripts/confirm_v1_7.py` → `confirmatory.json` |

Each amendment file lists everything inspected before it was written. The `tabformer_p5` test period held no
labelled fraud, so v1.6's H1, H2 and H5 are untestable; its remaining four seeds were stopped after the first
run (the split does not depend on the seed), and the registered `confirm_v1_6.py` stopped at its
seed-completeness check (`results/replication_v1_6/registered_script_output.txt`). Two guards for a replication
without test fraud were then added to `confirm_v1_6.py`; they do not touch the H3/H4 computation (see its git
history). To reproduce:

```bash
git clone https://github.com/sagarkorde/UPI-Transaction-Profiler && cd UPI-Transaction-Profiler
git checkout 996081e && python scripts/run_all.py --datasets banksim_u tabformer_p5 --out results/replication_v1_6/runs
git checkout c6c9488 && python scripts/run_all.py --datasets tabformer_e14 --out results/replication_v1_7/runs
git checkout main && python scripts/confirm_v1_6.py && python scripts/confirm_v1_7.py
```

A check of the registrations should confirm that, at the results commit, `git diff c6c9488 --
configs/amendment_v1_6.yaml configs/amendment_v1_7.yaml scripts/confirm_v1_7.py scripts/run_all.py src/` is
empty and `git diff 996081e -- configs/amendment_v1_6.yaml` is empty.

## Re-run records

| Record | Result |
|---|---|
| v1.1 re-run vs v1.0 (`results/reproduction_check.txt`) | 45 runs identical, 0 different; 10 ULB runs skipped (scenarios redefined by v1.1) |
| v1.2 re-run vs v1.1 (`results/reproduction_check_v1_2.txt`) | 55 runs identical, 0 different; 448,400 policy rows (new DRT-ADAPT-AUDIT rows excluded) |
| v1.3 re-run vs v1.2 (`results/reproduction_check_v1_3.txt`) | 55 runs identical, 0 different; 465,440 policy rows (new DRT-LD rows excluded) |
| v1.4 re-run vs v1.3 (`results/reproduction_check_v1_4.txt`) | 55 runs identical, 0 different; 482,480 policy rows (new v1.4 rows excluded) |
| v1.5 re-run vs v1.4 (`results/reproduction_check_v1_5.txt`) | 55 runs identical, 0 different; 533,600 policy rows (new v1.5 rows excluded) |

Archived outputs: `results/runs_v1_0/` (protocol v1.0), `results/runs_v1_1/` (v1.1), `results/runs_v1_2/` (v1.2), `results/runs_v1_3/` (v1.3), `results/runs_v1_4/` (v1.4).
`results/runs/` holds the v1.5 outputs used for every statistic.
