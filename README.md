# Disagreement-Reserved Triage (DRT): reproducibility package

Code, frozen protocol and results for the paper
**"Reserving Analyst Capacity for Disagreement: Budget-Constrained Dual-Signal Triage for Emerging Digital-Payment Fraud."**

Every number, table and figure in the paper is produced by the scripts in this repository from three
openly licensed corpora that are downloaded by script and verified by SHA-256.

## One command

```bash
bash reproduce.sh          # ~75 min on 2 CPU cores, ~7 GB RAM, ~12 GB disk
```

| Step | Script | Output |
|---|---|---|
| Data (download + SHA-256 + row/fraud counts) | `scripts/fetch_data.py` | `data/raw/` |
| Tests (leakage, budget accounting, determinism) | `pytest tests` | 15 tests |
| Feature tables + split summary | `scripts/prepare_data.py` | `results/dataset_summary.json` |
| Experiments (55 runs = 11 dataset/scenario cells × 5 seeds) | `scripts/run_all.py` | `results/runs/` |
| Statistics (bootstrap CIs, permutation tests, Holm) | `scripts/analyze.py` | `results/analysis/*.csv` |
| Figures | `scripts/make_figures.py` | `paper/figures/` |
| Manuscript | `paper/compose.py` + `paper/build_docx.js` | `paper/DRT_paper.docx` |

To check a re-run against the released numbers:
`python scripts/compare_results.py results/analysis <released>/results/analysis`.

## Pre-registration and audit trail

* `configs/protocol.yaml` fixes every hyper-parameter, policy, budget, scenario rule and statistical test.
  Its SHA-256 (`7d4914e7be97512a2b5867f532e0f8f7da51d63f5aff43c656ccb96a78a9a132`) is written into every
  result file, and `scripts/analyze.py` refuses to mix results from a different protocol.
* The git history shows the protocol and code committed (`aeddb8d`, "Freeze protocol v1.0 and code before
  any test-period evaluation") before the first test-period run.
* Development used `mode="dev"`, which deletes the real test period before any processing
  (`tests/test_experiment.py::test_dev_mode_never_reads_test_period`).

## Data

| Corpus | Source | SHA-256 (prefix) | Rows / frauds | Licence |
|---|---|---|---|---|
| ULB credit-card (real, 2 days) | GitHub mirror of Kaggle `mlg-ulb/creditcardfraud` | `33a178be…` | 284,807 / 492 | ODbL (contents DbCL) |
| BankSim (simulated bank payments) | GitHub mirror of Kaggle `ealaxi/banksim1` | `e37006f7…` | 594,643 / 7,200 | see Kaggle page |
| IBM TabFormer (simulated cards) | `IBM/TabFormer` Git-LFS | `e9f589a0…` | 24,386,900 / 29,757 | Apache-2.0 |

Full URLs, hashes and expected counts are in `data/manifest.json`. The ULB mirror rounds the PCA
features to nine decimals; counts match the original release.

## Determinism

All randomness is seeded (`seeds: [0..4]`, LightGBM `deterministic: true`, fixed thread count).
Re-running a configuration yields bit-identical per-window results on the same platform
(`test_deterministic_rerun`). Across CPU architectures, LightGBM may differ in the last floating-point
digits; the statistics in the paper are reported to three decimals.

## Environment

Python 3.13, packages pinned in `requirements.txt`; Node.js with the `docx` package for the manuscript only.

## Layout

```
configs/protocol.yaml      frozen protocol
data/manifest.json         sources, hashes, expected counts
src/dstriage/              data.py (features), models.py, policies.py (DRT etc.), experiment.py, stats.py
scripts/                   fetch, prepare, run_all, analyze, make_figures, compare_results
tests/                     leakage, budget, split and determinism tests
results/                   runs/ (per-window outputs), analysis/ (all statistics), logs
paper/                     compose.py + text.py (numbers pulled from results), build_docx.js, figures/
```
