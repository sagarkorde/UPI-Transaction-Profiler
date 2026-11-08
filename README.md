# Disagreement-Reserved Triage (DRT): reproducibility package

Code, frozen protocol and results for budget-constrained dual-signal fraud triage
(Disagreement-Reserved Triage, DRT): a share ρ of each analyst review batch is reserved for transactions
the supervised model ranks below its cut-off but an anomaly detector ranks highest.

Every result is produced by the scripts in this repository from three publicly accessible corpora
that are downloaded by script and verified by SHA-256.

## One command

```bash
bash reproduce.sh          # ~75 min on 2 CPU cores, ~7 GB RAM, ~12 GB disk
```

| Step | Script | Output |
|---|---|---|
| Data (download + SHA-256 + row/fraud counts) | `scripts/fetch_data.py` | `data/raw/` |
| Tests (leakage, budget accounting, feedback variants, determinism) | `pytest tests` | 18 tests |
| Feature tables + split summary | `scripts/prepare_data.py` | `results/dataset_summary.json` |
| Experiments (55 runs = 11 dataset/scenario cells × 5 seeds) | `scripts/run_all.py` | `results/runs/` |
| Statistics (bootstrap CIs, permutation tests, Holm) | `scripts/analyze.py` | `results/analysis/*.csv` |
| Figures | `scripts/make_figures.py`, `scripts/make_architecture.py` | `figures/` |
| Supplementary tables | `scripts/make_supplement.py` | `results/SUPPLEMENT.md` |

To check a re-run against the released numbers:
`python scripts/compare_results.py results/analysis <released>/results/analysis`.

## Pre-registration and audit trail

* `configs/protocol.yaml` fixes every hyper-parameter, policy, budget, scenario rule and statistical test.
  Its SHA-256 (`7d4914e7be97512a2b5867f532e0f8f7da51d63f5aff43c656ccb96a78a9a132`) is written into every
  result file, and `scripts/analyze.py` refuses to mix results from a different protocol.
* The git history shows the protocol and code committed (`0a5cc29`, "Freeze protocol v1.0 and code before
  any test-period evaluation") before the first test-period run.
* **Amendment v1.1** (`configs/amendment_v1_1.yaml`, written after the v1.0 test evaluation in response to
  peer review): ULB held-out clusters are fitted on pre-test frauds only; DRT-ADAPT is additionally evaluated
  with feedback delayed by 1/3/7 windows and with noisy analyst outcomes. `protocol.yaml` is unchanged.
* **Independent re-run.** All 55 runs were re-executed from scratch under v1.1. Every result not affected by
  the amendment (45 runs, all policies and windows) is bit-identical to v1.0
  (`results/reproduction_check.txt`, `scripts/check_reproduction.py`). v1.0 outputs are kept in
  `results/runs_v1_0/`.
* **Amendment v1.2** (`configs/amendment_v1_2.yaml`, exploratory): adds DRT-ADAPT-AUDIT, whose reserved share is
  driven only by a randomised audit of the bands just beyond each channel's cut-off, to test whether adaptive
  allocation is an artefact of selectively observed feedback. A second full re-run reproduced all 55 v1.1 runs
  bit-for-bit (`results/reproduction_check_v1_2.txt`, `scripts/compare_runs.py`). v1.1 outputs are kept in
  `results/runs_v1_1/`; `results/runs/` holds the v1.2 results used for all statistics.
* **Amendment v1.3** (`configs/amendment_v1_3.yaml`, exploratory): adds DRT-LD, a learned-deferral counterpart of DRT whose
  reserved share is filled by a rejector trained on calibration-period outcomes to predict the frauds SUP's queue missed.
  A third full re-run reproduced all 55 v1.2 runs bit-for-bit (`results/reproduction_check_v1_3.txt`); v1.2 outputs are kept in
  `results/runs_v1_2/`, and `results/runs/` holds the v1.3 results used for all statistics.
* **Amendment v1.4** (`configs/amendment_v1_4.yaml`, exploratory): adds DRT-AMT (reserve ranked by raw amount),
  SUP-NOID (supervised model without the features that identify held-out groups; a diagnostic, not a deployable policy)
  and DRT-NOID. A fourth full re-run reproduced all 55 v1.3 runs bit-for-bit (`results/reproduction_check_v1_4.txt`);
  v1.3 outputs are kept in `results/runs_v1_3/`, and `results/runs/` holds the v1.4 results used for all statistics.
  Descriptive analyses: `scripts/operational_metrics.py` (capacity in held-out scenarios, time to first detection) and
  `scripts/amount_signal.py` (how well raw amount ranks each held-out group).
* **Amendment v1.5** (`configs/amendment_v1_5.yaml`, exploratory): adds SUP-IDF, LightGBM without merchant, category and
  channel identifiers and their frequency encodings under one scenario-independent rule, and DRT-IDF, which fills DRT's
  reserve with its ranking. A fifth full re-run reproduced all 55 v1.4 runs bit-for-bit
  (`results/reproduction_check_v1_5.txt`); v1.4 outputs are kept in `results/runs_v1_4/`, and `results/runs/` holds the
  v1.5 results used for all statistics.
* **Scenario provenance.** `scripts/check_scenario_provenance.py` shows that every candidate held-out group is
  the same when chosen from pre-test frauds only (supplement S10b).
* **Audit map.** `REPRODUCTION_AUDIT.md` maps every table, figure and quoted number to its script and file.
* Development used `mode="dev", which deletes the real test period before any processing
  (`tests/test_experiment.py::test_dev_mode_never_reads_test_period`).

## Data

| Corpus | Source | SHA-256 (prefix) | Rows / frauds | Licence |
|---|---|---|---|---|
| ULB credit-card (real, 2 days) | GitHub mirror of Kaggle `mlg-ulb/creditcardfraud` | `33a178be…` | 284,807 / 492 | ODbL (contents DbCL) |
| BankSim (simulated bank payments) | GitHub mirror of Kaggle `ealaxi/banksim1` | `e37006f7…` | 594,643 / 7,200 | none stated on the public release; cited as the authors request |
| IBM TabFormer (simulated cards) | `IBM/TabFormer` Git-LFS | `e9f589a0…` | 24,386,900 / 29,757 | Apache-2.0 |

Full URLs, hashes and expected counts are in `data/manifest.json`. The ULB mirror rounds the PCA
features to nine decimals; counts match the original release.

## Determinism

All randomness is seeded (`seeds: [0..4]`, LightGBM `deterministic: true`, fixed thread count).
Re-running a configuration yields bit-identical per-window results on the same platform
(`test_deterministic_rerun`). Across CPU architectures, LightGBM may differ in the last floating-point
digits; reported statistics use three decimals.

## Environment

Python 3.13, packages pinned in `requirements.txt`.

## Layout

```
configs/protocol.yaml      frozen protocol
data/manifest.json         sources, hashes, expected counts
src/dstriage/              data.py (features), models.py, policies.py (DRT etc.), experiment.py, stats.py
scripts/                   fetch, prepare, run_all, analyze, make_figures, compare_results
tests/                     leakage, budget, split and determinism tests
results/                   runs/ (per-window outputs), analysis/ (all statistics), logs
figures/                   generated figures (created by the figure scripts)
```
