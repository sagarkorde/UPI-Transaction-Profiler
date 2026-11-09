#!/usr/bin/env bash
# Regenerate every result, table and figure from scratch.
# Requirements: Python 3.13, git, ~12 GB free disk, ~7 GB RAM. Runtime ~75 min on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. environment";  python -m pip install -r requirements.txt
echo "== 1. data";         python scripts/fetch_data.py                 # downloads + SHA-256 / count checks
echo "== 2. tests";        python -m pytest -q tests                    # leakage, budget, determinism
echo "== 3. features";     python scripts/prepare_data.py               # results/dataset_summary.json
echo "== 4. experiments";  python scripts/run_all.py                    # 55 runs -> results/runs/
echo "== 5. statistics";   python scripts/analyze.py                    # results/analysis/*.csv
echo "== 6. figures";      python scripts/make_figures.py && python scripts/make_architecture.py   # figures/
echo "== 7. provenance";   python scripts/check_scenario_provenance.py  # results/analysis/scenario_provenance.json
echo "== 8. dependence";   python scripts/check_dependence.py           # results/analysis/dependence.csv
echo "== 9. operational"; python scripts/operational_metrics.py && python scripts/amount_signal.py  # capacity, time to detection, amount
echo "== 10. replication"; python scripts/run_all.py --datasets tabformer_p5 banksim_u --out results/replication_v1_6/runs \
                         && python scripts/confirm_v1_6.py      # amendment v1.6, pre-registered
echo "== 11. supplement";  python scripts/make_supplement.py            # results/SUPPLEMENT.md
echo "done. Compare with the released results: scripts/compare_runs.py and scripts/compare_results.py (see REPRODUCTION_AUDIT.md)."
