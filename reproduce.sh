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
echo "== 7. supplement";   python scripts/make_supplement.py            # results/SUPPLEMENT.md
echo "done. Compare results/analysis/*.csv with the released copies (scripts/compare_results.py)."
