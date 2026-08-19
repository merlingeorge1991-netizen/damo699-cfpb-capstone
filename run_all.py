"""
Run the whole pipeline end to end.

    python run_all.py             # every step
    python run_all.py --from 4    # resume from step 04 (models onward)

Each step is also runnable on its own from src/. Steps are idempotent: rerunning
one overwrites its own outputs and nothing else.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    (1, "step01_acquire.py", "Download and verify the CFPB archive"),
    (2, "step02_audit.py", "Chunked data quality audit (Section 4 tables)"),
    (3, "step03_prepare.py", "Build the enriched training sample"),
    (4, "step04_train.py", "Train logistic regression, CatBoost, TF-IDF"),
    (5, "step05_threshold.py", "Validation scoring and threshold selection"),
    (6, "step06_test.py", "Final single-use test evaluation"),
    (7, "step07_figures.py", "Generate all report figures"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="first step to run (1-7)")
    ap.add_argument("--to", dest="end", type=int, default=7,
                    help="last step to run (1-7)")
    args = ap.parse_args()

    for num, script, desc in STEPS:
        if not (args.start <= num <= args.end):
            continue
        print("\n" + "=" * 70)
        print(f"STEP {num:02d}  {desc}")
        print("=" * 70)
        result = subprocess.run([sys.executable, str(SRC / script)])
        if result.returncode != 0:
            print(f"\nStep {num:02d} failed (exit {result.returncode}). Stopping.")
            return result.returncode

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("  tables  -> outputs/tables/")
    print("  figures -> outputs/figures/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
