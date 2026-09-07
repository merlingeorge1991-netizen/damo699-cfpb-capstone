"""
count_chunks.py
---------------
Confirms the last two checkable claims in the report.

Place next to config.py, then:  python count_chunks.py

  1. CatBoost iterations actually run  (instant)
     config.py sets iterations=500 with no early-stopping parameter.
     learn_error.tsv records what really happened.

  2. Appendix B.3 - "read in 35 sequential chunks of 500,000 rows"  (10-15 min)
     Reads complaints.csv exactly as step02_audit.py does and counts.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")
print(f"project root: {ROOT}")
print("=" * 66)

# ------------------------------------------------------- 1. CatBoost log
log = ROOT / "catboost_info" / "learn_error.tsv"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    iters = max(0, len(lines) - 1)          # minus header
    print("1. CatBoost iterations")
    print(f"     recorded in learn_error.tsv : {iters}")
    print(f"     configured in config.py     : 500")
    if iters >= 500:
        print("     -> ran to completion; no early stopping occurred")
    else:
        print(f"     -> stopped early at {iters}; early stopping was configured "
              "in step04_train.py")
else:
    print("1. catboost_info/learn_error.tsv not found; skipped")

# ------------------------------------------------------ 2. chunk count
csv = ROOT / "data" / "complaints.csv"
if not csv.exists():
    sys.exit(f"\n{csv} not found.")

print(f"\n2. Appendix B.3 - chunk count")
print(f"     reading {csv.name} in 500,000-row chunks.")
print("     This takes 10-15 minutes. Progress updates below.\n", flush=True)

n = rows = 0
for chunk in pd.read_csv(csv, usecols=["Complaint ID"], chunksize=500_000,
                         low_memory=False):
    n += 1
    rows += len(chunk)
    print(f"\r     chunk {n:>3}  |  {rows:>12,} rows", end="", flush=True)

print(f"\n\n     chunks read   : {n}            (report: 35)")
print(f"     source rows   : {rows:,}")
print(f"     -> {'CONFIRMED' if n == 35 else f'report says 35, actual is {n}'}")
print("=" * 66)
