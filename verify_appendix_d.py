"""
verify_appendix_d.py
--------------------
Checks the quantitative claims made in Appendix D's figure descriptions and
in Section 4.3, none of which have a saved table behind them.

Place next to config.py, then:  python verify_appendix_d.py

Claims tested:
  A. Section 4.3   - zero duplicate complaint IDs in the study population
  B. Figure 6      - precision holds near 0.10-0.15 through the middle of
                     the recall range; no-skill line is 0.0045
  C. Figure 7      - TPR above 0.9 at FPR below 0.1
  D. Figure 8      - substantial mass of untimely complaints between 0.1 and 0.6
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")
print(f"project root: {ROOT}")
print("=" * 70)
print("Loading (1-2 minutes, the 377 MB population file is the slow part)...\n",
      flush=True)

from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

scores = np.load(ROOT / "data/scores_test_logreg.npy")
pop = pd.read_pickle(ROOT / "data/study_population.pkl")

# ------------------------------------------------------------ A. duplicates
idcol = next((c for c in pop.columns if "Complaint ID" in c or c == "id"), None)
if idcol:
    dupes = int(pop[idcol].duplicated().sum())
    print(f"A. Section 4.3 - duplicate complaint IDs")
    print(f"     records            : {len(pop):,}   (report: 11,209,689)")
    print(f"     duplicate IDs      : {dupes:,}        (report: 0)")
    print(f"     -> {'CONFIRMED' if dupes == 0 else 'CONTRADICTED'}\n")
else:
    print("A. no complaint-ID column found; skipped\n")

ycol = next((c for c in ("target", "y", "untimely") if c in pop.columns), None)
y = pop.loc[pop["year"] == 2025, ycol].to_numpy().astype(int)
if len(y) != len(scores):
    sys.exit("score/label length mismatch - cannot continue")
del pop

# ------------------------------------------------------------- B. Figure 6
prec, rec, _ = precision_recall_curve(y, scores)
print("B. Figure 6 - precision through the middle of the recall range")
print(f"     no-skill line      : {y.mean():.4f}   (report: 0.0045)")
mid = []
for target in (0.3, 0.4, 0.5, 0.6, 0.7):
    i = int(np.argmin(np.abs(rec - target)))
    mid.append(prec[i])
    print(f"     precision @ recall {target:.1f} : {prec[i]:.4f}")
lo, hi = min(mid), max(mid)
inrange = all(0.10 <= p <= 0.15 for p in mid)
print(f"     range across mid    : {lo:.4f} - {hi:.4f}   (report: 0.10-0.15)")
print(f"     -> {'CONFIRMED' if inrange else 'DOES NOT MATCH - see range above'}\n")

# ------------------------------------------------------------- C. Figure 7
fpr, tpr, _ = roc_curve(y, scores)
tpr_at = float(np.interp(0.10, fpr, tpr))
fpr_at = float(np.interp(0.90, tpr, fpr))
print("C. Figure 7 - ROC shape")
print(f"     TPR at FPR = 0.10   : {tpr_at:.4f}   (report: above 0.9)")
print(f"     FPR at TPR = 0.90   : {fpr_at:.4f}   (report: below 0.1)")
print(f"     -> {'CONFIRMED' if tpr_at > 0.9 and fpr_at < 0.1 else 'CONTRADICTED'}\n")

# ------------------------------------------------------------- D. Figure 8
pos = scores[y == 1]
band = int(((pos >= 0.1) & (pos < 0.6)).sum())
below = int((pos < 0.1).sum())
above = int((pos >= 0.6).sum())
print("D. Figure 8 - where the untimely complaints actually score")
print(f"     untimely total      : {len(pos):,}")
print(f"     score < 0.1         : {below:,} ({below/len(pos):.1%})")
print(f"     0.1 <= score < 0.6  : {band:,} ({band/len(pos):.1%})  <- 'substantial mass'")
print(f"     score >= 0.6        : {above:,} ({above/len(pos):.1%})   (report: 3,242)")
print(f"     timely median score : {np.median(scores[y == 0]):.6f}")
print("=" * 70)
