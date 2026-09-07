"""
verify_remaining.py
-------------------
Closes the last unverified numbers in the capstone report using artifacts
that already exist in the project. Nothing is recomputed from complaints.csv.

Run from:  C:\\Users\\surya\\Downloads\\cfpb_capstone_pipeline_v2\\cfpb_capstone

    python verify_remaining.py

Checks:
  1. encoded feature count            -> report says 971
  2. narrative-bearing training rows  -> report says 53,452 (41.72% of 128,118)
  3. untimely rate in that subset     -> report says 21.23%
  4. calibration, top quantile bin    -> report says pred ~0.19 vs observed 0.044
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve every path from this file's own folder, so the script works
# no matter which directory it is launched from (VS Code Run button included).
ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script.\n"
             f"Script is at: {ROOT}\n"
             f"Move it into cfpb_capstone (the folder holding config.py and run_all.py).")
print(f"project root: {ROOT}")

print("=" * 68)
print("This takes 1-3 minutes. Loading libraries and large files is slow;\nthe pauses below are normal. Do NOT press Ctrl+C.\n", flush=True)

# ---------------------------------------------------------------- 1. features
print("   [1/4] loading model_logreg.pkl (imports scikit-learn, ~30s)...", flush=True)
try:
    with open(ROOT / "data/model_logreg.pkl", "rb") as f:
        obj = pickle.load(f)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    n = None
    for attr in ("n_features_in_", "coef_"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            n = int(v) if attr == "n_features_in_" else v.shape[1]
            break
    if n is None and hasattr(model, "named_steps"):
        n = model[-1].coef_.shape[1]
    print(f"1. encoded features          : {n}          (report: 971)")
except Exception as e:
    print(f"1. encoded features          : could not read ({e})")

# ------------------------------------------------------- 2/3. training subset
print("\n   [2/4] loading train_sample.pkl (56 MB)...", flush=True)
try:
    tr = pd.read_pickle(ROOT / "data/train_sample.pkl")
    print(f"\n2. training rows             : {len(tr):,}      (report: 128,118)")
    ycol = next((c for c in ("y", "target", "untimely") if c in tr.columns), None)
    ncol = next((c for c in tr.columns
                 if "narrative" in c.lower() or "has_text" in c.lower()), None)
    print(f"   columns available         : {list(tr.columns)}")

    if ycol:
        print(f"   untimely                  : {int(tr[ycol].sum()):,} "
              f"({tr[ycol].mean():.4%})   (report: 21,353 / 16.67%)")
    if ncol:
        sub = tr[tr[ncol].astype(bool)]
        print(f"   narrative-bearing rows    : {len(sub):,} "
              f"({len(sub)/len(tr):.2%})   (report: 53,452 / 41.72%)")
        if ycol:
            print(f"3. untimely rate in subset   : {sub[ycol].mean():.4%}"
                  f"        (report: 21.23%)")
    else:
        print("   no narrative flag column found - items 2/3 not verifiable here")
except Exception as e:
    print(f"\n2/3. training subset         : could not read ({e})")

# --------------------------------------------------------------- 4. calibration
print("\n   [3/4] loading scores_test_logreg.npy (21 MB)...", flush=True)
try:
    scores = np.load(ROOT / "data/scores_test_logreg.npy")
    print("   [4/4] loading study_population.pkl (377 MB, slowest step)...", flush=True)
    pop = pd.read_pickle(ROOT / "data/study_population.pkl")
    ycol = next((c for c in ("y", "target", "untimely") if c in pop.columns), None)
    ycol = ycol or pop.columns[-1]
    test = pop.loc[pop["year"] == 2025]
    y = test[ycol].to_numpy().astype(int)

    print(f"\n4. calibration")
    print(f"   scores loaded             : {len(scores):,}")
    print(f"   2025 rows in population   : {len(y):,}   (report: 5,442,974)")
    if len(scores) != len(y):
        print("   LENGTH MISMATCH - scores may be ordered differently; stopping")
    else:
        print(f"   untimely                  : {int(y.sum()):,}   (report: 24,557)")
        print(f"   scoring >= 0.60           : {int((scores >= 0.60).sum()):,}"
              f"   (report: 26,154)")
        print(f"   mean score                : {scores.mean():.6f}")

        for nb in (5, 10, 20):
            q = pd.qcut(pd.Series(scores).rank(method="first"), nb, labels=False)
            g = pd.DataFrame({"b": q, "p": scores, "y": y}).groupby("b")
            top = g.agg(pred=("p", "mean"), obs=("y", "mean")).iloc[-1]
            print(f"   top bin of {nb:2d} (quantile) : "
                  f"predicted {top.pred:.4f} | observed {top.obs:.4f}")

        # uniform-width bins, the sklearn calibration_curve default
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(scores, edges) - 1, 0, 9)
        df = pd.DataFrame({"b": idx, "p": scores, "y": y})
        occ = df.groupby("b").agg(n=("p", "size"), pred=("p", "mean"),
                                  obs=("y", "mean"))
        occ = occ[occ.n > 0]
        print("\n   uniform-width bins (sklearn default):")
        print(occ.to_string())
except Exception as e:
    print(f"\n4. calibration               : could not read ({e})")

print("=" * 68)
