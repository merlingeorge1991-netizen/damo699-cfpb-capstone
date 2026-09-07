"""
diagnostics.py
--------------
Computes the formal model diagnostics the capstone writing manual's sample
chapter expects, using artifacts already in the project. Nothing is refitted.

Place next to config.py, then:  python diagnostics.py

Produces:
  - Hosmer-Lemeshow goodness-of-fit test (chi-square, df, p) on 2024 and 2025
  - Brier score and log loss for both periods
  - Observed vs expected counts per risk group, for a table in the report
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")
print(f"project root: {ROOT}")
print("Loading (1-2 minutes)...\n", flush=True)

from scipy import stats                                    # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss     # noqa: E402

pop = pd.read_pickle(ROOT / "data/study_population.pkl")
ycol = next((c for c in ("target", "y", "untimely") if c in pop.columns), None)

periods = {}
for year, fn in ((2024, "scores_val_logreg.npy"), (2025, "scores_test_logreg.npy")):
    path = ROOT / "data" / fn
    if not path.exists():
        print(f"  {fn} missing; skipping {year}")
        continue
    s = np.load(path)
    y = pop.loc[pop["year"] == year, ycol].to_numpy().astype(int)
    if len(s) != len(y):
        print(f"  {year}: length mismatch ({len(s):,} vs {len(y):,}); skipping")
        continue
    periods[year] = (y, s)
del pop


def hosmer_lemeshow(y, p, groups=10):
    """
    Chi-square goodness-of-fit over equal-sized risk groups.

    Groups are formed on the rank of the predicted score, so each holds the
    same number of cases; ties do not collapse a group.
    """
    order = pd.Series(p).rank(method="first")
    g = pd.qcut(order, groups, labels=False)
    df = pd.DataFrame({"g": g, "p": p, "y": y})
    agg = df.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"),
                              mean_p=("p", "mean"))
    agg["obs_rate"] = agg.obs / agg.n
    agg["exp_rate"] = agg.exp / agg.n
    denom = agg.exp * (1 - agg.exp / agg.n)
    chi2 = float((((agg.obs - agg.exp) ** 2) / denom).sum())
    dof = groups - 2
    return chi2, dof, float(stats.chi2.sf(chi2, dof)), agg


for year, (y, p) in periods.items():
    p = np.clip(p.astype(np.float64), 1e-15, 1 - 1e-15)
    chi2, dof, pval, agg = hosmer_lemeshow(y, p)
    label = "validation" if year == 2024 else "test"
    print("=" * 72)
    print(f"{year} {label}   n = {len(y):,}   observed untimely = {int(y.sum()):,}")
    print("=" * 72)
    pstr = "< .001" if pval < 0.001 else f"= {pval:.3f}"
    print(f"  Hosmer-Lemeshow : chi-square({dof}) = {chi2:,.2f}, p {pstr}")
    print(f"  Brier score     : {brier_score_loss(y, p):.6f}")
    print(f"  Log loss        : {log_loss(y, p):.6f}")
    print(f"  Mean predicted  : {p.mean():.6f}   observed rate: {y.mean():.6f}")
    print(f"  Overprediction  : {p.mean() / y.mean():.2f}x\n")
    show = agg.copy()
    show.index = [f"{i+1}" for i in range(len(show))]
    show = show[["n", "obs", "exp", "obs_rate", "exp_rate"]]
    show.columns = ["Cases", "Observed", "Expected", "Observed rate", "Expected rate"]
    print(show.to_string(float_format=lambda v: f"{v:,.4f}"))
    print()

print("=" * 72)
print("NOTE ON INTERPRETATION")
print("=" * 72)
print("""The Hosmer-Lemeshow test is known to reject almost any model at very large
sample sizes, because the statistic scales with n while the degrees of freedom
stay fixed at 8. With populations of 2.7M and 5.4M, a significant result is
expected and is NOT by itself evidence that the model is unusable. Report the
statistic alongside the observed-versus-expected rates above, which show the
direction and size of the miscalibration, and interpret those rather than the
p-value alone. This is consistent with the report's existing position that the
scores be treated as an ordinal ranking rather than as probabilities.""")
