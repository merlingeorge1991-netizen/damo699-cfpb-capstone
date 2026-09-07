"""
year_ablation.py  (v3)
----------------------
Tests whether the year feature explains the 2024-to-2025 score shift.

This version reuses the FITTED categorical encoder from data/model_logreg.pkl
rather than refitting one, so the categorical block is identical to the
report's model by construction. Only the numeric block changes between arms,
and only the classifier is refitted.

Configuration read from the saved pipeline:
    cat  OneHotEncoder(min_frequency=10, handle_unknown='ignore')  -> 968 cols
    num  StandardScaler                                            ->   3 cols
    clf  LogisticRegression(solver='saga', C=1.0, max_iter=1000)

Arms:
    baseline  year as a standardised numeric feature   (the report's model)
    dropped   year removed
    capped    year clipped at 2023, its maximum in training

Run:  python year_ablation.py     (roughly 15-30 minutes)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")
print(f"project root: {ROOT}")
print("Loading (1-2 minutes)...\n", flush=True)

from scipy import sparse                                            # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

CATS = ["Product", "Sub-product", "Issue", "Sub-issue", "Company", "State", "Submitted via"]
NUMS = ["year", "month", "quarter"]
SEED, THRESHOLD = 42, 0.60

EXPECT = {"2024": {"recall": 0.4362, "pr_auc": 0.2403},
          "2025": {"recall": 0.1320, "pr_auc": 0.1156}}
TOL = 0.05

# ---- reuse the fitted categorical encoder from the saved pipeline ----------
with open(ROOT / "data/model_logreg.pkl", "rb") as f:
    saved = pickle.load(f)
saved = saved.get("model", saved) if isinstance(saved, dict) else saved
pre = saved.named_steps["pre"]
cat_enc = dict((n, t) for n, t, _ in pre.transformers_)["cat"]
print(f"reusing fitted encoder: {len(cat_enc.get_feature_names_out()):,} categorical columns")

train = pd.read_pickle(ROOT / "data/train_sample.pkl")
pop = pd.read_pickle(ROOT / "data/study_population.pkl")
ycol = next(c for c in ("target", "y", "untimely") if c in pop.columns)
val = pop.loc[pop["year"] == 2024]
test = pop.loc[pop["year"] == 2025]
print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}\n")


def numeric(df, variant):
    year = df["year"].astype(float)
    if variant == "capped":
        year = year.clip(upper=2023)
    cols = [df["month"].astype(float), df["quarter"].astype(float)]
    if variant != "dropped":
        cols.insert(0, year)
    return np.column_stack(cols)


def run(variant):
    Xc = cat_enc.transform(train[CATS])
    scaler = StandardScaler().fit(numeric(train, variant))
    X = sparse.hstack([Xc, sparse.csr_matrix(
        scaler.transform(numeric(train, variant)).astype(np.float32))], format="csr")
    if variant == "baseline":
        print(f"  total features: {X.shape[1]:,} (report states 971)")

    clf = LogisticRegression(solver="saga", C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(X, train[ycol].to_numpy())

    out = {}
    for name, df in (("2024", val), ("2025", test)):
        y = df[ycol].to_numpy().astype(int)
        s = np.empty(len(df), dtype=np.float32)
        for i in range(0, len(df), 250_000):
            part = df.iloc[i:i + 250_000]
            Xp = sparse.hstack([cat_enc.transform(part[CATS]), sparse.csr_matrix(
                scaler.transform(numeric(part, variant)).astype(np.float32))], format="csr")
            s[i:i + 250_000] = clf.predict_proba(Xp)[:, 1]
        pred = s >= THRESHOLD
        tp = int((pred & (y == 1)).sum())
        out[name] = dict(recall=tp / y.sum(),
                         precision=tp / pred.sum() if pred.sum() else 0.0,
                         flagged=int(pred.sum()),
                         pr_auc=average_precision_score(y, s),
                         roc_auc=roc_auc_score(y, s),
                         mean_score=float(s.mean()))
    return out


print("fitting 'baseline'...", flush=True)
results = {"baseline": run("baseline")}

print("\nBASELINE REPRODUCTION CHECK")
ok = True
for period, exp in EXPECT.items():
    for metric, want in exp.items():
        have = results["baseline"][period][metric]
        near = abs(have - want) <= TOL * want
        ok &= near
        print(f"  {period} {metric:<8} expected {want:.4f}  got {have:.4f}  "
              f"{'OK' if near else 'MISMATCH'}")

if not ok:
    print("\nThe baseline still does not reproduce. Send this output back rather than\n"
          "reporting these numbers.")
    sys.exit(1)

print("\nBaseline reproduces. Fitting the ablation arms...\n")
for variant in ("dropped", "capped"):
    print(f"fitting '{variant}'...", flush=True)
    results[variant] = run(variant)

print("\n" + "=" * 78)
print(f"{'variant':<10}{'period':<8}{'recall':>9}{'PR-AUC':>9}{'ROC-AUC':>9}"
      f"{'flagged':>10}{'mean score':>12}")
print("=" * 78)
for v, r in results.items():
    for p, m in r.items():
        print(f"{v:<10}{p:<8}{m['recall']:>9.4f}{m['pr_auc']:>9.4f}"
              f"{m['roc_auc']:>9.4f}{m['flagged']:>10,}{m['mean_score']:>12.6f}")

print("\n" + "=" * 78)
for name, r in results.items():
    fall = r["2024"]["recall"] - r["2025"]["recall"]
    shift = r["2025"]["mean_score"] - r["2024"]["mean_score"]
    print(f"  {name:<10} recall {r['2024']['recall']:.4f} -> {r['2025']['recall']:.4f}"
          f"  (fall {fall:.4f})   mean score shift {shift:+.6f}")
print("""
If dropped and capped fall much less than baseline, the extrapolation account
in Section 6.6 is demonstrated. If all three fall alike, it is refuted and the
cause lies elsewhere - equally worth reporting.""")

pd.DataFrame([{**{"variant": v, "period": p}, **m}
              for v, r in results.items() for p, m in r.items()]) \
  .to_csv(ROOT / "outputs/tables/year_ablation.csv", index=False)
print("\nwritten: outputs/tables/year_ablation.csv")
