"""
build_dashboard_data.py
-----------------------
Exports Power BI-ready CSVs from artifacts the project already has. Nothing is
refitted and nothing is overwritten; everything lands in outputs/dashboard/.

Run AFTER year_ablation.py finishes - both load study_population.pkl and
together they would exhaust 8 GB.

    python build_dashboard_data.py

Produces six files:
    dim_year.csv        complaint volume and untimely rate by year
    dim_company.csv     top companies with volume and untimely rate
    dim_product.csv     products with volume and untimely rate
    fact_threshold.csv  the 13-threshold sweep (drives the threshold slicer)
    fact_decile.csv     decile concentration and lift
    fact_queue.csv      the top-scored 2025 complaints, ready to triage
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")
OUT = ROOT / "outputs" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)
print(f"project root: {ROOT}\nwriting to  : {OUT}\n")

QUEUE_ROWS = 50_000          # top-scored 2025 complaints to expose in the dashboard
TOP_COMPANIES = 40

pop = pd.read_pickle(ROOT / "data/study_population.pkl")
ycol = next(c for c in ("target", "y", "untimely") if c in pop.columns)
print(f"study population: {len(pop):,} rows")


def rate_table(df, key, min_n=1):
    g = (df.groupby(key)[ycol]
           .agg(complaints="size", untimely="sum")
           .query("complaints >= @min_n")
           .assign(untimely_rate=lambda d: (d.untimely / d.complaints).round(6))
           .sort_values("complaints", ascending=False)
           .reset_index())
    return g


# --- dimensions -------------------------------------------------------------
year = rate_table(pop, "year").sort_values("year")
year["share_of_total"] = (year.complaints / year.complaints.sum()).round(6)
year.to_csv(OUT / "dim_year.csv", index=False)
print(f"  dim_year.csv        {len(year)} rows")

comp = rate_table(pop, "Company", min_n=500).head(TOP_COMPANIES)
comp.to_csv(OUT / "dim_company.csv", index=False)
print(f"  dim_company.csv     {len(comp)} rows")

prod = rate_table(pop, "Product")
prod.to_csv(OUT / "dim_product.csv", index=False)
print(f"  dim_product.csv     {len(prod)} rows")

# --- facts from the existing result tables ----------------------------------
T = ROOT / "outputs" / "tables"
for src, dst in (("table14_threshold_sweep.csv", "fact_threshold.csv"),
                 ("decile_lift_test.csv", "fact_decile.csv")):
    p = T / src
    if p.exists():
        df = pd.read_csv(p)
        df.to_csv(OUT / dst, index=False)
        print(f"  {dst:<20}{len(df)} rows")
    else:
        print(f"  {src} not found; skipped")

# --- the triage queue: top-scored 2025 complaints ---------------------------
sp = ROOT / "data" / "scores_test_logreg.npy"
if sp.exists():
    scores = np.load(sp)
    test = pop.loc[pop["year"] == 2025].copy()
    if len(scores) != len(test):
        print(f"  score/label length mismatch ({len(scores):,} vs {len(test):,}); queue skipped")
    else:
        test["score"] = scores
        test["rank"] = test["score"].rank(method="first", ascending=False).astype(int)
        test["percentile"] = (test["rank"] / len(test)).round(6)
        test["decile"] = np.minimum((test["percentile"] * 10).astype(int) + 1, 10)
        test["flagged_at_060"] = (test["score"] >= 0.60).astype(int)
        cols = [c for c in ["Complaint ID", "Company", "Product", "Sub-product", "Issue",
                            "State", "Submitted via", "year", "month", "quarter"]
                if c in test.columns]
        q = (test.nsmallest(QUEUE_ROWS, "rank")[cols + ["score", "rank", "percentile",
                                                        "decile", "flagged_at_060", ycol]]
                 .rename(columns={ycol: "actually_untimely"}))
        q.to_csv(OUT / "fact_queue.csv", index=False)
        print(f"  fact_queue.csv      {len(q):,} rows "
              f"({q.actually_untimely.sum():,} untimely, "
              f"{q.actually_untimely.mean():.1%} of the queue)")
else:
    print("  scores_test_logreg.npy not found; queue skipped")

print("\nAll files are small enough to import directly into Power BI.")
