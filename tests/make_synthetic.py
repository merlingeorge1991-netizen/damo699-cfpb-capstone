"""
Generate a small synthetic file with the CFPB schema.

This exists so the pipeline can be smoke-tested without the 1 GB+ download.
It is NOT data for the report - every number produced from it is meaningless.
Delete data/complaints.csv before running the real pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

rng = np.random.default_rng(7)
N = 60_000

PRODUCTS = ["Credit reporting or other personal consumer reports",
            "Debt collection", "Checking or savings account", "Credit card",
            "Mortgage", "Student loan"]
COMPANIES = ["EQUIFAX, INC.", "TRANSUNION INTERMEDIATE HOLDINGS, INC.",
             "Experian Information Solutions Inc.", "CAPITAL ONE FINANCIAL",
             "JPMORGAN CHASE & CO.", "Small Regional Bank"]
ISSUES = ["Incorrect information on your report", "Attempts to collect debt",
          "Managing an account", "Problem with a purchase"]
CHANNELS = ["Web", "Phone", "Referral", "Postal mail", "Fax"]
STATES = ["CA", "TX", "FL", "NY", "GA", "IL"]

years = rng.choice([2020, 2021, 2022, 2023, 2024, 2025], size=N,
                   p=[0.05, 0.06, 0.09, 0.13, 0.25, 0.42])
months = rng.integers(1, 13, size=N)
days = rng.integers(1, 28, size=N)

product = rng.choice(PRODUCTS, size=N, p=[0.66, 0.12, 0.08, 0.06, 0.05, 0.03])
company = rng.choice(COMPANIES, size=N, p=[0.28, 0.27, 0.24, 0.09, 0.08, 0.04])
issue = rng.choice(ISSUES, size=N)
channel = rng.choice(CHANNELS, size=N, p=[0.96, 0.02, 0.01, 0.005, 0.005])
state = rng.choice(STATES, size=N)

# Inject a mild, learnable signal so the models have something to find.
logit = -5.2
logit = logit + np.where(company == "Small Regional Bank", 1.6, 0.0)
logit = logit + np.where(product == "Debt collection", 0.9, 0.0)
logit = logit + np.where(channel == "Postal mail", 1.1, 0.0)
logit = logit + (years - 2022) * 0.05
p = 1 / (1 + np.exp(-logit))
untimely = rng.random(N) < p

has_narrative = rng.random(N) < 0.29
narratives = np.where(
    has_narrative,
    np.where(untimely,
             "the company never responded to my dispute XXXX after many weeks",
             "i submitted a dispute and received a reply from XXXX quickly"),
    None,
)

df = pd.DataFrame({
    cfg.COL_ID: np.arange(1, N + 1),
    cfg.COL_DATE: [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)],
    "Product": product,
    "Sub-product": rng.choice(["General", "Other", None], size=N),
    "Issue": issue,
    "Sub-issue": rng.choice(["Detail A", "Detail B", None], size=N),
    "Company": company,
    "State": state,
    "Submitted via": channel,
    cfg.COL_NARRATIVE: narratives,
    cfg.COL_TARGET: np.where(untimely, "No", "Yes"),
    "Company response to consumer": "Closed with explanation",
})

df.to_csv(cfg.RAW_CSV, index=False)
print(f"Wrote {cfg.RAW_CSV} with {len(df):,} synthetic rows")
print(f"Untimely rate: {100*untimely.mean():.2f}%")
