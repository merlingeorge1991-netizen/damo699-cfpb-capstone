"""
Central configuration for the CFPB untimely-response prediction pipeline.

Every tunable value in the project lives here so that the report can state
exactly what was used and a reader can reproduce the run by inspecting one file.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for _d in (DATA, OUTPUTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# Raw CFPB export. Script 01 downloads this; if you already have the CSV,
# drop it here and script 01 will verify rather than re-download.
RAW_ZIP = DATA / "complaints.csv.zip"
RAW_CSV = DATA / "complaints.csv"

# Intermediate artefacts (pickled DataFrames - fast, no extra dependency)
CLEAN_PKL = DATA / "study_population.pkl"
TRAIN_PKL = DATA / "train_sample.pkl"


def narrative_path(year: int) -> Path:
    """
    Narratives are stored per year, separately from the structured population.

    Holding 3.2M narrative strings alongside 11.2M structured rows exceeds
    available memory on a typical laptop, and no single step needs more than
    one or two years of narrative at a time. Splitting by year bounds peak
    memory for every downstream step.
    """
    return DATA / f"narratives_{year}.pkl"

CFPB_BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

# ---------------------------------------------------------------- schema
# Exact CFPB column headers. Do not rename these - the loader matches on them.
COL_ID = "Complaint ID"
COL_DATE = "Date received"
COL_TARGET = "Timely response?"
COL_NARRATIVE = "Consumer complaint narrative"

CATEGORICAL_FEATURES = [
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "Submitted via",
]

NUMERIC_FEATURES = ["year", "month", "quarter"]

# Columns read from the raw CSV. Everything else is discarded at read time,
# which is what makes a 1 GB+ file tractable in chunks.
USECOLS = [COL_ID, COL_DATE, COL_TARGET, COL_NARRATIVE] + CATEGORICAL_FEATURES

# Post-response fields. Listed explicitly so the leakage control is auditable
# rather than implicit: none of these may ever enter the feature matrix.
LEAKAGE_FIELDS = [
    "Company response to consumer",
    "Company public response",
    "Consumer disputed?",
    "Consumer consent provided?",
    "Date sent to company",
]

# ---------------------------------------------------------------- study design
STUDY_START_YEAR = 2020
STUDY_END_YEAR = 2025

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

# Majority-class undersampling ratio applied to the TRAINING period only.
# Validation and test keep their full natural populations.
NEGATIVE_PER_POSITIVE = 5

# Locked operating threshold. Selected on validation in script 05, then frozen.
DECISION_THRESHOLD = 0.60
THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

RANDOM_SEED = 42

# ---------------------------------------------------------------- model params
LOGREG_PARAMS = dict(
    solver="saga",
    penalty="l2",
    C=1.0,
    max_iter=1000,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

CATBOOST_PARAMS = dict(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_seed=RANDOM_SEED,
    verbose=100,
)

TFIDF_PARAMS = dict(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_features=50_000,
    strip_accents="unicode",
    sublinear_tf=True,
)

# CFPB redacts personal information with XXXX tokens. Left in place, these
# become high-frequency terms the model can key on, so they are suppressed.
REDACTION_STOPWORDS = ["xxxx", "xx", "xxxxxxxx", "xxxxxxxxxxxx"]

# Rows per chunk when reading or scoring. Lower this if you hit memory limits.
CHUNKSIZE = 500_000
