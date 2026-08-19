# Predicting Untimely Company Responses to CFPB Consumer Complaints

Analysis pipeline for the DAMO 699 capstone. Running it end to end produces
every number, table and figure that Sections 4, 6 and Appendices C–E of the
report require.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

The download is roughly 1 GB compressed. Budget 45–90 minutes for a full run on
a laptop; the two slowest parts are the chunked audit and scoring the test-year
population.

Steps are independent and idempotent, so you can resume:

```bash
python run_all.py --from 4        # skip re-downloading and re-auditing
python src/step05_threshold.py    # or run one step directly
```

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 01 | `step01_acquire.py` | Downloads the CFPB archive, verifies the ZIP end-of-central-directory signature, extracts the CSV |
| 02 | `step02_audit.py` | Chunked audit → report Tables 1, 4, 5, 6, 7 and the narrative-availability series |
| 03 | `step03_prepare.py` | Enriched training sample (all positives + 5 negatives each) → Table 9 |
| 04 | `step04_train.py` | Fits logistic regression, CatBoost and the TF–IDF text model → Table 17, coefficient tables |
| 05 | `step05_threshold.py` | Validation scoring, model comparison, threshold sweep → Tables 11, 12, 13, 14 |
| 06 | `step06_test.py` | Single-use test evaluation → Tables 15, 16, decile lift |
| 07 | `step07_figures.py` | All nine figures at 300 dpi |

## Where the report numbers come from

Every highlighted placeholder in the report maps to a file in `outputs/tables/`:

| Report item | File |
|---|---|
| Table 11 — validation performance | `table11_12_validation.csv` |
| Table 12 — LR vs CatBoost | `table11_12_validation.csv` |
| Table 13 — narrative subset | `table13_narrative_subset.csv` |
| Table 14 — threshold sweep | `table14_threshold_sweep.csv` |
| Table 15 — confusion matrix | `table15_confusion_matrix.csv` |
| Table 16 — validation vs test | `table16_val_vs_test.csv` |
| Table 17 — coefficients | `table17_logreg_coefficients.csv` |
| Section 6.8 — lift and calibration | `decile_lift_test.csv`, `fig05_calibration.png` |
| Section 9.2 — headline figures | `final_test_summary.csv` |

## Design decisions worth knowing before you run it

**Leakage control.** `config.LEAKAGE_FIELDS` lists the post-response columns
that must never become predictors. Step 02 drops them explicitly rather than
relying on them not being selected, so the guarantee is auditable.

**Training enrichment only.** The 5:1 undersampling touches the training years
alone. Validation and test keep their full natural populations, so reported
precision and recall reflect real prevalence.

**Scores are not calibrated probabilities.** Enrichment raises the training base
rate roughly thirty-five-fold, so raw scores sit well above true population
probabilities. Use them for ranking and thresholding. `fig05_calibration.png`
shows the size of the deviation.

**The test year is scored once.** Step 06 exists as a separate script for a
reason: if you change a feature, a hyperparameter or the threshold after running
it, you must re-freeze on validation before running it again. Otherwise the test
set has become a second validation set.

## Verifying against the report

After step 02, compare the console output with report Table 1. Complaint counts
per year, untimely counts, and the 71.11% narrative missingness should all
reproduce. If they don't, the CFPB file has been updated since the audit was
first run — note the retrieval date and update Section 4 rather than forcing the
old numbers.

## Configuration

Everything tunable lives in `config.py`: study years, split boundaries,
undersampling ratio, decision threshold, sweep grid, model hyperparameters,
TF–IDF settings and the random seed (42). Change values there, not in the
scripts, so the report can describe one authoritative source.

## Testing without the download

```bash
python tests/make_synthetic.py
python run_all.py --from 2
```

Generates a 60,000-row file with the CFPB schema so you can confirm the pipeline
runs before committing to the full download. **Numbers from synthetic data are
meaningless** — delete `data/complaints.csv` before the real run.

## Memory

The study population is 11.2M rows, ~3.2M of them carrying narrative text.
Narratives are therefore stored in one pickle per year (`data/narratives_YYYY.pkl`)
and merged only onto the rows a given step actually needs, while the seven
categorical columns are held as pandas `category` dtype. Scoring prepares and
predicts one slice at a time rather than materialising the whole frame.

If you still hit `_ArrayMemoryError`, lower `CHUNKSIZE` in `config.py` to
250_000 and close other applications. The peak is during step 02's concat.

## Requirements

Python 3.10+, roughly 8 GB RAM and 10 GB free disk. If memory is tight, lower
`CHUNKSIZE` in `config.py`. CatBoost is optional: if it isn't installed, that
benchmark is skipped with a message and the rest of the pipeline still runs.
