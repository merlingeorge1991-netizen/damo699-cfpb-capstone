"""
Step 05 - Validation scoring, model comparison and threshold selection.

Everything here happens on the 2024 validation population. The test period is
not touched. This separation is what allows step06 to report a genuine
out-of-sample estimate.

Produces report Tables 11, 12, 13 and 14.

Run:  python src/step05_threshold.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation as ev


def load(name):
    path = cfg.DATA / name
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> int:
    df = pd.read_pickle(cfg.CLEAN_PKL)
    val = df[df["year"] == cfg.VALIDATION_YEAR].reset_index(drop=True)
    if val.empty:
        print(f"No {cfg.VALIDATION_YEAR} records found.")
        return 1

    y = val["target"].values
    print(f"Validation population ({cfg.VALIDATION_YEAR}): {len(val):,} records, "
          f"{int(y.sum()):,} untimely ({100*y.mean():.3f}%)")

    # ------------------------------------------------ baseline (Table 11 col 1)
    base = ev.baseline_metrics(y)
    print("\n--- Majority-class baseline ---")
    print(f"  accuracy  {base['accuracy']:.4f}")
    print(f"  recall    {base['recall']:.4f}  "
          f"(identifies 0 of {base['untimely_total']:,} untimely complaints)")
    print(f"  PR-AUC    {base['pr_auc']:.4f}  (no-skill line = prevalence)")

    rows = [base]

    # ------------------------------------------------ structured logistic
    logreg = load("model_logreg.pkl")
    if logreg is None:
        print("\nmodel_logreg.pkl missing. Run step04_train.py first.")
        return 1

    print("\n--- Structured logistic regression ---")
    scores_lr = ev.score_in_chunks(logreg, val)
    np.save(cfg.DATA / "scores_val_logreg.npy", scores_lr)

    rank_lr = ev.ranking_metrics(y, scores_lr)
    at_lr = ev.metrics_at_threshold(y, scores_lr, cfg.DECISION_THRESHOLD)
    print(f"  PR-AUC    {rank_lr['pr_auc']:.4f}  "
          f"(no-skill {rank_lr['no_skill_pr_auc']:.4f})")
    print(f"  ROC-AUC   {rank_lr['roc_auc']:.4f}")
    print(f"  at {cfg.DECISION_THRESHOLD}: precision {at_lr['precision']:.4f}  "
          f"recall {at_lr['recall']:.4f}  F1 {at_lr['f1']:.4f}")
    rows.append({"model": "logistic regression", **rank_lr, **at_lr})

    # ------------------------------------------------ CatBoost (Table 12)
    cat_scores = None
    try:
        from catboost import CatBoostClassifier
        cbm = cfg.DATA / "model_catboost.cbm"
        if cbm.exists():
            model = CatBoostClassifier()
            model.load_model(str(cbm))
            print("\n--- CatBoost ---")
            cat_scores = ev.score_in_chunks(model, val, as_string=True)
            np.save(cfg.DATA / "scores_val_catboost.npy", cat_scores)
            rank_cb = ev.ranking_metrics(y, cat_scores)
            at_cb = ev.metrics_at_threshold(y, cat_scores, cfg.DECISION_THRESHOLD)
            print(f"  PR-AUC    {rank_cb['pr_auc']:.4f}")
            print(f"  ROC-AUC   {rank_cb['roc_auc']:.4f}")
            print(f"  at {cfg.DECISION_THRESHOLD}: precision {at_cb['precision']:.4f}  "
                  f"recall {at_cb['recall']:.4f}  F1 {at_cb['f1']:.4f}")
            rows.append({"model": "catboost", **rank_cb, **at_cb})
    except ImportError:
        print("\ncatboost not installed - benchmark skipped.")

    pd.DataFrame(rows).to_csv(cfg.TABLES / "table11_12_validation.csv", index=False)

    # ------------------------------------------------ text model (Table 13)
    text = load("model_text.pkl")
    if text is not None:
        # Narratives live in a per-year file; merge only the validation year.
        nar = ev.load_narratives(cfg.VALIDATION_YEAR)
        sub = val.merge(nar, on=cfg.COL_ID, how="inner").reset_index(drop=True)
        del nar
        if len(sub) > 50:
            print(f"\n--- Narrative subset ({len(sub):,} records, "
                  f"{100*len(sub)/len(val):.2f}% of validation) ---")
            y_sub = sub["target"].values

            s_text = text.predict_proba(sub[cfg.COL_NARRATIVE].astype(str))[:, 1]
            s_struct = ev.score_in_chunks(logreg, sub)
            # Simple average ensemble as the "combined" column of Table 13.
            s_comb = (s_text + s_struct) / 2.0

            trows = []
            for label, s in [("structured (subset)", s_struct),
                             ("tfidf text", s_text),
                             ("combined", s_comb)]:
                r = ev.ranking_metrics(y_sub, s)
                a = ev.metrics_at_threshold(y_sub, s, cfg.DECISION_THRESHOLD)
                trows.append({"model": label, **r, **a})
                print(f"  {label:<22} PR-AUC {r['pr_auc']:.4f}  "
                      f"precision {a['precision']:.4f}  recall {a['recall']:.4f}")
            pd.DataFrame(trows).to_csv(
                cfg.TABLES / "table13_narrative_subset.csv", index=False)

    # ------------------------------------------------ threshold sweep (Table 14)
    print("\n--- Threshold sweep on validation (Table 14) ---")
    sweep = pd.DataFrame([
        ev.metrics_at_threshold(y, scores_lr, t) for t in cfg.THRESHOLD_SWEEP
    ])
    sweep.to_csv(cfg.TABLES / "table14_threshold_sweep.csv", index=False)
    print(sweep[["threshold", "recall", "precision", "flagged",
                 "flag_rate", "reviews_per_catch"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    chosen = sweep[sweep["threshold"] == cfg.DECISION_THRESHOLD]
    if not chosen.empty:
        c = chosen.iloc[0]
        print(f"\nLocked threshold {cfg.DECISION_THRESHOLD}: "
              f"flags {int(c['flagged']):,} complaints "
              f"({100*c['flag_rate']:.2f}% of the year), "
              f"catches {int(c['tp']):,} of {int(y.sum()):,} untimely, "
              f"{c['reviews_per_catch']:.1f} reviews per case identified.")

    print("\nStep 05 complete. Threshold is now frozen.")
    print("Next: python src/step06_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
