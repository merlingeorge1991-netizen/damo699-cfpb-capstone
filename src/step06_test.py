"""
Step 06 - Final evaluation on the held-out test period.

Scores the complete test-year population ONCE at the threshold frozen in
step05. No tuning decision may be made after running this script; doing so
converts the test set into a second validation set and invalidates the
out-of-sample claim in report Section 5.5.

Produces report Tables 15 and 16 and the decile lift table.

Run:  python src/step06_test.py
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


def main() -> int:
    print("=" * 68)
    print(f"FINAL TEST EVALUATION - {cfg.TEST_YEAR}")
    print(f"Threshold frozen at {cfg.DECISION_THRESHOLD} (selected on "
          f"{cfg.VALIDATION_YEAR} validation)")
    print("=" * 68)

    df = pd.read_pickle(cfg.CLEAN_PKL)
    test = df[df["year"] == cfg.TEST_YEAR].reset_index(drop=True)
    if test.empty:
        print(f"No {cfg.TEST_YEAR} records found.")
        return 1

    y = test["target"].values
    print(f"\nTest population: {len(test):,} records, "
          f"{int(y.sum()):,} untimely ({100*y.mean():.3f}%)")

    with open(cfg.DATA / "model_logreg.pkl", "rb") as fh:
        model = pickle.load(fh)

    scores = ev.score_in_chunks(model, test)
    np.save(cfg.DATA / "scores_test_logreg.npy", scores)

    # ------------------------------------------------ baseline for contrast
    base = ev.baseline_metrics(y)
    print(f"\nMajority-class baseline accuracy: {base['accuracy']:.4f} "
          f"(identifies 0 of {base['untimely_total']:,} untimely)")

    # ------------------------------------------------ Table 15
    at = ev.metrics_at_threshold(y, scores, cfg.DECISION_THRESHOLD)
    print("\nTable 15 - confusion matrix")
    print(f"                     Predicted timely   Predicted untimely")
    print(f"  Actually timely    {at['tn']:>16,}   {at['fp']:>18,}")
    print(f"  Actually untimely  {at['fn']:>16,}   {at['tp']:>18,}")

    pd.DataFrame([{
        "": "Actually timely", "Predicted timely": at["tn"],
        "Predicted untimely": at["fp"]}, {
        "": "Actually untimely", "Predicted timely": at["fn"],
        "Predicted untimely": at["tp"]},
    ]).to_csv(cfg.TABLES / "table15_confusion_matrix.csv", index=False)

    # ------------------------------------------------ Table 16
    rank = ev.ranking_metrics(y, scores)
    print("\nTest metrics")
    print(f"  accuracy   {at['accuracy']:.4f}")
    print(f"  precision  {at['precision']:.4f}")
    print(f"  recall     {at['recall']:.4f}")
    print(f"  F1         {at['f1']:.4f}")
    print(f"  PR-AUC     {rank['pr_auc']:.4f}  "
          f"(no-skill {rank['no_skill_pr_auc']:.4f})")
    print(f"  ROC-AUC    {rank['roc_auc']:.4f}")
    print(f"  flag rate  {at['flag_rate']:.4f} "
          f"({at['flagged']:,} complaints flagged)")

    val_path = cfg.DATA / "scores_val_logreg.npy"
    comparison = None
    if val_path.exists():
        val = df[df["year"] == cfg.VALIDATION_YEAR].reset_index(drop=True)
        vs = np.load(val_path)
        vy = val["target"].values
        v_at = ev.metrics_at_threshold(vy, vs, cfg.DECISION_THRESHOLD)
        v_rank = ev.ranking_metrics(vy, vs)

        comparison = pd.DataFrame([
            {"metric": "Precision", "validation": v_at["precision"],
             "test": at["precision"]},
            {"metric": "Recall", "validation": v_at["recall"],
             "test": at["recall"]},
            {"metric": "F1", "validation": v_at["f1"], "test": at["f1"]},
            {"metric": "PR-AUC", "validation": v_rank["pr_auc"],
             "test": rank["pr_auc"]},
            {"metric": "ROC-AUC", "validation": v_rank["roc_auc"],
             "test": rank["roc_auc"]},
            {"metric": "Flag rate", "validation": v_at["flag_rate"],
             "test": at["flag_rate"]},
            {"metric": "Positive prevalence",
             "validation": v_rank["positive_prevalence"],
             "test": rank["positive_prevalence"]},
        ])
        comparison["change"] = comparison["test"] - comparison["validation"]
        comparison.to_csv(cfg.TABLES / "table16_val_vs_test.csv", index=False)
        print("\nTable 16 - validation vs test")
        print(comparison.to_string(index=False,
                                   float_format=lambda x: f"{x:.4f}"))
        print("\nNote for Section 6.6: precision moves with prevalence even when")
        print("ranking quality is unchanged. Compare the PR-AUC row against the")
        print("prevalence row before attributing any change to model drift.")

    # ------------------------------------------------ decile lift
    lift = ev.decile_lift(y, scores)
    lift.to_csv(cfg.TABLES / "decile_lift_test.csv", index=False)
    print("\nDecile lift (Section 6.8)")
    print(lift.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    top = lift.iloc[0]
    print(f"\nTop decile holds {top['cumulative_untimely_pct']:.1f}% of all "
          f"untimely complaints, a lift of {top['lift']:.2f}x over the base rate.")

    # ------------------------------------------------ summary for the report
    summary = pd.DataFrame([{
        "period": cfg.TEST_YEAR,
        "records": len(test),
        "untimely": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": cfg.DECISION_THRESHOLD,
        **{k: at[k] for k in ("precision", "recall", "f1", "accuracy",
                              "tp", "fp", "tn", "fn", "flagged",
                              "flag_rate", "reviews_per_catch")},
        "pr_auc": rank["pr_auc"],
        "roc_auc": rank["roc_auc"],
        "no_skill_pr_auc": rank["no_skill_pr_auc"],
        "baseline_accuracy": base["accuracy"],
    }])
    summary.to_csv(cfg.TABLES / "final_test_summary.csv", index=False)

    print("\nStep 06 complete. All tables in outputs/tables/.")
    print("Next: python src/step07_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
