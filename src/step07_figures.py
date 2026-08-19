"""
Step 07 - Generate every figure the report needs.

Writes 300 dpi PNGs to outputs/figures/, numbered to match the report:

    fig01  complaints by year                (Section 4.2)
    fig02  narrative availability by year    (Section 4.3)
    fig03  untimely rate by year             (Section 4.3)
    fig04  precision/recall vs threshold     (Section 6.5)
    fig05  calibration curve                 (Section 6.8)
    fig06  precision-recall curves           (Appendix D)
    fig07  ROC curves                        (Appendix D)
    fig08  score distribution by class       (Appendix D)
    fig09  cumulative gains                  (Appendix D)

Run:  python src/step07_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
NAVY, ORANGE, GREY = "#1F3864", "#C55A11", "#808080"


def save(fig, name):
    path = cfg.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def audit_figures():
    t1 = cfg.TABLES / "table01_study_population.csv"
    if not t1.exists():
        print("  audit tables missing - run step02_audit.py")
        return
    df = pd.read_csv(t1)
    df = df[df["year"].astype(str) != "Total"]
    df["year"] = df["year"].astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["year"], df["complaints"], color=NAVY)
    ax.set_xlabel("Year")
    ax.set_ylabel("Complaints")
    ax.set_title("Figure 1: CFPB complaints in the study population")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v/1e6:.1f}M"))
    save(fig, "fig01_complaints_by_year.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["year"], df["untimely_rate_pct"], marker="o", color=ORANGE)
    ax.set_xlabel("Year")
    ax.set_ylabel("Untimely response rate (%)")
    ax.set_title("Figure 3: Untimely response rate by year")
    ax.set_ylim(bottom=0)
    save(fig, "fig03_untimely_rate.png")

    nav = cfg.TABLES / "narrative_availability_by_year.csv"
    if nav.exists():
        n = pd.read_csv(nav)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(n["year"], n["availability_pct"], marker="s", color=NAVY)
        ax.set_xlabel("Year")
        ax.set_ylabel("Narrative available (%)")
        ax.set_title("Figure 2: Consumer narrative availability by year")
        ax.set_ylim(bottom=0)
        save(fig, "fig02_narrative_availability.png")


def threshold_figure():
    path = cfg.TABLES / "table14_threshold_sweep.csv"
    if not path.exists():
        print("  threshold sweep missing - run step05_threshold.py")
        return
    s = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s["threshold"], s["recall"], marker="o", color=NAVY, label="Recall")
    ax.plot(s["threshold"], s["precision"], marker="s", color=ORANGE,
            label="Precision")
    ax.axvline(cfg.DECISION_THRESHOLD, color=GREY, linestyle="--",
               label=f"Selected ({cfg.DECISION_THRESHOLD})")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Figure 4: Precision and recall vs threshold (validation)")
    ax.legend()
    save(fig, "fig04_threshold_tradeoff.png")


def _load_test():
    sp = cfg.DATA / "scores_test_logreg.npy"
    if not sp.exists():
        return None, None
    scores = np.load(sp)
    df = pd.read_pickle(cfg.CLEAN_PKL)
    y = df[df["year"] == cfg.TEST_YEAR]["target"].values
    if len(y) != len(scores):
        print("  score/label length mismatch - re-run step06_test.py")
        return None, None
    return y, scores


def performance_figures():
    y, scores = _load_test()
    if y is None:
        print("  test scores missing - run step06_test.py")
        return

    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color=NAVY, label="Model")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed untimely fraction")
    ax.set_title("Figure 5: Calibration curve (test period)")
    ax.legend()
    save(fig, "fig05_calibration.png")

    prec, rec, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rec, prec, color=NAVY, label="Logistic regression")
    ax.axhline(y.mean(), color=GREY, linestyle="--",
               label=f"No skill ({y.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Figure 6: Precision-recall curve (test period)")
    ax.legend()
    save(fig, "fig06_precision_recall.png")

    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=NAVY, label="Logistic regression")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="No skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 7: ROC curve (test period)")
    ax.legend()
    save(fig, "fig07_roc.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 51)
    ax.hist(scores[y == 0], bins=bins, alpha=0.6, color=NAVY,
            label="Timely", density=True)
    ax.hist(scores[y == 1], bins=bins, alpha=0.6, color=ORANGE,
            label="Untimely", density=True)
    ax.axvline(cfg.DECISION_THRESHOLD, color="black", linestyle="--",
               label=f"Threshold ({cfg.DECISION_THRESHOLD})")
    ax.set_xlabel("Model score")
    ax.set_ylabel("Density")
    ax.set_title("Figure 8: Score distribution by actual class")
    ax.legend()
    save(fig, "fig08_score_distribution.png")

    lift_path = cfg.TABLES / "decile_lift_test.csv"
    if lift_path.exists():
        lift = pd.read_csv(lift_path)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = lift["decile"] * 10
        ax.plot(x, lift["cumulative_untimely_pct"], marker="o", color=NAVY,
                label="Model")
        ax.plot([0, 100], [0, 100], "--", color=GREY, label="Random")
        ax.set_xlabel("Percentage of complaints reviewed (highest score first)")
        ax.set_ylabel("Percentage of untimely complaints captured")
        ax.set_title("Figure 9: Cumulative gains (test period)")
        ax.legend()
        save(fig, "fig09_cumulative_gains.png")


def main() -> int:
    print("Generating figures...")
    audit_figures()
    threshold_figure()
    performance_figures()
    print(f"\nAll figures in {cfg.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
