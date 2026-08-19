"""
Shared scoring and metric helpers used by steps 05 and 06.

Scoring is chunked because the test period contains millions of rows. Crucially
the categorical preparation happens INSIDE each chunk: converting 5.4M rows of
category dtype back to strings in one pass is exactly the kind of full-frame
materialisation that exhausts memory on a laptop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def prep_frame(df: pd.DataFrame, as_string: bool = False) -> pd.DataFrame:
    """
    Fill categorical NaNs with an explicit level (Section 5.3).

    Category dtype needs the fill value added as a category first, otherwise
    fillna raises. Pass as_string=True for CatBoost, which wants plain strings
    in its cat_features columns.
    """
    out = df.copy()
    for col in cfg.CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "MISSING"
            continue
        s = out[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if "MISSING" not in s.cat.categories:
                s = s.cat.add_categories(["MISSING"])
            s = s.fillna("MISSING")
            if as_string:
                s = s.astype(str)
        else:
            s = s.fillna("MISSING").astype(str)
        out[col] = s
    return out


def score_in_chunks(model, df: pd.DataFrame, chunk: int = 250_000,
                    as_string: bool = False) -> np.ndarray:
    """Return P(untimely) for every row, preparing and scoring slice by slice."""
    cols = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
    frame = df[cols]
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk):
        end = min(start + chunk, len(frame))
        piece = prep_frame(frame.iloc[start:end], as_string=as_string)
        out[start:end] = model.predict_proba(piece)[:, 1]
        del piece
        pct = 100.0 * end / len(frame)
        print(f"\r    scoring {end:,}/{len(frame):,} ({pct:5.1f}%)", end="")
    print()
    return out


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": threshold, "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / n,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "flagged": int(tp + fp), "flag_rate": (tp + fp) / n,
        "reviews_per_catch": (tp + fp) / tp if tp else float("nan"),
    }


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-independent summary. PR-AUC is the primary metric."""
    prevalence = float(np.mean(y_true))
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "no_skill_pr_auc": prevalence,
        "positive_prevalence": prevalence,
        "n": len(y_true),
        "n_positive": int(np.sum(y_true)),
    }


def decile_lift(y_true, scores) -> pd.DataFrame:
    """
    Concentration of untimely cases by score decile.

    This is the diagnostic that matters operationally: prioritisation depends on
    whether the positives cluster at the top of the ranking, not on calibration.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(scores)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) // max(1, len(df) // 10)).clip(0, 9) + 1

    base = df["y"].mean()
    out = (
        df.groupby("decile")
        .agg(records=("y", "size"), untimely=("y", "sum"))
        .reset_index()
    )
    out["rate"] = out["untimely"] / out["records"]
    out["lift"] = out["rate"] / base if base else np.nan
    out["cumulative_untimely_pct"] = (
        100 * out["untimely"].cumsum() / out["untimely"].sum()
    )
    return out


def baseline_metrics(y_true) -> dict:
    """Majority-class classifier: predict timely for everything."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    pos = int(y_true.sum())
    return {
        "model": "majority-class baseline",
        "accuracy": (n - pos) / n, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "pr_auc": pos / n, "roc_auc": 0.5,
        "untimely_identified": 0, "untimely_total": pos,
    }


def load_narratives(year: int) -> pd.DataFrame:
    """Load one year of narrative text, or an empty frame if none was written."""
    path = cfg.narrative_path(year)
    if not path.exists():
        return pd.DataFrame(columns=[cfg.COL_ID, cfg.COL_NARRATIVE])
    return pd.read_pickle(path)
