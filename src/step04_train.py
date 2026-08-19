"""
Step 04 - Train the three model streams.

    A. Structured logistic regression  (principal model, full population)
    B. CatBoost                        (nonlinear benchmark, same features)
    C. TF-IDF + logistic regression    (narrative-available subset only)

The majority-class baseline needs no fitting and is computed in step06.

Models are pickled to data/ so scoring and evaluation can be re-run without
retraining.

Run:  python src/step04_train.py
"""

import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def build_structured_pipeline() -> Pipeline:
    """
    Encoder + classifier as one object.

    handle_unknown="ignore" is the mechanism described in Section 5.3: a company
    or issue value that first appears in 2025 produces an all-zero block rather
    than an error, so every complaint still receives a score.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  min_frequency=10,
                                  sparse_output=True),
             cfg.CATEGORICAL_FEATURES),
            ("num", StandardScaler(), cfg.NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(**cfg.LOGREG_PARAMS)),
    ])


sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import prep_frame  # noqa: E402  (category-aware NaN handling)


def train_structured(train: pd.DataFrame):
    print("\n=== A. Structured logistic regression ===")
    X = prep_frame(train)[cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES]
    y = train["target"]

    pipe = build_structured_pipeline()
    t0 = time.time()
    pipe.fit(X, y)
    print(f"  fitted in {time.time()-t0:.1f}s")

    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    print(f"  features: {len(names):,}")

    coef_df = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs(),
                direction=lambda d: d["coefficient"].apply(
                    lambda v: "increases untimely risk" if v > 0
                    else "decreases untimely risk"))
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    coef_df.to_csv(cfg.TABLES / "table17_logreg_coefficients.csv", index=False)
    print("\n  Top 10 coefficients by magnitude (report Table 16):")
    print(coef_df.head(10)[["feature", "coefficient", "direction"]]
          .to_string(index=False))

    with open(cfg.DATA / "model_logreg.pkl", "wb") as fh:
        pickle.dump(pipe, fh)
    return pipe


def train_catboost(train: pd.DataFrame):
    print("\n=== B. CatBoost benchmark ===")
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        print("  catboost not installed - skipping this stream.")
        print("  Install with: pip install catboost")
        return None

    df = prep_frame(train, as_string=True)
    X = df[cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES]
    y = df["target"]

    model = CatBoostClassifier(**cfg.CATBOOST_PARAMS)
    t0 = time.time()
    model.fit(Pool(X, y, cat_features=cfg.CATEGORICAL_FEATURES))
    print(f"  fitted in {time.time()-t0:.1f}s")

    imp = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(cfg.TABLES / "catboost_feature_importance.csv", index=False)
    print("\n  Feature importance:")
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    model.save_model(str(cfg.DATA / "model_catboost.cbm"))
    return model


def train_text(train: pd.DataFrame):
    print("\n=== C. TF-IDF text model (narrative subset) ===")
    if cfg.COL_NARRATIVE not in train.columns:
        print("  no narrative column in the training sample - skipping.")
        return None
    sub = train[train[cfg.COL_NARRATIVE].notna()].copy()
    if len(sub) < 100:
        print(f"  only {len(sub)} narratives in training sample - skipping.")
        return None

    print(f"  narratives available: {len(sub):,} "
          f"({100*len(sub)/len(train):.2f}% of training sample)")
    print(f"  untimely rate in subset: {100*sub['target'].mean():.2f}%")

    vec = TfidfVectorizer(stop_words=cfg.REDACTION_STOPWORDS, **cfg.TFIDF_PARAMS)
    pipe = Pipeline([
        ("tfidf", vec),
        ("clf", LogisticRegression(**cfg.LOGREG_PARAMS)),
    ])
    t0 = time.time()
    pipe.fit(sub[cfg.COL_NARRATIVE].astype(str), sub["target"])
    print(f"  fitted in {time.time()-t0:.1f}s")

    terms = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    tdf = (
        pd.DataFrame({"term": terms, "coefficient": coefs})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    tdf.to_csv(cfg.TABLES / "tfidf_coefficients.csv", index=False)
    print("\n  Terms most associated with UNTIMELY:")
    print(tdf.head(15).to_string(index=False))
    print("\n  Terms most associated with TIMELY:")
    print(tdf.tail(15).to_string(index=False))

    with open(cfg.DATA / "model_text.pkl", "wb") as fh:
        pickle.dump(pipe, fh)
    return pipe


def main() -> int:
    if not cfg.TRAIN_PKL.exists():
        print(f"{cfg.TRAIN_PKL} not found. Run step03_prepare.py first.")
        return 1

    train = pd.read_pickle(cfg.TRAIN_PKL)
    print(f"Training sample: {len(train):,} rows, "
          f"{100*train['target'].mean():.2f}% untimely")

    train_structured(train)
    train_catboost(train)
    train_text(train)

    print("\nStep 04 complete. Next: python src/step05_threshold.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
