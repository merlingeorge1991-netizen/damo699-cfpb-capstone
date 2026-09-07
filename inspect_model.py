"""
inspect_model.py
----------------
Reads data/model_logreg.pkl and prints exactly how the pipeline was built, so
the ablation can reuse the real configuration instead of guessing at it.

    python inspect_model.py

Runs in under a minute.
"""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").is_dir():
    sys.exit(f"No 'data' folder next to this script. Script is at: {ROOT}")

print("loading model_logreg.pkl (imports scikit-learn, ~30s)...", flush=True)
with open(ROOT / "data/model_logreg.pkl", "rb") as f:
    obj = pickle.load(f)

print(f"\ntop-level object: {type(obj).__name__}")
model = obj.get("model", obj) if isinstance(obj, dict) else obj
if isinstance(obj, dict):
    print(f"dict keys: {list(obj.keys())}")

print(f"model type      : {type(model).__name__}")


def describe(name, est, indent="  "):
    print(f"{indent}{name}: {type(est).__name__}")
    for attr in ("min_frequency", "max_categories", "handle_unknown", "drop",
                 "sparse_output", "with_mean", "with_std", "categories",
                 "solver", "C", "penalty", "max_iter", "class_weight"):
        if hasattr(est, attr):
            v = getattr(est, attr)
            if attr == "categories" and not isinstance(v, str):
                v = f"<{len(v)} field(s)>" if hasattr(v, "__len__") else v
            print(f"{indent}    {attr} = {v}")
    if hasattr(est, "n_features_in_"):
        print(f"{indent}    n_features_in_ = {est.n_features_in_}")
    if hasattr(est, "get_feature_names_out"):
        try:
            print(f"{indent}    n_features_out = {len(est.get_feature_names_out())}")
        except Exception:
            pass


if hasattr(model, "named_steps"):
    print("\npipeline steps:")
    for name, step in model.named_steps.items():
        describe(name, step)
        if hasattr(step, "transformers_"):
            print("    column transformer blocks:")
            for tname, trans, cols in step.transformers_:
                ncols = len(cols) if hasattr(cols, "__len__") and not isinstance(cols, str) else cols
                print(f"      - {tname} on {ncols} column(s)")
                describe(tname, trans, indent="        ")
else:
    describe("estimator", model)

if hasattr(model, "coef_"):
    print(f"\nfinal coefficient count: {model.coef_.shape[1]:,}")
elif hasattr(model, "named_steps"):
    last = list(model.named_steps.values())[-1]
    if hasattr(last, "coef_"):
        print(f"\nfinal coefficient count: {last.coef_.shape[1]:,}")

print("""
Send this output back. The encoder settings above are what the ablation needs
in order to reproduce the baseline; guessing at min_frequency produced 254
features against the 971 the report states.""")
