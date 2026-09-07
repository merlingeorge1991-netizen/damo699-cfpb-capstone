from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> int:
    input_path = Path("scored_complaints.csv")
    if not input_path.exists():
        alt = ROOT / "data" / "scored_complaints.csv"
        if alt.exists():
            input_path = alt
        else:
            print("ERROR: scored_complaints.csv was not found.")
            print("Put the file in this project folder or in ./data/ and run again.")
            print(f"Expected paths: {ROOT / 'scored_complaints.csv'} or {alt}")
            return 1

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    required = {"pred_proba"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"ERROR: missing required column(s): {missing}")
        print(f"Available columns: {list(df.columns[:20])}")
        return 1

    scores = df["pred_proba"]
    if scores.nunique() <= 1:
        print("WARNING: pred_proba has only one unique value; assigning all rows to decile 1.")
        df["decile"] = 1
    else:
        df["decile"] = pd.qcut(scores, q=10, labels=False, duplicates="drop") + 1
        if df["decile"].isna().any():
            df["decile"] = pd.qcut(scores.rank(method="first"), q=10, labels=False, duplicates="drop") + 1

    output_path = input_path.with_name("scored_with_decile.csv")
    print(df.groupby("decile").size().to_string())
    if "actually_untimely" in df.columns:
        print(df.groupby("decile")["actually_untimely"].agg(["count", "mean"]).to_string())

    df.to_csv(output_path, index=False)
    print(f"Saved decile output to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())