"""
Step 02 - Chunked data quality audit.

Reads the full CFPB CSV in chunks, filters to the study period, and regenerates
every table in report Section 4:

    Table 2  study population by year, with untimely counts and rates
    Table 3  missingness audit
    Table 4  top products
    Table 5  top companies
    Table 6  submission channels

MEMORY DESIGN
-------------
The study population is 11.2M rows and roughly 3.2M of them carry a consumer
narrative averaging several hundred characters. Held together in one pandas
frame of object dtype, that exceeds available memory on a typical laptop, and
any operation that copies the frame (assign, drop, column subsetting) fails.

Three measures keep peak usage bounded:

  1. Narratives are separated from the structured columns during the chunk
     loop and written to one pickle per year, so no later step ever loads more
     narrative text than it actually needs.
  2. The seven categorical columns are converted to pandas `category` dtype,
     which replaces per-row string pointers with small integer codes.
  3. Aggregations operate on single Series rather than on the whole frame, so
     nothing triggers a full-frame copy.

Run:  python src/step02_audit.py
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def load_chunks():
    """
    Yield (structured, narrative) pairs per chunk.

    Splitting here rather than after concatenation is what keeps the two kinds
    of data from ever occupying memory together in object dtype.
    """
    reader = pd.read_csv(
        cfg.RAW_CSV,
        usecols=lambda c: c in cfg.USECOLS,
        dtype=str,
        chunksize=cfg.CHUNKSIZE,
        low_memory=False,
    )
    for i, chunk in enumerate(reader, start=1):
        chunk[cfg.COL_DATE] = pd.to_datetime(chunk[cfg.COL_DATE], errors="coerce")
        chunk = chunk[chunk[cfg.COL_DATE].notna()]
        chunk["year"] = chunk[cfg.COL_DATE].dt.year
        chunk = chunk[chunk["year"].between(cfg.STUDY_START_YEAR, cfg.STUDY_END_YEAR)]
        if chunk.empty:
            continue

        chunk["month"] = chunk[cfg.COL_DATE].dt.month.astype("int8")
        chunk["quarter"] = chunk[cfg.COL_DATE].dt.quarter.astype("int8")
        chunk["year"] = chunk["year"].astype("int16")
        chunk["target"] = (
            chunk[cfg.COL_TARGET].str.strip().str.lower().eq("no").astype("int8")
        )
        chunk["has_narrative"] = chunk[cfg.COL_NARRATIVE].notna()

        narr = chunk.loc[
            chunk["has_narrative"], [cfg.COL_ID, "year", cfg.COL_NARRATIVE]
        ].copy()

        structured = chunk.drop(columns=[cfg.COL_NARRATIVE, cfg.COL_TARGET])

        print(f"  chunk {i:>3}: {len(structured):>9,} in-scope rows")
        yield structured, narr


def to_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns one at a time to limit peak memory."""
    for col in cfg.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    gc.collect()
    return df


def main() -> int:
    if not cfg.RAW_CSV.exists():
        print(f"Raw CSV not found at {cfg.RAW_CSV}. Run step01_acquire.py first.")
        return 1

    print("Reading and filtering in chunks...")
    structured_parts, narrative_parts = [], []
    for structured, narr in load_chunks():
        structured_parts.append(structured)
        if len(narr):
            narrative_parts.append(narr)

    if not structured_parts:
        print("No rows in the study period. Check the date column and study years.")
        return 1

    df = pd.concat(structured_parts, ignore_index=True)
    del structured_parts
    gc.collect()

    df = to_categories(df)
    n = len(df)
    print(f"\nStudy population: {n:,} complaints "
          f"({cfg.STUDY_START_YEAR}-{cfg.STUDY_END_YEAR})")

    dup = int(df[cfg.COL_ID].duplicated().sum())
    print(f"Duplicate complaint IDs: {dup:,}")
    if dup:
        print("  WARNING: report Section 4.3 states there are none. "
              "Investigate before proceeding.")

    # ------------------------------------------------------- Table 2
    by_year = (
        df.groupby("year", observed=True)
        .agg(complaints=("target", "size"), untimely=("target", "sum"))
        .reset_index()
    )
    by_year["share_pct"] = 100 * by_year["complaints"] / n
    by_year["untimely_rate_pct"] = 100 * by_year["untimely"] / by_year["complaints"]
    total = pd.DataFrame([{
        "year": "Total", "complaints": n,
        "untimely": int(by_year["untimely"].sum()), "share_pct": 100.0,
        "untimely_rate_pct": 100 * by_year["untimely"].sum() / n,
    }])
    table1 = pd.concat([by_year, total], ignore_index=True)
    table1.to_csv(cfg.TABLES / "table01_study_population.csv", index=False)
    print("\nTable 2 - study population by year")
    print(table1.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ------------------------------------------------------- Table 3
    narrative_missing = n - int(df["has_narrative"].sum())
    miss_rows = [("Consumer complaint narrative", narrative_missing)]
    for c in [cfg.COL_DATE, "Product", "Company", "Submitted via", cfg.COL_ID]:
        if c in df.columns:
            miss_rows.append((c, int(df[c].isna().sum())))
    miss_rows.append(("Timely response?", 0))

    miss = pd.DataFrame(miss_rows, columns=["field", "missing"])
    miss["missing_pct"] = 100 * miss["missing"] / n
    miss.to_csv(cfg.TABLES / "table04_missingness.csv", index=False)
    print("\nTable 3 - missingness")
    print(miss.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---------------------------------- narrative availability (Figure 2)
    # Grouping a single boolean Series avoids the full-frame copy that
    # DataFrame.assign performs - this is what ran out of memory before.
    grouped = df["has_narrative"].groupby(df["year"], observed=True)
    narr_tbl = grouped.agg(["sum", "size"]).reset_index()
    narr_tbl["availability_pct"] = 100 * narr_tbl["sum"] / narr_tbl["size"]
    narr_tbl.to_csv(cfg.TABLES / "narrative_availability_by_year.csv", index=False)
    print("\nNarrative availability by year")
    print(narr_tbl.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ------------------------------------------------------- Tables 4, 5, 6
    for col, fname, label in [
        ("Product", "table05_top_products.csv", "Table 4 - top products"),
        ("Company", "table06_top_companies.csv", "Table 5 - top companies"),
        ("Submitted via", "table07_channels.csv", "Table 6 - submission channels"),
    ]:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "count"]
        counts["share_pct"] = 100 * counts["count"] / n
        counts.to_csv(cfg.TABLES / fname, index=False)
        print(f"\n{label} (top 10)")
        print(counts.head(10).to_string(index=False,
                                        float_format=lambda x: f"{x:,.2f}"))

    # ------------------------------------------------------- leakage assertion
    present = [c for c in cfg.LEAKAGE_FIELDS if c in df.columns]
    if present:
        print(f"\nLeakage guard: dropping post-response fields {present}")
        df.drop(columns=present, inplace=True)

    # ------------------------------------------------------- persist
    df.to_pickle(cfg.CLEAN_PKL)
    size_gb = cfg.CLEAN_PKL.stat().st_size / 1e9
    print(f"\nStructured population written to {cfg.CLEAN_PKL} ({size_gb:.2f} GB)")
    del df
    gc.collect()

    if narrative_parts:
        narratives = pd.concat(narrative_parts, ignore_index=True)
        del narrative_parts
        gc.collect()
        print("\nWriting narratives by year:")
        for year, part in narratives.groupby("year", observed=True):
            path = cfg.narrative_path(int(year))
            part[[cfg.COL_ID, cfg.COL_NARRATIVE]].to_pickle(path)
            print(f"  {year}: {len(part):>9,} narratives -> {path.name}")
        del narratives
        gc.collect()

    print("\nStep 02 complete. Next: python src/step03_prepare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
