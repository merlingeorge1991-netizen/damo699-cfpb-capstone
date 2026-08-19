"""
Step 03 - Build the enriched training sample.

Implements report Section 4.7 / 5.4: retain every untimely complaint from the
training years and draw NEGATIVE_PER_POSITIVE timely complaints for each.
Validation and test periods are deliberately left untouched at their natural
prevalence, so all reported performance is measured on real distributions.

Narratives are merged onto the sampled rows only, after sampling. Merging
before would pull several million narrative strings into memory to keep the
few thousand the training sample actually uses.

Run:  python src/step03_prepare.py
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def attach_narratives(sample: pd.DataFrame) -> pd.DataFrame:
    """Merge narrative text onto the sampled rows, one training year at a time."""
    wanted = set(sample.loc[sample["has_narrative"], cfg.COL_ID])
    if not wanted:
        sample[cfg.COL_NARRATIVE] = None
        return sample

    parts = []
    for year in cfg.TRAIN_YEARS:
        path = cfg.narrative_path(year)
        if not path.exists():
            continue
        nar = pd.read_pickle(path)
        parts.append(nar[nar[cfg.COL_ID].isin(wanted)])
        del nar
        gc.collect()

    if not parts:
        sample[cfg.COL_NARRATIVE] = None
        return sample

    merged = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    out = sample.merge(merged, on=cfg.COL_ID, how="left")
    print(f"  narratives attached: {out[cfg.COL_NARRATIVE].notna().sum():,}")
    return out


def main() -> int:
    if not cfg.CLEAN_PKL.exists():
        print(f"{cfg.CLEAN_PKL} not found. Run step02_audit.py first.")
        return 1

    df = pd.read_pickle(cfg.CLEAN_PKL)
    train_pool = df[df["year"].isin(cfg.TRAIN_YEARS)]

    positives = train_pool[train_pool["target"] == 1]
    negatives = train_pool[train_pool["target"] == 0]

    n_neg_wanted = len(positives) * cfg.NEGATIVE_PER_POSITIVE
    if n_neg_wanted > len(negatives):
        print(f"WARNING: requested {n_neg_wanted:,} timely cases but only "
              f"{len(negatives):,} available. Using all of them.")
        n_neg_wanted = len(negatives)

    sampled_neg = negatives.sample(n=n_neg_wanted, random_state=cfg.RANDOM_SEED)

    train = (
        pd.concat([positives, sampled_neg], ignore_index=True)
        .sample(frac=1.0, random_state=cfg.RANDOM_SEED)
        .reset_index(drop=True)
    )

    n_pos, n_neg = len(positives), len(sampled_neg)

    val_stats, test_stats = [], []
    for year, bucket in [(cfg.VALIDATION_YEAR, val_stats), (cfg.TEST_YEAR, test_stats)]:
        sub = df[df["year"] == year]
        if len(sub):
            bucket.append((year, len(sub), int(sub["target"].sum()),
                           100 * sub["target"].mean()))

    del df, train_pool, positives, negatives, sampled_neg
    gc.collect()

    train = attach_narratives(train)
    train.to_pickle(cfg.TRAIN_PKL)

    rate = 100 * train["target"].mean()
    print("\nTraining sample (report Table 8)")
    print(f"  training years      : {cfg.TRAIN_YEARS}")
    print(f"  untimely (class 1)  : {n_pos:,}")
    print(f"  timely   (class 0)  : {n_neg:,}")
    print(f"  total observations  : {len(train):,}")
    print(f"  untimely rate       : {rate:.2f}%")

    for label, bucket in [("validation", val_stats), ("test", test_stats)]:
        for year, records, untimely, pct in bucket:
            print(f"  {label:<10} {year}    : {records:,} records, "
                  f"{untimely:,} untimely ({pct:.2f}%)")

    pd.DataFrame([{
        "period": "train", "years": str(cfg.TRAIN_YEARS),
        "records": len(train), "untimely": n_pos,
        "untimely_rate_pct": rate,
        "sampling": f"all positives + {cfg.NEGATIVE_PER_POSITIVE} negatives each",
    }]).to_csv(cfg.TABLES / "table09_split_design.csv", index=False)

    print(f"\nWritten to {cfg.TRAIN_PKL}")
    print("Step 03 complete. Next: python src/step04_train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
