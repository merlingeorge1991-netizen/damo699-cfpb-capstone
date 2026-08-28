\# Ablation: removing the `year` feature



Run 2026-08-28 on the full CFPB snapshot (9.15 GB, 5.44M test-year records).



`year` carried 26.0% of CatBoost feature importance, which is a concern on a

temporally split problem: the test year (2025) falls outside the training range,

so any level effect learned from `year` cannot extrapolate.



Removing `year` from `config.NUMERIC\_FEATURES` (keeping `month` and `quarter`,

which are cyclical and do transfer) changes validation ranking almost not at all:



| metric | with `year` | without `year` |

|---|---|---|

| LR ROC-AUC | 0.98465 | 0.98458 |

| CatBoost ROC-AUC | 0.98906 | 0.98770 |

| LR PR-AUC | 0.2431 | 0.2351 |

| CatBoost PR-AUC | 0.2623 | 0.2366 |



Feature importance measures split frequency, not contribution to generalization.

The simpler model is preferred: equal ranking quality, no non-extrapolable feature.



\*\*Caveat.\*\* This comparison is validation-only. The 2025 test year was scored once,

with the `year`-inclusive model, and was not rescored. All reported test metrics

(Tables 15, 16, decile lift, final\_test\_summary) come from that original run.

Baseline table retained as `outputs/tables/ablation\_validation\_with\_year.csv`.

