# Iteration Tracker

## Goal
- Primary metric: `policy_score`
- Hard constraints: `violations = 0`, `lr_total` in `0.69..0.71`
- Current milestone target: `+0.03` vs current production baseline

## Baseline Snapshot
- Run ID: `20260220_171846`
- Winner: `catboost_freq_sev`
- `policy_score`: `0.4877901493`
- `AUC`: `0.7212097897`
- `Gini`: `0.4424195794`
- `lr_total`: `0.6911251494`
- `in_target`: `true`

## Experiment Log
| Timestamp UTC | Stage | Candidate | Config/Params | policy_score | lr_total | violations | in_target | Decision |
|---|---|---|---|---:|---:|---:|---|---|
| 20260225_115943 | stability_check | catboost_freq_sev | params=artifacts/tuning/catboost/best_params_catboost_tuning_v1.json; splits=configs/experiments/stability_splits.json | 0.4698063375 (std=0.0259856246) | 0.6980113073 | 0 | 1.00 | PASS: all stability gates satisfied |
| 20260226_080110 | stability_check | catboost_freq_sev | params=artifacts/tuning/catboost/best_params_catboost_tuning_balanced_v3.json; splits=configs/experiments/stability_splits.json | 0.4063787604 (std=0.0410992250) | 0.7004813006 | 0 | 1.00 | REJECT: std_policy_score above gate (0.0411 > 0.03) |
| 20260302_064948 | full_pipeline | catboost_freq_sev | config=configs/default_with_tuned_catboost.json; params=artifacts/tuning/catboost/20260301_094709/best_params.json | 0.3581395718 | 0.7001672655 | 0 | true | REJECT: large policy_score drop and near-zero share_group1 (0.00094) despite valid constraints |
| 20260302_085309 | full_pipeline | catboost_freq_sev | config=configs/default_with_tuned_catboost.json; params=artifacts/tuning/catboost/best_params_catboost_tuning_v1.json | 0.4572403210 | 0.6925828549 | 0 | true | PASS: champion restored; constraints satisfied and group mix normalized |

## Stability Gate
- Target pass-rate for `in_target`: `>= 0.80`
- Target pass-rate for constraints: `1.00`
- Target std(`policy_score`): `<= 0.03`
- Source of truth: `artifacts/stability/<timestamp>/summary.json`

## Notes
- Fill this file after each tuning/stability cycle.
- Keep only decision-relevant details.
