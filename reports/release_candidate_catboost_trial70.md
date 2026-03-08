# Release Candidate: CatBoost trial 70 (severity-only)

- Date: 2026-03-08
- Source study: `catboost_tuning_stability_severity_only_20260304_140253`
- Trial: `70`
- Objective (`stability_adjusted`): `0.482271379678`
- Policy score mean/std: `0.484483448892` / `0.006320197754`
- Constraints pass rate: `1.00`
- In-target rate: `1.00`
- Splits: `5`

## Decision

Selected as release candidate because it is currently the best full 5-split stability-adjusted result with 100% constraints pass and 100% in-target rate.

## Frozen artifacts

- Config: `configs/release_catboost_trial70_threads4.json`
- Params snapshot: `artifacts/tuning/catboost/20260308_release_trial70/best_params_trial70_severity_only.json`

## Next run command

```bash
python scripts/run_pipeline.py --config configs/release_catboost_trial70_threads4.json
```
