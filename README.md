# Risk Case Championship Architecture Scaffold

This repository now contains an executable architecture scaffold for the OGPO
risk case:

- deterministic data pipeline with validation gates
- baseline frequency-severity model
- multi-model benchmark (baseline + xgboost + lightgbm + catboost + OOF blend) with auto winner selection
- pricing policy optimizer with business constraints
- experiment run artifacts and leaderboard
- lightweight agent tool contracts
- FastAPI demo service
- unit + e2e tests via `unittest`

## Quick start

Run an end-to-end experiment:

```bash
python scripts/run_pipeline.py --config configs/default.json
```

Each run writes step-by-step logs both to console and to:

`artifacts/runs/<run_id>/pipeline.log`

With `benchmark.enabled=true`, each run also writes:

- `artifacts/runs/<run_id>/benchmark/results.csv`
- `artifacts/runs/<run_id>/benchmark/results.json`
- `artifacts/runs/<run_id>/benchmark/winner.json`
- `artifacts/runs/<run_id>/benchmark/failed_candidates.json`

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run full EDA:

```bash
python scripts/run_eda.py --config configs/eda.json
```

Run CatBoost hyperparameter tuning (Optuna, business-metric objective):

```bash
python scripts/tune_catboost.py --config configs/default.json --n_trials 240
```

Run CatBoost tuning in explicit phase mode:

```bash
python scripts/tune_catboost.py --config configs/default.json --phase coarse --n_trials 160
python scripts/tune_catboost.py --config configs/default.json --phase fine --n_trials 80
```

Run CatBoost tuning in severity-only mode (fixed frequency baseline, tune severity block only):

```bash
python scripts/tune_catboost.py --config configs/default.json --phase focused --mode severity_only --n_trials 80
```

Recommended stability-first training flow (current project setup):

1. Long stability-adjusted CatBoost tuning (5 time splits, hard business gates):

```bash
python scripts/tune_catboost.py --config configs/default.json --phase fine --n_trials 60 --timeout_sec 43200 --study_name catboost_tuning_stability_full_YYYYMMDD --storage catboost_tuning.db --objective_mode stability_adjusted --splits_config configs/experiments/stability_splits.json --stability_lambda 0.35 --stability_min_constraints_rate 1.0 --stability_min_in_target_rate 0.8 --catboost_threads 4
```

2. Validate best params on the same 5-split stability protocol:

```bash
python scripts/run_stability_checks.py --config configs/default.json --splits_config configs/experiments/stability_splits.json --candidate catboost_freq_sev --params_json artifacts/tuning/catboost/<timestamp>/best_params.json --catboost_threads 4
```

Optional: set the objective lambda used in stability summary (`objective_value = mean_policy_score - lambda * std_policy_score`):

```bash
python scripts/run_stability_checks.py --config configs/default.json --splits_config configs/experiments/stability_splits.json --objective_lambda 0.35
```

3. Apply approved params to `configs/default_with_tuned_catboost.json`, then run full pipeline:

```bash
python scripts/run_pipeline.py --config configs/default_with_tuned_catboost.json
```

4. OOF orchestration candidate (enabled by default in active configs):

- Keep/adjust params in `benchmark.candidate_params.oof_blend_freq_sev`:
  - `base_candidates` (recommended: `["catboost_freq_sev", "xgboost_freq_sev", "lightgbm_freq_sev"]`)
  - `oof_folds`
  - `oof_group_column`
  - `weight_grid_step`

5. Review outputs:

- tuning summary: `artifacts/tuning/catboost/<timestamp>/summary.json`
- stability summary: `artifacts/stability/<timestamp>/summary.json`
- full run metrics: `artifacts/runs/<run_id>/metrics.json`

If you are using the local virtual environment on Windows PowerShell, run the
same commands with:

```bash
.\.venv\Scripts\python.exe <command>
```

Examples on Windows PowerShell:

```bash
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\default_with_tuned_catboost.json
.\.venv\Scripts\python.exe scripts\tune_catboost.py --config configs\default.json --phase fine --n_trials 60
.\.venv\Scripts\python.exe scripts\tune_catboost.py --config configs\default.json --phase focused --mode severity_only --n_trials 80
.\.venv\Scripts\python.exe scripts\run_stability_checks.py --config configs\default.json --splits_config configs\experiments\stability_splits.json
```

Run stability checks on multiple time holdout splits:

```bash
python scripts/run_stability_checks.py --config configs/default.json --splits_config configs/experiments/stability_splits.json
```

Google Colab GPU run:

```bash
git clone <your-repo-url>
cd <repo-dir>
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/colab_gpu.json
```

Google Colab GPU stability run with resume:

```bash
python scripts/run_stability_checks.py --config configs/colab_gpu.json --splits_config configs/experiments/stability_splits.json --candidate oof_blend_freq_sev
python scripts/run_stability_checks.py --config configs/colab_gpu.json --splits_config configs/experiments/stability_splits.json --candidate oof_blend_freq_sev --resume_from artifacts/stability/<timestamp>/oof_blend_checkpoint
```

Notes for Colab:

- `configs/colab_gpu.json` enables GPU for `catboost` and `xgboost`.
- The Colab config intentionally excludes `lightgbm_freq_sev` because the default Colab LightGBM wheel is often CPU-only.
- The blend weight search and pricing evaluation still run on CPU, so GPU helps most on model fitting, not on every stage.

Start API (after at least one pipeline run):

```bash
uvicorn risk_case.api.main:app --app-dir src --reload
```

## Project layout

- `src/risk_case/data`: loading + validation contracts
- `src/risk_case/features`: feature schema and transformers
- `src/risk_case/models`: frequency-severity baseline and ML metrics
- `src/risk_case/pricing`: pricing function + portfolio KPI evaluation
- `src/risk_case/orchestration`: run manager and artifact tracking
- `src/risk_case/agent`: JSON contracts and safe tool stubs
- `src/risk_case/api`: scoring/repricing/metrics endpoints
- `configs/default.json`: default run configuration
- `configs/eda.json`: full EDA configuration
- `tests/`: quality gates and e2e scenario

## EDA outputs

`scripts/run_eda.py` generates:

- `notebooks/03_eda_full_ml_ready.ipynb` (interactive analysis notebook)
- `reports/eda_summary.md` (compact findings)
- `artifacts/eda/tables` (CSV diagnostics)
- `artifacts/eda/figures` (EDA plots)
- `artifacts/eda/metadata/eda_profile.json` (key metrics)
- `artifacts/eda/feature_selection/feature_whitelist.csv`
- `artifacts/eda/feature_selection/feature_droplist.csv`
- `artifacts/eda/feature_selection/feature_review_list.csv`
- `reports/feature_selection_report.md`

## Logging

- Configure log level via `logging.level` in `configs/default.json`.
- Pipeline logs include stage boundaries (`START/END`) and timings for:
  - ingest
  - validation
  - preprocessing
  - train/metrics
  - pricing
  - artifact persistence
  - submission generation

## Feature Selection Integration

`configs/default.json` now supports EDA-driven feature lists:

- `preprocessing.feature_whitelist_path`
- `preprocessing.feature_droplist_path`
- `preprocessing.selection_rules.force_keep`
- `preprocessing.selection_rules.force_drop`
- `preprocessing.feature_pruning` (`enabled`, `drop_exact_duplicates`, `drop_missing_share`, `corr_threshold`)
- `preprocessing.drift_pruning` (`enabled`, `time_column`, `reference_share`, `psi_threshold`, `bins`, `min_rows`, `exclude_columns`, `exclude_patterns`)
- `preprocessing.interaction_features_mvp` (`enabled`, `definitions`, `max_features`, `corr_filter_threshold`, `psi_filter_threshold`, `require_business_whitelist`)

Recommended workflow:

1. `python scripts/run_eda.py --config configs/eda.json`
2. `python scripts/run_pipeline.py --config configs/default.json`

## Benchmark configuration

`configs/default.json` supports benchmark controls:

- `benchmark.enabled`
- `benchmark.candidates`
- `benchmark.selection_metric`
- `benchmark.constraints`
- `benchmark.fallback_strategy`
- `benchmark.fallback_candidate`
- `benchmark.random_state`
- `benchmark.calibration` (`enabled`, `method`, `oof_folds`, `group_column`, `min_samples`, `clip_eps`)
- `benchmark.candidate_params.oof_blend_freq_sev` (OOF blend orchestration candidate)

`configs/default.json` pricing controls:

- `pricing.target_lr`
- `pricing.alpha_grid`
- `pricing.beta_grid`
- `pricing.target_band`

CatBoost tuning artifacts are written to:

- `artifacts/tuning/catboost/<timestamp>/results.csv`
- `artifacts/tuning/catboost/<timestamp>/best_params.json`
- `artifacts/tuning/catboost/<timestamp>/summary.json`
- `artifacts/tuning/catboost/<timestamp>/best_5_trials_summary.csv`

Stability-check artifacts are written to:

- `artifacts/stability/<timestamp>/results.csv`
- `artifacts/stability/<timestamp>/summary.json`

Optional speed-up control:

- `ablation.enabled` (set `false` for faster iterative tuning runs)
