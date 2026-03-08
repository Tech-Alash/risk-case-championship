# Контекст проекта для LLM (актуализация на 3 марта 2026)

Дата обновления: 2026-03-03  
Проект: `Риск кейс чемпоинат`  
Домен: страховой риск и перепрайсинг (`frequency-severity + pricing optimization`)

## 1) Цель и бизнес-ограничения

Цель пайплайна:
- прогноз риска по договору (`p_claim`, `expected_severity`, `expected_loss`),
- перепрайсинг через policy-функцию,
- выбор лучшей модели по бизнес-метрике.

Ключевая метрика:
- `policy_score` (максимизируем),
- при жестких ограничениях:
  - `violations = 0`,
  - `lr_total` в диапазоне `0.69..0.71`.

`policy_score` учитывает:
- близость к целевому LR,
- баланс групп (`lr_group1`, `lr_group2`),
- штрафы за нарушения ограничений,
- бонус за попадание в target-band.

## 2) Текущая архитектура проекта

Основной E2E-пайплайн:
1. Ingest + validation.
2. Агрегация к policy-level (`contract_number`).
3. Preprocessing + feature engineering.
4. Benchmark кандидатов моделей.
5. Pricing optimization (`alpha`, `beta`, grid).
6. Артефакты run/leaderboard/metadata.
7. Inference и submission.

Ключевые модули:
- `src/risk_case/data` — загрузка, валидация, split, policy aggregation.
- `src/risk_case/features` — preprocessing, feature store.
- `src/risk_case/models` — baseline + benchmark (xgb/lgbm/catboost), blend.
- `src/risk_case/pricing` — evaluator и оптимизация pricing.
- `src/risk_case/orchestration/run_pipeline.py` — оркестрация полного запуска.

Конфигурация (рабочая):
- `configs/default_with_tuned_catboost.json`
- validation scheme: `group_time`, holdout start: `2022-09-22`.
- benchmark candidates: `baseline_freq_sev`, `xgboost_freq_sev`, `lightgbm_freq_sev`, `catboost_freq_sev`.
- реализован `oof_blend_freq_sev` (пока не включен по умолчанию в список кандидатов).

## 3) Feature engineering и отбор признаков

В preprocessing включены:
- date features,
- target encoding,
- frequency encoding,
- interaction features,
- `log1p` трансформации,
- missing flags и missing-агрегаты,
- whitelist/droplist.

Текущие механизмы pruning:
- удаление точных дублей признаков,
- удаление `score_missing_share`,
- корреляционный pruning (`corr_threshold`),
- новый `drift_pruning` по PSI во времени.

Параметры `drift_pruning` (включены в `default` и `default_with_tuned_catboost`):
- `enabled=true`,
- `time_column=operation_date`,
- `reference_share=0.7`,
- `psi_threshold=0.25`,
- `bins=10`,
- `min_rows=500`.

## 4) Сводка ключевых экспериментов

### 4.1 Лучшие full pipeline run (single holdout)

Топ run по `policy_score` (валидный, с ограничениями):

1. `20260303_091029`  
   `policy_score=0.5133352741`, `AUC=0.7190061106`, `Gini=0.4380122212`, `lr_total=0.6989040832`, `violations=0`, winner=`catboost_freq_sev`.

2. `20260220_093233`  
   `policy_score=0.5043840154`, `AUC=0.7201479975`, `Gini=0.4402959949`, `lr_total=0.6918722292`, winner=`catboost_freq_sev`.

3. `20260220_171846`  
   `policy_score=0.4877901493`, `AUC=0.7212097897`, `Gini=0.4424195794`, `lr_total=0.6911251494`, winner=`catboost_freq_sev`.

Артефакты:
- `artifacts/runs/20260303_091029/metrics.json`
- `artifacts/runs/20260303_091029/benchmark/results.csv`

### 4.2 Stability checks (5 split)

Лучшие результаты по `mean_policy_score` при `constraints_pass_rate=1.0` и `in_target_rate=1.0`:

1. `20260302_062439`  
   `mean_policy_score=0.4724037981`, `std_policy_score=0.0226595841`, `n_splits=5`.  
   Источник params: `artifacts/tuning/catboost/20260301_094709/best_params.json`.

2. `20260225_115943`  
   `mean_policy_score=0.4698063375`, `std_policy_score=0.0259856246`, `n_splits=5`.

3. `20260303_060948`  
   `mean_policy_score=0.4394485847`, `std_policy_score=0.0358598003`, `n_splits=5`.

Артефакты:
- `artifacts/stability/20260302_062439/summary.json`
- `artifacts/stability/20260225_115943/summary.json`

### 4.3 CatBoost tuning (stability-adjusted)

`artifacts/tuning/catboost/20260303_050925/summary.json`:
- study: `catboost_tuning_stability_full_20260302_long`,
- objective: `stability_adjusted`,
- `n_trials_requested=60`, `n_trials_finished=18`,
- best trial value: `0.4268976546`,
- best trial user attrs: `policy_score_mean=0.4394485847`, `policy_score_std=0.0358598003`, `constraints_pass_rate=1.0`, `in_target_rate=1.0`.

`artifacts/tuning/catboost/20260301_094709/summary.json` (smoke 2-split):
- best trial value: `0.4710243029`,
- `policy_score_mean=0.4801418874`, `policy_score_std=0.0260502413`.

## 5) Эксперимент с drift-pruning (03 марта 2026)

Full run с новым `drift_pruning`:
- run: `20260303_105410`,
- preprocessing: `before=209`, `after=161`, `dropped_total=48`.

Разложение удаления:
- exact duplicates: `44`,
- manual (`score_missing_share`): `1`,
- drift PSI drops: `3`.

Удаленные по PSI:
- `operation_date_sin_month` (`psi=10.7329`),
- `operation_date_month` (`psi=7.5728`),
- `operation_date_cos_month` (`psi=5.4732`).

Метрики run `20260303_105410`:
- `policy_score=0.4456735025`,
- `AUC=0.7174859715`,
- `Gini=0.4349719431`,
- `lr_total=0.6924988120`,
- `violations=0`, `in_target=true`.

Stability после этого изменения (`20260303_113006`):
- `mean_policy_score=0.3290606417`,
- `std_policy_score=0.1654336264`,
- `constraints_pass_rate=1.0`, `in_target_rate=1.0`.

Сравнение с предыдущим stability baseline (`20260303_094520`):
- mean: `0.3613510038 -> 0.3290606417` (хуже),
- std: `0.1451917900 -> 0.1654336264` (хуже).

Вывод по текущей итерации:
- в текущих параметрах `drift_pruning` ухудшил устойчивость и качество.

## 6) Текущее состояние (на 2026-03-03)

Что уже работает стабильно:
- Полный pipeline от ingestion до submission.
- Benchmark из 4 моделей с авто-выбором winner.
- Stability-check протокол на 5 time split.
- CatBoost tuning в режиме `stability_adjusted`.

Что важно зафиксировать:
- Лучший single-holdout run: `20260303_091029` (`policy_score=0.5133`).
- Лучший 5-split stability: `20260302_062439` (`mean=0.4724`, `std=0.02266`).
- Последний drift-pruning эксперимент не дал улучшения.

## 7) Рекомендации на следующий цикл

1. Вернуться к best-known stable params (`20260302_062439` / params из `20260301_094709`).
2. `drift_pruning` оставить выключенным или сделать мягче (`psi_threshold` выше, исключить сезонные date-синусы из pruning).
3. Включить и оценить `oof_blend_freq_sev` на 5 split.
4. Решение о прод-версии принимать по stability-критерию (`mean` и `std`), а не только по single split.

## 8) Полезные команды

Full pipeline:
```bash
python scripts/run_pipeline.py --config configs/default_with_tuned_catboost.json
```

Stability checks (5 split):
```bash
python scripts/run_stability_checks.py --config configs/default_with_tuned_catboost.json --splits_config configs/experiments/stability_splits.json --candidate catboost_freq_sev --catboost_threads 4
```

CatBoost tuning (stability-adjusted):
```bash
python scripts/tune_catboost.py --config configs/default.json --phase fine --n_trials 60 --timeout_sec 43200 --study_name catboost_tuning_stability_full_YYYYMMDD --storage catboost_tuning.db --objective_mode stability_adjusted --splits_config configs/experiments/stability_splits.json --stability_lambda 0.35 --stability_min_constraints_rate 1.0 --stability_min_in_target_rate 0.8 --catboost_threads 4
```
