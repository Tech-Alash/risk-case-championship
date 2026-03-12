# MASTER: Risk Case Project Context (Глубокий Snapshot)

Проект: `Риск кейс чемпоинат`  
Домен: страховой скоринг убыточности (frequency/severity + pricing)

## 0) Snapshot и границы актуальности

- Версия документа: `master_project_context_20260308.md` (обновлённая, углублённая)
- Снимок времени (локально): `2026-03-08 21:29:12 +05:00`
- Снимок времени (UTC): `2026-03-08 16:29:12 UTC`
- Важно: live-статусы Optuna/process и динамические метрики отражают состояние **на момент генерации snapshot**. Для актуализации используйте операционные команды из раздела 8.

## 1) Цель проекта и метрики отбора

### 1.1 Бизнес-цель

Построить policy-level модель, которая:
- прогнозирует риск и ожидаемый убыток (`expected_loss`),
- переводит прогноз в тариф (`new_premium`) через pricing-оптимизацию,
- максимизирует бизнес-метрику отбора при жёстких ограничениях портфеля.

### 1.2 Основная метрика и constraints

- Основная метрика отбора: `policy_score` (чем выше, тем лучше).
- Жёсткие ограничения:
  - `violations = 0`
  - `lr_total` в диапазоне `0.69..0.71`
  - `share_group1` в пределах `0.0..1.0` (в конфиге)

### 1.3 Метрика для stability-тюнинга

Для режима `objective_mode=stability_adjusted`:

- `objective_value = mean(policy_score) - lambda * std(policy_score)`
- текущий `lambda = 0.35`
- доп. гейты на серию сплитов:
  - `constraints_pass_rate >= 1.0`
  - `in_target_rate >= 0.8` (в текущем запуске фактически лучшие решения дают 1.0)

## 2) Архитектура pipeline (decision-level)

Ниже — фактический поток `ingest -> policy aggregation -> preprocessing -> benchmark -> pricing -> artifact persistence -> inference/submission`.

| Шаг | Вход | Ключевое решение/логика | Выход | Ключевые модули |
|---|---|---|---|---|
| Ingest | `train.csv`, `test_final.csv` | Чтение сырых таблиц | DataFrame train/test | `src/risk_case/data/io.py` |
| Validation | Сырые данные train | Контрактные проверки качества/схемы | `ok/errors/warnings` | `src/risk_case/data/validation.py` |
| Policy Aggregation | Driver-level train/test | Агрегация до `contract_number` для консистентного pricing/KPI | Policy-level train/test | `src/risk_case/data/policy_aggregation.py` |
| Preprocessing | Policy-level train/test + config | Impute, winsorize, date-features, target/freq encoding, interaction features, pruning | `train_split/valid_split`, preprocessor state, metadata | `src/risk_case/features/preprocessing.py`, `src/risk_case/features/feature_store.py` |
| Benchmark | Preprocessed train/valid + список кандидатов | Обучение кандидатов, pricing на valid, фильтр по constraints, выбор winner | winner model + candidate leaderboard | `src/risk_case/models/benchmark.py` |
| Pricing | `expected_loss` на valid/test | Поиск `alpha,beta` (grid/slsqp), проверка таргет-бэнда LR | `new_premium`, LR/constraints/policy_score | `src/risk_case/pricing/evaluator.py`, `src/risk_case/pricing/policy.py` |
| Artifact Persistence | run state и метрики | Сериализация модели, метрик, summary, benchmark-артефактов, pointer-файлов | `metrics.json`, `summary.md`, `leaderboard.csv`, `latest_run.json`, и др. | `src/risk_case/orchestration/run_pipeline.py` |
| Inference/Submission | test + fitted preprocessor/model | Transform test, predict, repricing, merge в submission | `test_policy_predictions.csv`, `submission.csv` | `src/risk_case/features/feature_store.py`, `src/risk_case/orchestration/run_pipeline.py` |

### 2.1 Каталог модулей `src/risk_case/*`

- `src/risk_case/data`: загрузка, валидация, policy aggregation, quality report.
- `src/risk_case/features`: preprocessing, feature store, feature builder.
- `src/risk_case/models`: benchmark-оркестрация кандидатов (baseline/xgb/lgbm/catboost/oof blend), метрики.
- `src/risk_case/pricing`: оптимизация и оценка pricing policy.
- `src/risk_case/orchestration`: end-to-end run manager и запись артефактов.
- `src/risk_case/eda`: EDA и feature-selection артефакты.
- `src/risk_case/api`: FastAPI inference/demo слой.

## 3) Данные и структура признаков

Источники: `artifacts/eda/metadata/eda_profile.json`, `artifacts/runs/20260308_140918/preprocessed/preprocess_metadata.json`.

### 3.1 Форма данных

- Train raw: `569,508 x 159`
- Test raw: `244,073 x 156`
- Train-only колонки: `is_claim`, `claim_amount`, `claim_cnt`
- Contract unique (train): `180,635`
- Среднее строк на контракт: `3.1528`
- Driver->Policy premium duplication factor: `3.1776`

### 3.2 Таргет и разреженность

- Claim-rate (`is_claim` mean): `0.01948` (~1.95%)
- `claim_amount` и `claim_cnt` высоко sparse по дизайну (событийность редкая)

### 3.3 Почему сейчас признаков меньше, чем раньше (~200 vs 164)

Факт из `feature_pruning_report` (latest run `20260308_140918`):

- До pruning: `209`
- После pruning: `164`
- Удалено всего: `45`
  - exact duplicates: `44`
  - manual drop: `1` (`score_missing_share`)
  - high-corr drop: `0`
  - drift-psi drop: `0`

Итого уменьшение обусловлено не “потерей сигнала”, а главным образом чисткой дублирующих missing-флагов и ручным дропом технического столбца.

### 3.4 Ключевые преобразования (из `configs/default.json`)

- Missing handling:
  - `numeric_default=median`
  - financial fill `0.0`
  - missing flags + missing aggregates
- Outliers: winsorize `0.01..0.99`
- Date features: `month`, `quarter`, `dayofweek`, `is_month_end`, `sin_month`, `cos_month`
- Target encoding (OOF on train split) для: `model`, `mark`, `ownerkato`, `region_name`, `car_year`, `bonus_malus`
- Frequency encoding для того же набора
- Interaction features (MVP):
  - `premium_per_driver`
  - `premium_wo_term_per_driver`
  - `premium_per_power`
  - `premium_wo_term_per_power`
  - `car_age_x_bonus_malus`
  - `region_x_vehicle_type`
- Forbidden/leakage-sensitive columns в drop: `unique_id`, `driver_iin`, `insurer_iin`, `car_number`, `contract_number`

## 4) Эксперименты и подтверждённые результаты

### 4.1 Сводная таблица: single-holdout vs stable vs stability-adjusted

| Категория | Источник истины | Лучший результат | Комментарий |
|---|---|---|---|
| Лучший single-holdout run | `artifacts/runs/20260303_091029/metrics.json` | `policy_score=0.5133352741`, `AUC=0.7190061106`, `Gini=0.4380122212`, `lr_total=0.6989040832` | Отличный на одном holdout, но требует проверки устойчивости |
| Лучший stability-adjusted trial | Optuna study `catboost_tuning_stability_severity_only_20260304_140253` в `artifacts/tuning/catboost/catboost_tuning.db` | **Trial 74**: `objective=0.4871532902`, `mean=0.4905352446`, `std=0.0096627270`, pass-rate=1.0 | Лучший баланс mean/std по формуле objective |
| Самый стабильный full-pass trial | та же study | **Trial 70**: `std=0.0063201978`, `objective=0.4822713797`, `mean=0.4844834489` | Минимальный разброс при 100% pass |

### 4.2 Проверка устойчивости лучшего single-holdout

Проверенный run: `20260303_091029`  
Артефакты:
- `artifacts/stability/20260308_151309/summary.json`
- `artifacts/stability/20260308_151309/results.csv`
- params snapshot: `artifacts/stability/params/run_20260303_091029_params.json`

Итоги multi-split (5 holdout-start):
- `mean_policy_score = 0.3933546598`
- `std_policy_score = 0.1434816549` (очень высокий разброс)
- `objective_value = 0.3431360805`
- `constraints_pass_rate = 1.0`
- `in_target_rate = 1.0`
- `mean_lr_group1 = 0.8626712233` (дрифт вверх)

### 4.3 Почему результаты расходятся

Причины расхождения `single-holdout` vs `stability-adjusted`:
- Single-holdout отражает одну временную точку; multi-split выявляет сильную чувствительность к сдвигу распределений.
- Для run `20260303_091029` внутри 5 сплитов есть провалы `policy_score` (например ~0.198 и ~0.238) при сохранении формального `in_target`.
- Стабильность ухудшается из-за volatility сегментов (в т.ч. `lr_group1`) и shift по времени.

### 4.4 Live-состояние study на момент snapshot

Study: `catboost_tuning_stability_severity_only_20260304_140253`
- Trials: `74 COMPLETE`, `2 FAIL`, `2 RUNNING`
- RUNNING trial numbers: `60`, `77`
- Текущий best trial: `74`

Примечание: `trial 60` выглядит как исторически “зависший RUNNING” после рестартов; учитывайте это при live-мониторинге (см. раздел 8).

### 4.5 Последний полный pipeline run

Исторический snapshot в этом master был зафиксирован на run `20260308_140918`, но на текущий момент active pointer уже обновлён.

Текущий active latest run:
- `artifacts/latest_run.json` указывает на run `20260312_061840`
- config freeze: `configs/release_catboost_trial70_threads4.json`
- winner: `catboost_freq_sev`

Ключевые значения (`artifacts/runs/20260312_061840/metrics.json`):
- `policy_score = 0.4744704479`
- `AUC = 0.7213892576`
- `Gini = 0.4427785152`
- `lr_total = 0.6956910980`
- `feature_count = 164`
- `violations = 0`
- `in_target = true`

Статус веток на текущий момент:
- stable baseline `trial 70` — active champion candidate
- `catboost_dep_freq_sev` — paused research, не release path
- live freeze-status: `artifacts/ops/20260312_111528_freeze_trial70/freeze_status.json`

## 5) Полный каталог скриптов (`scripts/*.py`)

Проверка полноты каталога: в `scripts/` найдено `6` Python-скриптов, в master-доке описаны все 6.

### 5.1 `scripts/run_pipeline.py`

- Назначение:
  - Полный end-to-end pipeline (ingest -> preprocess -> benchmark -> pricing -> artifacts -> submission).
- Когда использовать:
  - Для финального/контрольного прогона конфигурации и получения production-ready артефактов.
- Минимальный запуск (PowerShell):
```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\default.json
```
- Ключевые CLI аргументы:
  - `--config` (JSON конфиг пайплайна).
- Входные данные/конфиги:
  - `configs/default.json` (или release-конфиг)
  - `final_dataset/final_dataset/train.csv`
  - `final_dataset/final_dataset/test_final.csv`
- Выходные артефакты:
  - `artifacts/runs/<run_id>/pipeline.log`
  - `artifacts/runs/<run_id>/metrics.json`
  - `artifacts/runs/<run_id>/summary.md`
  - `artifacts/runs/<run_id>/model.joblib`
  - `artifacts/runs/<run_id>/valid_predictions.csv`
  - `artifacts/runs/<run_id>/benchmark/*`
  - `artifacts/runs/<run_id>/submission.csv`
  - `artifacts/leaderboard.csv`
  - `artifacts/latest_run.json`
- Типичная длительность:
  - По run `20260308_140918`: ~9–11 минут (от старта до `Run completed`).
- CPU/память профиль:
  - CPU heavy на `benchmark_models` и preprocessing.
  - RAM обычно около 1–2+ ГБ в зависимости от кандидатов/признаков.
- Частые ошибки и быстрая диагностика:
  - `Validation failed`: проверить сырые поля/типы и `validation.py`.
  - `TARGET_* missing after preprocessing`: проверить `target_columns`, drop/forbidden list.
  - `No compliant benchmark candidates`: проверить constraints и candidate params.
  - Быстрая проверка прогресса:
```powershell
Get-Content artifacts\runs\<run_id>\pipeline.log -Tail 80 -Wait
```

### 5.2 `scripts/tune_catboost.py`

- Назначение:
  - Optuna tuning для `catboost_freq_sev`, включая режим `stability_adjusted` по нескольким time-split.
- Когда использовать:
  - Для поиска hyperparams под бизнес-метрику и устойчивость.
- Минимальный запуск (single split):
```powershell
.\.venv\Scripts\python.exe scripts\tune_catboost.py --config configs\default.json --n_trials 40
```
- Минимальный запуск (stability-adjusted, текущий рабочий режим):
```powershell
.\.venv\Scripts\python.exe scripts\tune_catboost.py --config configs\default.json --phase focused --mode severity_only --n_trials 20 --study_name catboost_tuning_stability_severity_only_20260304_140253 --storage catboost_tuning.db --objective_mode stability_adjusted --splits_config configs\experiments\stability_splits.json --stability_lambda 0.35 --stability_min_constraints_rate 1.0 --stability_min_in_target_rate 0.8 --catboost_threads 4
```
- Ключевые CLI аргументы:
  - Базовые: `--config`, `--n_trials`, `--timeout_sec`, `--study_name`, `--storage`, `--seed`, `--n_jobs`
  - Ресурсы: `--catboost_threads`
  - Search-space: `--phase {coarse,fine,focused}`, `--mode {full,severity_only}`
  - Objective: `--objective_mode {single_split,stability_adjusted}`, `--splits_config`, `--stability_lambda`, `--stability_min_constraints_rate`, `--stability_min_in_target_rate`
  - Split override: `--time_holdout_start_override`
- Входные данные/конфиги:
  - `configs/default.json`
  - `configs/experiments/stability_splits.json` (для stability-adjusted)
  - train CSV из `paths.train_csv`
- Выходные артефакты:
  - `artifacts/tuning/catboost/<timestamp>/results.csv`
  - `artifacts/tuning/catboost/<timestamp>/study_trials.json`
  - `artifacts/tuning/catboost/<timestamp>/best_params.json`
  - `artifacts/tuning/catboost/<timestamp>/best_5_trials_summary.csv`
  - `artifacts/tuning/catboost/<timestamp>/summary.json`
  - Опционально study storage: `artifacts/tuning/catboost/catboost_tuning.db`
  - Cache: `artifacts/tuning/catboost/preprocessed_cache/*`
- Типичная длительность:
  - `stability_adjusted + 5 split + threads=4`: ~20–30 минут на trial.
  - Оценочно: `20 trials ~8–10 часов`, `60 trials ~1–1.5 суток`.
- CPU/память профиль:
  - Самый тяжёлый скрипт проекта.
  - CPU высокая, контролируется `--catboost_threads`.
  - RAM обычно 1–3+ ГБ (кэш, модели, сплиты).
- Частые ошибки и быстрая диагностика:
  - Зависшие `RUNNING` trial после прерываний процесса.
  - Неконсистентный `storage` путь (relative path не там, где ожидается).
  - `--catboost_threads must be > 0`.
  - Быстрая диагностика:
```powershell
Get-Item artifacts\tuning\catboost\catboost_tuning.db | Select-Object LastWriteTime,Length
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts/tune_catboost.py*'} | Select-Object ProcessId,CreationDate,CommandLine
```

### 5.3 `scripts/run_stability_checks.py`

- Назначение:
  - Прогон фиксированного кандидата/параметров по нескольким временным holdout split и расчёт устойчивости.
- Когда использовать:
  - После выбора candidate params из tuning перед release-решением.
- Минимальный запуск:
```powershell
.\.venv\Scripts\python.exe scripts\run_stability_checks.py --config configs\default.json --splits_config configs\experiments\stability_splits.json --candidate catboost_freq_sev --params_json artifacts\tuning\catboost\20260308_release_trial70\best_params_trial70_severity_only.json --catboost_threads 4 --objective_lambda 0.35
```
- Ключевые CLI аргументы:
  - `--config`, `--splits_config`, `--candidate`, `--params_json`, `--catboost_threads`, `--objective_lambda`
- Входные данные/конфиги:
  - `configs/default.json`
  - `configs/experiments/stability_splits.json`
  - params JSON (если хотим проверить конкретный trial)
- Выходные артефакты:
  - `artifacts/stability/<timestamp>/results.csv`
  - `artifacts/stability/<timestamp>/summary.json`
- Типичная длительность:
  - На 5 сплитов для CatBoost обычно сопоставимо с одним stability trial: ~20–40 минут.
- CPU/память профиль:
  - CPU heavy (последовательные train/eval по split).
  - Для CatBoost рекомендуется ограничение `--catboost_threads 4` на shared машине.
- Частые ошибки и быстрая диагностика:
  - Неправильный `splits_config` (нет `time_holdout_starts`).
  - Params JSON не соответствует candidate.
  - Быстрая проверка результата:
```powershell
Get-Content artifacts\stability\<timestamp>\summary.json
```

### 5.4 `scripts/tune_xgboost.py`

- Назначение:
  - Быстрый grid-sweep по встроенному набору XGBoost параметров (до `max_trials`).
- Когда использовать:
  - Быстрый sanity-check альтернативы CatBoost, без долгого Optuna.
- Минимальный запуск:
```powershell
.\.venv\Scripts\python.exe scripts\tune_xgboost.py --config configs\default.json --max_trials 4
```
- Ключевые CLI аргументы:
  - `--config`, `--max_trials`
- Входные данные/конфиги:
  - `configs/default.json`
  - train CSV из конфига
- Выходные артефакты:
  - `artifacts/tuning/xgboost/<timestamp>/results.csv`
  - `artifacts/tuning/xgboost/<timestamp>/summary.json`
  - `artifacts/tuning/xgboost/preprocessed_cache/*`
- Типичная длительность:
  - Обычно быстрый (минуты, реже десятки минут), зависит от `max_trials`.
- CPU/память профиль:
  - Средняя CPU нагрузка, заметно легче долгого CatBoost stability-tuning.
- Частые ошибки и быстрая диагностика:
  - Ошибки preprocessing (если target/forbidden columns сломаны в конфиге).
  - Проверить `summary.json` на факт выполнения и лучший trial.

### 5.5 `scripts/run_eda.py`

- Назначение:
  - Полный EDA пакет: schema/missing/leakage watchlist/feature selection.
- Когда использовать:
  - Перед массовым тюнингом, после изменения данных/feature rules.
- Минимальный запуск:
```powershell
.\.venv\Scripts\python.exe scripts\run_eda.py --config configs\eda.json
```
- Ключевые CLI аргументы:
  - `--config` (по умолчанию `configs/eda.json`)
- Входные данные/конфиги:
  - `configs/eda.json`
  - train/test CSV
- Выходные артефакты:
  - `artifacts/eda/tables/*`
  - `artifacts/eda/figures/*`
  - `artifacts/eda/metadata/eda_profile.json`
  - `artifacts/eda/feature_selection/feature_whitelist.csv`
  - `artifacts/eda/feature_selection/feature_droplist.csv`
  - `artifacts/eda/feature_selection/feature_review_list.csv`
  - `reports/eda_summary.md`
  - `reports/feature_selection_report.md`
- Типичная длительность:
  - От нескольких минут до ~30+ минут (зависит от I/O и генерации графиков).
- CPU/память профиль:
  - Умеренная CPU, RAM зависит от full read CSV.
- Частые ошибки и быстрая диагностика:
  - Отсутствуют входные CSV/битые пути в `eda.json`.
  - Проверка: наличие `artifacts/eda/metadata/eda_profile.json`.

### 5.6 `scripts/generate_deep_eda_notebook.py`

- Назначение:
  - Генерация глубокого Jupyter-ноутбука для leakage/drift/segment diagnostics.
- Когда использовать:
  - Когда нужно быстро получить шаблон расширенного EDA workflow.
- Минимальный запуск:
```powershell
.\.venv\Scripts\python.exe scripts\generate_deep_eda_notebook.py
```
- Ключевые CLI аргументы:
  - Нет (скрипт без argparse).
- Входные данные/конфиги:
  - Не требует данных на этапе генерации файла.
- Выходные артефакты:
  - `notebooks/04_eda_deep_leakage_drift_segments.ipynb`
  - (после исполнения ноутбука) `artifacts/eda/deep/*` и `reports/eda_deep_summary.md`
- Типичная длительность:
  - Генерация файла: секунды.
  - Выполнение всех ячеек ноутбука: может быть существенно дольше (зависит от данных).
- CPU/память профиль:
  - На этапе генерации минимальный.
- Частые ошибки и быстрая диагностика:
  - Если `nbformat` не установлен — установить зависимости окружения.

## 6) Текущий pipeline/state snapshot (операционный минимум)

### 6.1 Что сейчас считать “источником истины”

| Тип задачи | Файл истины | Комментарий |
|---|---|---|
| Последний полный run | `artifacts/latest_run.json` | Только pointer на текущий run |
| Финальные метрики run | `artifacts/runs/<run_id>/metrics.json` | Главный источник по run quality |
| История run | `artifacts/leaderboard.csv` | Быстрый индекс, но не заменяет `metrics.json` |
| Stability результата | `artifacts/stability/<timestamp>/summary.json` | Главный источник для multi-split устойчивости |
| Детали stability по split | `artifacts/stability/<timestamp>/results.csv` | Разбивка по каждому holdout_start |
| Тюнинг/лучший trial | `artifacts/tuning/catboost/catboost_tuning.db` | Главный источник состояния Optuna |
| Экспорт тюнинга за запуск | `artifacts/tuning/catboost/<timestamp>/summary.json` | Снимок по завершившемуся запуску |

### 6.2 Где смотреть прогресс и как отличить “идёт” от “зависло”

- Full pipeline:
```powershell
Get-Content artifacts\runs\<run_id>\pipeline.log -Tail 80 -Wait
```

- Tuning (CatBoost + Optuna):
```powershell
Get-Item artifacts\tuning\catboost\catboost_tuning.db | Select-Object LastWriteTime,Length
Get-ChildItem artifacts\tuning\catboost\launch_logs\*.err.log | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,LastWriteTime,Length
```

- Процесс в ОС:
```powershell
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts/tune_catboost.py*'} | Select-Object ProcessId,CreationDate,CommandLine
Get-Process -Id <pid> | Select-Object Id,StartTime,CPU,WorkingSet64
```

Признаки “скорее зависло”:
- База Optuna (`catboost_tuning.db`) и launch-log не меняются существенно дольше типичной длительности trial.
- В study есть старые `RUNNING` trial, у которых `datetime_start` сильно в прошлом (пример snapshot: trial `60` с `2026-03-06`).

### 6.3 Live Freeze Override

После этого snapshot был выполнен отдельный operational freeze baseline:
- active stable run: `20260312_061840`
- active config: `configs/release_catboost_trial70_threads4.json`
- dependent FS research остановлен и сохранён как snapshot:
  - `artifacts/ops/20260312_111528_freeze_trial70/dependent_fs_pause_summary.md`
- финальный same-code stability rerun запущен из:
  - `artifacts/ops/20260312_111528_freeze_trial70/stability_check_retry.log`
- текущий status file freeze-процедуры:
  - `artifacts/ops/20260312_111528_freeze_trial70/freeze_status.json`

## 7) Исполнимый playbook: Trial 74 vs Trial 70

Цель: выбрать production champion между “лучший objective” (74) и “лучший std” (70).

### 7.1 Критерии выбора

Гейт-уровень 1 (обязательные):
- `constraints_pass_rate = 1.0` на 5 split
- `in_target_rate = 1.0` на 5 split

Гейт-уровень 2 (основной ранжир):
- максимальный `objective_value = mean - 0.35*std`

Гейт-уровень 3 (tie-break/риск):
- ниже `policy_score_std`
- ниже drift по `mean_lr_group1`
- сопоставимый/лучший full-run `policy_score` на свежем holdout

### 7.2 Последовательность запусков

1. Зафиксировать params trial 74 в отдельный params JSON + release config (`thread_count=4`, candidate only `catboost_freq_sev`).
2. Прогнать full pipeline для trial 70 (уже есть релизный конфиг):
```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\release_catboost_trial70_threads4.json
```
3. Прогнать full pipeline для trial 74 (аналогичный release-конфиг).
4. Прогнать stability-check для каждого trial на одинаковом `splits_config`:
```powershell
.\.venv\Scripts\python.exe scripts\run_stability_checks.py --config configs\default.json --splits_config configs\experiments\stability_splits.json --candidate catboost_freq_sev --params_json <trial70_params.json> --catboost_threads 4 --objective_lambda 0.35
.\.venv\Scripts\python.exe scripts\run_stability_checks.py --config configs\default.json --splits_config configs\experiments\stability_splits.json --candidate catboost_freq_sev --params_json <trial74_params.json> --catboost_threads 4 --objective_lambda 0.35
```
5. Сравнить артефакты и зафиксировать champion.

### 7.3 Какие артефакты сравнивать

Для каждого trial (70/74):
- Full run:
  - `artifacts/runs/<run_id>/metrics.json`
  - `artifacts/runs/<run_id>/pipeline.log`
- Stability:
  - `artifacts/stability/<timestamp>/summary.json`
  - `artifacts/stability/<timestamp>/results.csv`
- Params/config freeze:
  - `configs/release_catboost_trialXX_threads4.json`
  - `artifacts/tuning/catboost/<...>/best_params_trialXX_*.json`

## 8) Проверки консистентности, выполненные при обновлении master

1. Каталог скриптов: найдено `6` `.py` в `scripts/`, все 6 описаны в разделе 5.
2. Ключевые числа сверены с источниками:
- `artifacts/runs/20260303_091029/metrics.json`
- `artifacts/stability/20260308_151309/summary.json`
- `artifacts/tuning/catboost/catboost_tuning.db` (study `catboost_tuning_stability_severity_only_20260304_140253`, trials 70/74)
3. Пути на основные артефакты в документе соответствуют реальным файлам в рабочем каталоге.
4. Команды запуска согласованы с `--help` соответствующих CLI-скриптов.

## 9) Changelog (относительно предыдущей версии master)

Обновлено:
- Добавлен snapshot-блок с фиксированными временными метками (UTC + local) и пометкой динамичности live-метрик.
- Архитектура расширена до decision-level pipeline с входами/выходами/модулями по шагам.
- Добавлен полный практический каталог всех 6 скриптов из `scripts/` в едином формате (назначение, CLI, I/O, артефакты, runtime, CPU/RAM, диагностика).
- Экспериментальные результаты разделены на:
  - лучший single-holdout,
  - лучший stability-adjusted,
  - самый стабильный trial,
  - с явным объяснением расхождений.
- Добавлен операционный блок мониторинга и критерии “идёт/зависло”.
- Добавлен исполнимый playbook выбора champion между trial 74 и trial 70.
- Добавлен блок проверок консистентности обновления.
- После snapshot выполнен live freeze baseline `trial 70`: `latest_run.json` переведён на run `20260312_061840`, dependent FS помечен как paused research, финальный stability rerun вынесен в отдельный `ops`-status.

Причина обновления:
- Предыдущая версия была удобным short handover, но не покрывала глубоко эксплуатационные детали и полный каталог скриптов для нового инженера/большой модели.
