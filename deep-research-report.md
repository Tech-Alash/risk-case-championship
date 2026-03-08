# Архитектура и итеративный план для победы в кейс-чемпионате по скорингу убыточности ОГПО с LLM‑агентом‑автотестировщиком

**Executive summary.** В кейсе требуется построить скоринг‑модель для оценки риска ДТП по полису ОГПО и на её основе предложить новую «справедливую» премию так, чтобы коэффициент выплат (выплаты / премии за вычетом расторжений) оказался около 70% по портфелю и отдельно по двум группам: полисы без увеличения цены (или со снижением) и полисы с повышением цены, при этом желательно максимизировать долю первой группы и соблюдать ограничения на изменение цены (снижение не более 100%, рост не более чем в 3 раза). fileciteturn0file0  
Оптимальная стратегия под победу — не «просто ML‑модель», а воспроизводимый **experiment‑driven** проект: (1) частота–тяжесть (frequency–severity) или Tweedie‑подход для предсказания ожидаемого ущерба, (2) отдельный модуль оптимизации тарифной политики под бизнес‑ограничения, (3) агентовый контур (Codex/аналог) как «автоматизированный тестировщик и экспериментатор», который генерирует гипотезы, запускает эксперименты через оркестратор, валидирует результаты и формирует артефакты отчётности. Методологически это «eval → улучшение → re‑eval» цикл, в духе современных практик оценивания и оптимизации LLM/агентных систем и MLOps‑подходов к экспериментам и версиионированию. citeturn12search2turn12search29turn6search11turn0search11  
Таймлайн соревнования фиксирует ключевые точки: кейс разослан 10 февраля 2026, драфт проекта — до 23:59 **17 марта 2026**, финальная сдача — до 23:59 **6 апреля 2026**, финал — **28 апреля 2026** (Алматы). citeturn2view0  
Ниже — строгая целевая архитектура и недельный итеративный план, оптимизированные именно под «быстрый, доказуемый бизнес‑эффект» и сильную демонстрацию для жюри.

## Контекст, цели и критерии успеха

**Контекст задачи.** Требуется оценивать риск ДТП/выплат по полису на момент продажи (известны премия и признаки клиента/ТС) и менять стоимость полиса только там, где это оправдано риском, чтобы снизить убыточность и сохранить клиентов. Ключевая бизнес‑метрика — коэффициент выплат = сумма выплат / сумма премий за вычетом расторжений; ориентир «здорового» уровня — около 70%. fileciteturn0file0  

**Цели (формально).**  
1) Построить модель, оценивающую вероятность ДТП/выплаты по полису и/или ожидаемый ущерб, и на её основе рассчитать новую премию для каждого полиса. fileciteturn0file0  
2) Добиться коэффициента выплат ~70%: (a) по всему портфелю, (b) отдельно по группе 1 (цена не выросла) и группе 2 (цена выросла), при максимизации доли группы 1. fileciteturn0file0  
3) Соблюсти ограничения на изменение премии: снижение — не более 100%, увеличение — не более чем в 3 раза. fileciteturn0file0  

**Критерии успеха (если точный скоринг жюри не указан — “не указано”).** В документации кейса явно подчёркивается, что важна не только точность, но и бизнес‑эффект, а также ожидаются метрики качества модели (GINI/ROC‑AUC/F1/R² и др.), расчёт портфельных коэффициентов выплат и обоснование тарифной стратегии. fileciteturn0file0  
Поскольку **официальная формула ранжирования команд: не указано**, практически целесообразно оптимизировать сразу по двум слоям — **ML‑качество** и **policy‑качество**. Ниже — рекомендуемая «матрица критериев», с тем как это измерять (всё — измеримо на train/valid и воспроизводимо в отчёте).

| Блок оценки | Критерий | Как измерять (рекомендовано) | Порог/таргет (рекоменд.) |
|---|---|---|---|
| Бизнес‑эффект | LR (коэффициент выплат) портфеля после применения новой цены | `LR_total = sum(payout) / sum(new_premium_net_cancel)` | близко к 0.70 (±1–3 п.п.) |
| Бизнес‑эффект | LR по группе 1 и группе 2 | отдельно по сегментам `Δprice<=0` и `Δprice>0` | обе группы близко к 0.70 |
| Бизнес‑эффект | Доля группы 1 | `share_group1 = N(Δprice<=0)/N_all` | максимум при соблюдении LR |
| ML‑качество (частота) | ROC‑AUC / Gini | AUC на out‑of‑fold; Gini = 2·AUC − 1 | максимизировать (Gini полезен как «страховой» стандарт) citeturn10search1 |
| ML‑качество (калибровка) | Brier / ECE / reliability curve | Brier + графики калибровки; post‑hoc калибровка sigmoid/isotonic | минимизировать Brier/ECE citeturn10search0 |
| ML‑качество (тяжесть) | RMSE/MAE на log‑scale или deviance | для severity: только для `claim>0`, для aggregate: Tweedie deviance | минимизировать |
| Инженерия | Воспроизводимость | один командный запуск `make reproduce` даёт те же метрики/артефакты | must‑have |
| Презентация | Объяснимость и обоснование тарифа | SHAP/feature importance + story про fairness и ограничения | must‑have |

**Почему frequency–severity (или Tweedie) — “правильный” стандарт.** В кейсе прямо отмечены редкие события выплат с высокой стоимостью и рекомендуется подход, учитывающий одновременно частоту и тяжесть (frequency–severity). fileciteturn0file0 Это соответствует общепринятой актуарной практике: либо раздельные модели частоты и тяжести, либо моделирование агрегированного убытка с распределениями типа Tweedie. citeturn0search16turn0search31turn0search10turn0search19  

## Архитектура решения и компонентов

Ниже — **модульная архитектура**, оптимизированная под две цели:  
(а) быстро находить лучший вариант модели/тарифа через интенсивные эксперименты,  
(б) делать это безопасно и воспроизводимо через CI/CD, тест‑контуры и строгие логи.

### Концепция: LLM‑агент как «надстройка» над детерминированным экспериментальным конвейером

Ключевой принцип для надёжности: **LLM не “обучает модель” напрямую**, а управляет конфигурациями, тестами и идеями; выполнение — строго в оркестраторе (код) с валидацией схем/правил. Это совпадает с рекомендациями строить агентные системы через инструменты, guardrails и детерминируемые workflows. citeturn13search0turn12search3turn4search8turn4search4  

**Что даёт Codex‑подобный агент.** Современный Codex позиционируется как агент, который выполняет инженерные задачи end‑to‑end (код, фиксы, PR‑предложения) в sandbox‑окружениях; это удобно для генерации тестов, рефакторинга пайплайна, и автоматизации экспериментов. citeturn4search4turn4search19turn4search0turn4search33  

### Модули системы (обязательные компоненты из требований)

**Агент Codex (или аналог LLM).**  
Роль: «автотестировщик+экспериментатор». Он:  
- генерирует гипотезы (новые признаки, веса классов, виды калибровки, виды прайс‑функции),  
- пишет/обновляет unit‑тесты и data‑tests,  
- предлагает изменения в конфигурациях экспериментов (Hydra/Optuna),  
- анализирует результаты прогонов и формирует короткие итоговые summary для отчёта.  
Технически рекомендуется строить агента на базе tool‑calling + Structured Outputs (JSON Schema), чтобы команды агента были формально валидны. citeturn12search3turn4search1turn13search0turn4search8  

**Оркестратор экспериментов.**  
Роль: исполнять планы агента, гарантировать воспроизводимость, собирать метрики/артефакты. Минимально‑достаточный стек для соревнования:  
- конфигурационный менеджмент и мультизапуски: Hydra (multi‑run для sweep’ов) citeturn7search1turn7search4  
- HPO/ранняя остановка/прунинг: Optuna citeturn7search0turn7search18  
- трекинг экспериментов и версионирование моделей: MLflow Model Registry citeturn0search11turn0search21  
- хранение артефактов: локально + (опционально) S3/MinIO (если нужно; параметр: не указано).

**Пайплайны данных.**  
Роль: превратить сырые данные в стабильные, проверенные фичи и обучающую выборку, с защитой от утечек и дрейфа схем. Для “быстрого MLOps без инфраструктуры” — DVC pipelines + Git‑версионирование (или аналог). DVC позиционируется как git‑подобное управление данными и пайплайнами, позволяющее воспроизводить DAG обработки/обучения. citeturn6search3turn6search0turn6search11  

**Модели (типы и семейства).**  
Держите “портфель” моделей, потому что победа часто получается из сочетания: (A) сильный скоринг, (B) корректная калибровка, (C) грамотный тарифный модуль.

- Frequency–Severity:
  - Частота (claim/ДТП): GBDT (CatBoost/LightGBM) или логрег с WoE.
  - Тяжесть (размер выплаты | claim>0): Gamma/lognormal‑регрессия, GBDT‑регрессия, либо нейросети. citeturn0search10turn0search13  
- Агрегированный убыток:
  - Tweedie regression как популярная смесь с массой в нуле и непрерывной частью для положительных убытков. citeturn0search31turn0search19  

CatBoost особенно уместен для табличных страховых данных с категориальными признаками (встроенная обработка категорий, рекомендации не делать OHE заранее). citeturn0search2turn0search5  

**CI/CD для моделей.**  
Роль: сделать “как в проде”: каждая правка кода/признаков проходит автоматические проверки, а каждая модель имеет версию, метрики, артефакты и воспроизводимый рецепт. Для CI подойдёт entity["company","GitHub","code hosting company"] Actions или entity["company","GitLab","devops platform company"] CI; оба описывают пайплайны через YAML и поддерживают артефакты. citeturn8search12turn8search13turn8search2turn8search14  

**Мониторинг и логирование.**  
В кейсе мониторинг “в проде” может быть не нужен, но как сильный аргумент для жюри-доменных экспертов полезно показать готовность к эксплуатации:  
- техническая наблюдаемость: OpenTelemetry (traces/metrics/logs) citeturn9search0  
- метрики: Prometheus + Grafana dashboards citeturn9search1turn9search2turn9search14  
В «соревновательном режиме» это можно заменить на MLflow/W&B + структурированные JSON‑логи, но архитектурно компонент должен быть описан.

**Тестовые среды.**  
- unit‑тесты (инварианты фич, корректность агрегаций, проверка утечек),  
- data‑tests (schema, null‑rate, диапазоны) через Great Expectations‑подобные “expectations” (тип инструмента),  
- model‑tests (метрики не хуже baseline, калибровка, стабильность),  
- end‑to‑end smoke (прогон пайплайна на маленькой выборке). Концепция data‑pipelines и data‑testing типична для DVC‑подхода и data‑quality фреймворков. citeturn6search0turn6search10  

**Интерфейсы (API/GUI).**  
- CLI для воспроизводимости (`make train`, `make predict`, `make report`),  
- API для скоринга: FastAPI (быстрый способ сделать демонстрационный endpoint). citeturn9search3  
- GUI для демо: Streamlit/Gradio (параметр: не указано), или MLflow UI.

**Безопасность и управление версиями.**  
- Git: версия кода; DVC: версия данных/артефактов; MLflow Registry: версия модели;  
- секреты: запрет коммита ключей, pre-commit hooks (pre‑commit управляет мульти‑языковыми хуками до коммита). citeturn8search3turn8search26  
- “sandbox‑режим” агента: агент не имеет произвольного доступа к внешней сети (параметр: зависит от инфраструктуры; в кейсе рекомендуется “закрытый” режим), любые действия — через whitelisted tools.

### Диаграмма архитектуры (Mermaid)

```mermaid
flowchart LR
  U[Команда/Аналитик] -->|цели, ограничения, приоритеты| A[LLM-агент Codex/аналог]
  A -->|JSON-план эксперимента (strict schema)| O[Оркестратор экспериментов]
  O --> DQ[Data QA + Leakage Guardrails]
  O --> FE[Feature Pipeline]
  FE --> TR[Train: Frequency/Severity или Tweedie]
  TR --> CAL[Calibration + Thresholding]
  CAL --> PR[Pricing Engine: оптимизация премий под LR~70% и ограничения]
  PR --> EV[Backtesting/Validation: LR, share_group1, AUC/Gini, stability]
  EV --> REG[Model Registry + Artifacts Store]
  EV --> REP[Auto Report Builder]
  REG --> CI[CI/CD Gate: unit+data+model tests]
  REP --> DEMO[Demo API/GUI + Slides]
  EV -->|результаты| A
```

### Диаграмма потоков данных и артефактов (Mermaid)

```mermaid
flowchart TB
  RAW[raw: train.csv/test.csv] --> VAL[validate schema/ranges]
  VAL --> CLEAN[clean/impute]
  CLEAN --> AGG[entity aggregation: driver↔car↔policy]
  AGG --> SPLIT[time/contract-safe split]
  SPLIT --> FIT1[fit freq model: P(claim)]
  SPLIT --> FIT2[fit severity model: E(amount|claim)]
  FIT1 --> EL[expected loss = P * E]
  FIT2 --> EL
  EL --> PRICE[new_premium = f(expected loss, constraints)]
  PRICE --> METRICS[metrics: AUC/Gini, calib, LR_total, LR_g1, LR_g2, share_g1]
  PRICE --> SUB[submissions: contract_number, prob, loss, new_premium]
  METRICS --> ART[artifacts: model file, params, plots, report]
```

## Итеративный план и таймлайн

Ниже — план в неделях, привязанный к официальному расписанию (ключевые даты и дедлайны указаны на сайте чемпионата). citeturn2view0  
Текущая дата для ориентира: **15 февраля 2026** (Алматы).

### Принципы итераций и “gates” (критерии перехода)

Каждая итерация заканчивается **gate‑проверкой**:  
- пайплайн воспроизводим (одна команда → те же метрики),  
- тесты проходят (unit+data),  
- есть измеримый прирост по целевой функции (см. ниже),  
- артефакты (конфиги/логи/модель/таблицы) автоматически сохраняются.

**Единая целевая функция для ранжирования экспериментов** (рекомендовано):  
\[
Score = -\alpha |LR_{total}-0.70| -\beta |LR_{g1}-0.70| -\gamma |LR_{g2}-0.70| + \delta \cdot share_{g1} + \eta \cdot Gini
\]
где веса \(\alpha,\beta,\gamma,\delta,\eta\) задаются командой (параметры: не указано) и фиксируются в конфиге эксперимента.

### Таймлайн (таблица)

| Неделя (даты) | Цель спринта | Выходные артефакты | Gate (переход) |
|---|---|---|---|
| 1 (10–16 фев) | Репозиторий + скелет пайплайна + baseline | `make reproduce`, baseline LR до/после (на простом правиле), MLflow/DVC структура, 5–10 unit/data tests | E2E smoke проходит; есть baseline‑таблица метрик |
| 2 (17–23 фев) | EDA + утечки + entity‑агрегации + простой скоринг | безопасный split, агрегаты driver/car/policy, baseline CatBoost/LogReg, первичная калибровка | AUC/Gini измеряются out‑of‑fold; нет очевидной leakage |
| 3 (24 фев–2 мар) | Frequency–severity v1 или Tweedie v1 | два‑этапа freq+sev (или Tweedie), baseline pricing `new = old * k(score)` | LR_total можно “подогнать” к 0.70 без нарушения ограничений |
| 4 (3–9 мар) | Авто‑эксперименты: HPO + абляции + feature factory | Hydra multirun + Optuna, leaderboard, отчёт “топ‑10 экспериментов” | стабильный прогресс по Score; воспроизводимость |
| 5 (10–16 мар) | Подготовка драфта к 17 марта | полный отчёт v0, экспорт модели, генерация submission файла | все deliverables формируются одной командой |
| 6 (17–23 мар) | После сдачи драфта: усиление тарифа (policy optimization) | оптимизация порогов/кривой скидок/надбавок, групповая стабилизация LR | LR_total/LR_g1/LR_g2 в коридоре; share_g1 улучшен |
| 7 (24–30 мар) | Надёжность: калибровка, стабильность, explainability | calibration, PSI/drift offline, SHAP/feature impact, стресс‑тесты | метрики и выводы устойчивы (bootstrap) |
| 8 (31 мар–6 апр) | Финальная сдача до 6 апреля | финальный отчёт+код+submission+model artifacts | “release candidate”: CI зелёный, результаты воспроизводимы |
| 9–11 (7–27 апр) | Подготовка к финалу: демо + storytelling | демо‑приложение, слайды, 3‑минутный питч, ответы на Q&A | прогон демо без сбоев |
| 12 (28 апр) | Финал | live‑демо + защита | — |

**Привязка к официальным контрольным точкам.** Даты встреч/дедлайнов (10 фев — рассылка кейса; 17 мар — драфт; 6 апр — финал; 28 апр — финал очно/онлайн) указаны в расписании чемпионата. citeturn2view0  

## Автотестирование и валидация гипотез агентом

Здесь цель — превратить агента в “машину научного метода”: формулировка гипотезы → эксперимент → проверка → документирование → следующий шаг. Для агентных систем важно строить оценивание системно (evals), чтобы снижать вариативность и получать воспроизводимые решения. citeturn12search2turn12search29turn12search26  

### Что именно тестирует агент

1) **Data‑контроль (до обучения)**: схема, пропуски, неожиданные категории, дубликаты ключей, “невозможные” значения; это снижает риск скрытых багов и утечек. citeturn6search10turn6search3  
2) **Leakage‑guardrails**: любые признаки, напрямую зависящие от payout/claim после даты продажи, должны быть исключены (конкретный список зависит от полей; параметр: не указано).  
3) **Model CI tests**: метрики на фиксированном сплите/seed не падают ниже baseline более чем на X (X: не указано).  
4) **Policy tests**: ограничения на цену соблюдены на 100% строк; LR_total/LR_g1/LR_g2 попадают в целевой коридор; share_g1 не деградирует без причины. fileciteturn0file0  
5) **Робастность**: bootstrap‑оценка метрик и доверительные интервалы (чтобы жюри видело стабильность).  
6) **Регрессионные тесты отчёта**: таблицы/цифры в отчёте совпадают с артефактами MLflow run.

### Метрики и протоколы экспериментов: A/B, bandit, ранжирование

**A/B‑подход (offline).** В соревновании нет онлайн‑трафика, поэтому A/B интерпретируем как сравнение двух тарифных политик (или двух моделей) на одинаковом hold‑out / OOF прогнозе.  
- A: baseline policy (например, линейный множитель)  
- B: новая policy (piecewise по квантилям риска)  
Сравниваем по вектору метрик (LR_total/LR_g1/LR_g2/share_g1/Gini, плюс калибровка). citeturn10search1turn10search0  

**Мультиарм‑бандит (управление бюджетом вычислений).** Чтобы не тратить время на 200 “слабых” вариантов, распределяем бюджет на эксперименты адаптивно. Примеры:  
- Arms = {комбинации (model_family, feature_set, pricing_policy)}  
- Reward = Score из раздела таймлайна  
Алгоритмы: Thompson Sampling / UCB / successive halving (реализация — руками или через Optuna pruning; конкретная: не указано). Optuna прямо ориентирован на эффективный поиск и pruning плохих trial’ов. citeturn7search0turn7search18  

**Ранжирование (leaderboard).** Оркестратор после каждого прогона обновляет таблицу экспериментов и сохраняет top‑K (например, K=10) с полным lineage: git_sha, data_hash, config, метрики, артефакты. MLflow Model Registry как раз предоставляет версионирование, lineage и метаданные для жизненного цикла модели. citeturn0search11turn0search21  

### Как “заземлить” агента: строгие схемы, инструменты и трассировка

Чтобы агент был **не “болтливым консультантом”**, а детерминируемым исполнителем, используйте:

- **Structured Outputs (JSON Schema)** для формата “план эксперимента/патч/отчёт”, чтобы агент не мог вернуть некорректную структуру. citeturn12search3turn4search1  
- **Tool calling**: агент не запускает shell напрямую, а вызывает whitelisted инструменты оркестратора. (Это соответствует подходу “agents + tools + guardrails”.) citeturn4search8turn13search14  
- **Agents SDK**: для трассировки шагов, handoff’ов, воспроизводимых прогонов. citeturn13search0turn4search8  
- **AGENTS.md / repository rules**: закрепите “как агент работает в репозитории” (структура, запреты, тест‑правила). Практика использования AGENTS.md в инженерных workflow’ах с Codex описана в кейс‑материалах OpenAI. citeturn4search33turn4search4  

Пример минимального «контракта» для агента (логика, не код):  
- инструмент `propose_experiment(config_schema)` → валидирует схему, создаёт PR с YAML‑конфигом;  
- инструмент `run_experiment(run_id)` → запускает, возвращает run summary;  
- инструмент `evaluate_pricing(run_id)` → считает LR_total/LR_g1/LR_g2/share_g1;  
- инструмент `write_report_snippet(run_id)` → добавляет абзац в отчёт, с ссылками на артефакты.

## Выбор моделей и инструментов

### Рекомендованные семейства моделей для скоринга и ценообразования

Ниже — практичный “набор” моделей, который закрывает (1) качество, (2) объяснимость, (3) риск дисбаланса.

**Базовый интерпретируемый слой (must‑have).**  
- WoE/IV + логистическая регрессия как baseline для вероятности. Требование включать WoE/IV анализ присутствует в ожидаемых материалах отчёта. fileciteturn0file0  

**Основной performance‑слой (скорее всего победный).**  
- CatBoost для бинарной классификации (частота) и регрессии (тяжесть) — особенно если много категориальных признаков; официальная документация подчёркивает поддержку категорий и предупреждает против one‑hot encoding на препроцессинге. citeturn0search2turn0search5  
- Калибровка вероятностей через CalibratedClassifierCV (sigmoid или isotonic) для того, чтобы “вероятность” действительно была вероятностью, что критично для тарифной функции. citeturn10search0turn10search4  

**Актуарная “правильность” агрегата.**  
- Tweedie‑регрессия (вместо двухэтапа) как популярный способ моделировать суммарный убыток с нулевой массой и положительной непрерывной частью. citeturn0search31turn0search19  

#### Таблица сравнения модельных подходов (модели)

| Подход | Что предсказывает | Сильные стороны | Ограничения/риски | Практичный старт гиперпараметров |
|---|---|---|---|---|
| WoE+LogReg | P(claim) | объяснимость, быстрый baseline | может проиграть по качеству GBDT | регуляризация C∈[0.1..10], бины WoE: 5–20 |
| CatBoost (binary) | P(claim) | сильный tabular‑перфоманс, категории “из коробки” citeturn0search2 | нужен контроль утечек и калибровка | depth 6–10; lr 0.02–0.08; iterations 2000–8000; class_weights/scale_pos_weight |
| CatBoost (regression) | E(amount|claim) или log(amount) | устойчивость, удобные метрики | выбросы; нужна трансформация таргета | loss RMSE/log; depth 6–10; l2 3–10 |
| LightGBM Tweedie | E(total_loss) | “одной моделью” покрывает нули+положительные | настройка tweedie_variance_power важна | learning_rate 0.02–0.08; num_leaves 31–255; tweedie_variance_power ≈1.1–1.9 citeturn10search23 |
| Нейросети Freq‑Sev | freq+sev с NN‑функциями | гибкость, может дать прирост citeturn0search13 | сложно объяснять/тюнить в срок | MLP 2–4 слоя; dropout 0.1–0.3; early stopping |
| Зависимые freq‑sev (hurdle) | учёт корреляции | ближе к реальности портфеля citeturn0search4 | усложнение и риск по срокам | используйте только если baseline уже силён |

**Важно про Gini.** В страховом/скоринговом контексте часто используют Gini, и связь с AUC может считаться стандартной: \(G=2\cdot AUC-1\). citeturn10search1  

### Инструменты и стек: агент, оркестрация, трекинг

#### Таблица сравнения инструментов (инструменты)

| Задача/слой | Рекомендация (быстро) | Альтернатива (enterprise/комплексно) | Почему это разумно в кейсе |
|---|---|---|---|
| LLM‑агент | Codex/модель OpenAI для кода (например, GPT‑5.3‑Codex) citeturn4search0 | Open‑source coder (Qwen2.5‑Coder, DeepSeek‑Coder, Code Llama) citeturn5search8turn5search5turn5search30 | Codex‑подобные модели оптимальны для генерации тестов/рефакторинга; open‑source — если нужен локальный режим |
| Агентный фреймворк | OpenAI Agents SDK citeturn13search0 | entity["company","LangChain","framework company"] / LangGraph citeturn13search2; entity["company","LlamaIndex","ai framework company"] citeturn13search3 | Agents SDK даёт трассы и управляемый toolcalling; LangGraph удобен для граф‑workflow |
| Structured outputs | JSON Schema strict mode citeturn12search3 | function calling + валидация schema | минимизирует “галлюцинации” конфигов |
| Оркестрация экспериментов | Hydra + Optuna citeturn7search1turn7search0 | Prefect/Dagster (если нужен UI‑оркестратор) citeturn7search2turn7search7 | Hydra/Optuna быстрее поднять, хватает для соревнования |
| Трекинг | MLflow (runs + registry) citeturn0search11 | W&B (dashboards + sweeps) citeturn6search1 | MLflow проще self‑host; W&B сильнее в визуализациях sweep |
| Версионирование данных | DVC pipelines citeturn6search3 | LakeFS/Delta (не обязательно) | DVC даёт “git‑подобную” воспроизводимость |
| CI/CD | GitHub Actions workflow + artifacts citeturn8search0turn8search13 | GitLab CI/CD pipelines citeturn8search2 | обе платформы понятны жюри, легко показать инженерную зрелость |

### Подбор моделей для агента: коммерческие и open‑source варианты

**Коммерческие/облачные (быстро и мощно).**  
- entity["company","OpenAI","ai company"]: Codex как инженерный агент и модели семейства Codex/GPT для code‑tasks и tool‑use; в 2025–2026 описаны Codex как sandbox‑агент и линейка специализированных agentic coding моделей. citeturn4search4turn4search19turn4search0turn4search7  
- Mistral code‑модели (Codestral) как коммерческая альтернатива для coding/FIM задач. citeturn5search3  

**Open‑source (если нужен локальный режим или бюджет).**  
- entity["company","Mistral AI","ai company"] (open‑weight, Codestral в разных версиях) citeturn5search39turn5search3  
- entity["company","Meta Platforms","tech company"] Code Llama (семейство моделей для кода) citeturn5search30turn5search10  
- entity["company","Alibaba Cloud","cloud company"] Qwen2.5‑Coder (техрепорт и релиз) citeturn5search8turn5search16  
- entity["company","Hugging Face","ai platform company"] как стандартное место для model cards/weights и paper pages (например, DeepSeek‑Coder). citeturn5search5turn5search13  

### Fine-tuning: что реально стоит делать (и что нет) в рамках кейса

**Рекомендация под кейс:** чаще выигрывает **не fine‑tuning**, а строгие схемы + toolcalling + eval‑контур. Structured Outputs как раз предназначен для гарантии соответствия JSON Schema. citeturn12search3turn12search29  

Тем не менее, если вы хотите показать “взрослость” и у вас есть бюджет/время:

- **Supervised fine‑tuning (SFT)**: для улучшения “agent formatting” (правильные планы экспериментов, корректные YAML/JSON). В API описаны гиперпараметры `n_epochs`, `batch_size`, `learning_rate_multiplier` (часто доступны как `auto`). citeturn12search16turn12search20turn12search4  
- **Reinforcement fine‑tuning (RFT)**: если нужно “натренировать” агента следовать рубрикам/грейдерам; RFT привязан к grader‑оценке качества, совместимой с evals/graders. citeturn12search1turn12search13turn12search9  

**Пример “разумных” SFT гиперпараметров для аккуратного старта** (тюнинг зависит от данных; конкретные значения — стартовые, не догма; “идеальные” — не указано):  
- `n_epochs`: 2–4 (или `auto`)  
- `batch_size`: `auto`  
- `learning_rate_multiplier`: 0.05–0.2 (или `auto`)  
Обоснование — общая логика fine‑tuning best practices и описание роли epochs/LR multiplier в документации. citeturn12search4turn12search16  

## Шаблоны артефактов экспериментов и примеры автоматизации

Эта секция даёт «готовые кирпичики» для проекта: лог‑схемы, метрики, отчёт эксперимента, CI/CD и скрипты.

### Шаблон структурированного лога эксперимента (JSON Lines)

```json
{
  "timestamp_utc": "2026-02-15T12:00:00Z",
  "run_id": "mlflow:12345",
  "git_sha": "abc123",
  "data_version": "dvc:train@v7",
  "config": {
    "model_family": "catboost_freq_sev",
    "split": "time_based",
    "seed": 42
  },
  "metrics": {
    "auc_oof": 0.78,
    "gini_oof": 0.56,
    "brier": 0.11,
    "lr_total": 0.701,
    "lr_group1": 0.698,
    "lr_group2": 0.705,
    "share_group1": 0.83
  },
  "constraints": {
    "price_decrease_min": 0.0,
    "price_increase_max_mult": 3.0,
    "violations_count": 0
  },
  "artifacts": {
    "model_path": "models/catboost_freq.cbm",
    "submission_path": "submissions/submission_run12345.csv",
    "report_path": "reports/run12345.md"
  }
}
```

### Шаблон “карточки эксперимента” (Markdown)

```markdown
## Experiment <run_id>

**Hypothesis**: (1–2 предложения)

**Dataset / Split**:
- data_version:
- split_strategy:
- leakage checks: pass/fail

**Model**:
- family:
- target:
- key hyperparams:

**Pricing policy**:
- group split rule:
- multiplier curve:
- constraints handling:

**Results (OOF / Holdout)**:
- AUC / Gini:
- Calibration (Brier/ECE):
- LR_total / LR_g1 / LR_g2:
- share_group1:
- Sensitivity / bootstrap CI:

**Decision**:
- ship / iterate
- next experiments:
```

### Пример конфигурации эксперимента (Hydra YAML)

```yaml
# configs/exp/catboost_freq_sev.yaml
seed: 42

data:
  split:
    type: time_based   # или kfold_grouped; параметр: зависит от данных
  target:
    freq: is_claim
    sev: claim_amount

model:
  freq:
    family: catboost
    params:
      loss_function: Logloss
      eval_metric: AUC
      depth: 8
      learning_rate: 0.04
      iterations: 4000
      l2_leaf_reg: 6
      auto_class_weights: Balanced
  sev:
    family: catboost
    params:
      loss_function: RMSE
      depth: 8
      learning_rate: 0.05
      iterations: 3000

calibration:
  method: sigmoid  # sigmoid|isotonic
  enabled: true

pricing:
  objective_lr: 0.70
  max_increase_mult: 3.0
  max_decrease_pct: 1.0
  policy_family: piecewise_quantile
  params:
    q_increase_start: 0.85
    discount_floor: 0.15
    increase_cap_mult: 3.0
```

Калибровка sigmoid/isotonic как стандартный вариант описана в scikit‑learn. citeturn10search0  

### Пример пайплайна DVC (dvc.yaml)

```yaml
stages:
  validate:
    cmd: python -m src.data.validate --input data/raw/train.csv --out data/interim/validated.parquet
    deps:
      - data/raw/train.csv
      - src/data/validate.py
    outs:
      - data/interim/validated.parquet

  features:
    cmd: python -m src.features.build --input data/interim/validated.parquet --out data/processed/features.parquet
    deps:
      - data/interim/validated.parquet
      - src/features/build.py
    outs:
      - data/processed/features.parquet

  train:
    cmd: python -m src.train.run --config configs/exp/catboost_freq_sev.yaml --features data/processed/features.parquet
    deps:
      - data/processed/features.parquet
      - src/train/run.py
      - configs/exp/catboost_freq_sev.yaml
    outs:
      - models/model_artifact/
    metrics:
      - reports/metrics.json
```

DVC описывает pipelines как DAG из стадий и подчёркивает воспроизводимость и версионирование. citeturn6search0turn6search14  

### Пример CI/CD (GitHub Actions) для “Model CI” и артефактов

```yaml
name: ml-ci

on:
  pull_request:
  push:
    branches: [ main ]

jobs:
  test-and-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          pip install -r requirements.txt

      - name: Unit tests
        run: |
          pytest -q

      - name: Data + pipeline smoke (small sample)
        run: |
          python -m src.pipeline.smoke --rows 5000

      - name: Upload artifacts (metrics, logs, sample submission)
        uses: actions/upload-artifact@v4
        with:
          name: ml-artifacts
          path: |
            reports/metrics.json
            reports/experiment.log.jsonl
            submissions/sample_submission.csv
```

Факт, что workflow’и определяются YAML, а артефакты используются для сохранения результатов прогонов, описан в документации GitHub Actions. citeturn8search12turn8search0  

### Пример “скрипта автоматизации” (Makefile)

```makefile
.PHONY: setup validate features train predict report reproduce

setup:
\tpip install -r requirements.txt

validate:
\tpython -m src.data.validate --input data/raw/train.csv --out data/interim/validated.parquet

features:
\tpython -m src.features.build --input data/interim/validated.parquet --out data/processed/features.parquet

train:
\tpython -m src.train.run --config configs/exp/catboost_freq_sev.yaml --features data/processed/features.parquet

predict:
\tpython -m src.predict.run --model models/model_artifact --input data/raw/test.csv --out submissions/final.csv

report:
\tpython -m src.report.build --run_id latest --out reports/final_report.pdf

reproduce: validate features train predict report
\t@echo "Reproducibility pipeline done."
```

## Демо, риски и первоисточники

### План презентации/демо для жюри (структура и “ключевые аргументы”)

Жюри (и риск‑эксперты) обычно ценят сочетание **точности, бизнеса и инженерной зрелости**; на сайте конкурса подчёркнуты «модель + презентация с аргументацией подходов». citeturn2view0  

**Рекомендуемая структура 8–10 слайдов (7–10 минут).**  
1) Проблема ОГПО как хронически убыточного продукта и почему «усреднённые тарифы» не работают; формула коэффициента выплат и бизнес‑цель ~70%. fileciteturn0file0  
2) Данные и сложности: дисбаланс, сложные связи водитель↔авто↔полис, необходимость frequency–severity. fileciteturn0file0  
3) Архитектура решения (1 диаграмма): где агент, где оркестратор, где pricing engine.  
4) Модельный подход: freq+sev или Tweedie, калибровка вероятностей. citeturn0search31turn10search0  
5) Тарифная политика: как соблюдаем ограничения (−100%…+3×) и как добиваемся LR_total/LR_g1/LR_g2≈0.70. fileciteturn0file0  
6) Результаты: таблица “до/после” (LR_total, share_g1, Gini/AUC) + графики по децилям риска (lift/price change).  
7) Explainability: топ‑факторы, sanity checks, отсутствие утечек (коротко).  
8) Надёжность: CI/CD, воспроизводимость, авто‑эксперименты агентом. Здесь сильный аргумент: Codex‑подобные агенты выполняют инженерные задачи в sandbox‑средах; вы используете их как автотестировщика и ускоритель итераций при строгих guardrails. citeturn4search4turn4search19turn12search3  
9) План внедрения (1 слайд): как это “по‑взрослому” катится в прод (monitoring, registry).  
10) Q&A backup: ограничения, риски, почему выбранная стратегия отвечает соц‑аспекту (аккуратные водители не переплачивают). fileciteturn0file0  

**Демо‑скрипт (2–3 минуты).**  
- Введите (или выберите из тестовых примеров) профиль водителя/ТС → получите P(ДТП), ожидаемый ущерб, коэффициент риска, новую премию и объяснение “почему”.  
- Показать “ползунок” политики (например, порог q_increase_start) и как меняются LR_total и share_g1 (это визуально объясняет оптимизацию).  
- Показать “экспериментальную панель”: последний run, метрики, ссылка на артефакты, воспроизводимость (`make reproduce`).  

### Риски и план смягчения (risk register)

| Риск | Вероятность | Impact | Смягчение |
|---|---:|---:|---|
| Leakage через признаки, связанные с выплатами/расторжениями после даты продажи | Высокая | Критический | запрет “post‑event” фич; time‑split; тесты утечки; ручной аудит top features |
| Несовпадение оффлайн‑качества и бизнес‑метрик (модель “красивая”, LR плохой) | Средняя | Высокий | оптимизировать policy отдельно; multi‑objective Score; калибровка вероятностей citeturn10search0 |
| Переобучение на дисбалансе | Высокая | Высокий | class weights; PR‑AUC доп. метрика; bootstrap‑проверка стабильности |
| Невыполнение LR≈0.70 по группам одновременно | Средняя | Высокий | делайте policy‑оптимизацию как задачу поиска параметров (grid/Optuna); штрафы в Score |
| Срыв сроков из‑за “слишком продовой” архитектуры | Средняя | Высокий | “contest mode”: минимальный стек; всё лишнее — опционально |
| Агент ломает репозиторий/вносит ошибки | Средняя | Высокий | только PR‑режим, обязательные тесты+линтеры, structured outputs, ограниченный toolset citeturn12search3turn8search3 |
| Невоспроизводимость результатов | Средняя | Высокий | фикс seeds, DVC/MLflow lineage, артефакты в CI, один “reproduce” путь citeturn6search11turn0search11 |
| Перекос “соц‑справедливости” (слишком много повышений) | Средняя | Средний | оптимизация share_g1 как явная часть Score; отчёт по децилям риска; ограничение на рост цены fileciteturn0file0 |

### Список первоисточников и ресурсов

- Сайт чемпионата и расписание (ключевые даты 10.02.2026, 17.03.2026, 06.04.2026, 28.04.2026). citeturn2view0  
- Условия кейса и бизнес‑метрики (коэффициент выплат, цель ~70% по портфелю и группам, ограничения на изменение премии, рекомендация frequency–severity, состав сдачи). fileciteturn0file0  
- Frequency–severity как типовой страховой подход (обзор в актуарной литературе). citeturn0search16turn0search10  
- Tweedie как популярная модель агрегированного убытка (смесь для нулей/положительных значений). citeturn0search31turn0search19  
- CatBoost: обработка категориальных признаков и рекомендации по работе с ними. citeturn0search2turn0search5  
- Связь Gini и ROC‑AUC (G = 2·AUC − 1) в scikit‑learn. citeturn10search1  
- Калибровка вероятностей (sigmoid/isotonic) в scikit‑learn. citeturn10search0  
- DVC: пайплайны и versioning данных/моделей. citeturn6search0turn6search11  
- MLflow Model Registry: версионирование моделей и lineage. citeturn0search11turn0search21  
- OpenAI Structured Outputs (JSON Schema) и Evals/Agent evals как основа eval‑driven разработки агента. citeturn12search3turn12search2turn12search26  
- OpenAI Codex и агентные workflow’и (sandbox‑подход, Codex app, инженерные практики). citeturn4search4turn4search19turn4search33