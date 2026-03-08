# Deep EDA Summary (утечки / drift / сегменты)

- Generated at (UTC): 2026-02-20T16:29:05.089896+00:00
- Train rows: 569,508, columns: 159
- Test rows: 244,073, columns: 156
- Policy rows (aggregated): 180,635

## Ключевые выводы
- Claim rate: 0.019480
- Leakage high-risk features (risk_score >= 90): 3
- Drift alerts (PSI >= 0.25): 0
- Proposed final drop features: 8
- Proposed review features: 7

## Артефакты
- Tables: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\tables`
- Figures: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\figures`
- Main leakage table: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\tables\leakage_risk_rank.csv`
- Main drift table: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\tables\drift_psi_by_feature.csv`
- Segment KPI table: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\tables\segment_kpi_table.csv`
- Action board: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\deep\tables\feature_action_board.csv`