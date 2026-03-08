# Feature Selection Report

## Summary
- Candidate features: 152
- Whitelist: 133
- Droplist: 11
- Review list: 12

## Rules
- missing_drop_threshold: 0.9
- missing_review_threshold: 0.6
- corr_review_threshold: 0.08

## Force Lists
- force_keep: ['premium', 'premium_wo_term']
- force_drop: ['unique_id', 'contract_number', 'driver_iin', 'insurer_iin', 'car_number', 'is_claim', 'claim_amount', 'claim_cnt']

## Artifacts
- whitelist: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\feature_selection\feature_whitelist.csv`
- droplist: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\feature_selection\feature_droplist.csv`
- review list: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\feature_selection\feature_review_list.csv`
- summary json: `C:\Users\Admin\Desktop\Риск кейс чемпоинат\artifacts\eda\feature_selection\feature_selection_summary.json`