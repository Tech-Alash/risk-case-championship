# EDA Summary

## Executive Summary
- Train shape: [569508, 159]
- Test shape: [244073, 156]
- Claim rate: 0.019480
- Contracts in train: 180635
- Mean rows per contract: 3.1528

## Key Findings
- Train-only target columns: ['claim_amount', 'claim_cnt', 'is_claim']
- Top sparse features are concentrated in SCORE_11/SCORE_12 blocks.
- Driver-level accounting can multiply financial totals if not aggregated by contract.
- Missing share for claim-related fields is high by design (sparse events).

## Policy vs Driver
- Driver-level premium sum: 7568585612.00
- Policy-level premium sum (max per contract): 2381834852.00
- Duplication factor (driver/policy): 3.1776

## Recommended Actions
- Keep policy-level as the canonical training layer for financial KPI consistency.
- Apply strict coverage and signal filtering for SCORE_11 and SCORE_12 blocks.
- Retain explicit missing flags for sparse feature groups.
- Keep identifier fields (iin, car_number, unique_id) in the training blacklist.
- Track distribution shifts introduced by driver->policy aggregation.

## Artifacts
- tables: `artifacts/eda/tables`
- figures: `artifacts/eda/figures`
- metadata: `artifacts/eda/metadata/eda_profile.json`
- feature selection: `artifacts/eda/feature_selection`