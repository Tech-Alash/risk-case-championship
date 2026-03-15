"""Quick integration test for all new architecture modules."""
import sys
import numpy as np

# -- 1. WoE baseline --
from risk_case.models.woe_baseline import WoEFrequencySeverityModel, compute_woe_iv
from risk_case.models.benchmark import _build_candidate_model
woe = _build_candidate_model("woe_freq_sev", 300, 1.0, 42, {})
assert type(woe).__name__ == "WoEFrequencySeverityModel"
print(f"[OK] WoE model created: {type(woe).__name__}")

# -- 2. Tweedie aggregate --
from risk_case.models.tweedie_model import TweedieAggregateLossModel
tweedie = _build_candidate_model("tweedie_aggregate", 300, 1.0, 42, {})
assert type(tweedie).__name__ == "TweedieAggregateLossModel"
print(f"[OK] Tweedie model created: {type(tweedie).__name__}")

# -- 3. Bootstrap CI --
from risk_case.models.bootstrap_ci import compute_bootstrap_ci, bootstrap_ci_dataframe
y_true = np.random.RandomState(42).randint(0, 2, 200)
p_pred = np.random.RandomState(42).rand(200)
ci = compute_bootstrap_ci(y_true, p_pred, n_bootstrap=100)
assert "auc" in ci
assert ci["auc"]["lower"] is not None
ci_df = bootstrap_ci_dataframe(ci)
assert len(ci_df) >= 2
print(f"[OK] Bootstrap CI: AUC=[{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}]")

# -- 4. SHAP module import --
from risk_case.explainability.shap_analysis import generate_shap_report, compute_shap_values
print("[OK] SHAP module imported successfully")

# -- 5. Benchmark _build_candidate_model knows all candidates --
known = ["baseline_freq_sev", "woe_freq_sev", "catboost_freq_sev",
         "catboost_dep_freq_sev", "tweedie_aggregate"]
for name in known:
    try:
        m = _build_candidate_model(name, 300, 1.0, 42, {})
        print(f"[OK] Candidate '{name}': {type(m).__name__}")
    except ImportError as e:
        print(f"[SKIP] Candidate '{name}': optional dependency missing ({e})")
    except Exception as e:
        print(f"[FAIL] Candidate '{name}': {e}")
        sys.exit(1)

print("\n=== ALL INTEGRATION TESTS PASSED ===")
