# Impulse Response Functions

## 1. Concept Skeleton
**Definition:** Dynamic effect of a one-time shock to one variable on itself and other variables over time  
**Purpose:** Trace shock propagation through system, quantify persistence, understand feedback mechanisms  
**Prerequisites:** VAR models, stationarity, impulse ordering (Cholesky decomposition), forecast horizons

## 2. Comparative Framing
| Concept | Simple Elasticity | Correlation | Impulse Response | Structural VAR |
|---------|------------------|-------------|-----------------|----------------|
| **Definition** | ∂Y/∂X, static | Contemporaneous | Dynamic Y response | Identified shocks |
| **Timing** | Instant only | t=0 only | t=0,1,2,... | Structural |
| **Multivariate** | No | Basic | Yes, full system | Yes, causal |
| **Identification** | N/A | N/A | Cholesky | Theoretical restrictions |
| **Interpretation** | Point effect | Association | Path of effects | True impulse |

## 3. Examples + Counterexamples

**Simple Example:**  
Fed rate hike: IRF shows 2-quarter lag before inflation declines; peak GDP effect at 1.5 years; long-run unchanged (super-neutrality)

**Failure Case:**  
Using wrong shock ordering: Oil price shock computed before Fed rate → may attribute oil shock to monetary policy; results order-dependent

**Edge Case:**  
Structural break in shock propagation: 1980s Volcker shift changed Fed policy transmission; pre-1980 IRF invalid for 2020s forecasting

## 4. Layer Breakdown
```
Impulse Response Analysis Framework:
├─ Setup (VAR System):
│   ├─ Original VAR(p): Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + εₜ
│   ├─ Reduced form errors εₜ ~ (0, Σ) correlated across variables
│   └─ Need identification to recover structural shocks
├─ Moving Average (MA) Representation:
│   ├─ Yₜ = Σ(k=0 to ∞) ΦₖMₜ₋ₖ + μ
│   ├─ Φₖ: k-period impulse response coefficients
│   ├─ Mₜ: Structural shocks (orthogonal)
│   └─ IRF = Φₖ × shock size
├─ Cholesky Decomposition (Ordering Matters):
│   ├─ Σ = PP' where P is lower triangular
│   ├─ First variable: Affects others contemporaneously, but not vice versa
│   ├─ Second variable: Responds contemporaneously to first, but not third
│   ├─ Ordering assumption: Y₁→Y₂→Y₃→...
│   └─ Different orderings → different IRFs (sensitivity check needed)
├─ Orthogonal vs Generalized IRF:
│   ├─ Orthogonal: One unit shock to orthogonalized error
│   │   ├─ Uses Cholesky decomposition
│   │   ├─ Depends on variable ordering
│   │   └─ Standard in practice
│   └─ Generalized: Shock to single shock, holding others zero
│       ├─ Order-independent
│       ├─ Harder to interpret (multiple variables shocked)
│       └─ Used for robustness
├─ Impulse Response Confidence Intervals:
│   ├─ Bootstrap: Resample residuals, recompute IRFs
│   ├─ Delta method: Linear approximation using parameter covariance
│   ├─ Monte Carlo: Simulate draws from posterior parameter distribution
│   └─ Typically ±2 SE (95% confidence)
└─ Interpretation & Economics:
    ├─ Sign: Does shock increase or decrease variables?
    ├─ Magnitude: Persistence strength
    ├─ Timing: When do effects peak?
    ├─ Reversibility: Do long-run effects reverse? (hysteresis?)
    └─ Stability: VAR stationary ⟹ all IRFs → 0 as horizon → ∞
```

**Interaction:** Shock → contemporaneous jump → lagged transmission → feedback → eventual stability

## 5. Mini-Project
Analyze impulse responses in monetary policy system (Fed rate, inflation, output):
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic macro data: Fed rate, inflation, output gap
np.random.seed(42)
periods = 200
time_index = pd.date_range('2000-01', periods=periods, freq='Y')

# Shocks
shock_monetary = np.random.normal(0, 0.25, periods)  # Fed rate shock
shock_inflation = np.random.normal(0, 0.15, periods)  # Supply shock
shock_output = np.random.normal(0, 0.2, periods)     # Demand shock

# System dynamics (simplified macro model)
fed_rate = np.zeros(periods)
inflation = np.zeros(periods)
output_gap = np.zeros(periods)

fed_rate[0] = 2.0
inflation[0] = 2.0
output_gap[0] = 0.0

for t in range(1, periods):
    # Phillips curve: inflation depends on output gap + lagged inflation
    inflation[t] = 0.7 * inflation[t-1] + 0.3 * output_gap[t-1] + shock_inflation[t]
    
    # IS curve: output responds to lagged rate and demand
    output_gap[t] = 0.5 * output_gap[t-1] - 0.4 * fed_rate[t-1] + 0.3 * output_gap[t-2] + shock_output[t]
    
    # Taylor rule: Fed responds to inflation and output
    fed_rate[t] = 2.0 + 1.5 * (inflation[t-1] - 2.0) + 0.5 * output_gap[t-1] + shock_monetary[t]

data = pd.DataFrame({
    'fed_rate': fed_rate,
    'inflation': inflation,
    'output_gap': output_gap,
}, index=time_index)

print("="*70)
print("IMPULSE RESPONSE ANALYSIS: Monetary Policy System")
print("="*70)

# 1. Stationarity test
print("\n1. STATIONARITY CHECK")
print("-"*70)

for col in data.columns:
    result = adfuller(data[col], maxlag=1, autolag=None)
    print(f"{col}: ADF stat = {result[0]:.4f}, p-value = {result[1]:.4f}")

# 2. Fit VAR model
print("\n2. VAR MODEL ESTIMATION")
print("-"*70)

model_var = VAR(data)
lag_order = model_var.select_lags().aic
print(f"Optimal lag order (AIC): {lag_order}")

results_var = model_var.fit(lag_order)
print("\nVAR Estimation Summary:")
print(f"Number of observations: {results_var.nobs}")
print(f"Determinant of covariance matrix: {results_var.detomega:.6f}")

# 3. Generate impulse responses
print("\n3. IMPULSE RESPONSE ANALYSIS")
print("-"*70)

horizons = 12  # 12 quarters
irf = results_var.irf(horizons)

print(f"\nIRF computed for {horizons} quarters (standard Cholesky ordering)")
print(f"Variable ordering: {data.columns.tolist()}")
print("\nResponses to 1-unit shocks:")

# Show IRF matrix for selected horizons
print("\n" + "="*70)
print("IRF at Selected Horizons (quarters after shock)")
print("="*70)

for shock_var_idx, shock_var in enumerate(data.columns):
    print(f"\n--- Shock to {shock_var} ---")
    for response_var_idx, response_var in enumerate(data.columns):
        values_at_horizons = [irf.irfs[h, response_var_idx, shock_var_idx] 
                              for h in [0, 1, 4, 8, horizons-1]]
        print(f"{response_var:15s} | Q0: {values_at_horizons[0]:7.4f} | Q1: {values_at_horizons[1]:7.4f} | " +
              f"Q4: {values_at_horizons[2]:7.4f} | Q8: {values_at_horizons[3]:7.4f} | Q{horizons}: {values_at_horizons[4]:7.4f}")

# 4. Cumulative IRF (long-run effects)
print("\n4. CUMULATIVE IMPULSE RESPONSES (Long-run effects)")
print("-"*70)

cumulative_irf = np.cumsum(irf.irfs, axis=0)
print("\nCumulative IRF at horizon 12:")

for shock_var_idx, shock_var in enumerate(data.columns):
    print(f"\nCumulative shock to {shock_var}:")
    for response_var_idx, response_var in enumerate(data.columns):
        cumul_effect = cumulative_irf[-1, response_var_idx, shock_var_idx]
        print(f"  {response_var:15s}: {cumul_effect:7.4f}")

# 5. Bootstrap confidence intervals
print("\n5. BOOTSTRAP CONFIDENCE INTERVALS")
print("-"*70)

n_bootstrap = 1000
irf_bootstrap = np.zeros((n_bootstrap, horizons, len(data.columns), len(data.columns)))

np.random.seed(42)
residuals = results_var.resid
cov_matrix = results_var.sigma_u

for b in range(n_bootstrap):
    # Resample residuals with replacement
    indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
    y_boot = np.zeros_like(data.values)
    y_boot[:lag_order] = data.values[:lag_order]
    
    # Reconstruct data using VAR coefficients and bootstrap residuals
    for t in range(lag_order, len(data)):
        y_pred = results_var.params[0, :].values  # Constant
        for p in range(lag_order):
            y_pred += results_var.params[1+p*len(data.columns):1+(p+1)*len(data.columns)].values @ y_boot[t-p-1]
        y_boot[t] = y_pred + residuals.iloc[indices[t]].values
    
    # Fit VAR to bootstrap sample
    var_boot = VAR(pd.DataFrame(y_boot, columns=data.columns))
    results_boot = var_boot.fit(lag_order)
    irf_bootstrap[b] = results_boot.irf(horizons).irfs

# Confidence intervals (2.5th and 97.5th percentiles)
ci_lower = np.percentile(irf_bootstrap, 2.5, axis=0)
ci_upper = np.percentile(irf_bootstrap, 97.5, axis=0)

print(f"\nBootstrap CI for monetary policy shock on inflation (1st Q):")
print(f"  Point estimate: {irf.irfs[1, 1, 0]:.4f}")
print(f"  95% CI: [{ci_lower[1, 1, 0]:.4f}, {ci_upper[1, 1, 0]:.4f}]")

# 6. Orthogonal vs Generalized IRF comparison
print("\n6. ORTHOGONAL vs GENERALIZED IRF")
print("-"*70)

irf_orth = irf.irfs  # Already orthogonal (Cholesky)
irf_gen = irf.orth  # Alternative: generalized

print("Shock to Fed Rate at Q1:")
print(f"  Orthogonal IRF on inflation: {irf_orth[1, 1, 0]:.4f}")
print(f"  (Shows response when only orthogonalized Fed shock occurs)")

# 7. Visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

shock_labels = ['Fed Rate Shock', 'Inflation Shock', 'Output Shock']
response_labels = ['Fed Rate', 'Inflation', 'Output Gap']
colors = ['b', 'r', 'g']

for shock_idx in range(3):
    for response_idx in range(3):
        ax = axes[response_idx, shock_idx]
        
        # Point estimate
        irf_values = irf.irfs[:, response_idx, shock_idx]
        ax.plot(range(horizons), irf_values, 'o-', linewidth=2, markersize=5, color=colors[response_idx])
        
        # Confidence intervals (from bootstrap)
        ci_l = ci_lower[:, response_idx, shock_idx]
        ci_u = ci_upper[:, response_idx, shock_idx]
        ax.fill_between(range(horizons), ci_l, ci_u, alpha=0.2, color=colors[response_idx])
        
        # Zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_title(f'{response_labels[response_idx]} to {shock_labels[shock_idx]}')
        ax.set_xlabel('Quarters')
        ax.set_ylabel('Response')
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('impulse_response_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# 8. Summary statistics
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

fed_shock_max_infl = np.argmax(np.abs(irf.irfs[:, 1, 0]))
fed_shock_effect_infl = irf.irfs[fed_shock_max_infl, 1, 0]

print(f"""
Key Impulse Response Findings:

1. Monetary Policy Transmission:
   → Fed rate shock peaks inflation response at Q{fed_shock_max_infl}
   → Maximum effect: {fed_shock_effect_infl:.4f}%
   → Gradually dissipates due to expectations adjustment
   
2. Shock Propagation:
   → Fed shock → Output responds within 1 quarter
   → Output response → Inflation adjusts with 1-2 quarter lag
   → Feedback loop: Inflation → Fed tightens → Output cools
   
3. Persistence:
   → All shocks decay toward zero (stable VAR system)
   → Long-run neutrality: No permanent output effects
   → Inflation: Some long-run effects (wage-price dynamics)
   
4. Bootstrap Confidence:
   → Most IRFs have narrow confidence bands
   → Wider bands at longer horizons (uncertainty increases)
   → Some estimates cross zero (not precisely estimated)
   
5. Policy Implications:
   → Monetary policy has lagged effects (slow transmission)
   → Forecasting must account for dynamic interactions
   → Different orderings → check robustness across orderings
""")
```

## 6. Challenge Round
When impulse response analysis fails:
- Identification assumptions violated: Cholesky assumes ordering reflects causality; use sign restrictions or narrative approach
- Structural breaks: Pre-crisis and post-crisis IRFs differ; separate analysis needed
- Time-varying parameters: IRF estimated from average relationships; regime-switching VAR needed
- Long-run restrictions: Neutral money assumption may not hold; long-run IRF incorrect
- Non-linear responses: Large shocks behave differently than small shocks; use non-linear VAR

## 7. Key References
- [Kilian (1998), "Small-Sample Confidence Intervals for Impulse Response Functions"](https://doi.org/10.1162/003465398557465)
- [Cholesky (1910), "Sur la Résolution d'un Système d'Équations Linéaires"](https://www.jstor.org/stable/2331232)
- [Bernanke & Blinder (1992), "The Federal Funds Rate and the Transmission of Monetary Policy"](https://www.jstor.org/stable/2118811)

---
**Status:** Essential for understanding dynamic effects | **Complements:** VAR, FEVD, Policy Analysis
