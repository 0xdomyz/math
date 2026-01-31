# Granger Causality

## 1. Concept Skeleton
**Definition:** X Granger-causes Y if past values of X improve predictions of Y beyond Y's own history  
**Purpose:** Test predictive precedence (not true causality), identify lead-lag relationships, structure VAR models  
**Prerequisites:** Stationarity, lag selection, no simultaneous causality, two-way causality possible

## 2. Comparative Framing
| Concept | Correlation | Granger Causality | True Causality |
|---------|------------|-----------------|----------------|
| **Definition** | Co-movement | Predictive precedence | Mechanism |
| **Direction** | Symmetric | Directional (X→Y) | Directional |
| **Timing** | Concurrent | X leads Y | X before Y |
| **Test Method** | Correlation coefficient | F-test on coefficients | Experiment/IV |
| **Interpretation** | Association | Prediction | Cause → Effect |

## 3. Examples + Counterexamples

**Simple Example:**  
Oil price → stock returns: Past 3 months oil price predicts stock returns significantly; Granger-causal at 5% level. Not true causation (both respond to geopolitics)

**Failure Case:**  
Interest rates → inflation (reverse of true causation): Rates Granger-cause inflation when inflation actually drives rate expectations → test both directions

**Edge Case:**  
Contemporaneous causality missed: Two variables move simultaneously, neither Granger-causes other; bidirectional feedback → need higher frequency data or simultaneous system

## 4. Layer Breakdown
```
Granger Causality Testing Framework:
├─ Setup (Two Variables):
│   ├─ Assumption: Both Yₜ and Xₜ stationary I(0)
│   ├─ If I(1): Use differenced data or VECM (cointegrated)
│   ├─ If cointegrated: Granger causality exists (Representation Theorem)
│   └─ No causality: Use VAR in levels or differences
├─ Granger Causality Definition:
│   ├─ X Granger-causes Y if:
│   │   E[Yₜ | Yₜ₋₁, Yₜ₋₂, ...] ≠ E[Yₜ | Yₜ₋₁, Yₜ₋₂, ..., Xₜ₋₁, Xₜ₋₂, ...]
│   ├─ Past X improves Y forecasts
│   └─ Precedence in time (not true causation)
├─ Restricted vs Unrestricted Model:
│   ├─ Restricted: Yₜ = φ₁·Yₜ₋₁ + ... + φₚ·Yₜ₋ₚ + εₜ
│   ├─ Unrestricted: Yₜ = φ₁·Yₜ₋₁ + ... + φₚ·Yₜ₋ₚ + ψ₁·Xₜ₋₁ + ... + ψₚ·Xₜ₋ₚ + εₜ
│   ├─ H₀: ψ₁ = ψ₂ = ... = ψₚ = 0 (X does NOT Granger-cause Y)
│   └─ H₁: ∃ ψⱼ ≠ 0 (X Granger-causes Y)
├─ Test Statistic:
│   ├─ F-test: F = [(SSRᵣ - SSRᵤ)/p] / [SSRᵤ/(T-2p-1)]
│   ├─ Follows F(p, T-2p-1) under H₀
│   ├─ Reject H₀ if F > Fα(p, T-2p-1)
│   └─ Also: Wald test, likelihood ratio test
├─ Lag Selection:
│   ├─ Too few lags: Omitted variables bias
│   ├─ Too many lags: Loss of power (degrees of freedom)
│   ├─ AIC/BIC: Balance fit and parsimony
│   └─ Typically p=4-12 for economic data
└─ Interpretation Issues:
    ├─ Granger ≠ True causation (X could be proxy for Z)
    ├─ Bidirectional possible (X→Y and Y→X)
    ├─ Non-linear relationships: Linear Granger misses
    └─ Leads/lags: X₊₁ may "cause" Yₜ (information leakage)
```

**Interaction:** Lag structure → coefficient significance → prediction improvement

## 5. Mini-Project
Test Granger causality between money supply and inflation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, var_model
from statsmodels.tsa.api import VAR
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic money supply and inflation (money → inflation with lag)
np.random.seed(42)
periods = 200
time_index = pd.date_range('2000-01', periods=periods, freq='Y')

# Money supply: random walk (I(1))
money = np.cumsum(np.random.normal(0.02, 0.015, periods))

# Inflation: depends on lagged money supply + shocks
inflation = np.zeros(periods)
inflation[0] = np.random.normal(0.02, 0.01)
for t in range(1, periods):
    # Inflation depends on lagged money (Granger causality)
    inflation[t] = 0.3 * inflation[t-1] + 0.4 * money[t-1] + np.random.normal(0, 0.008)

data = pd.DataFrame({
    'money_supply': money,
    'inflation': inflation,
}, index=time_index)

print("="*70)
print("GRANGER CAUSALITY: Money Supply → Inflation")
print("="*70)

# 1. Test stationarity
print("\n1. STATIONARITY TESTS")
print("-"*70)

for col in data.columns:
    result = adfuller(data[col], autolag='AIC')
    print(f"\n{col}:")
    print(f"  ADF stat: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    is_stationary = result[1] < 0.05
    print(f"  I(1) (needs differencing): {not is_stationary}")

# Difference non-stationary series
data_diff = data.diff().dropna()

print("\n\nAfter First Differencing:")
for col in data_diff.columns:
    result = adfuller(data_diff[col])
    print(f"\n{col}:")
    print(f"  ADF stat: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")

# 2. Granger causality test: Money → Inflation
print("\n" + "="*70)
print("2. GRANGER CAUSALITY TEST (Differenced Data)")
print("="*70)

print("\nH₀: Money Supply does NOT Granger-cause Inflation")
print("-"*70)

# Test with different lag lengths
lag_range = range(1, 5)

for lag in lag_range:
    print(f"\n\nLag length: {lag}")
    print("-"*40)
    
    gc_result = grangercausalitytests(data_diff[['inflation', 'money_supply']], 
                                      maxlag=lag, verbose=True)

# 3. Bidirectional test
print("\n" + "="*70)
print("3. BIDIRECTIONAL CAUSALITY CHECK")
print("="*70)

print("\nH₀: Inflation does NOT Granger-cause Money Supply")
print("-"*70)

gc_result_reverse = grangercausalitytests(data_diff[['money_supply', 'inflation']], 
                                          maxlag=4, verbose=True)

# 4. Manual implementation with VAR
print("\n" + "="*70)
print("4. VAR MODEL & CAUSALITY INTERPRETATION")
print("="*70)

# Fit VAR(2)
model_var = VAR(data_diff)
lag_order = model_var.select_lags().aic
print(f"\nOptimal lag order (AIC): {lag_order}")

results_var = model_var.fit(lag_order)
print("\nVAR Summary:")
print(results_var.summary())

# Extract coefficients
print("\n\nCoefficient Analysis:")
print("-"*70)

# Money equation
money_eq_coefs = results_var.params.iloc[:, 0]
print(f"\nMoney Supply equation (ΔMoney):")
for idx, coef in enumerate(money_eq_coefs):
    print(f"  {money_eq_coefs.index[idx]}: {coef:.6f}")

# Inflation equation
infl_eq_coefs = results_var.params.iloc[:, 1]
print(f"\nInflation equation (ΔInflation):")
for idx, coef in enumerate(infl_eq_coefs):
    print(f"  {infl_eq_coefs.index[idx]}: {coef:.6f}")

# Check if money terms significant in inflation equation
money_cols = [i for i, col in enumerate(infl_eq_coefs.index) if 'money' in col.lower()]
infl_money_coefs = infl_eq_coefs.iloc[money_cols]
print(f"\nMoney coefficients in inflation equation: {infl_money_coefs.values}")
print(f"Significant at 5% level: {(np.abs(infl_money_coefs.values) > 1.96*0.01).any()}")

# 5. Impulse response analysis
print("\n" + "="*70)
print("5. IMPULSE RESPONSE: Money Shock → Inflation")
print("="*70)

irf = results_var.irf(10)
print("\nImpulse Response: 1-unit shock to Money Supply")
print(irf.irfs)

# 6. Forecast Error Variance Decomposition (FEVD)
print("\n" + "="*70)
print("6. FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("="*70)

fevd = results_var.fevd(10)
print("\nVariance of Inflation explained by:")
print("  Money Supply shock: ", fevd.decomp[:, 1, 0])  # 2nd variable's shock on inflation
print("  Inflation shock: ", fevd.decomp[:, 1, 1])    # 1st variable's own shock

# 7. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Raw series
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(data.index, data['money_supply'], 'b-', linewidth=2, label='Money Supply')
ax2.plot(data.index, data['inflation'], 'r-', linewidth=2, label='Inflation')
ax.set_ylabel('Money Supply', color='b')
ax2.set_ylabel('Inflation', color='r')
ax.set_title('Money Supply and Inflation')
ax.grid(alpha=0.3)

# Plot 2: First differences
ax = axes[0, 1]
ax.plot(data_diff.index, data_diff['money_supply'], 'b-', linewidth=1, label='ΔMoney', alpha=0.7)
ax.plot(data_diff.index, data_diff['inflation'], 'r-', linewidth=1, label='ΔInflation', alpha=0.7)
ax.set_title('First Differences (Stationary)')
ax.set_ylabel('Change')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Scatter (lagged relationship)
ax = axes[0, 2]
lags_plot = 1
ax.scatter(data_diff['money_supply'].shift(lags_plot), data_diff['inflation'], alpha=0.6, s=50)
# Fit line
slope, intercept, r_value, _, _ = stats.linregress(
    data_diff['money_supply'].shift(lags_plot).dropna(), 
    data_diff['inflation'].loc[data_diff['inflation'].index >= data_diff['money_supply'].shift(lags_plot).index.min()]
)
x_line = np.array([data_diff['money_supply'].min(), data_diff['money_supply'].max()])
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'β={slope:.4f}, R²={r_value**2:.3f}')
ax.set_xlabel('ΔMoney (t-1)')
ax.set_ylabel('ΔInflation (t)')
ax.set_title('Lagged Relationship')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Impulse response
ax = axes[1, 0]
horizons = range(10)
irf_money_on_infl = irf.irfs[:, 1, 0]  # Money shock → Inflation
ax.plot(horizons, irf_money_on_infl, 'o-', linewidth=2, markersize=6)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Quarters after shock')
ax.set_ylabel('Inflation response')
ax.set_title('Impulse Response: Money Shock → Inflation')
ax.grid(alpha=0.3)

# Plot 5: FEVD stacked area
ax = axes[1, 1]
horizons = range(10)
money_contrib = fevd.decomp[:, 1, 0]  # Money shock contribution to inflation var
infl_contrib = fevd.decomp[:, 1, 1]   # Inflation shock contribution
ax.fill_between(horizons, 0, money_contrib*100, label='Money shock', alpha=0.7)
ax.fill_between(horizons, money_contrib*100, 100, label='Inflation shock', alpha=0.7)
ax.set_xlabel('Quarters')
ax.set_ylabel('Variance %')
ax.set_title('Forecast Error Variance Decomposition')
ax.set_ylim([0, 100])
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Residuals
ax = axes[1, 2]
residuals = results_var.resid
ax.plot(residuals.iloc[:, 1], linewidth=1, alpha=0.7, label='Inflation residuals')
ax.axhline(y=0, color='k', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Residual')
ax.set_title('VAR Residuals')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('granger_causality_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION SUMMARY")
print("="*70)
print(f"""
Key Findings:
1. Money Granger-causes Inflation:
   → Past money supply predicts inflation beyond past inflation alone
   → Supports monetary theory of inflation
   
2. VAR coefficients show:
   → Inflation responds to lagged money with {infl_eq_coefs.iloc[2]:.4f} elasticity
   → Money supply responds to lagged inflation with {money_eq_coefs.iloc[3]:.4f} elasticity
   
3. Impulse response shows:
   → Money shock impact peaks at quarter {np.argmax(irf_money_on_infl)}
   → Cumulative effect: {irf_money_on_infl.sum():.4f}
   
4. FEVD shows:
   → Money explains {money_contrib[-1]*100:.1f}% of inflation variance
   → But causation vs correlation unclear (both respond to expectations)
   
5. Important caveats:
   → Granger causality ≠ True causality
   → Both could respond to central bank credibility announcements
   → Use as diagnostic tool, not causal proof
""")
```

## 6. Challenge Round
When Granger causality is misleading:
- Omitted common cause: Z causes both X and Y → spurious Granger causality
- Lead/lag misspecification: Using forward-looking data (info leakage) → false causality
- Non-linear relationships: Quarterly aggregation masks weekly causality
- Structural breaks: Causality direction reverses pre/post shock; subsample test
- Multiple time scales: Daily causality differs from monthly; use multiresolution approach

## 7. Key References
- [Granger (1969), "Investigating Causal Relations by Econometric Models and Cross-Spectral Methods"](https://www.jstor.org/stable/1912791)
- [Toda & Yamamoto (1995), "Statistical Inference on Cointegrating (or Cofeature) Rank in High-Dimensional Systems"](https://www.jstor.org/stable/2171817)
- [Newey & West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"](https://www.jstor.org/stable/1912934)

---
**Status:** Diagnostic tool for causality direction | **Complements:** VAR, VECM, Cointegration
