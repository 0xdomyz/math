# Error Correction Model (ECM)

## 1. Concept Skeleton
**Definition:** Dynamic model capturing short-run adjustments toward long-run equilibrium; decomposes changes into error correction and short-run dynamics  
**Purpose:** Model transitory deviations from equilibrium, quantify adjustment speed, separate long-run from short-run effects  
**Prerequisites:** Cointegration, VECM, error correction terms, dynamic relationships

## 2. Comparative Framing
| Model | OLS Regression | Differenced Model | ECM | VECM |
|-------|----------------|-------------------|-----|------|
| **Levels vs Differences** | Levels | Differences only | Both | Both |
| **Long-run** | Implicit | Lost (overcorrection) | Explicit | Explicit |
| **Short-run** | Mixed | Only | Explicit | Explicit |
| **Cointegration** | Ignores | Ignores | Requires | Requires |
| **Efficiency** | Inefficient if I(1) | Loses information | Efficient | Efficient |

## 3. Examples + Counterexamples

**Simple Example:**  
Inflation target = 2%, actual = 3.5%: ECM shows inflation partially reverts toward target each period with adjustment speed λ = 0.1, implying 10% gap closure per period

**Failure Case:**  
No cointegration but force ECM: Error term explosive, adjustment coefficient unreliable, forecasts worse than differenced model

**Edge Case:**  
Lag length misspecification: Include insufficient lags → residual autocorrelation biases adjustment coefficient; test with Ljung-Box

## 4. Layer Breakdown
```
Error Correction Model (ECM):
├─ Setup: Two cointegrated variables Xₜ ~ I(1), Yₜ ~ I(1)
│   ├─ Long-run: Yₜ = β₀ + β₁·Xₜ (cointegrating equation)
│   ├─ Equilibrium error: ECₜ = Yₜ - β₀ - β₁·Xₜ
│   └─ If ECₜ ≠ 0, system out of equilibrium → adjustment needed
├─ Single Equation ECM:
│   ├─ ΔYₜ = α·ECₜ₋₁ + Σφᵢ·ΔYₜ₋ᵢ + Σψⱼ·ΔXₜ₋ⱼ + εₜ
│   ├─ α: adjustment coefficient (typically -0.1 to -0.5)
│   ├─ If α close to 0: Slow adjustment (inertia)
│   ├─ If α = -1: Full correction each period (overshooting risk)
│   └─ Half-life = ln(0.5)/ln(1+α) quarters to half-adjust
├─ Vector ECM (VECM):
│   ├─ ΔYₜ = Π·Yₜ₋₁ + ΓΔYₜ₋₁ + εₜ
│   ├─ Π = α·β' (cointegrating vector β, adjustment α)
│   ├─ Multiple cointegrating relationships r > 1
│   └─ Johansen decomposition identifies ranks
├─ Granger Representation Theorem:
│   ├─ Cointegrated system ⟺ ECM representation exists
│   ├─ No VECM without cointegration
│   └─ Speed of adjustment quantifies mean-reversion strength
└─ Interpretation:
    ├─ EC coefficient: % gap closed per period
    ├─ AR coefficients: Short-run momentum
    ├─ Exogenous shifts: Long-run vs transitory effects
    └─ Impulse responses: Shock propagation toward equilibrium
```

**Interaction:** Disequilibrium → adjustment pressure → short-run dynamics → convergence

## 5. Mini-Project
Model consumption and income with ECM, decompose short/long-run effects:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import OLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Generate cointegrated consumption and income data
np.random.seed(42)
periods = 200
time_index = pd.date_range('2000-01', periods=periods, freq='Y')

# Common stochastic trend (permanent income)
trend = np.cumsum(np.random.normal(0.03, 0.02, periods))
permanent_income = 30000 + 5000 * trend

# Income: permanent + transitory component
transitory = np.random.normal(0, 1000, periods)
income = permanent_income + transitory

# Consumption: cointegrated with income (long-run: C = 0.8·Y)
consumption = 0.8 * permanent_income + 2000 + np.random.normal(0, 500, periods)

data = pd.DataFrame({
    'consumption': consumption,
    'income': income,
}, index=time_index)

print("="*70)
print("ERROR CORRECTION MODEL: Consumption vs Income")
print("="*70)

# 1. Unit root and cointegration tests
print("\n1. PRELIMINARY TESTS")
print("-"*70)

for col in data.columns:
    result = adfuller(data[col])
    print(f"\n{col}: ADF stat={result[0]:.4f}, p-value={result[1]:.4f}")
    print(f"  Conclusion: {'I(1)' if result[1] > 0.05 else 'Stationary'}")

score, pvalue, _ = coint(data['consumption'], data['income'])
print(f"\nCointegration test: stat={score:.4f}, p-value={pvalue:.4f}")
print(f"Cointegrated: {'Yes' if pvalue < 0.05 else 'No'}")

# 2. Estimate long-run relationship (OLS)
print("\n2. LONG-RUN RELATIONSHIP (OLS on Levels)")
print("-"*70)

Y = data['consumption']
X = data['income']
X_with_const = np.column_stack((np.ones(len(X)), X))

# OLS regression
from numpy.linalg import lstsq
beta_lr, residuals, rank, s = lstsq(X_with_const, Y, rcond=None)
print(f"Long-run: Consumption = {beta_lr[0]:.2f} + {beta_lr[1]:.4f}·Income")
print(f"Interpretation: 1% income increase → {beta_lr[1]:.4f}% consumption increase (long-run)")

# Error correction term
ec_term = Y - beta_lr[0] - beta_lr[1] * X
print(f"\nError Correction Term:")
print(f"  Mean: {ec_term.mean():.2f}")
print(f"  Std Dev: {ec_term.std():.2f}")

# 3. Single Equation ECM
print("\n3. SINGLE EQUATION ECM")
print("-"*70)

# Prepare data for ECM: ΔCₜ = α·ECₜ₋₁ + φ·ΔCₜ₋₁ + ψ·ΔIₜ₋₁ + εₜ
dc = data['consumption'].diff().dropna()
di = data['income'].diff().dropna()
ec_lag = ec_term.shift(1).iloc[1:]  # Align with ΔC
dc_lag = dc.shift(1).iloc[1:]
di_lag = di.shift(1).iloc[1:]

# Align all variables
alignment_idx = max(dc.index[0], di.index[0], ec_lag.index[0])
dc_aligned = dc[dc.index >= alignment_idx]
di_aligned = di[di.index >= alignment_idx]
ec_aligned = ec_lag[ec_lag.index >= alignment_idx]
dc_lag_aligned = dc_lag[dc_lag.index >= alignment_idx]
di_lag_aligned = di_lag[di_lag.index >= alignment_idx]

# OLS for ECM
X_ecm = np.column_stack((np.ones(len(dc_aligned)), 
                         ec_aligned.values, 
                         dc_lag_aligned.values, 
                         di_lag_aligned.values))
y_ecm = dc_aligned.values

beta_ecm, residuals_ecm, _, _ = lstsq(X_ecm, y_ecm, rcond=None)
se_ecm = np.sqrt(np.sum(residuals_ecm**2) / (len(y_ecm) - len(beta_ecm))) / np.sqrt(np.diag(np.linalg.inv(X_ecm.T @ X_ecm)))

print(f"\nECM Specification:")
print(f"ΔCₜ = {beta_ecm[0]:.4f} + {beta_ecm[1]:.4f}·ECₜ₋₁ + {beta_ecm[2]:.4f}·ΔCₜ₋₁ + {beta_ecm[3]:.4f}·ΔIₜ₋₁ + εₜ")

print(f"\nCoefficients:")
print(f"  Constant: {beta_ecm[0]:.4f} (SE: {se_ecm[0]:.4f})")
print(f"  EC term: {beta_ecm[1]:.4f} (SE: {se_ecm[1]:.4f})")
print(f"    → Adjustment speed: {beta_ecm[1]:.2%} gap closed per year")
print(f"  ΔC lag: {beta_ecm[2]:.4f} (SE: {se_ecm[2]:.4f})")
print(f"  ΔI lag: {beta_ecm[3]:.4f} (SE: {se_ecm[3]:.4f})")

# Half-life of adjustment
if beta_ecm[1] < 0:
    half_life = np.log(0.5) / np.log(1 + beta_ecm[1])
    print(f"\nHalf-life of adjustment: {half_life:.2f} years")
    print(f"  (Time to close 50% of disequilibrium)")

# 4. VECM estimation
print("\n4. VECTOR ERROR CORRECTION MODEL (VECM)")
print("-"*70)

model_vecm = VECM(data, deterministic='ci', k_ar_diff=1, coint_rank=1)
results_vecm = model_vecm.fit()
print(results_vecm.summary())

# 5. Extract long-run and short-run effects
print("\n5. LONG-RUN vs SHORT-RUN DECOMPOSITION")
print("-"*70)

alpha = results_vecm.alpha  # Adjustment coefficients
beta = results_vecm.beta    # Cointegrating vector

print(f"\nCointegrating vector (β): {beta.flatten()}")
print(f"Adjustment speeds (α): {alpha.flatten()}")

# Long-run effect = cumulative multiplier
print(f"\nShort-run effects (impact at t=0):")
print(f"  ∂ΔConsumption/∂ΔIncome: {results_vecm.gamma[0, 1]:.4f}")

# 6. Impulse response analysis
print("\n6. IMPULSE RESPONSE ANALYSIS")
print("-"*70)

# Simulate shock to income, track adjustment path
initial_shock = 1000
adjustment_periods = 20

consumption_path = [0]
income_path = [initial_shock]

# Use VECM coefficients to simulate
current_ec_error = 0
for t in range(1, adjustment_periods):
    # Income adjustment (AR term)
    dincome_t = -alpha[1, 0] * current_ec_error
    income_path.append(income_path[-1] + dincome_t)
    
    # Consumption adjustment
    dconsumption_t = -alpha[0, 0] * current_ec_error
    consumption_path.append(consumption_path[-1] + dconsumption_t)
    
    # Update EC error (assumes income sticks at new level)
    current_ec_error = income_path[-1] - beta_lr[1] * income_path[-1]

print(f"Income shock of ${initial_shock:,.0f}:")
print(f"  Year 1 consumption response: ${consumption_path[1]:,.2f}")
print(f"  Year 5 cumulative response: ${sum(consumption_path[:5]):,.2f}")

# 7. Visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Raw series
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(data.index, data['consumption'], 'b-', linewidth=2, label='Consumption')
ax2.plot(data.index, data['income'], 'r-', linewidth=2, label='Income')
ax.set_ylabel('Consumption ($)', color='b')
ax2.set_ylabel('Income ($)', color='r')
ax.set_title('Consumption and Income Over Time')
ax.grid(alpha=0.3)

# Plot 2: Error correction term
ax = axes[0, 1]
ax.plot(data.index, ec_term, 'g-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.fill_between(data.index, ec_term.min(), ec_term.max(), alpha=0.2, color='g')
ax.set_ylabel('Disequilibrium ($)')
ax.set_title('Error Correction Term (Disequilibrium)')
ax.grid(alpha=0.3)

# Plot 3: First differences
ax = axes[0, 2]
ax.plot(data.index[1:], dc, 'b-', linewidth=1, label='ΔConsumption', alpha=0.7)
ax.plot(data.index[1:], di, 'r-', linewidth=1, label='ΔIncome', alpha=0.7)
ax.set_ylabel('Change ($)')
ax.set_title('First Differences (Stationarity)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Adjustment dynamics (scatter)
ax = axes[1, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(dc_aligned)))
scatter = ax.scatter(ec_aligned.values, dc_aligned.values, c=range(len(dc_aligned)), 
                     cmap='viridis', s=50, alpha=0.6)
# Fitted line from ECM
ec_range = np.linspace(ec_aligned.min(), ec_aligned.max(), 100)
dc_fitted = beta_ecm[0] + beta_ecm[1] * ec_range
ax.plot(ec_range, dc_fitted, 'r--', linewidth=2, label='ECM fit')
ax.set_xlabel('Error Correction Term (Disequilibrium)')
ax.set_ylabel('ΔConsumption ($)')
ax.set_title('Error Correction Dynamics')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Impulse response
ax = axes[1, 1]
ax.plot(range(adjustment_periods), consumption_path, 'o-', linewidth=2, markersize=6)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Years after shock')
ax.set_ylabel('Cumulative Consumption Response ($)')
ax.set_title('Impulse Response: Income Shock')
ax.grid(alpha=0.3)

# Plot 6: Residual diagnostics
ax = axes[1, 2]
ax.plot(residuals_ecm, 'k-', linewidth=1, alpha=0.7)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_ylabel('Residuals')
ax.set_title('ECM Residuals (Should be white noise)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ecm_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION SUMMARY")
print("="*70)
print(f"""
Key ECM Results:
1. Long-run elasticity: {beta_lr[1]:.4f}
   → 1% permanent income increase → {beta_lr[1]:.4f}% consumption increase
   
2. Adjustment speed: {beta_ecm[1]:.2%} per year
   → System closes {abs(beta_ecm[1]):.1%} of disequilibrium annually
   → Half-life: {half_life:.1f} years
   
3. Short-run effects:
   → Past consumption changes: {beta_ecm[2]:.4f}
   → Past income changes: {beta_ecm[3]:.4f}
   
4. Economic interpretation:
   → Consumption responds partially to income changes in short-run
   → Over time, consumption adjusts to long-run equilibrium
   → Disequilibrium from temporary income shocks
""")
```

## 6. Challenge Round
When ECM interpretation breaks down:
- Weak cointegration: Small adjustment coefficient, slow convergence; ECM may not capture dynamics
- Multiple cointegrating vectors: VECM with rank > 1; each vector has different speed
- Structural instability: EC coefficient changes pre/post break; subsample analysis needed
- Endogenous regressor: Reverse causality biases adjustment coefficient; use IV
- Misspecified lag length: Too few lags → autocorrelated residuals; test with LM test

## 7. Key References
- [Engle & Granger (1987), "Co-Integration and Error Correction"](https://www.jstor.org/stable/1913236)
- [Banerjee et al. (1993), "Co-Integration, Error Correction, and the Econometric Analysis of Non-Stationary Data"](https://doi.org/10.1093/0198288107.001.0001)
- [Johansen (1995), "Likelihood-Based Inference in Cointegrated Vector Autoregressive Models"](https://doi.org/10.1093/0198288107.001.0001)

---
**Status:** Bridges long-run and short-run dynamics | **Complements:** VECM, Cointegration, Granger Causality
