# Credit Correlation

## 1. Concept Skeleton
**Definition:** Co-movement between default events; probability two (or more) borrowers default together; measures default clustering  
**Purpose:** Quantify portfolio concentration risk, model tail scenarios, price multi-name derivatives (baskets, CDOs)  
**Prerequisites:** Copulas, correlation matrices, default processes, portfolio theory, risk aggregation

## 2. Comparative Framing
| Approach | Data | Estimation | Stability | Use Case |
|----------|------|-----------|----------|----------|
| **Pairwise Correlation** | Returns, spreads | Sample or model | Medium | Simple case |
| **Factor Model** | Market data | Regression | High | Large portfolios |
| **Copula** | Default times | Simulation | Medium | Tail modeling |
| **Implied** | Market prices | Reverse engineering | Low | Exotic pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
Default correlation between two firms = 0.30. If Firm A defaults, probability Firm B defaults increases from 5% to ~8%

**Failure Case:**  
2008: Portfolio assumed low correlation (0.30), actual crisis correlation ≈ 0.95 (all defaults cluster). VaR model underestimated loss 10x

**Edge Case:**  
Negative correlation rare in credit; borrowers move together in cycle. Possible with hedges or short positions

## 4. Layer Breakdown
```
Credit Correlation Framework:
├─ Definition and Types:
│   ├─ Pairwise default correlation: ρ(i,j) = Cov[D_i, D_j] / (σ_i × σ_j)
│   ├─ Asset correlation: ρ_A = correlation of firm values
│   ├─ Conditional PD: P(default|other defaults) > P(default)
│   └─ Default intensity correlation: λ_i and λ_j co-move
├─ Sources of Correlation:
│   ├─ Systematic: Macro factors (interest rates, GDP, unemployment)
│   ├─ Sector: Industry-specific (real estate, tech, energy)
│   ├─ Contagion: Firm failure triggers others (counterparty risk)
│   ├─ Liquidity: Market stress affects all borrowers
│   └─ Common ownership: Shared investors, collateral
├─ Estimation Methods:
│   ├─ Historical defaults:
│   │   ├─ Joint default frequency approach
│   │   ├─ Tetrachoric correlation on 2x2 table
│   │   └─ Issue: Few joint defaults; high estimation error
│   ├─ Market-implied:
│   │   ├─ From CDS prices via copula
│   │   ├─ More forward-looking than historical
│   │   └─ Sensitive to liquidity, bid-ask spreads
│   ├─ Factor models:
│   │   ├─ ρ_shared = β_i × β_j × ρ_factor + idiosyncratic
│   │   ├─ Single-factor: Merton-style asset correlation
│   │   └─ Multi-factor: Systemic + sector + idio
│   └─ Copula approach:
│       ├─ Model marginal defaults + joint structure
│       ├─ Gaussian copula: Easy but tail-underestimating
│       └─ Student-t copula: Fat tails, higher correlation in crisis
├─ Correlation Dynamics:
│   ├─ Stability: Pairwise correlation ≈ 0.3-0.5 in normal times
│   ├─ Contagion: Correlation spikes during crisis (0.7-0.95)
│   ├─ Regime-switching: Low vol normal, high vol crisis
│   ├─ Term structure: Longer maturities show higher correlation
│   └─ Tail dependence: Correlation higher in tail (1% scenarios)
├─ Portfolio Impact:
│   ├─ Low correlation: Portfolio VaR << Σ individual VaR (diversification)
│   ├─ High correlation: Portfolio VaR ≈ Σ individual VaR (concentration)
│   ├─ Diversification ratio: √N in uncorrelated, 1 in perfectly correlated
│   └─ Convexity: Non-linear relationship between correlation and risk
└─ Copulas (Joint Distribution):
    ├─ Gaussian copula: C(u,v) = Φ(Φ⁻¹(u), Φ⁻¹(v), ρ)
    ├─ Clayton copula: Lower tail dependence
    ├─ Gumbel copula: Upper tail dependence
    └─ Student-t copula: Symmetric tail dependence
```

## 5. Mini-Project
Estimate and analyze credit correlation:
```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Credit Correlation Analysis ===")

# Simulate default data for 2 firms over 20 years
n_years = 20
n_simulations = 1000

# True underlying correlation in asset returns
rho_true = 0.35

print(f"True asset correlation: {rho_true:.2f}")

# Generate correlated default times
# Single factor model: V_i = β_i × M + √(1 - β_i²) × ε_i
beta_1, beta_2 = 0.6, 0.5  # Factor loadings

# Common factor and idiosyncratic shocks
M = np.random.normal(0, 1, n_simulations)
eps_1 = np.random.normal(0, 1, n_simulations)
eps_2 = np.random.normal(0, 1, n_simulations)

# Firm values
V_1 = beta_1 * M + np.sqrt(1 - beta_1**2) * eps_1
V_2 = beta_2 * M + np.sqrt(1 - beta_2**2) * eps_2

# Default thresholds (based on PD)
pd_1, pd_2 = 0.05, 0.05
threshold_1 = stats.norm.ppf(pd_1)
threshold_2 = stats.norm.ppf(pd_2)

# Defaults
default_1 = (V_1 < threshold_1).astype(int)
default_2 = (V_2 < threshold_2).astype(int)

# Empirical statistics
emp_pd_1 = default_1.mean()
emp_pd_2 = default_2.mean()
joint_default = (default_1 * default_2).sum()
joint_pd = joint_default / n_simulations

print(f"\nEmpirical Results (n={n_simulations}):")
print(f"Firm 1 default rate: {emp_pd_1:.2%}")
print(f"Firm 2 default rate: {emp_pd_2:.2%}")
print(f"Joint default rate: {joint_pd:.2%}")

# Calculate correlations
# Binary default correlation
p_11 = joint_pd
p_10 = (default_1 * (1 - default_2)).mean()
p_01 = ((1 - default_1) * default_2).mean()
p_00 = ((1 - default_1) * (1 - default_2)).mean()

corr_default = (p_11 - emp_pd_1 * emp_pd_2) / np.sqrt(emp_pd_1 * (1 - emp_pd_1) * emp_pd_2 * (1 - emp_pd_2))

print(f"\nDefault Correlation (Tetrachoric): {corr_default:.3f}")

# Asset value correlation
corr_asset = np.corrcoef(V_1, V_2)[0, 1]
print(f"Asset Correlation (Realized): {corr_asset:.3f}")
print(f"Expected Asset Correlation: {np.corrcoef(beta_1 * M, beta_2 * M)[0, 1]:.3f}")

# Conditional default probability
# P(D_2 | D_1)
if default_1.sum() > 0:
    cond_prob_d2_given_d1 = (default_1 * default_2).sum() / default_1.sum()
    uncond_prob_d2 = emp_pd_2
    print(f"\nP(Firm 2 defaults | Firm 1 defaults): {cond_prob_d2_given_d1:.2%}")
    print(f"P(Firm 2 defaults | unconditional): {uncond_prob_d2:.2%}")
    print(f"Contagion effect: {cond_prob_d2_given_d1 / uncond_prob_d2:.2f}x")

# 2x2 Contingency table
contingency = np.array([[p_00, p_01], [p_10, p_11]]) * n_simulations
print(f"\nContingency Table (n={n_simulations}):")
print(f"              No Default D2  | Default D2")
print(f"No Default D1      {p_00*n_simulations:6.0f}      |    {p_01*n_simulations:6.0f}")
print(f"Default D1         {p_10*n_simulations:6.0f}      |    {p_11*n_simulations:6.0f}")

# Correlation under different scenarios
print("\n=== Scenario Analysis: Correlation Variation ===")

# Scenario 1: Normal times (low common factor realization)
normal_mask = M < stats.norm.ppf(0.5)  # M < 0 (lower half of distribution)
default_1_normal = default_1[normal_mask]
default_2_normal = default_2[normal_mask]
corr_normal = np.corrcoef(default_1_normal, default_2_normal)[0, 1]

# Scenario 2: Crisis times (high common factor realization)
crisis_mask = M > stats.norm.ppf(0.9)  # M > high value (upper 10%)
default_1_crisis = default_1[crisis_mask]
default_2_crisis = default_2[crisis_mask]
corr_crisis = np.corrcoef(default_1_crisis, default_2_crisis)[0, 1]

print(f"Normal times correlation: {np.nan_to_num(corr_normal):.3f}")
print(f"Crisis times correlation: {np.nan_to_num(corr_crisis):.3f}")
print(f"Correlation increase in crisis: {(np.nan_to_num(corr_crisis) - np.nan_to_num(corr_normal))*100:.1f}pp")

# Portfolio impact: Two-firm portfolio
print("\n=== Portfolio Risk Impact ===")

portfolio_losses_indep = (default_1 + default_2) * 0.5  # Equal-weight, independence
portfolio_losses_actual = (default_1 * default_2) * 2 + (default_1 * (1 - default_2)) * 0.5 + ((1 - default_1) * default_2) * 0.5

var_indep = np.percentile(portfolio_losses_indep, 99)
var_actual = np.percentile(portfolio_losses_actual, 99)

print(f"99% VaR (assuming independence): {var_indep:.2%} of portfolio")
print(f"99% VaR (actual correlation): {var_actual:.2%} of portfolio")
print(f"Underestimation from independence assumption: {(var_actual - var_indep)/var_indep * 100:.1f}%")

# Multi-firm portfolio with correlation matrix
print("\n=== Multi-Firm Portfolio with Correlation Matrix ===")

n_firms = 10
corr_matrix = np.zeros((n_firms, n_firms))

# Build correlation matrix: sector structure
for i in range(n_firms):
    for j in range(n_firms):
        if i == j:
            corr_matrix[i, j] = 1.0
        elif i % 3 == j % 3:  # Same sector
            corr_matrix[i, j] = 0.50
        else:  # Different sector
            corr_matrix[i, j] = 0.20

# Ensure positive definite
eigs = np.linalg.eigvalsh(corr_matrix)
if (eigs < 0).any():
    # Adjust to make positive definite
    corr_matrix = corr_matrix + (abs(eigs.min()) + 0.01) * np.eye(n_firms)

print("\nCorrelation Matrix Structure:")
print(f"Within-sector correlation: 0.50")
print(f"Cross-sector correlation: 0.20")
print(f"Condition number: {np.linalg.cond(corr_matrix):.1f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Joint default scatter
ax1 = axes[0, 0]
scatter = ax1.scatter(V_1, V_2, c=default_1 + default_2, cmap='RdYlGn_r', s=20, alpha=0.5)
ax1.axhline(threshold_2, color='r', linestyle='--', alpha=0.5, label='Default threshold (Firm 2)')
ax1.axvline(threshold_1, color='r', linestyle='--', alpha=0.5, label='Default threshold (Firm 1)')
ax1.set_xlabel('Firm 1 Asset Value')
ax1.set_ylabel('Firm 2 Asset Value')
ax1.set_title('Default Correlation via Common Factor\n(Red = both default)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Contingency table
ax2 = axes[0, 1]
contingency_norm = contingency / n_simulations
im = ax2.imshow(contingency_norm, cmap='Blues', aspect='auto')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Survive', 'Default'])
ax2.set_yticklabels(['Survive', 'Default'])
ax2.set_xlabel('Firm 2')
ax2.set_ylabel('Firm 1')
ax2.set_title('Joint Probability Distribution')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{contingency_norm[i, j]:.4f}', ha='center', va='center', color='black')

# Plot 3: Correlation by market state
ax3 = axes[0, 2]
states = ['Normal\n(M<median)', 'Crisis\n(M>90th pctile)']
corr_values = [np.nan_to_num(corr_normal), np.nan_to_num(corr_crisis)]
colors_state = ['green', 'red']
ax3.bar(states, corr_values, color=colors_state, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Default Correlation')
ax3.set_title('Regime-Dependent Correlation\n(Higher in crisis)')
for i, (state, corr) in enumerate(zip(states, corr_values)):
    ax3.text(i, corr + 0.02, f'{corr:.3f}', ha='center', va='bottom')
ax3.set_ylim([0, max(corr_values) * 1.2])
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Common factor vs defaults
ax4 = axes[1, 0]
ax4.scatter(M, default_1, alpha=0.3, s=20, label='Firm 1')
ax4.scatter(M, default_2, alpha=0.3, s=20, label='Firm 2')
ax4.axvline(threshold_1 / beta_1, color='r', linestyle='--', alpha=0.5)
ax4.set_xlabel('Common Factor (M)')
ax4.set_ylabel('Default (0/1)')
ax4.set_title('Factor Model: Common Shock Drives Correlation')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Correlation matrix heatmap
ax5 = axes[1, 1]
im5 = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
ax5.set_title('Multi-Firm Correlation Matrix\n(10 firms, 3 sectors)')
ax5.set_xlabel('Firm')
ax5.set_ylabel('Firm')
plt.colorbar(im5, ax=ax5)

# Plot 6: Portfolio loss distribution
ax6 = axes[1, 2]
ax6.hist(portfolio_losses_indep, bins=30, alpha=0.5, label='Independent', edgecolor='black')
ax6.hist(portfolio_losses_actual, bins=30, alpha=0.5, label='Correlated', edgecolor='black')
ax6.axvline(var_indep, color='steelblue', linestyle='--', linewidth=2, label=f'99% VaR (indep)')
ax6.axvline(var_actual, color='orange', linestyle='--', linewidth=2, label=f'99% VaR (actual)')
ax6.set_xlabel('Portfolio Loss (as % of exposure)')
ax6.set_ylabel('Frequency')
ax6.set_title('Portfolio Loss Distribution')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Credit Correlation Summary ===")
print(f"Pairwise correlation captures co-movement in defaults")
print(f"Factor models explain correlation through systemic risk")
print(f"Correlation is regime-dependent (increases in crisis)")
```

## 6. Challenge Round
When is correlation estimation problematic?
- **Few joint defaults**: Historical correlation estimates have high error when joint defaults rare
- **Regime shifts**: Crisis correlation ≠ normal correlation; model may assume wrong regime
- **Non-stationarity**: Correlation changes with economic cycle; past ≠ future
- **Causality vs correlation**: Common factor vs direct contagion hard to distinguish
- **Portfolio-specific**: Correlation depends on composition; not portfolio-invariant

## 7. Key References
- [Copula Methods for Credit Risk](https://en.wikipedia.org/wiki/Copula_(probability_theory)) - Joint distribution modeling
- [Vasicek Asset Correlation](https://en.wikipedia.org/wiki/Vasicek_model) - Single-factor framework
- [Credit Correlation Dynamics](https://www.bis.org/publ/work374.pdf) - BIS research on regime changes

---
**Status:** Critical driver of portfolio tail risk | **Complements:** Credit VaR, Concentration Risk, Portfolio modeling
