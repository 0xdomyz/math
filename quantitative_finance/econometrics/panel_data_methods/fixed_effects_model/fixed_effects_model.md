# Fixed Effects Model

## 1. Concept Skeleton
**Definition:** Panel data estimator controlling for time-invariant unobserved heterogeneity by within-transformation or dummy variables  
**Purpose:** Eliminate omitted variable bias from unit-specific effects; identify causal effects using within-unit variation  
**Prerequisites:** Panel data structure, strict exogeneity, unobserved heterogeneity, demeaning transformation

## 2. Comparative Framing
| Method | Fixed Effects (FE) | Random Effects (RE) | Pooled OLS | First Differences |
|--------|-------------------|---------------------|------------|-------------------|
| **Unobserved αᵢ** | Correlated with Xᵢₜ | Uncorrelated with Xᵢₜ | Ignored (biased) | Differenced out |
| **Efficiency** | Less efficient than RE | Most efficient if valid | Most if αᵢ = 0 | Loses one period |
| **Time-Invariant X** | Cannot estimate | Can estimate | Can estimate | Differenced out |
| **Consistency** | Consistent if strict exog. | Inconsistent if Cov(αᵢ,X)≠0 | Inconsistent | Consistent |

## 3. Examples + Counterexamples

**Classic Example:**  
Wage equation: ln(wage) on education, experience. Ability (αᵢ) unobserved, correlated with education. FE uses within-person variation over time.

**Failure Case:**  
Gender wage gap: Gender time-invariant, absorbed by FE. Cannot estimate gender coefficient. Need between variation or interactions.

**Edge Case:**  
Short panel (T=2): FE equivalent to first differences. No efficiency gain. Long panels (T large) improve precision of FE.

## 4. Layer Breakdown
```
Fixed Effects Model Framework:
├─ Model Specification:
│   Yᵢₜ = Xᵢₜ'β + αᵢ + εᵢₜ
│   ├─ i = 1,...,N units (individuals, firms, countries)
│   ├─ t = 1,...,T time periods
│   ├─ αᵢ: Unobserved individual effect (time-invariant)
│   ├─ εᵢₜ: Idiosyncratic error (time-varying)
│   └─ Key: Cov(αᵢ, Xᵢₜ) ≠ 0 (FE allows correlation)
├─ Estimation Methods:
│   ├─ Within Transformation (Demeaning):
│   │   Ÿᵢₜ = Yᵢₜ - Ȳᵢ,  Ẍᵢₜ = Xᵢₜ - X̄ᵢ
│   │   └─ Ÿᵢₜ = Ẍᵢₜ'β + ε̈ᵢₜ  (αᵢ eliminated)
│   ├─ LSDV (Least Squares Dummy Variable):
│   │   Yᵢₜ = Xᵢₜ'β + Σⱼ δⱼDⱼ + εᵢₜ
│   │   └─ Dⱼ = 1 if i=j, 0 otherwise (N-1 dummies)
│   └─ First Differences:
│       ΔYᵢₜ = ΔXᵢₜ'β + Δεᵢₜ  (αᵢ differenced out)
│       └─ Equivalent to FE when T=2, less efficient when T>2
├─ Assumptions:
│   ├─ Strict Exogeneity: E[εᵢₜ | Xᵢ₁,...,Xᵢₜ, αᵢ] = 0
│   │   └─ Past, present, future X uncorrelated with current ε
│   ├─ No Perfect Collinearity: Rank condition after demeaning
│   ├─ Homoskedasticity (optional): Var(εᵢₜ | X, α) = σ²
│   └─ No Serial Correlation (optional): Cov(εᵢₜ, εᵢₛ) = 0
├─ Standard Errors:
│   ├─ Default: Assumes iid εᵢₜ
│   ├─ Robust (Heteroskedasticity): White/Huber sandwich
│   ├─ Clustered: By unit (i) for serial correlation
│   └─ Two-Way Clustering: By unit and time
├─ Hypothesis Tests:
│   ├─ F-test for FE: H₀: α₁ = α₂ = ... = αₙ
│   │   └─ Test if FE needed vs pooled OLS
│   ├─ t-tests on β: Standard after within transformation
│   └─ Joint Tests: Wald, F-test for multiple coefficients
├─ Time-Varying Effects:
│   ├─ Time Fixed Effects: λₜ for common time shocks
│   │   └─ Two-Way FE: Yᵢₜ = Xᵢₜ'β + αᵢ + λₜ + εᵢₜ
│   ├─ Interactions: Xᵢₜ × Dᵢ for heterogeneous effects
│   └─ Time Trends: Linear or quadratic trends by unit
├─ What FE Cannot Estimate:
│   ├─ Time-Invariant Variables: Gender, race, birth year
│   ├─ Slow-Moving Variables: Education (if T small)
│   └─ Between Effects: Level differences across units
└─ Diagnostics:
    ├─ Hausman Test: FE vs RE specification
    ├─ Serial Correlation: Wooldridge test, Breusch-Godfrey
    ├─ Heteroskedasticity: Modified Wald test for panel
    └─ Cross-Sectional Dependence: Pesaran CD test
```

**Interaction:** Within-transformation → Eliminates αᵢ → OLS on demeaned data → Cluster SE by unit

## 5. Mini-Project
Estimate fixed effects model with diagnostics and comparisons:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, PooledOLS, FirstDifferenceOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from scipy import stats

np.random.seed(42)

# ===== Simulate Panel Data =====
N = 200  # Number of individuals
T = 10   # Time periods
n_obs = N * T

# Individual fixed effects (ability, productivity, etc.)
alpha_i = np.random.normal(10, 3, N)

# Time-varying variables
# X1: Education/experience (slow-moving), correlated with alpha
# X2: Health status (time-varying)
# X3: Policy treatment (time-varying)

data_list = []
for i in range(N):
    for t in range(T):
        # X1 correlated with individual effect (endogeneity)
        x1 = 5 + 0.3 * alpha_i[i] + np.random.normal(0, 1)
        
        # X2 and X3 exogenous
        x2 = 7 + 0.1 * t + np.random.normal(0, 1.5)
        x3 = np.random.binomial(1, 0.3 + 0.01 * t)  # Treatment increases over time
        
        # Outcome: Wage
        # True effects: β1=1.5, β2=0.8, β3=2.0
        epsilon = np.random.normal(0, 2)
        y = 5 + 1.5*x1 + 0.8*x2 + 2.0*x3 + alpha_i[i] + epsilon
        
        data_list.append({
            'id': i,
            'time': t,
            'y': y,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'alpha_true': alpha_i[i]
        })

df = pd.DataFrame(data_list)

print("="*70)
print("FIXED EFFECTS MODEL: PANEL DATA ANALYSIS")
print("="*70)
print(f"\nPanel Dimensions:")
print(f"  Units (N): {N}")
print(f"  Time Periods (T): {T}")
print(f"  Total Observations: {n_obs}")
print(f"\nTrue Coefficients:")
print(f"  β₁ (X1): 1.5")
print(f"  β₂ (X2): 0.8")
print(f"  β₃ (X3): 2.0")
print(f"\nDescriptive Statistics:")
print(df[['y', 'x1', 'x2', 'x3']].describe().round(3))

# Set multi-index for panel data
df_panel = df.set_index(['id', 'time'])

# ===== Pooled OLS (Biased - Ignores αᵢ) =====
print("\n" + "="*70)
print("POOLED OLS (Ignoring Individual Effects)")
print("="*70)

exog_vars = ['x1', 'x2', 'x3']
pooled_model = PooledOLS(df_panel['y'], df_panel[exog_vars]).fit()
print(pooled_model.summary)

print("\nBias in Pooled OLS:")
print(f"  X1 coefficient: {pooled_model.params['x1']:.4f} (True: 1.5)")
print(f"  Bias: {pooled_model.params['x1'] - 1.5:.4f}")

# ===== Fixed Effects Model =====
print("\n" + "="*70)
print("FIXED EFFECTS MODEL (Within Estimator)")
print("="*70)

fe_model = PanelOLS(df_panel['y'], df_panel[exog_vars], 
                    entity_effects=True).fit(cov_type='clustered', 
                                             cluster_entity=True)
print(fe_model.summary)

print("\nFixed Effects Estimates:")
print(f"  X1 coefficient: {fe_model.params['x1']:.4f} (True: 1.5)")
print(f"  X2 coefficient: {fe_model.params['x2']:.4f} (True: 0.8)")
print(f"  X3 coefficient: {fe_model.params['x3']:.4f} (True: 2.0)")

# ===== Two-Way Fixed Effects =====
print("\n" + "="*70)
print("TWO-WAY FIXED EFFECTS (Entity + Time)")
print("="*70)

twfe_model = PanelOLS(df_panel['y'], df_panel[exog_vars],
                      entity_effects=True, time_effects=True).fit(
                          cov_type='clustered', cluster_entity=True)
print(twfe_model.summary)

# ===== First Differences =====
print("\n" + "="*70)
print("FIRST DIFFERENCES MODEL")
print("="*70)

fd_model = FirstDifferenceOLS(df_panel['y'], df_panel[exog_vars]).fit(
    cov_type='robust')
print(fd_model.summary)

# ===== Model Comparison =====
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison = compare({
    'Pooled OLS': pooled_model,
    'Fixed Effects': fe_model,
    'Two-Way FE': twfe_model,
    'First Diff': fd_model
})
print(comparison.summary)

# ===== Extract Fixed Effects =====
print("\n" + "="*70)
print("ESTIMATED INDIVIDUAL EFFECTS")
print("="*70)

# Get estimated fixed effects
estimated_effects = fe_model.estimated_effects

# Compare with true effects
fe_comparison = pd.DataFrame({
    'True_Alpha': alpha_i,
    'Estimated_Alpha': estimated_effects.values
})
fe_comparison['Error'] = fe_comparison['Estimated_Alpha'] - fe_comparison['True_Alpha']

print("\nSummary of Fixed Effects Estimation:")
print(f"  Correlation (True vs Estimated): {fe_comparison[['True_Alpha', 'Estimated_Alpha']].corr().iloc[0,1]:.4f}")
print(f"  Mean Absolute Error: {np.abs(fe_comparison['Error']).mean():.4f}")
print(f"  RMSE: {np.sqrt((fe_comparison['Error']**2).mean()):.4f}")

print("\nFirst 10 Individuals:")
print(fe_comparison.head(10).round(4))

# ===== F-test for Fixed Effects =====
print("\n" + "="*70)
print("F-TEST FOR FIXED EFFECTS")
print("="*70)

# Manual F-test: Compare pooled OLS vs FE
rss_pooled = pooled_model.resid.T @ pooled_model.resid
rss_fe = fe_model.resid.T @ fe_model.resid

n = len(df_panel)
k = len(exog_vars)
f_stat = ((rss_pooled - rss_fe) / (N - 1)) / (rss_fe / (n - N - k))
p_value = 1 - stats.f.cdf(f_stat, N - 1, n - N - k)

print(f"H₀: All individual effects αᵢ are equal (use Pooled OLS)")
print(f"H₁: Individual effects differ (use Fixed Effects)")
print(f"\nF-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("✓ Reject H₀: Fixed effects are present (use FE model)")
else:
    print("  Fail to reject: No evidence for fixed effects")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: True vs Estimated Fixed Effects
axes[0, 0].scatter(fe_comparison['True_Alpha'], 
                   fe_comparison['Estimated_Alpha'],
                   alpha=0.5, s=30)
axes[0, 0].plot([fe_comparison['True_Alpha'].min(), 
                 fe_comparison['True_Alpha'].max()],
                [fe_comparison['True_Alpha'].min(), 
                 fe_comparison['True_Alpha'].max()],
                'r--', linewidth=2, label='45° line')
axes[0, 0].set_xlabel('True Fixed Effect (αᵢ)')
axes[0, 0].set_ylabel('Estimated Fixed Effect')
axes[0, 0].set_title('Fixed Effects: True vs Estimated')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Coefficient Comparison
coef_comparison = pd.DataFrame({
    'True': [1.5, 0.8, 2.0],
    'Pooled OLS': [pooled_model.params['x1'], 
                   pooled_model.params['x2'],
                   pooled_model.params['x3']],
    'Fixed Effects': [fe_model.params['x1'],
                     fe_model.params['x2'],
                     fe_model.params['x3']],
    'First Diff': [fd_model.params['x1'],
                   fd_model.params['x2'],
                   fd_model.params['x3']]
}, index=['X1', 'X2', 'X3'])

x_pos = np.arange(len(coef_comparison))
width = 0.2

for i, model_name in enumerate(['True', 'Pooled OLS', 'Fixed Effects', 'First Diff']):
    offset = (i - 1.5) * width
    axes[0, 1].bar(x_pos + offset, coef_comparison[model_name], 
                   width, label=model_name, alpha=0.8)

axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(coef_comparison.index)
axes[0, 1].set_ylabel('Coefficient')
axes[0, 1].set_title('Coefficient Comparison Across Models')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Within vs Between Variation
# Calculate within and between variation
df_within = df.copy()
df_within['y_within'] = df_within.groupby('id')['y'].transform(lambda x: x - x.mean())
df_within['x1_within'] = df_within.groupby('id')['x1'].transform(lambda x: x - x.mean())

df_between = df.groupby('id')[['y', 'x1']].mean().reset_index()

axes[0, 2].scatter(df_within['x1_within'], df_within['y_within'],
                   alpha=0.3, s=10, label='Within Variation')
axes[0, 2].scatter(df_between['x1'], df_between['y'],
                   alpha=0.5, s=50, color='red', label='Between Variation')

# Fit lines
from sklearn.linear_model import LinearRegression
within_reg = LinearRegression().fit(df_within[['x1_within']], 
                                     df_within['y_within'])
between_reg = LinearRegression().fit(df_between[['x1']], 
                                      df_between['y'])

x_range_within = np.linspace(df_within['x1_within'].min(), 
                             df_within['x1_within'].max(), 100)
x_range_between = np.linspace(df_between['x1'].min(), 
                              df_between['x1'].max(), 100)

axes[0, 2].plot(x_range_within, 
                within_reg.predict(x_range_within.reshape(-1, 1)),
                'b-', linewidth=2, label=f'Within: {within_reg.coef_[0]:.2f}')
axes[0, 2].plot(x_range_between, 
                between_reg.predict(x_range_between.reshape(-1, 1)),
                'r-', linewidth=2, label=f'Between: {between_reg.coef_[0]:.2f}')

axes[0, 2].set_xlabel('X1')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_title('Within vs Between Variation')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3)

# Plot 4: Time Series for Selected Individuals
selected_ids = [0, 5, 10, 15, 20]
for unit_id in selected_ids:
    unit_data = df[df['id'] == unit_id]
    axes[1, 0].plot(unit_data['time'], unit_data['y'], 
                   marker='o', linewidth=1, markersize=4, 
                   alpha=0.7, label=f'ID {unit_id}')

axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_title('Time Series: Selected Individuals')
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# Plot 5: Residuals Distribution
axes[1, 1].hist(pooled_model.resid, bins=50, alpha=0.5, 
               label='Pooled OLS', density=True)
axes[1, 1].hist(fe_model.resid, bins=50, alpha=0.5, 
               label='Fixed Effects', density=True)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Plot 6: Standard Errors Comparison
se_comparison = pd.DataFrame({
    'FE (Default)': fe_model.std_errors.values,
    'FE (Clustered)': fe_model.std_errors.values,  # Already clustered
    'Pooled OLS': pooled_model.std_errors.values
}, index=exog_vars)

x_pos_se = np.arange(len(se_comparison))
width_se = 0.25

for i, col in enumerate(se_comparison.columns):
    offset = (i - 1) * width_se
    axes[1, 2].bar(x_pos_se + offset, se_comparison[col], 
                   width_se, label=col, alpha=0.8)

axes[1, 2].set_xticks(x_pos_se)
axes[1, 2].set_xticklabels(se_comparison.index)
axes[1, 2].set_ylabel('Standard Error')
axes[1, 2].set_title('Standard Error Comparison')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ===== Test for Serial Correlation =====
print("\n" + "="*70)
print("DIAGNOSTIC TESTS")
print("="*70)

# Wooldridge test for serial correlation in FE models
print("\nSerial Correlation:")
print("  Note: Use clustered SE by entity to address serial correlation")
print(f"  FE model used clustered SE: {fe_model.cov_type}")

# ===== Balance Check =====
print("\n" + "="*70)
print("PANEL BALANCE")
print("="*70)

panel_counts = df.groupby('id').size()
print(f"Observations per unit:")
print(f"  Min: {panel_counts.min()}")
print(f"  Max: {panel_counts.max()}")
print(f"  Mean: {panel_counts.mean():.2f}")

if panel_counts.nunique() == 1:
    print("  ✓ Balanced panel (all units have same T)")
else:
    print("  ✗ Unbalanced panel (varying T across units)")

# ===== Summary =====
print("\n" + "="*70)
print("SUMMARY AND INTERPRETATION")
print("="*70)

print("\nKey Findings:")
print(f"1. Pooled OLS suffers from omitted variable bias:")
print(f"   - X1 coefficient: {pooled_model.params['x1']:.4f} (biased upward)")
print(f"   - Bias due to Cov(X1, αᵢ) ≠ 0")

print(f"\n2. Fixed Effects eliminates bias:")
print(f"   - X1 coefficient: {fe_model.params['x1']:.4f} (close to true 1.5)")
print(f"   - Uses within-unit variation only")

print(f"\n3. F-test strongly rejects pooled OLS:")
print(f"   - F = {f_stat:.2f}, p < {p_value:.4f}")
print(f"   - Individual effects are significant")

print(f"\n4. Fixed effects well estimated:")
print(f"   - Correlation with true αᵢ: {fe_comparison[['True_Alpha', 'Estimated_Alpha']].corr().iloc[0,1]:.4f}")

print("\nLimitations of FE:")
print("  • Cannot estimate time-invariant variables")
print("  • Requires strict exogeneity (no lagged Y)")
print("  • Less efficient than RE if RE assumptions hold")
print("  • Short panels (small T) may have large SE")
```

## 6. Challenge Round
When does fixed effects fail or need modification?
- **Time-invariant variables**: Gender, ethnicity absorbed by FE → Cannot estimate, use random effects or between estimator
- **Lagged dependent variable**: FE with Yᵢₜ₋₁ inconsistent (Nickell bias) → Use Arellano-Bond GMM
- **Short panels (T=2,3)**: FE has large variance, weak instrument if IV used → Consider first differences
- **Measurement error**: Within transformation amplifies attenuation bias → Use between estimator or IV
- **Heterogeneous treatment effects**: Two-way FE with staggered treatment gives negative weights → Use stacked DiD, CS estimator
- **Incidental parameters**: N→∞, T fixed has bias of order 1/T → Not problematic for linear, issue for nonlinear

## 7. Key References
- [Wooldridge - Econometric Analysis of Cross Section and Panel Data (Ch 10-11)](https://mitpress.mit.edu/9780262232586/)
- [Angrist & Pischke - Mostly Harmless Econometrics (Ch 5)](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- [Cameron & Trivedi - Microeconometrics (Ch 21-22)](https://www.cambridge.org/core/books/microeconometrics/1CA8C6B23A6FD32300AB5D5954CF1B73)

---
**Status:** Core panel data method | **Complements:** Random Effects, Hausman Test, Dynamic Panel, First Differences
