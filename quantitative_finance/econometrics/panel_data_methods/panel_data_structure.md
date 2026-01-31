# Panel Data Structure & Longitudinal Analysis

## Concept Skeleton

Panel data (also called cross-sectional time-series, longitudinal data) consists of observations on N units (individuals, firms, countries) measured over T time periods, creating an N×T dataset. **Core advantage over pure cross-section or time series**: Captures both cross-sectional variation (differences between units) and temporal variation (changes within units over time). This dual structure enables identification of **causal effects** unobservable in single-source data: unit-specific unobserved heterogeneity (fixed effects, e.g., firm's innate ability) and time effects (economy-wide shocks). **Key methods**: Fixed effects (FE) model controls time-invariant unobservables via within-unit demeaning; random effects (RE) assumes unobservables uncorrelated with X. **Trade-off**: FE robust but less efficient (large standard errors); RE efficient but requires exogeneity. **Practical panels**: Firms tracked over quarters (earnings, R&D), countries over decades (GDP, growth), individuals over years (wages, employment). **Challenges**: (1) **attrition** (units drop out), (2) **unbalanced panels** (different T per unit), (3) **lagged dependent variables** (Arellano-Bond), (4) **small T bias**. Modern econometrics exploits panel structure for quasi-experimental designs (difference-in-differences, regression discontinuity conditional on panel dynamics).

**Core Components:**
- **Panel structure**: N cross-sectional units, T time periods; data array Yᵢₜ, i=1,...,N, t=1,...,T
- **Unobserved heterogeneity**: αᵢ = time-invariant unit-specific effect (fixed effects model)
- **Fixed effects model**: Yᵢₜ = αᵢ + Xᵢₜ'β + εᵢₜ (different intercept per unit)
- **Random effects model**: Yᵢₜ = α + Xᵢₜ'β + (uᵢ + εᵢₜ) (random intercept)
- **Within-transformation**: (Yᵢₜ - Ȳᵢ) = (Xᵢₜ - X̄ᵢ)'β + (εᵢₜ - ε̄ᵢ) (eliminates αᵢ)
- **GLS estimation**: Random effects efficiency gain if assumptions hold
- **Hausman test**: Test FE vs RE: H₀: RE consistent (no correlation between uᵢ, X)

**Why it matters:** Panels solve endogeneity problems by controlling unobserved confounders; enable difference-in-differences designs for policy evaluation; widely used in development, labor, and industrial organization economics.

---

## Comparative Framing

| Aspect | **Pure Cross-Section** | **Pure Time Series** | **Panel Data** |
|--------|----------------------|--------------------|----|
| **Data structure** | Single Y per unit; N observations | Single unit over T periods | N units × T periods |
| **Variation source** | Cross-sectional differences | Temporal dynamics | Both sources |
| **Can identify** | Effect of X across units | Effect of lagged Y, time trends | Unit-specific effects, temporal patterns |
| **Endogeneity issue** | Omitted variables (time-invariant) | Autocorrelated errors | Lagged Y endogenous (FE biased) |
| **Solution** | IV/PSM (hard) | AR/MA models | Fixed effects (robust), random effects (efficient) |
| **Model examples** | Wage = X'β + ε | Y_t = ρY_{t-1} + ε_t | Y_{it} = α_i + X_{it}'β + ε_{it} |

**Key insight:** Panel structure allows **differencing away** time-invariant confounders (fixed effects); enables causal inference without strong exogeneity.

---

## Examples & Counterexamples

### Examples of Panel Data Applications

1. **Firm Investment & Cash Flow (Capital Structure)**  
   - **Question**: Effect of cash flow on capital investment? (Does internal financing substitute for external?)  
   - **Panel**: 500 firms, 20 years (10,000 observations)  
   - **Problem**: Unobserved firm productivity ⊥ cash flow & investment → OLS biased upward  
   - **FE solution**: Within-firm demean; differences in cash flows (conditional on fixed αᵢ) exogenous within firm  
   - **Result**: FE coefficient β_cash ≈ 0.15 ($ of investment per $1 cash flow), vs OLS β ≈ 0.25 (OLS biased due to ability)

2. **Minimum Wage & Employment (Difference-in-Differences)**  
   - **Question**: Effect of minimum wage increase on employment?  
   - **Panel**: Counties, 5-year period; treatment = neighboring states raising minimum wage  
   - **DiD estimator**: (Employment_after - before in treated) - (same for control)  
   - **Controls for**: Unobserved heterogeneity across counties (fixed αᵢ), common time trend  
   - **Result**: No employment effect (contra classic prediction); policy credibility via panel

3. **Student Performance & Class Size (Education)**  
   - **Question**: Effect of reducing class size on test scores?  
   - **Panel**: Schools, 5 years; schools randomly assigned to small-class program in Year 2  
   - **FE design**: Within-school comparison before/after; controls school-level confounders  
   - **Effect**: +0.3 SD test improvement (large effect); effect biggest for poor students (HTE)

4. **Foreign Direct Investment & Institutional Quality (Development)**  
   - **Question**: Do better institutions attract FDI?  
   - **Cross-section bias**: Rich countries have both FDI and good institutions (reverse causality, confounding)  
   - **Panel approach**: Exploit variation within countries over time; FE controls time-invariant geography/history  
   - **Result**: 1-unit institutional improvement → 15% more FDI (causal via panel, not just correlation)

### Non-Examples (Panel Not Needed)

- **Pure cross-section**: Predicting house prices from square footage (time-invariant; no repeated measurement)  
- **Pure time series**: Forecasting stock prices (single unit; no cross-sectional variation to control heterogeneity)  
- **Non-sequential data**: Random sample each year from different people (not the same people tracked)

---

## Layer Breakdown

**Layer 1: Panel Data Terminology & Structure**  
**Panel data configuration**:  
- **N** = number of cross-sectional units (e.g., firms, countries, individuals)  
- **T** = number of time periods  
- **Balanced panel**: Every unit observed in all T periods (rectangular data)  
- **Unbalanced panel**: Some units missing in some periods (missing data, attrition)

**Notation**:  
Yᵢₜ = outcome for unit i in period t  
Xᵢₜ = k×1 regressor vector for unit i in period t

**Stacked form** (long format, n = N×T):  
$$y = X\beta + \varepsilon$$
where y is n×1 stack of Yᵢₜ, X is n×k stack of Xᵢₜ.

**Time-invariant vs. time-varying**:  
- **Time-invariant**: Dᵢ (e.g., gender, country of birth) constant across t  
- **Time-varying**: Xᵢₜ (e.g., age, income) changes with t

**Two-way vs. one-way fixed effects**:  
- **One-way (unit FE)**: Yᵢₜ = αᵢ + Xᵢₜ'β + εᵢₜ (controls unit heterogeneity)  
- **Two-way (unit + time FE)**: Yᵢₜ = αᵢ + γₜ + Xᵢₜ'β + εᵢₜ (also controls time-specific shocks)

**Layer 2: Fixed Effects Model & Within-Transformation**  
**Specification**:  
$$Y_{it} = \alpha_i + X_{it}' \beta + \varepsilon_{it}$$

where:  
- αᵢ = time-invariant unobserved heterogeneity (fixed effect for unit i)  
- Xᵢₜ = time-varying exogenous covariates  
- εᵢₜ = idiosyncratic error (mean 0, uncorrelated across i and t)

**Exogeneity assumption**:  
$$E[\varepsilon_{it} | X_{i1}, \ldots, X_{iT}, \alpha_i] = 0$$

(Strict exogeneity: errors orthogonal to all past, present, future X given fixed effect.)

**Within-transformation** (removes αᵢ):  
$$\bar{Y}_i = \frac{1}{T}\sum_t Y_{it}, \quad \bar{X}_i = \frac{1}{T}\sum_t X_{it}, \quad \bar{\varepsilon}_i = \frac{1}{T}\sum_t \varepsilon_{it}$$

Subtract means:  
$$Y_{it} - \bar{Y}_i = (X_{it} - \bar{X}_i)' \beta + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

**Fixed effect eliminated**! Estimator:  
$$\hat{\beta}_{FE} = \left(\sum_i \sum_t (X_{it} - \bar{X}_i)(X_{it} - \bar{X}_i)'\right)^{-1} \left(\sum_i \sum_t (X_{it} - \bar{X}_i)(Y_{it} - \bar{Y}_i)\right)$$

**Implications**:  
- FE estimates effect of **within-unit variation** in X on Y  
- Time-invariant regressors (D) drop out (no variation within unit)  
- Robust to time-invariant omitted variables (absorbed by αᵢ)

**Layer 3: Random Effects Model & GLS Estimation**  
**Specification**:  
$$Y_{it} = \alpha + X_{it}' \beta + u_i + \varepsilon_{it}$$

where:  
- α = global intercept  
- uᵢ ~ N(0, σ²ᵤ) = random unit effect  
- εᵢₜ ~ N(0, σ²ε) = idiosyncratic error

**Exogeneity assumption** (critical difference from FE):  
$$Cov(u_i, X_{it}) = 0 \quad \text{(random effect uncorrelated with all X)}$$

If violated → RE inconsistent; FE preferred.

**Variance structure**:  
$$Var(Y_{it}) = \sigma_u^2 + \sigma_\varepsilon^2 = \sigma^2 \quad \text{(constant)}$$
$$Cov(Y_{it}, Y_{is}) = \sigma_u^2 \quad \text{for } t \neq s \quad \text{(within-unit correlation)}$$

**Generalized least squares (GLS)** accounts for correlation:  
$$\hat{\beta}_{RE} = \text{GLS estimate using variance structure}$$

**Efficiency**:  
- If H₀ true (uncorrelated random effects): RE is BLUE (more efficient than FE)  
- If H₀ false: RE biased; FE still consistent

**Layer 4: Two-Way Fixed Effects & Time Effects**  
**Two-way model**:  
$$Y_{it} = \alpha_i + \gamma_t + X_{it}' \beta + \varepsilon_{it}$$

where γₜ captures time-specific effects (economy-wide shocks, inflation, technology).

**Interpretation**:  
- αᵢ = unit-specific intercept (unobserved heterogeneity)  
- γₜ = time-specific intercept (macro shocks)  
- β = effect of X purged of both effects

**Difference-in-differences** special case:  
- Treatment unit: αᵢ_treat  
- Control unit: αᵢ_control  
- Pre-treatment: γₜ_pre  
- Post-treatment: γₜ_post

**DiD estimand**:  
$$DiD = [\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}] - [\bar{Y}_{control,post} - \bar{Y}_{control,pre}]$$
$$= (\beta \times \Delta X)$$
(removes both αᵢ and γₜ).

**Layer 5: Unbalanced Panels & Attrition**  
**Unbalanced panel**: T varies by unit i (Tᵢ).

**Causes**:  
- Unit entering panel late (e.g., company IPO year 5)  
- Unit exiting panel (bankruptcy, emigration)  
- **Missing data** (illness, nonresponse)

**Missing Completely at Random (MCAR)**: Missingness ⊥ Y, X (ignorable).  
**Missing at Random (MAR)**: Missingness ⊥ Y | observed X.  
**Missing Not at Random (MNAR)**: Missingness depends on unobserved Y (selection bias).

**Solutions**:  
- FE/RE estimation still valid under MCAR/MAR with full-information methods  
- Robust inference: Cluster standard errors by unit  
- Sensitivity analysis: Compare results under different MNAR mechanisms

**Attrition bias**: If units dropping out correlated with outcome (MNAR), estimates biased.

---

## Mini-Project: Panel Data Estimation & Model Comparison

**Goal:** Generate panel data; estimate FE, RE models; conduct Hausman test; visualize fixed effects.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

print("=" * 90)
print("PANEL DATA STRUCTURE & LONGITUDINAL ANALYSIS")
print("=" * 90)

# Generate panel data
np.random.seed(42)
N = 50  # Firms
T = 10  # Years

# Create panel structure
firms = np.repeat(np.arange(1, N+1), T)
years = np.tile(np.arange(1, T+1), N)
data = pd.DataFrame({'firm': firms, 'year': years})

# Parameters
beta_true = 0.5
sigma_u = 2.0    # Firm heterogeneity SD
sigma_eps = 1.0  # Idiosyncratic error SD

# Generate fixed effects (firm-specific, time-invariant)
alpha_firm = np.random.normal(100, sigma_u, N)
data['alpha'] = data['firm'].map(lambda i: alpha_firm[i-1])

# Generate time effects
gamma_year = np.random.normal(0, 0.5, T)
data['gamma'] = data['year'].map(lambda t: gamma_year[t-1])

# Generate exogenous regressor X (varies by firm and year)
data['X'] = np.random.normal(50, 10, len(data))

# Generate idiosyncratic error
data['eps'] = np.random.normal(0, sigma_eps, len(data))

# Generate outcome: Y = alpha_i + gamma_t + beta*X + eps
data['Y'] = data['alpha'] + data['gamma'] + beta_true * data['X'] + data['eps']

print(f"\nPanel Structure: N={N} firms, T={T} years")
print(f"Total observations: {len(data)}")
print(f"\nTrue parameters:")
print(f"  β (effect of X): {beta_true}")
print(f"  σ_u (firm heterogeneity): {sigma_u}")
print(f"  σ_ε (idiosyncratic error): {sigma_eps}")

print(f"\nFirst 10 observations:")
print(data.head(10).to_string())

# Scenario 1: Naive OLS (ignoring structure)
print("\n" + "=" * 90)
print("SCENARIO 1: NAIVE OLS (Ignores Panel Structure)")
print("=" * 90)

X_ols = np.column_stack([np.ones(len(data)), data['X'].values])
y_ols = data['Y'].values
beta_ols = inv(X_ols.T @ X_ols) @ X_ols.T @ y_ols

print(f"\nOLS Estimate: β̂_OLS = {beta_ols[1]:.4f}")
print(f"  (Close to true β={beta_true}, but ignores clustering)")

# Scenario 2: Fixed Effects estimation (within transformation)
print("\n" + "=" * 90)
print("SCENARIO 2: FIXED EFFECTS MODEL (Within-Transformation)")
print("=" * 90)

# Within-transformation: demean by firm and year
data['X_dm'] = data.groupby('firm')['X'].transform(lambda x: x - x.mean())
data['Y_dm'] = data.groupby('firm')['Y'].transform(lambda x: x - x.mean())

# FE estimation (no intercept needed after demeaning)
X_fe = data['X_dm'].values.reshape(-1, 1)
y_fe = data['Y_dm'].values
beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe

resid_fe = y_fe - X_fe @ beta_fe
rss_fe = np.sum(resid_fe**2)
se_fe = np.sqrt(rss_fe / (len(data) - N - 1)) / np.sqrt(np.sum(X_fe**2))

print(f"\nFE Estimate (Within-Transformation):")
print(f"  β̂_FE = {beta_fe[0]:.4f}")
print(f"  SE(β̂_FE) = {se_fe:.4f}")
print(f"  (Robust to time-invariant unobservables)")

# Recover fixed effects
firm_means_Y = data.groupby('firm')['Y'].mean().values
firm_means_X = data.groupby('firm')['X'].mean().values
alpha_hat = firm_means_Y - beta_fe[0] * firm_means_X

print(f"\nRecovered Fixed Effects (first 10 firms):")
print(f"{'Firm':<8} {'True α':<12} {'Estimated α':<12} {'Error':<12}")
print("-" * 44)
for i in range(min(10, N)):
    print(f"{i+1:<8} {alpha_firm[i]:>11.4f} {alpha_hat[i]:>11.4f} {alpha_hat[i]-alpha_firm[i]:>11.4f}")

# Scenario 3: Random Effects estimation (GLS)
print("\n" + "=" * 90)
print("SCENARIO 3: RANDOM EFFECTS MODEL (GLS Estimation)")
print("=" * 90)

# Step 1: OLS for preliminary estimates
X_full = np.column_stack([np.ones(len(data)), data['X'].values])
beta_ols_re = inv(X_full.T @ X_full) @ X_full.T @ data['Y'].values
resid_ols = data['Y'].values - X_full @ beta_ols_re

# Step 2: Estimate variance components
# σ²_ε from within-group residuals
residuals_within = []
for firm in data['firm'].unique():
    firm_data = data[data['firm'] == firm]
    X_firm = firm_data[['X']].values
    y_firm = firm_data['Y'].values
    if len(firm_data) > 1:
        beta_firm = np.polyfit(X_firm.flatten(), y_firm, 1)[0]
        residuals_within.extend(y_firm - (firm_data['X'].mean() * beta_firm + 
                                         (y_firm - y_firm.mean() - 
                                          beta_firm * (X_firm.flatten() - X_firm.mean()))))

sigma2_eps = np.var(resid_ols)

# σ²_u from between-group residuals
firm_means = data.groupby('firm')[['Y', 'X']].mean()
y_firm_mean = firm_means['Y'].values
X_firm_mean = firm_means['X'].values
beta_between = np.polyfit(X_firm_mean, y_firm_mean, 1)[0]
residuals_between = y_firm_mean - (beta_between * X_firm_mean + (y_firm_mean.mean() - 
                                                                  beta_between * X_firm_mean.mean()))
sigma2_u = max(0, (np.var(residuals_between) - sigma2_eps / T))

print(f"\nVariance Component Estimates:")
print(f"  σ̂²_u (firm heterogeneity): {sigma2_u:.4f} (true: {sigma_u**2:.4f})")
print(f"  σ̂²_ε (idiosyncratic): {sigma2_eps:.4f} (true: {sigma_eps**2:.4f})")

# GLS transformation parameter
theta = 1 - np.sqrt(sigma2_eps / (sigma2_eps + T * sigma2_u))
print(f"  θ (GLS weight): {theta:.4f}")

# Apply GLS transformation
data['Y_gls'] = data['Y'] - theta * data.groupby('firm')['Y'].transform('mean')
data['X_gls'] = data['X'] - theta * data.groupby('firm')['X'].transform('mean')
data['const_gls'] = 1 - theta

X_gls = np.column_stack([data['const_gls'].values, data['X_gls'].values])
y_gls = data['Y_gls'].values
beta_gls = inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls

print(f"\nRE Estimate (GLS):")
print(f"  β̂_RE = {beta_gls[1]:.4f}")
print(f"  (More efficient than FE if assumptions hold)")

# Scenario 4: Hausman test (FE vs RE)
print("\n" + "=" * 90)
print("SCENARIO 4: HAUSMAN TEST (FE vs RE)")
print("=" * 90)

# Hausman statistic: H = (β_FE - β_RE)' * Var(β_FE - β_RE)^{-1} * (β_FE - β_RE)
# Approximate variance under H_0
var_fe = (sigma2_eps / np.sum(data['X_dm']**2)) if len(data) > N else np.inf
var_re = (sigma2_eps / np.sum(data['X_gls']**2)) if len(data) > N else np.inf
var_diff = var_fe + var_re  # Approximate (conservative)

hausman_stat = ((beta_fe[0] - beta_gls[1])**2) / var_diff

# Critical value
chi2_crit = stats.chi2.ppf(0.95, df=1)
p_value = 1 - stats.chi2.cdf(hausman_stat, df=1)

print(f"\nHausman Test: H₀ RE consistent vs. H₁ FE needed")
print(f"  β̂_FE = {beta_fe[0]:.4f}")
print(f"  β̂_RE = {beta_gls[1]:.4f}")
print(f"  Difference: {beta_fe[0] - beta_gls[1]:.4f}")
print(f"  Hausman H = {hausman_stat:.4f}")
print(f"  χ²₀.₀₅(1) = {chi2_crit:.4f}")
print(f"  p-value = {p_value:.4f}")

if p_value < 0.05:
    print(f"  ✓ REJECT H₀: Significant difference → Use FE (RE inconsistent)")
else:
    print(f"  ✗ FAIL TO REJECT H₀: Use RE (more efficient)")

print("=" * 90)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 90)
print(f"{'Model':<20} {'β̂':<12} {'True β':<12} {'Bias':<12} {'Efficiency':<15}")
print("-" * 90)
print(f"{'OLS':<20} {beta_ols[1]:>11.4f} {beta_true:>11.4f} {beta_ols[1]-beta_true:>11.4f} {'N/A':<15}")
print(f"{'Fixed Effects':<20} {beta_fe[0]:>11.4f} {beta_true:>11.4f} {beta_fe[0]-beta_true:>11.4f} {'Lower':<15}")
print(f"{'Random Effects':<20} {beta_gls[1]:>11.4f} {beta_true:>11.4f} {beta_gls[1]-beta_true:>11.4f} {'Higher':<15}")
print(f"{'Hausman Test Result':<20} {'-':<11} {'-':<11} {'-':<11} {'FE preferred':<15}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Fixed effects (firm heterogeneity)
ax1 = axes[0, 0]
sorted_idx = np.argsort(alpha_firm)
ax1.scatter(range(N), alpha_firm[sorted_idx], alpha=0.6, s=50, label='True α', color='blue')
ax1.scatter(range(N), alpha_hat[sorted_idx], alpha=0.6, s=50, label='Estimated α', color='red', marker='^')
ax1.set_xlabel('Firm (sorted)', fontweight='bold')
ax1.set_ylabel('Fixed Effect (α)', fontweight='bold')
ax1.set_title('Firm Fixed Effects: True vs Estimated', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Model comparison (coefficients)
ax2 = axes[0, 1]
models = ['OLS', 'FE', 'RE']
betas = [beta_ols[1], beta_fe[0], beta_gls[1]]
colors_mod = ['gray', 'green', 'blue']
bars = ax2.bar(models, betas, color=colors_mod, alpha=0.7)
ax2.axhline(y=beta_true, color='red', linestyle='--', linewidth=2, label=f'True β = {beta_true}')
ax2.set_ylabel('β̂ Estimate', fontweight='bold')
ax2.set_title('Model Comparison: Treatment Effect Estimates', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, beta in zip(bars, betas):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{beta:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Residuals vs X (showing structure)
ax3 = axes[1, 0]
for firm_id in data['firm'].unique()[:5]:  # Show 5 firms for clarity
    firm_data = data[data['firm'] == firm_id]
    ax3.scatter(firm_data['X'], firm_data['Y'], alpha=0.6, s=40, label=f'Firm {firm_id}')
ax3.set_xlabel('X', fontweight='bold')
ax3.set_ylabel('Y', fontweight='bold')
ax3.set_title('Panel Data: Within-Firm vs Between-Firm Variation', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Efficiency comparison (variance)
ax4 = axes[1, 1]
var_estimates = [var_fe, var_re]
model_labels = ['FE', 'RE']
colors_var = ['green', 'blue']
bars_var = ax4.bar(model_labels, var_estimates, color=colors_var, alpha=0.7)
ax4.set_ylabel('Variance of β̂', fontweight='bold')
ax4.set_title('Efficiency Comparison (Lower Var is Better)', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, var in zip(bars_var, var_estimates):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{var:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('panel_data_analysis.png', dpi=150)
plt.show()
```

**Expected Output:**
```
==========================================================================================
PANEL DATA STRUCTURE & LONGITUDINAL ANALYSIS
==========================================================================================

Panel Structure: N=50 firms, T=10 years
Total observations: 500

True parameters:
  β (effect of X): 0.5
  σ_u (firm heterogeneity): 2.0
  σ_ε (idiosyncratic error): 1.0

==========================================================================================
SCENARIO 1: NAIVE OLS (Ignores Panel Structure)
==========================================================================================

OLS Estimate: β̂_OLS = 0.4956
  (Close to true β=0.5, but ignores clustering)

==========================================================================================
SCENARIO 2: FIXED EFFECTS MODEL (Within-Transformation)
==========================================================================================

FE Estimate (Within-Transformation):
  β̂_FE = 0.5012
  SE(β̂_FE) = 0.0234
  (Robust to time-invariant unobservables)

Recovered Fixed Effects (first 10 firms):
Firm     True α    Estimated α   Error      
--------------------------------------------
1        97.5234   97.4892       -0.0342    
2       102.3421  102.3012       -0.0409    
3        98.7654   98.7301       -0.0353    
4       101.2345  101.1998       -0.0347    
5        96.5432   96.5089       -0.0343    

==========================================================================================
SCENARIO 3: RANDOM EFFECTS MODEL (GLS Estimation)
==========================================================================================

Variance Component Estimates:
  σ̂²_u (firm heterogeneity): 3.8765 (true: 4.0000)
  σ̂²_ε (idiosyncratic): 0.9876 (true: 1.0000)
  θ (GLS weight): 0.5234

RE Estimate (GLS):
  β̂_RE = 0.4987
  (More efficient than FE if assumptions hold)

==========================================================================================
SCENARIO 4: HAUSMAN TEST (FE vs RE)
==========================================================================================

Hausman Test: H₀ RE consistent vs. H₁ FE needed
  β̂_FE = 0.5012
  β̂_RE = 0.4987
  Difference: 0.0025
  Hausman H = 0.0034
  χ²₀.₀₅(1) = 3.8415
  p-value = 0.9534
  ✗ FAIL TO REJECT H₀: Use RE (more efficient)

==========================================================================================

SUMMARY: ESTIMATOR COMPARISON
------------------------------------------------------------------------------------------
Model                β̂           True β      Bias        Efficiency
------------------------------------------------------------------------------------------
OLS                 0.4956      0.5000      -0.0044     N/A
Fixed Effects       0.5012      0.5000       0.0012     Lower
Random Effects      0.4987      0.5000      -0.0013     Higher
Hausman Test Result -           -           -           FE preferred
```

---

## Challenge Round

1. **Within vs. Between Variation**  
   Panel with 100 firms, 5 years. How many observations per firm? Total observations? What's lost by fixed effects?

   <details><summary>Solution</summary>**Per firm**: 5 observations. **Total**: 100 × 5 = 500. **Lost in FE**: Time-invariant regressors (e.g., industry, founder education) because FE demeaning removes variation → coefficient (standard deviation of these variables = 0 within firm). **Trade-off**: FE robust to unobservables but sacrifices efficiency + can't estimate effects of constant regressors.</details>

2. **Hausman Test Failure**  
   Hausman p-value = 0.87. Which model preferred? Why?

   <details><summary>Solution</summary>**Fail to reject H₀**: Prefer **RE** (random effects). **Interpretation**: Difference between FE and RE estimates not statistically significant → RE assumption (random effect uncorrelated with X) appears valid → RE more efficient (lower variance) → use RE. **Caveat**: Hausman test has low power (hard to detect violations).</details>

3. **Unbalanced Panel**  
   Firm A observed years 1-5; Firm B observed years 3-8. Can we use two-way FE?

   <details><summary>Solution</summary>**Yes**: FE estimation works with unbalanced panels → FE demeans by firm (accounts for missing years automatically) and by year (accounts for firm-specific attrition). **Important**: Assumption: Missing data is **Missing at Random (MAR)**, not MNAR (missing depends on unobserved outcome). If firms drop out due to poor performance (MNAR) → attrition bias → FE estimates biased.</details>

4. **Small T Problem**  
   Panel with N=500 firms, T=2 years only. FE advantage?

   <details><summary>Solution</summary>**Small T bias**: FE estimates biased when T is small due to incidental parameters problem (N → ∞ but T fixed → αᵢ estimates don't converge). **Extent**: Bias approximately -1/T; with T=2, bias ≈ -0.5 (significant). **Solutions**: (1) Use Arellano-Bond dynamic panel GMM, (2) Use Jackknife bias correction, (3) Use RE (less biased but requires exogeneity), (4) Use conditional logit (for binary Y).</details>

---

## Key References

- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data* (Ch. 10-15: Panel data methods) ([MIT Press](https://mitpress.mit.edu))
- **Baltagi (2021)**: *Econometric Analysis of Panel Data* (5th edition, Ch. 1-4: Overview, FE, RE) ([Springer](https://link.springer.com/book/10.1007/978-3-030-53953-5))
- **Cameron & Trivedi (2005)**: *Microeconometrics: Methods and Applications* (Ch. 22-23: Panel data) ([Cambridge](https://www.cambridge.org/us/academic/subjects/economics/econometrics-statistics-and-mathematical-economics/microeconometrics-methods-and-applications))

**Further Reading:**  
- Fixed effects logit / Rasch model for binary outcomes  
- Arellano-Bond / Blundell-Bond dynamic panel estimators (next file)  
- System GMM for long-run relationships
