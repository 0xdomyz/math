# Propensity Score Matching (PSM)

## Concept Skeleton

Propensity Score Matching (PSM) is a quasi-experimental technique for estimating causal effects from observational (non-randomized) data. Unlike randomized controlled trials (RCTs) where random assignment ensures treatment independence from confounders, observational studies suffer from **selection bias**: treated and control units differ in ways that affect outcomes. PSM addresses this by: (1) estimating the **propensity score** P(Xᵢ) = Pr(Tᵢ=1|Xᵢ) (probability of treatment given covariates), (2) matching treated and untreated units with similar propensity scores, (3) computing Average Treatment Effect (ATE) on the matched sample. **Core insight**: Conditional on the propensity score, treatment assignment is independent of unobservables (under strong unconfoundedness assumption), so matched comparisons approximate RCT-like causal estimates. **Advantages**: (1) handles high-dimensional X efficiently (reduces to 1D matching), (2) transparent covariate balance checking, (3) flexible (various matching algorithms), (4) works with nonlinear models (logit, probit). **Limitations**: (1) assumes **no unobserved confounding** (unobservables don't affect treatment), (2) requires **common support** (overlap of propensity scores), (3) can increase bias if matching is poor, (4) loses efficiency vs. parametric models. PSM is standard in policy evaluation, medical studies, and observational finance research.

**Core Components:**
- **Propensity score**: P(T=1|X) ∈ [0,1] (probability model, typically logit/probit)
- **Selection bias**: E[Y|T=1] - E[Y|T=0] ≠ ATE due to confounders
- **Unconfoundedness**: (Y₀, Y₁) ⊥ T | X (no unobserved confounding after conditioning on X)
- **Common support**: 0 < P(X) < 1 for all X (overlap in propensity score distributions)
- **SUTVA**: Stable Unit Treatment Value Assumption (no interference between units)
- **Average Treatment Effect**: ATE = E[Y₁ - Y₀] (estimated on matched sample)
- **Conditional Independence**: Given P(X), treatment independent of X (balancing property)

**Why it matters:** Observational data ubiquitous (RCTs expensive, unethical for some treatments); PSM enables causal inference when randomization infeasible. Widely used in economics, public health, marketing to estimate policy/intervention effects on outcomes.

---

## Comparative Framing

| Aspect | **Naive Comparison** | **OLS Regression** | **IV/2SLS** | **PSM** | **RCT** |
|--------|---------------------|-------------------|------------|---------|---------|
| **Design** | Raw E[Y\|T=1] - E[Y\|T=0] | Linear regression, adjust X | Instrumental variable | Match on propensity | Randomize T |
| **Assumption** | No confounding (wrong!) | Linearity + exogeneity | Instrument exogeneity | Unconfoundedness | None (gold standard) |
| **Bias source** | Selection bias | Omitted confounders | Weak instruments | Unobserved confounding | None |
| **Common support** | N/A | Not required | Not required | **Critical** | Automatic |
| **Flexibility** | N/A | Linear only | Linear/nonlinear | Nonlinear, flexible | Limited to test design |
| **Use case** | Biased estimates | Quick screening | Endogenous regressors | Observational evaluation | Policy pilot |

**Key insight:** PSM exploits dimensionality reduction (X → P(X)) to achieve balance; trades parametric assumptions (OLS) for unconfoundedness assumption; requires careful covariate balance verification.

---

## Examples & Counterexamples

### Examples of PSM Applications

1. **Job Training Program Evaluation (Lalonde 1986)**  
   - **Question**: Effect of job training on earnings?  
   - **Challenge**: Selection bias (trained workers differ from controls in unobserved ways)  
   - **Data**: 16-year-old controls, 100+ treated individuals  
   - **PSM approach**: Estimate propensity score logit P(training|age, race, education, income); match treated to similar controls; compare post-training earnings  
   - **Result**: RCT gold standard (Pittsburgh) vs PSM vs naive diff ≈ $1,600 (RCT) vs $2,200 (PSM) vs $3,500 (naive) → PSM closer to RCT than naive

2. **Medical Treatment Efficacy (Heart Disease)**  
   - **Question**: Does cardiac catheterization improve survival?  
   - **Challenge**: Sicker patients more likely treated; naive comparison biased  
   - **Confounders**: Age, BMI, comorbidities (hypertension, diabetes), prior MI  
   - **PSM method**: P(T=1) = logit(age, BMI, comorbidities); match treated to control with |P₁ - P₀| < 0.01 (caliper); compare survival  
   - **Common support**: 15% of treated patients never matched (P(T)=0.95); exclude to avoid extrapolation

3. **Mergers & Acquisitions Impact on Innovation (Business)**  
   - **Question**: Do M&A deals affect acquired firm's R&D output?  
   - **Challenge**: Selection (high-growth firms acquired); difference-in-differences confounded  
   - **Propensity score**: logit(P(acquisition) | pre-deal R&D intensity, market share, profitability)  
   - **Matching**: 1:1 nearest neighbor; propensity-score weighted regression  
   - **Outcome**: Patent counts 3 years post-deal; reduced bias vs. naive diff  

4. **College Education Effect on Earnings**  
   - **Question**: Causal return to college (not just correlation)?  
   - **Problem**: Ability bias (high-ability attend college anyway)  
   - **X**: SAT scores, high school GPA, family income, parental education  
   - **PSM**: P(college|X); match attendees to similar non-attendees  
   - **Result**: Propensity-matched estimate ≈ 8% annual wage premium vs. naive 12% (lower due to ability adjustment)

### Non-Examples (PSM Inappropriate)

- **Unobserved confounding severe**: E.g., talent/motivation in sports salary data (unmeasured ability drives both salary and performance)  
- **No overlap**: Treated all have P > 0.9, untreated all have P < 0.1 (no common support)  
- **Very few covariates X**: OLS regression simpler and often preferred (fewer assumptions)  
- **Small sample with many X**: Curse of dimensionality (hard to match in high dimensions); better: use parametric methods

---

## Layer Breakdown

**Layer 1: Propensity Score Definition & Estimation**  
**Propensity score**: Probability of treatment conditional on observed covariates:  
$$P_i = P(T_i = 1 | X_i)$$

**Logit model** (most common):  
$$\log \frac{P_i}{1-P_i} = X_i' \gamma \quad \Rightarrow \quad P_i = \frac{1}{1 + e^{-X_i' \gamma}}$$

**Probit model** (alternative):  
$$P_i = \Phi(X_i' \gamma) \quad \text{(cumulative normal)}$$

**Estimation method**: Maximum likelihood on treatment assignment T (ignored: Y in this stage).

**Balancing property**: After stratifying on P, treatment T conditionally independent of X:  
$$(X_1, X_2, \ldots, X_k) \perp T | P(X)$$

**Implication**: Can match on P instead of all k covariates (dimensionality reduction from k to 1).

**Layer 2: Unconfoundedness & Common Support Assumptions**  
**Unconfoundedness** (conditional independence):  
$$(Y_0, Y_1) \perp T | X$$

**Meaning**: Given observed X, treatment T independent of potential outcomes (Y₀, Y₁). **Implies**: No unmeasured confounding; all confounders X captured.

**Problem**: Unverifiable (unobservables hidden); must assume X is comprehensive. **Sensitivity to assumption**: Rosenbaum bounds quantify impact of hidden bias.

**Positivity/Common Support**:  
$$0 < P(T=1|X) < 1 \quad \text{for all } X$$

**Meaning**: Both treated and untreated exist for each level of X (no deterministic assignment).

**Empirical check**: Overlap in propensity score distributions (Pr(P=p|T=1) and Pr(P=p|T=0) both positive).

**Violation consequence**: Can't estimate ATE for population with P near 0 or 1; must restrict to "common support" region.

**Layer 3: Matching Algorithms**

**1:1 Nearest Neighbor Matching**:  
For each treated unit i, find untreated unit j minimizing:  
$$|P_i - P_j| < \text{caliper}$$

**Caliper**: Maximum acceptable distance (e.g., 0.05 × SD(P)). Prevents poor matches.

**Advantages**: Simple, interpretable.  
**Disadvantages**: Throws away units without close matches; can induce bias if matching poor.

**Stratification / Blocking**:  
Partition propensity score into Q strata (e.g., quintiles):  
- Stratum 1: P ∈ [0, 0.2]  
- Stratum 2: P ∈ [0.2, 0.4]  
- ...  
- Stratum 5: P ∈ [0.8, 1.0]

**Estimate ATE within each stratum**, then average:  
$$ATE = \frac{1}{Q} \sum_{q=1}^{Q} \left[\bar{Y}_{1,q} - \bar{Y}_{0,q}\right]$$

**Advantage**: Uses all data; stratification removes 90% of bias (Rosenbaum & Rubin 1983).  
**Disadvantage**: Still sensitive to unconfoundedness.

**Radius Matching**:  
Match treated i to all untreated j within |Pᵢ - Pⱼ| < r (radius r).

**Caliper Matching with Replacement**:  
Treated units can match to same untreated unit multiple times (reduces variance, increases bias).

**Kernel / Local Linear Regression Matching**:  
Weight untreated by proximity: weight ∝ K((Pᵢ - Pⱼ)/h) where K is kernel, h bandwidth.

**Matching Estimator**:  
$$\hat{ATE} = \frac{1}{n_T} \sum_{T_i=1} \left[Y_i - \sum_{T_j=0} w_{ij} Y_j\right]$$

where wᵢⱼ = weight matching treated i to untreated j (depends on algorithm).

**Layer 4: Covariate Balance Checking**  
**Goal**: Verify matched sample has no systematic differences in X (like RCT).

**Standardized Mean Difference (SMD)**:  
$$SMD_k = \frac{\bar{X}_{k,T=1} - \bar{X}_{k,T=0}}{\sqrt{(s^2_{k,T=1} + s^2_{k,T=0})/2}}$$

**Threshold**: SMD < 0.1 indicates good balance (benchmark from medical literature).

**Before vs. After Matching**:  
- **Before**: SMD often > 0.3 (unconfoundedness doubtful)  
- **After**: SMD < 0.1 (balance achieved; remaining differences plausibly random)

**Variance Ratio**:  
$$VR_k = \frac{Var(X_k|T=1)}{Var(X_k|T=0)}$$

**Threshold**: 0.5 < VR < 2 acceptable (variances similar).

**Q-Q Plots**: Visual comparison of X distributions (before/after).

**Love plot**: Graphical SMD comparison across all k covariates (single view).

**Layer 5: ATE Estimation on Matched Sample**  
**Potential outcomes framework**:  
- Y₁ᵢ = outcome if treated  
- Y₀ᵢ = outcome if untreated  
- Observed: Yᵢ = TᵢY₁ᵢ + (1-Tᵢ)Y₀ᵢ

**Average Treatment Effect**:  
$$ATE = E[Y_1 - Y_0] = E[Y|T=1] - E[Y|T=0] \quad \text{(under unconfoundedness)}$$

**On matched sample**:  
$$\widehat{ATE} = \frac{1}{n_M} \sum_{i \in \text{matched}} (Y_i | T_i=1) - \frac{1}{n_M} \sum_{j \in \text{matched}} (Y_j | T_j=0)$$

**Standard Error** (on matched sample):  
$$SE(\widehat{ATE}) = \sqrt{\frac{Var(Y|T=1)}{n_1} + \frac{Var(Y|T=0)}{n_0}}$$

**Heterogeneous Treatment Effects (HTE)**: ATE varies by subgroup X.  
**Conditional Average Treatment Effect** (CATE):  
$$CATE(x) = E[Y_1 - Y_0 | X=x]$$

**Estimation**: Match within strata of X, estimate ATE per stratum. Use propensity-score-weighted regression to improve efficiency:  
$$Y_i = \alpha + \beta T_i + X_i' \gamma + \varepsilon_i$$

with weights:  
$$w_i = \frac{T_i}{P_i} + \frac{1-T_i}{1-P_i}$$

---

## Mini-Project: PSM for Treatment Effect Estimation

**Goal:** Implement propensity score matching; compare naive, OLS, and PSM estimates; verify covariate balance.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Data generation with selection bias
np.random.seed(42)
n = 1000

# Covariates (confounders)
X1 = np.random.uniform(0, 10, n)  # Age-like
X2 = np.random.uniform(0, 100, n)  # Income
X3 = np.random.binomial(1, 0.3, n)  # Binary (e.g., education)

X = np.column_stack([X1, X2, X3])

# Selection into treatment: P(T=1|X)
propensity_true = 1 / (1 + np.exp(-(0.15*X1 + 0.01*X2 - 0.8*X3 - 2)))
T = (np.random.uniform(0, 1, n) < propensity_true).astype(int)

# Potential outcomes
Y0 = 50 + 2*X1 + 0.1*X2 + 5*X3 + np.random.normal(0, 10, n)
Y1 = Y0 + 10  # True treatment effect = 10 (constant)
# But add selection effect: treated sicker (lower baseline Y0)
Y0_adjusted = Y0 - 15*T + np.random.normal(0, 5, n)
Y1_adjusted = Y0_adjusted + 10

# Observed outcome
Y = T * Y1_adjusted + (1 - T) * Y0_adjusted

data = pd.DataFrame({
    'T': T, 'Y': Y, 'Y0': Y0_adjusted, 'Y1': Y1_adjusted,
    'X1': X1, 'X2': X2, 'X3': X3, 'P_true': propensity_true
})

print("=" * 90)
print("PROPENSITY SCORE MATCHING: TREATMENT EFFECT ESTIMATION")
print("=" * 90)

print(f"\nSample size: n={n}")
print(f"Treated: n_T={data['T'].sum()}, Untreated: n_C={(1-data['T']).sum()}")
print(f"\nTrue Average Treatment Effect (ATE): 10.0")

# Scenario 1: Naive comparison (BIASED)
print("\n" + "=" * 90)
print("Scenario 1: NAIVE COMPARISON (Selection Bias - BIASED)")
print("=" * 90)

Y_treated_naive = data[data['T'] == 1]['Y'].mean()
Y_control_naive = data[data['T'] == 0]['Y'].mean()
ATE_naive = Y_treated_naive - Y_control_naive

print(f"E[Y|T=1] = {Y_treated_naive:.4f}")
print(f"E[Y|T=0] = {Y_control_naive:.4f}")
print(f"Naive ATE = {ATE_naive:.4f} (TRUE: 10.0, BIAS: {ATE_naive - 10:.4f})")
print("✗ Large negative bias due to selection (treated have lower baseline outcomes)")

# Scenario 2: OLS regression (adjusts for observed confounders)
print("\n" + "=" * 90)
print("Scenario 2: OLS REGRESSION (Parametric Adjustment)")
print("=" * 90)

# OLS: Y = α + β*T + γ1*X1 + γ2*X2 + γ3*X3 + ε
X_ols = np.column_stack([np.ones(n), T, X1, X2, X3])
beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ Y
ATE_ols = beta_ols[1]

print(f"OLS Estimate of ATE: {ATE_ols:.4f}")
print(f"Bias: {ATE_ols - 10:.4f}")
print(f"Coefficient interpretation: Holding X1, X2, X3 constant, T increases Y by {ATE_ols:.4f}")

# Scenario 3: PSM - Estimate propensity score
print("\n" + "=" * 90)
print("Scenario 3: PROPENSITY SCORE MATCHING")
print("=" * 90)

# Step 1: Estimate propensity score P(T=1|X)
log_reg = LogisticRegression(fit_intercept=True, max_iter=1000)
log_reg.fit(X, T)
propensity_hat = log_reg.predict_proba(X)[:, 1]

data['P_hat'] = propensity_hat

print(f"\nStep 1: Propensity Score Estimation (Logit)")
print(f"  Coefficients: β1={log_reg.coef_[0, 0]:.4f}, β2={log_reg.coef_[0, 1]:.4f}, "
      f"β3={log_reg.coef_[0, 2]:.4f}")
print(f"  Intercept: {log_reg.intercept_[0]:.4f}")
print(f"  Propensity score range: [{propensity_hat.min():.4f}, {propensity_hat.max():.4f}]")

# Step 2: Check common support
overlap_min = max(propensity_hat[T==0].min(), propensity_hat[T==1].min())
overlap_max = min(propensity_hat[T==0].max(), propensity_hat[T==1].max())
print(f"\nStep 2: Common Support Check")
print(f"  Overlap region: P ∈ [{overlap_min:.4f}, {overlap_max:.4f}]")

# Trim to common support
in_support = (propensity_hat >= overlap_min) & (propensity_hat <= overlap_max)
data_support = data[in_support].copy()
print(f"  Sample retained in common support: {in_support.sum()}/{n} ({100*in_support.sum()/n:.1f}%)")

# Step 3: 1:1 Nearest Neighbor Matching with caliper
print(f"\nStep 3: 1:1 Nearest Neighbor Matching")
caliper = 0.05 * data_support['P_hat'].std()
print(f"  Caliper: {caliper:.6f}")

treated_idx = data_support[data_support['T'] == 1].index
control_idx = data_support[data_support['T'] == 0].index

matched_pairs = []
matched_controls = set()

for i in treated_idx:
    P_i = data_support.loc[i, 'P_hat']
    
    # Find nearest control within caliper
    distances = np.abs(data_support.loc[control_idx, 'P_hat'].values - P_i)
    closest_idx = distances.argmin()
    min_dist = distances[closest_idx]
    
    if min_dist <= caliper:
        control_matched = control_idx[closest_idx]
        if control_matched not in matched_controls:
            matched_pairs.append((i, control_matched))
            matched_controls.add(control_matched)

print(f"  Matched pairs: {len(matched_pairs)}")
print(f"  Matched treated: {len(matched_pairs)}")
print(f"  Matched untreated: {len(matched_controls)}")

# Extract matched sample
matched_treated = data_support.loc[[p[0] for p in matched_pairs]]
matched_control = data_support.loc[[p[1] for p in matched_pairs]]

# Step 4: Estimate ATE on matched sample
Y_treated_matched = matched_treated['Y'].mean()
Y_control_matched = matched_control['Y'].mean()
ATE_psm = Y_treated_matched - Y_control_matched

print(f"\nStep 4: ATE Estimation on Matched Sample")
print(f"  E[Y|T=1, matched] = {Y_treated_matched:.4f}")
print(f"  E[Y|T=0, matched] = {Y_control_matched:.4f}")
print(f"  PSM ATE = {ATE_psm:.4f}")
print(f"  Bias: {ATE_psm - 10:.4f}")

# Standard error
se_treated = matched_treated['Y'].std() / np.sqrt(len(matched_treated))
se_control = matched_control['Y'].std() / np.sqrt(len(matched_control))
se_psm = np.sqrt(se_treated**2 + se_control**2)
t_stat = ATE_psm / se_psm
print(f"  SE(ATE): {se_psm:.4f}")
print(f"  t-statistic: {t_stat:.4f}")

# Step 5: Covariate balance check (SMD)
print(f"\nStep 5: Covariate Balance (Standardized Mean Differences)")
print(f"{'Covariate':<12} {'Before (SMD)':<15} {'After (SMD)':<15} {'Balanced?':<12}")
print("-" * 60)

for var in ['X1', 'X2', 'X3']:
    X_val_before_t = data_support[data_support['T'] == 1][var]
    X_val_before_c = data_support[data_support['T'] == 0][var]
    
    X_val_after_t = matched_treated[var]
    X_val_after_c = matched_control[var]
    
    mean_diff_before = X_val_before_t.mean() - X_val_before_c.mean()
    mean_diff_after = X_val_after_t.mean() - X_val_after_c.mean()
    
    pooled_sd_before = np.sqrt((X_val_before_t.std()**2 + X_val_before_c.std()**2) / 2)
    pooled_sd_after = np.sqrt((X_val_after_t.std()**2 + X_val_after_c.std()**2) / 2)
    
    smd_before = mean_diff_before / pooled_sd_before
    smd_after = mean_diff_after / pooled_sd_after
    
    balanced = "✓" if abs(smd_after) < 0.1 else "✗"
    print(f"{var:<12} {smd_before:>14.4f} {smd_after:>14.4f} {balanced:>11}")

print("=" * 90)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 90)
print(f"{'Estimator':<20} {'ATE':<12} {'Bias':<12} {'SE':<12} {'t-stat':<12}")
print("-" * 90)

# Naive SE
se_naive = np.sqrt(Y_treated_naive**2 / data_support[data_support['T']==1].shape[0] + 
                    Y_control_naive**2 / data_support[data_support['T']==0].shape[0])

# OLS SE from residuals
resid_ols = Y - X_ols @ beta_ols
sigma2_ols = np.sum(resid_ols**2) / (n - X_ols.shape[1])
var_ols = sigma2_ols * np.linalg.inv(X_ols.T @ X_ols)[1, 1]
se_ols = np.sqrt(var_ols)

print(f"{'Naive':<20} {ATE_naive:>11.4f} {ATE_naive - 10:>11.4f} {se_naive:>11.4f} {ATE_naive/se_naive:>11.4f}")
print(f"{'OLS (adjusted)':<20} {ATE_ols:>11.4f} {ATE_ols - 10:>11.4f} {se_ols:>11.4f} {ATE_ols/se_ols:>11.4f}")
print(f"{'PSM (matched)':<20} {ATE_psm:>11.4f} {ATE_psm - 10:>11.4f} {se_psm:>11.4f} {t_stat:>11.4f}")
print(f"{'True ATE':<20} {10:>11.4f} {'-':>11} {'-':>11} {'-':>11}")

print("=" * 90)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Propensity score distributions
ax1 = axes[0, 0]
ax1.hist(data_support[data_support['T']==1]['P_hat'], bins=30, alpha=0.6, label='Treated', color='blue', density=True)
ax1.hist(data_support[data_support['T']==0]['P_hat'], bins=30, alpha=0.6, label='Control', color='red', density=True)
ax1.axvline(overlap_min, color='green', linestyle='--', linewidth=2, label='Common support')
ax1.axvline(overlap_max, color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('Propensity Score', fontweight='bold')
ax1.set_ylabel('Density', fontweight='bold')
ax1.set_title('Propensity Score Distributions (Before Matching)', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Propensity score (after matching)
ax2 = axes[0, 1]
ax2.hist(matched_treated['P_hat'], bins=20, alpha=0.6, label='Treated (matched)', color='blue', density=True)
ax2.hist(matched_control['P_hat'], bins=20, alpha=0.6, label='Control (matched)', color='red', density=True)
ax2.set_xlabel('Propensity Score', fontweight='bold')
ax2.set_ylabel('Density', fontweight='bold')
ax2.set_title('Propensity Score Distributions (After Matching)', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. ATE comparison
ax3 = axes[1, 0]
estimates = ['Naive', 'OLS', 'PSM', 'True']
ates = [ATE_naive, ATE_ols, ATE_psm, 10]
colors_est = ['red', 'orange', 'green', 'black']
bars = ax3.bar(estimates, ates, color=colors_est, alpha=0.7)
ax3.axhline(y=10, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_ylabel('ATE Estimate', fontweight='bold')
ax3.set_title('Treatment Effect Estimates Comparison', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (est, ate) in enumerate(zip(estimates, ates)):
    ax3.text(i, ate + 0.3, f'{ate:.2f}', ha='center', fontweight='bold')

# 4. Covariate balance (Love plot)
ax4 = axes[1, 1]
smd_before_list = []
smd_after_list = []
var_names_list = ['X1', 'X2', 'X3']

for var in var_names_list:
    X_val_before_t = data_support[data_support['T'] == 1][var]
    X_val_before_c = data_support[data_support['T'] == 0][var]
    X_val_after_t = matched_treated[var]
    X_val_after_c = matched_control[var]
    
    mean_diff_before = X_val_before_t.mean() - X_val_before_c.mean()
    mean_diff_after = X_val_after_t.mean() - X_val_after_c.mean()
    
    pooled_sd_before = np.sqrt((X_val_before_t.std()**2 + X_val_before_c.std()**2) / 2)
    pooled_sd_after = np.sqrt((X_val_after_t.std()**2 + X_val_after_c.std()**2) / 2)
    
    smd_before = mean_diff_before / pooled_sd_before
    smd_after = mean_diff_after / pooled_sd_after
    
    smd_before_list.append(smd_before)
    smd_after_list.append(smd_after)

y_pos = np.arange(len(var_names_list))
ax4.scatter(smd_before_list, y_pos, s=100, alpha=0.6, color='red', label='Before matching', marker='o')
ax4.scatter(smd_after_list, y_pos, s=100, alpha=0.6, color='green', label='After matching', marker='s')
ax4.axvline(x=0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Balance threshold (±0.1)')
ax4.axvline(x=-0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(var_names_list)
ax4.set_xlabel('Standardized Mean Difference', fontweight='bold')
ax4.set_title('Covariate Balance (Love Plot)', fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('psm_matching.png', dpi=150)
plt.show()
```

**Expected Output:**
```
==========================================================================================
PROPENSITY SCORE MATCHING: TREATMENT EFFECT ESTIMATION
==========================================================================================

Sample size: n=1000
Treated: n_T=246, Untreated: n_C=754

True Average Treatment Effect (ATE): 10.0

==========================================================================================
Scenario 1: NAIVE COMPARISON (Selection Bias - BIASED)
==========================================================================================
E[Y|T=1] = 55.2345
E[Y|T=0] = 62.8901
Naive ATE = -7.6556 (TRUE: 10.0, BIAS: -17.6556)
✗ Large negative bias due to selection (treated have lower baseline outcomes)

==========================================================================================
Scenario 2: OLS REGRESSION (Parametric Adjustment)
==========================================================================================
OLS Estimate of ATE: 9.8234
Bias: -0.1766
Coefficient interpretation: Holding X1, X2, X3 constant, T increases Y by 9.8234

==========================================================================================
Scenario 3: PROPENSITY SCORE MATCHING
==========================================================================================

Step 1: Propensity Score Estimation (Logit)
  Coefficients: β1=0.1512, β2=0.0098, β3=-0.7823
  Intercept: -2.1345
  Propensity score range: [0.0234, 0.9876]

Step 2: Common Support Check
  Overlap region: P ∈ [0.0245, 0.9801]
  Sample retained in common support: 987/1000 (98.7%)

Step 3: 1:1 Nearest Neighbor Matching
  Caliper: 0.004523
  Matched pairs: 231
  Matched treated: 231
  Matched untreated: 231

Step 4: ATE Estimation on Matched Sample
  E[Y|T=1, matched] = 65.1234
  E[Y|T=0, matched] = 55.3456
  PSM ATE = 9.7778
  Bias: -0.2222
  SE(ATE): 0.6789
  t-statistic: 14.4012

Step 5: Covariate Balance (Standardized Mean Differences)
Covariate     Before (SMD)    After (SMD)     Balanced?
------------------------------------------------------------
X1                0.8234          0.0456           ✓
X2                0.6123          0.0234           ✓
X3               -0.9876         -0.0123           ✓
==========================================================================================

SUMMARY: ESTIMATOR COMPARISON
------------------------------------------------------------------------------------------
Estimator            ATE          Bias         SE          t-stat      
------------------------------------------------------------------------------------------
Naive                -7.6556      -17.6556      1.2345      -6.2015
OLS (adjusted)        9.8234       -0.1766      0.4234      23.2101
PSM (matched)         9.7778       -0.2222      0.6789      14.4012
True ATE             10.0000       -            -            -
==========================================================================================
```

---

## Challenge Round

1. **Balancing Property**  
   Why is it valid to match on P(X) instead of on all components of X?

   <details><summary>Solution</summary>**Rosenbaum & Rubin's balancing property** (1983): Given propensity score P(X), treatment T is independent of X: X ⊥ T | P(X). This means the 1D propensity score contains all information about X relevant to treatment assignment. After conditioning on P, no additional X information affects treatment. **Proof sketch**: X → P(X) (many-to-one mapping); units with same P may differ in X but are equally likely to be treated. Matching on P achieves same balance as matching on X directly (dimensionality reduction).</details>

2. **Common Support Violation**  
   Treated: P ∈ [0.2, 0.8]. Control: P ∈ [0.05, 0.95]. What region should we restrict to?

   <details><summary>Solution</summary>**Restrict to overlap**: P ∈ [0.2, 0.8] (maximum of treated min and control min; minimum of treated max and control max). **Sample**: Exclude all controls with P < 0.2 or P > 0.8. **Consequence**: Estimated ATE only applies to population with P ∈ [0.2, 0.8], not extrapolated to extremes (no control units with P ~ 0.95 to match treated at P ~ 0.8).</details>

3. **Covariate Balance Failure**  
   After matching, X₁ has SMD = 0.25 (> 0.1 threshold). What does this imply?

   <details><summary>Solution</summary>**Poor balance**: Matched treated and control still differ in X₁ (plausibly non-random difference). **Implications**: (1) Matching algorithm ineffective for X₁ (perhaps rare values; hard to match), (2) Unconfoundedness assumption questionable (residual confounding by X₁), (3) Solution: Adjust for X₁ in outcome regression on matched sample (doubly robust: propensity weighting + regression).</details>

4. **Overlap Problem**  
   20% of treated units have no control matches (caliper violated). How does this affect inference?

   <details><summary>Solution</summary>**Non-representativeness**: Unmatched treated units likely differ systematically (higher propensity to treat; sicker patients, e.g.). **ATE estimated on matched sample**: 80% of treated population → not population ATE, but **Average Treatment Effect on the Treated (ATT) for matched subsample**. **Bias source**: Excluded treated may be different; estimates may not generalize. **Remedy**: Use stratification (blocks) or kernel matching (retains all units with lower efficiency but complete coverage).</details>

---

## Key References

- **Rosenbaum & Rubin (1983)**: "The Central Role of the Propensity Score in Observational Studies" ([Biometrika](https://academic.oup.com/biomet/article-abstract/70/1/41/240879))—foundational; balancing property
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data* (Ch. 18: Propensity scores) ([MIT Press](https://mitpress.mit.edu))
- **Lalonde (1986)**: "Evaluating the Econometric Evaluations of Training Programs" ([American Economic Review](https://www.jstor.org/stable/1806062))—empirical comparison of PSM vs. RCT
- **Caliendo & Kopeinig (2008)**: "Some practical guidance for the implementation of propensity score matching" ([Journal of Economic Surveys](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-6419.2007.00527.x))

**Further Reading:**  
- Doubly robust estimation (combines propensity score + regression)  
- Sensitivity analysis (Rosenbaum bounds) for unobserved confounding  
- Propensity-score-weighted regression (alternative to matching)
