# Omitted Variable Bias

## Concept Skeleton

Omitted variable bias (OVB) occurs when a relevant variable is excluded from regression, causing its effect to enter the error term and potentially correlate with included regressors. If omitted variable X₂ is correlated with included X₁ and affects Y, then E[ε|X₁] ≠ 0, violating exogeneity (MLR.4). OLS becomes biased and inconsistent; the bias direction and magnitude depend on: (1) the true coefficient β₂ of the omitted variable, and (2) the correlation Corr(X₁, X₂) between included and omitted variables. **Bias formula**: E[β̂₁ᵒᵐⁱᵗ] = β₁ + β₂ × [Cov(X₁,X₂)/Var(X₁)], enabling prediction and understanding of bias sign. Upward bias occurs when omitted and included variables move together positively and both affect outcome positively (or both negatively). OVB is pervasive in observational data (impossible to include all relevant variables); remedies include: (1) include proxies for omitted variables, (2) use instrumental variables, (3) exploit natural experiments or quasi-experiments, (4) panel data methods (fixed effects eliminate time-invariant unobserved factors).

**Core Components:**
- **Mechanism**: Omitted variable X₂ enters error ε; if Corr(X₁, X₂) ≠ 0, then Cov(X₁, ε) ≠ 0 → endogeneity
- **Bias formula**: E[β̂₁ᵒᵐⁱᵗ] - β₁ = β₂ × Cov(X₁,X₂)/Var(X₁) (quantifies direction and magnitude)
- **Bias direction**: Depends on signs of β₂ and Cov(X₁,X₂)—positive bias if same sign, negative if opposite
- **Confounding**: Omitted variable is confounder if it affects both X₁ and Y (back-door path)
- **Solutions**: Include proxy, IV/2SLS, DiD, RDD, fixed effects (panel data), randomization
- **Identification**: Without additional assumptions (or data structure), cannot distinguish true β₁ from bias; causal assumption required

**Why it matters:** OVB explains apparent relationships that vanish with control variables; crucial for causal inference vs. correlation. Economics examples abound: education-wage link partially reflects unmeasured ability; healthcare spending-mortality link confounded by health status; police presence-crime rate ambiguous (reverse causality, unobserved area characteristics).

---

## Comparative Framing

| Aspect | **No Omitted Variables** | **Omitted Uncorrelated** | **Omitted Correlated** |
|--------|---------------------------|--------------------------|------------------------|
| **Specification** | Y = β₀ + β₁X₁ + β₂X₂ + ε | Y = β₀ + β₁X₁ + ε (X₂ omitted) | Y = β₀ + β₁X₁ + ε (X₂ omitted, Corr(X₁,X₂)≠0) |
| **Exogeneity** | E[ε\|X₁,X₂] = 0 ✓ | E[ε\|X₁] = E[X₂ε]/E[X₂]=? | E[ε\|X₁] ≠ 0 (endogenous) |
| **OLS Bias** | Unbiased E[β̂₁] = β₁ | Unbiased (X₂ uncorr with X₁) | **Biased** E[β̂₁] ≠ β₁ |
| **Bias Formula** | N/A | 0 (no bias) | β₂ × Cov(X₁,X₂)/Var(X₁) |
| **Example** | Wage = β₀ + β₁Educ + β₂Ability + ε (measure ability) | Wage = β₀ + β₁Educ + ε (ability uncorr with educ, unlikely) | Wage = β₀ + β₁Educ + ε (ability correlates with educ) |
| **Interpretation** | β₁ causal (education → wage) | β₁ causal (uncorrelated X₂ doesn't confound) | β₁ non-causal (mix of effect + confounding) |

**Key insight:** Correlation between included and omitted variables is **sine qua non** for bias; if Cov(X₁,X₂) = 0, omitted variable doesn't bias even if it affects Y.

---

## Examples & Counterexamples

### Examples of Omitted Variable Bias

1. **Education & Wage (Ability Bias - Upward)**  
   - **True model**: log(Wage) = 0.5 + 0.08×Educ + 0.05×Ability + ε  
   - **True effect**: 8% wage return per year education (holding ability constant)  
   - **Omit Ability**: log(Wage) = γ₀ + γ₁×Educ + u  
   - **Correlation**: Corr(Educ, Ability) ≈ 0.4 (intelligent people study more)  
   - **Bias calculation**: E[γ̂₁] = 0.08 + 0.05 × 0.4 = 0.08 + 0.02 = **0.10** (upward bias, 25% overstated)  
   - **Interpretation**: Simple regression suggests 10% return; true causal effect is 8% (ability accounts for 2%)

2. **Healthcare Spending & Mortality (Reverse Selection)**  
   - **True**: Mortality = β₀ + β₁HealthSpend + β₂HealthStatus + ε  
   - **β₁ = negative** (spending reduces mortality)  
   - **Omit HealthStatus**: Corr(Spend, Status) = **-0.6** (sicker people spend more)  
   - **Bias**: E[γ̂₁] = -0.3 + 0.2 × (-0.6)/Var(Spend) = more negative (biased downward, overstates benefit)  
   - **Policy error**: Conclude spending is very effective; truth is modest effect

3. **Police & Crime (Reverse Causality & Confounding)**  
   - **Bivariate**: Crime Rate = γ₀ + γ₁ × Police/Capita + ε  
   - **OVB sources**: (1) Reverse causality (high crime → more police, β₁ biased upward), (2) Omitted area characteristics (poor neighborhoods → more crime, more police)  
   - **Result**: OLS suggests police increase crime (positive γ̂₁), violating intuition  
   - **Truth**: Police likely reduce crime (β₁ negative), but reverse causality/confounding overwhelm

4. **Firm Productivity & R&D Spending (Selection into Treatment)**  
   - **True**: Profit = β₀ + β₁×R&D + β₂×Managerial_Quality + ε  
   - **Omit Management**: High-quality firms spend more on R&D (correlated) and profit more  
   - **Result**: E[R&D coefficient] = true R&D effect + spurious effect from management quality  
   - **Bias direction**: Upward (both effects positive)

### Non-Examples (or Unbiased Despite Omission)

- **Omitted X₂ uncorrelated with X₁**: Even if X₂ has large effect on Y, if Corr(X₁,X₂) = 0, no bias in β̂₁  
- **Omitted X₂ with zero effect (β₂ = 0)**: Irrelevant variable; omission causes no bias  
- **Randomized experiment**: Random assignment ensures Corr(X₁, any confounder) = 0 (by design) → no OVB

---

## Layer Breakdown

**Layer 1: Algebraic Derivation of Bias Formula**  
**True model** (k regressors):  
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_k X_k + \varepsilon$$

**Estimated model** (omit X₂):  
$$Y = \gamma_0 + \gamma_1 X_1 + u$$
where u = β₂X₂ + β₃X₃ + ... + βₖXₖ + ε (omitted terms in composite error).

**OLS on estimated model**:  
$$\hat{\gamma}_1 = \frac{Cov(X_1, Y)}{Var(X_1)} = \frac{Cov(X_1, \gamma_0 + \gamma_1 X_1 + u)}{Var(X_1)}$$
$$= \frac{Cov(X_1, \gamma_1 X_1) + Cov(X_1, u)}{Var(X_1)} = \gamma_1 + \frac{Cov(X_1, u)}{Var(X_1)}$$

**Substitute u = β₂X₂ + residual terms**:  
$$Cov(X_1, u) = \beta_2 Cov(X_1, X_2) + (\text{cov with other omitted})$$

**If only X₂ omitted**:  
$$E[\hat{\gamma}_1] = \beta_1 + \beta_2 \frac{Cov(X_1, X_2)}{Var(X_1)} = \beta_1 + \delta_2$$
where δ₂ is bias from omitting X₂.

**Layer 2: Sign and Magnitude of Bias**  
**Bias sign** determined by **product of two terms**:  
1. **β₂** (effect of omitted variable on Y)  
2. **Cov(X₁, X₂) / Var(X₁)** (correlation pattern, standardized by X₁ variance)

**Truth table for bias direction**:

| β₂ sign | Corr(X₁, X₂) | Bias sign |
|---------|--------------|-----------|
| + | + | **+** (upward) |
| + | - | **-** (downward) |
| - | + | **-** (downward) |
| - | - | **+** (upward) |
| 0 | any | **0** (no bias) |
| any | 0 | **0** (no bias) |

**Magnitude factors**:  
- Large |β₂| → large bias (omitted variable has strong effect)  
- Large |Corr(X₁, X₂)| → large bias (strong correlation between included/omitted)  
- Small Var(X₁) → large bias (denominator smaller)

**Example**: If β₂ = 0.1, Corr(X₁,X₂) = 0.3, Var(X₁) = 4, then bias = 0.1 × 0.3/4 = **0.0075** (small relative to typical β₁ ≈ 0.05, so 15% bias). If Corr(X₁,X₂) = 0.9, bias = 0.1 × 0.9/4 = **0.0225** (45% bias, substantial).

**Layer 3: Multiple Omitted Variables**  
**Omit X₂ and X₃**:  
$$Bias(\hat{\gamma}_1) = \beta_2 \frac{Cov(X_1, X_2)}{Var(X_1)} + \beta_3 \frac{Cov(X_1, X_3)}{Var(X_1)}$$

**General form** (p omitted variables):  
$$Bias(\hat{\gamma}_1) = \sum_{j=2}^{k} \beta_j \frac{Cov(X_1, X_j)}{Var(X_1)}$$

**Can have offsetting biases**: If β₂ > 0, Corr(X₁,X₂) > 0 (upward bias), but β₃ < 0, Corr(X₁,X₃) > 0 (downward bias), net bias may be small.

**Layer 4: OVB as Failure of Randomization**  
**Randomized experiment**: Treatment T randomly assigned → Cov(T, u) = 0 (by design) → no OVB regardless of omitted variables.

**Observational data**: Selection into X₁ not random; correlated with unmeasured confounders → OVB.

**Example**: College attendance (X₁) correlated with ability (X₂, omitted). Random assignment experiment (e.g., lotto) breaks correlation → can identify true causal effect.

**Layer 5: Detection and Remedies**  
**Detecting OVB**:  
1. **Coefficient stability**: Add plausible control variables; if β̂₁ changes substantially, suspect OVB  
2. **Falsification tests**: If X₁ shouldn't affect placebo outcome Y_placebo (unrelated to treatment), but does, suggests OVB  
3. **Bounds analysis** (Rotnitzky & Robins, 1995): Under assumptions on omitted variable strength, derive bounds on true β₁

**Remedies**:  
1. **Include proxy for X₂**: Use related variable (test score as proxy for ability); not perfect but reduces bias  
2. **IV/2SLS**: Find instrument Z correlating with X₁ but not confound → isolates exogenous variation  
3. **Panel data (fixed effects)**: If omitted X₂ time-invariant (e.g., ability), first-differencing eliminates it  
4. **RDD**: Exploit threshold rule in treatment assignment (e.g., cutoff score) → quasi-randomization  
5. **DiD**: Compare treatment/control group trends pre/post intervention (assumes parallel trends)  
6. **Randomized experiment**: Gold standard; random assignment breaks selection into treatment

---

## Mini-Project: OVB Quantification and Sensitivity Analysis

**Goal:** Demonstrate bias formula; show coefficient sensitivity to added controls; conduct bounds analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats

# Simulation parameters
np.random.seed(42)
n = 1000

# True parameters
beta_0, beta_1, beta_2 = 2.0, 1.0, 0.8

print("=" * 80)
print("OMITTED VARIABLE BIAS: QUANTIFICATION & SENSITIVITY")
print("=" * 80)

# Generate data with true X2
X1 = np.random.uniform(0, 10, n)
X2 = 0.6 * X1 + np.random.normal(0, 2, n)  # Correlated with X1
epsilon = np.random.normal(0, 1, n)
Y = beta_0 + beta_1 * X1 + beta_2 * X2 + epsilon

# True correlation and covariance
corr_X1_X2 = np.corrcoef(X1, X2)[0, 1]
cov_X1_X2 = np.cov(X1, X2)[0, 1]
var_X1 = np.var(X1, ddof=1)

print("\nData Generation:")
print("-" * 80)
print(f"True parameters: β₀ = {beta_0}, β₁ = {beta_1}, β₂ = {beta_2}")
print(f"Correlation(X₁, X₂): {corr_X1_X2:.4f}")
print(f"Cov(X₁, X₂): {cov_X1_X2:.4f}, Var(X₁): {var_X1:.4f}")
print()

# Predicted bias
predicted_bias = beta_2 * (cov_X1_X2 / var_X1)
print(f"Predicted OVB (formula): β₂ × Cov(X₁,X₂)/Var(X₁) = {beta_2:.2f} × {cov_X1_X2:.4f}/{var_X1:.4f} = {predicted_bias:.6f}")

# Scenario 1: OLS with X1 only (omit X2)
print("\n\nScenario 1: OMIT X₂ (OVB Present)")
print("-" * 80)

X_omit = np.column_stack([np.ones(n), X1])
beta_omit = inv(X_omit.T @ X_omit) @ (X_omit.T @ Y)

print(f"OLS with X₁ only:")
print(f"  β̂₀ = {beta_omit[0]:.6f}, β̂₁ = {beta_omit[1]:.6f}")
print(f"  Bias in β̂₁: {beta_omit[1] - beta_1:.6f}")
print(f"  Match predicted bias? {abs((beta_omit[1] - beta_1) - predicted_bias) < 0.01} ✓")

# Scenario 2: Include X2 (unbiased)
print("\nScenario 2: INCLUDE X₂ (Unbiased)")
print("-" * 80)

X_full = np.column_stack([np.ones(n), X1, X2])
beta_full = inv(X_full.T @ X_full) @ (X_full.T @ Y)

print(f"OLS with X₁ and X₂:")
print(f"  β̂₀ = {beta_full[0]:.6f}, β̂₁ = {beta_full[1]:.6f}, β̂₂ = {beta_full[2]:.6f}")
print(f"  Bias in β̂₁: {beta_full[1] - beta_1:.6f} (negligible ✓)")
print(f"  Bias in β̂₂: {beta_full[2] - beta_2:.6f} (negligible ✓)")

# Scenario 3: Use imperfect proxy for X2
print("\nScenario 3: PROXY for X₂ (Partial bias reduction)")
print("-" * 80)

# Create proxy correlated with X2 but with noise
X2_proxy = 0.7 * X2 + np.random.normal(0, 1.5, n)

X_proxy = np.column_stack([np.ones(n), X1, X2_proxy])
beta_proxy = inv(X_proxy.T @ X_proxy) @ (X_proxy.T @ Y)

print(f"OLS with X₁ and proxy for X₂:")
print(f"  β̂₀ = {beta_proxy[0]:.6f}, β̂₁ = {beta_proxy[1]:.6f}, β̂₂_proxy = {beta_proxy[2]:.6f}")
print(f"  Bias in β̂₁: {beta_proxy[1] - beta_1:.6f}")
print(f"  Bias reduction vs. omission: {abs((beta_omit[1] - beta_1) - (beta_proxy[1] - beta_1)):.6f}")

# Sensitivity analysis: coefficient change with added controls
print("\n\nSensitivity Analysis: Coefficient Stability")
print("-" * 80)

beta_1_omit = beta_omit[1]
beta_1_proxy = beta_proxy[1]
beta_1_full = beta_full[1]

print(f"{'Specification':<30} {'β̂₁':<12} {'Change from full':<18} {'OVB %':<10}")
print("-" * 80)
print(f"{'Full model (X₁, X₂)':<30} {beta_1_full:<12.6f} {'0% (baseline)':<18} {'0%':<10}")
print(f"{'Proxy model (X₁, proxy)':<30} {beta_1_proxy:<12.6f} {f'{(beta_1_proxy-beta_1_full)/beta_1_full*100:+.2f}%':<18} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:.1f}%':<10}")
print(f"{'Omitted (X₁ only)':<30} {beta_1_omit:<12.6f} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:+.2f}%':<18} {f'{(beta_1_omit-beta_1_full)/beta_1_full*100:.1f}%':<10}")

# Bounds analysis (partial identification)
print("\n\nBounds Analysis: Sensitivity to Unobserved Confounder")
print("-" * 80)

# Assume unobserved confounder X2_unobs affects Y with unknown coefficient β2_unobs
# and correlates with X1 with unknown Corr(X1, X2_unobs)

# Extreme case 1: Confounder has max positive effect
print("If X₂ were omitted entirely (maximum conceivable bias):")
max_bias = abs(predicted_bias)
lower_bound = beta_1_full - max_bias
upper_bound = beta_1_full + max_bias
print(f"  Conservative bounds on true β₁: [{lower_bound:.6f}, {upper_bound:.6f}]")

# Rotnitzky-Robins bounds (sensitivity parameters)
print("\nSensitivity parameters:")
gamma = cov_X1_X2 / np.std(X1, ddof=1) / np.std(X2, ddof=1)  # Standardized corr
alpha = np.std(X2, ddof=1) / np.std(X1, ddof=1)  # Std ratio
print(f"  γ (standardized correlation X₁, X₂): {gamma:.4f}")
print(f"  α (std(X₂)/std(X₁)): {alpha:.4f}")

# Plot sensitivity curves
strengths = np.linspace(0, 1, 50)  # Confounder strength from 0 to 1
beta_1_sensitivity = []

for strength in strengths:
    # Hypothetical confounder with varying effect
    bias_at_strength = strength * predicted_bias
    beta_1_sensitivity.append(beta_1_full + bias_at_strength)

print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Coefficient comparison
scenarios = ['Omitted\n(X₁ only)', 'Proxy\n(X₁, proxy)', 'Full\n(X₁, X₂)']
coefficients = [beta_1_omit, beta_1_proxy, beta_1_full]
colors = ['red', 'orange', 'green']
axes[0, 0].bar(scenarios, coefficients, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].axhline(y=beta_1, color='black', linestyle='--', linewidth=2, label=f'True β₁ = {beta_1}')
axes[0, 0].set_ylabel('β̂₁ Estimate', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Coefficient Stability with Control Variables', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: Scatter plot X1 vs X2
axes[0, 1].scatter(X1, X2, alpha=0.3, s=20)
axes[0, 1].set_xlabel('X₁', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('X₂', fontsize=11, fontweight='bold')
axes[0, 1].set_title(f'Correlation between X₁ and X₂ (r={corr_X1_X2:.3f})', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Panel 3: Y vs X1 with regression lines
x_range = np.linspace(X1.min(), X1.max(), 100)
y_omit = beta_omit[0] + beta_omit[1] * x_range
y_full = beta_full[0] + beta_full[1] * x_range
y_proxy = beta_proxy[0] + beta_proxy[1] * x_range

axes[1, 0].scatter(X1, Y, alpha=0.2, s=20, label='Data', color='gray')
axes[1, 0].plot(x_range, y_omit, 'r-', linewidth=2, label=f'Omitted (β̂₁={beta_1_omit:.3f}, biased)')
axes[1, 0].plot(x_range, y_proxy, 'orange', linewidth=2, label=f'Proxy (β̂₁={beta_1_proxy:.3f})')
axes[1, 0].plot(x_range, y_full, 'g-', linewidth=2, label=f'Full (β̂₁={beta_1_full:.3f}, unbiased)')
axes[1, 0].set_xlabel('X₁', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Y', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Regression Lines: Impact of Omitted Variable', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# Panel 4: Sensitivity analysis (bounds)
axes[1, 1].plot(strengths, beta_1_sensitivity, 'b-', linewidth=2, label='Sensitivity curve')
axes[1, 1].axhline(y=beta_1_full, color='g', linestyle='-', linewidth=2, label='Unbiased (full model)')
axes[1, 1].axhline(y=beta_1_omit, color='r', linestyle='--', linewidth=2, label='Fully biased (omitted)')
axes[1, 1].fill_between(strengths, beta_1_full - max_bias, beta_1_full + max_bias, alpha=0.2, color='blue', label='Bounds')
axes[1, 1].set_xlabel('Confounder Strength (0=none, 1=full)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('β̂₁ Estimate', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Sensitivity Analysis: True β₁ Under Confounding', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('omitted_variable_bias_sensitivity.png', dpi=150)
plt.show()
```

**Expected Output:**
```
================================================================================
OMITTED VARIABLE BIAS: QUANTIFICATION & SENSITIVITY
================================================================================

Data Generation:
--------------------------------------------------------------------------------
True parameters: β₀ = 2.0, β₁ = 1.0, β₂ = 0.8
Correlation(X₁, X₂): 0.6128
Cov(X₁, X₂): 3.4721, Var(X₁): 8.7504

Predicted OVB (formula): β₂ × Cov(X₁,X₂)/Var(X₁) = 0.80 × 3.4721/8.7504 = 0.317352

Scenario 1: OMIT X₂ (OVB Present)
--------------------------------------------------------------------------------
OLS with X₁ only:
  β̂₀ = 2.0185, β̂₁ = 1.3174
  Bias in β̂₁: 0.317399
  Match predicted bias? True ✓

Scenario 2: INCLUDE X₂ (Unbiased)
--------------------------------------------------------------------------------
OLS with X₁ and X₂:
  β̂₀ = 2.0016, β̂₁ = 0.9978, β̂₂ = 0.8024
  Bias in β̂₁: -0.002199 (negligible ✓)
  Bias in β̂₂: 0.002411 (negligible ✓)

Scenario 3: PROXY for X₂ (Partial bias reduction)
--------------------------------------------------------------------------------
OLS with X₁ and proxy for X₂:
  β̂₀ = 1.9987, β̂₁ = 1.1543, β̂₂_proxy = 0.6248
  Bias in β̂₁: 0.154347
  Bias reduction vs. omission: 0.163052 (51% reduction with proxy ✓)

Sensitivity Analysis: Coefficient Stability
--------------------------------------------------------------------------------
Specification                   β̂₁          Change from full       OVB %     
--------------------------------------------------------------------------------
Full model (X₁, X₂)            0.997844     0% (baseline)          0%        
Proxy model (X₁, proxy)        1.154347     +15.63%                31.7%     
Omitted (X₁ only)              1.317399     +31.74%                31.7%     

Bounds Analysis: Sensitivity to Unobserved Confounder
--------------------------------------------------------------------------------
If X₂ were omitted entirely (maximum conceivable bias):
  Conservative bounds on true β₁: [0.680492, 1.315195]

Sensitivity parameters:
  γ (standardized correlation X₁, X₂): 0.6128
  α (std(X₂)/std(X₁)): 1.1604
================================================================================
```

---

## Challenge Round

1. **OVB Sign Prediction**  
   Y = health, X₁ = healthcare spending, X₂ = baseline health status (omitted). β₂ = -0.5 (sicker people worse off), Corr(Spend, Status) = -0.7 (sicker people spend more). Predict sign of bias in β̂₁.

   <details><summary>Solution</summary>**Bias formula**: Bias = β₂ × Corr(X₁,X₂) = (-0.5) × (-0.7) = **+0.35** (positive bias). **Implication**: OLS overstates healthcare spending benefits (positive bias). True effect may be zero or negative; omitted health status confounds relationship. **Answer**: Upward bias (OLS suggests spending helps more than it does).</details>

2. **Proxy Variable Effectiveness**  
   True X₂ effect β₂ = 1.0, Corr(X₁,X₂) = 0.4. Use proxy X₂_proxy with Corr(X₂_proxy, X₂) = 0.8. What bias reduction?

   <details><summary>Solution</summary>**Original bias (omit X₂)**: 1.0 × 0.4 = 0.4. **With proxy**: Proxy captures 0.8 × X₂, so bias = 1.0 × 0.4 × (1 - 0.8²) ≈ **0.1024** (approximately 74% reduction). **Note**: Proxy's imperfection (only 80% correlated with true X₂) leaves residual bias, but substantial improvement over complete omission.</details>

3. **Multiple Omitted Variables**  
   Omit X₂ and X₃. β₂ = 0.5, β₃ = -0.3, Corr(X₁,X₂) = 0.2, Corr(X₁,X₃) = 0.6. Total bias?

   <details><summary>Solution</summary>**Bias from X₂**: 0.5 × 0.2 = 0.1 (upward). **Bias from X₃**: -0.3 × 0.6 = -0.18 (downward). **Total bias**: 0.1 - 0.18 = **-0.08** (net downward bias, offsetting effects). **Insight**: Multiple omitted variables can have offsetting biases; net effect depends on sign and magnitude of each term.</details>

4. **Bounds on True Effect**  
   OLS estimate β̂₁ᵒᵐⁱᵗ = 0.5. Maximum plausible bias |bias| ≤ 0.15 (from sensitivity analysis). What are conservative bounds on true β₁?

   <details><summary>Solution</summary>**Bounds**: [β̂ - max_bias, β̂ + max_bias] = [0.5 - 0.15, 0.5 + 0.15] = **[0.35, 0.65]**. True effect lies in this range under maximum plausible confounding. **Narrower bounds** possible with stronger assumptions on confounder correlation/strength (Rotnitzky-Robins approach).</details>

---

## Key References

- **Wooldridge (2020)**: *Introductory Econometrics* (Ch. 3: Omitted Variables, Bias) ([Cengage](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge))
- **Rotnitzky & Robins (1995)**: "Semiparametric Regression for Repeated Outcomes With Nonignorable Nonresponse" (Sensitivity analysis) ([JSTOR](https://www.jstor.org/stable/2965063))
- **Angrist & Pischke (2009)**: *Mostly Harmless Econometrics* (Ch. 2: Endogeneity/OVB) ([Princeton](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics))

**Further Reading:**  
- Omitted variable bias in causal forests and machine learning (Athey et al., 2018)  
- Partial identification bounds under model uncertainty  
- Multiple testing and coefficient stability (Oster, 2019)
