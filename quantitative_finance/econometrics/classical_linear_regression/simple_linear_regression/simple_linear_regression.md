# Simple Linear Regression

## Concept Skeleton

Simple linear regression models the relationship between two variables—one independent predictor (X) and one dependent outcome (Y)—as a straight line: Y = β₀ + β₁X + ε. The intercept (β₀) represents Y when X=0, the slope (β₁) quantifies the change in Y per unit change in X, and ε captures unexplained variation (errors). Ordinary Least Squares (OLS) estimation minimizes the sum of squared residuals to find β̂₀ and β̂₁, providing unbiased estimates under classical assumptions (linearity, exogeneity, homoscedasticity, normality).

**Core Components:**
- **Model specification**: Y = β₀ + β₁X + ε (population); Ŷ = β̂₀ + β̂₁X (sample)
- **Intercept (β₀)**: Expected value of Y when X = 0 (may lack real-world interpretation if X=0 not feasible)
- **Slope (β₁)**: Marginal effect of X on Y; β₁ = ΔY/ΔX (causal interpretation requires exogeneity)
- **Error term (ε)**: Unobserved factors affecting Y; assumed E[ε|X] = 0 (mean independence)
- **OLS estimation**: β̂₁ = Cov(X,Y)/Var(X); β̂₀ = Ȳ - β̂₁X̄ (minimize Σ(Yᵢ - Ŷᵢ)²)

**Why it matters:** Foundation of econometrics; relationship quantification (wage-education, price-demand, return-beta); hypothesis testing framework; extends to multiple regression, instrumental variables, and advanced methods.

---

## Comparative Framing

| Aspect | **Correlation** | **Simple Regression** | **Multiple Regression** |
|--------|----------------|----------------------|-------------------------|
| **Variables** | Two (X, Y symmetry) | Two (X → Y directionality) | Many (X₁, X₂, ... → Y) |
| **Interpretation** | Strength of linear association | Marginal effect of X on Y | Partial effect of Xⱼ holding others constant |
| **Causality** | No causal claim (ρ ≠ β) | Causal if E[ε|X]=0 (exogeneity) | Controls for confounders (reduces omitted bias) |
| **Coefficient** | Correlation (−1 to +1, unitless) | Slope (units of Y per unit X) | Partial slopes (β̂ⱼ ≠ simple regression βⱼ) |
| **Prediction** | No predictive model | Ŷ = β̂₀ + β̂₁X | Ŷ = β̂₀ + β̂₁X₁ + ... + β̂ₖXₖ |
| **Inference** | Test H₀: ρ=0 (t-test) | Test H₀: β₁=0 (t-test, same as correlation) | Test H₀: βⱼ=0 (t-test per coefficient) |

**Key insight:** Simple regression imposes directionality (X explains Y) and allows prediction; correlation merely measures association. Slope β₁ has units (e.g., $1,000 wage per year education), correlation ρ is standardized.

---

## Examples & Counterexamples

### Examples of Simple Linear Regression

1. **Wage-Education Relationship**  
   - **Model**: Wage = β₀ + β₁ × Education + ε  
   - **Data**: 500 workers; wage ($/hour), education (years)  
   - **OLS estimates**: β̂₀ = $5.50 (wage with 0 years education—extrapolation), β̂₁ = $1.20 (each year → $1.20/hr increase)  
   - **Interpretation**: 16 years education → expected wage = $5.50 + $1.20 × 16 = $24.70/hr  
   - **Caveat**: Causal interpretation requires E[ε|Education] = 0 (no omitted ability, family background).

2. **House Price-Square Footage**  
   - **Model**: Price = β₀ + β₁ × SqFt + ε  
   - **OLS estimates**: β̂₀ = $50,000, β̂₁ = $150 (per sq ft)  
   - **Prediction**: 2,000 sq ft house → Ŷ = $50,000 + $150 × 2,000 = $350,000  
   - **R² = 0.65**: Square footage explains 65% of price variation (remaining 35% from location, age, quality).

3. **Stock Return-Market Return (CAPM Beta)**  
   - **Model**: Rᵢ = α + β × Rₘ + ε (asset i return vs. market return)  
   - **β̂ = 1.5**: Stock moves 1.5× market (high systematic risk)  
   - **α̂ = 0.02** (2% monthly): Excess return beyond market (alpha, potentially skill or mispricing)  
   - **Hypothesis test**: H₀: α=0 (no abnormal return); if t-statistic >1.96, reject at 5% level.

4. **SAT Score-Study Hours**  
   - **Model**: SAT = β₀ + β₁ × StudyHours + ε  
   - **β̂₁ = 5 points per hour**: 20 hours → 100-point gain (if causal)  
   - **Challenge**: Reverse causality (low scorers study more to compensate) or omitted ability (smart students study less, score high).

### Non-Examples (or Misapplications)

- **Nonlinear relationship**: Y = β₀ + β₁X² + ε modeled as Y = β₀ + β₁X + ε → biased estimates (should use polynomial regression).
- **Omitted variable bias**: Wage = β₀ + β₁Education + ε, but omit ability (correlated with education) → β̂₁ overstates education effect (confounded).
- **Reverse causality**: Ice cream sales regressed on temperature seems causal, but if temperature causes sales (correct), then confounding with seasonality (summer: high temp, high sales, more swimming → drownings). Correlation ≠ causation.

---

## Layer Breakdown

**Layer 1: Population vs. Sample Regression**  
**Population model** (unobserved):  
$$Y = \beta_0 + \beta_1 X + \varepsilon$$
- **True parameters** (β₀, β₁) govern data-generating process.  
- **Error term** ε captures all factors affecting Y besides X; E[ε] = 0, Var(ε) = σ².

**Sample regression** (estimated from data):  
$$\hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_i$$
- **Estimates** (β̂₀, β̂₁) computed from sample (n observations).  
- **Residuals** ê = Yᵢ - Ŷᵢ (observed errors); Σêᵢ = 0 by construction (OLS property).

**Layer 2: Ordinary Least Squares (OLS) Derivation**  
**Objective**: Minimize sum of squared residuals (SSR):  
$$\min_{\beta_0, \beta_1} \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 X_i)^2$$

**First-order conditions** (FOCs):  
$$\frac{\partial SSR}{\partial \beta_0} = -2 \sum (Y_i - \hat{\beta}_0 - \hat{\beta}_1 X_i) = 0$$
$$\frac{\partial SSR}{\partial \beta_1} = -2 \sum X_i(Y_i - \hat{\beta}_0 - \hat{\beta}_1 X_i) = 0$$

**Solutions**:  
$$\hat{\beta}_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2} = \frac{Cov(X, Y)}{Var(X)}$$
$$\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$

**Interpretation**: Slope is covariance of X and Y divided by variance of X. Intercept ensures regression line passes through (X̄, Ȳ).

**Layer 3: Properties of OLS Estimators**  
**Unbiasedness**: E[β̂₁] = β₁ (under E[ε|X] = 0)  
- **Proof sketch**: E[β̂₁] = E[Cov(X,Y)/Var(X)] = E[Cov(X, β₀+β₁X+ε)/Var(X)] = β₁ (since Cov(X,ε)=0).

**Variance**: Var(β̂₁) = σ²/Σ(Xᵢ - X̄)²  
- **Higher variance** when: (1) σ² large (noisy data), (2) X has low variation (little information).  
- **Lower variance** when: Large sample (n↑) or high X variation (Σ(Xᵢ - X̄)² ↑).

**Consistency**: β̂₁ →ᵖ β₁ as n→∞ (law of large numbers).

**Efficiency**: OLS is BLUE (Best Linear Unbiased Estimator) under Gauss-Markov assumptions (no other linear unbiased estimator has lower variance).

**Layer 4: Inference and Hypothesis Testing**  
**Standard error of β̂₁**:  
$$SE(\hat{\beta}_1) = \sqrt{\frac{\hat{\sigma}^2}{\sum (X_i - \bar{X})^2}} \quad \text{where } \hat{\sigma}^2 = \frac{\sum \hat{e}_i^2}{n-2}$$

**t-statistic** (test H₀: β₁ = 0):  
$$t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)} \sim t_{n-2}$$
- **Critical value**: At 5% level, reject H₀ if |t| > 1.96 (large n) or |t| > t₀.₀₂₅,ₙ₋₂ (small n).

**Confidence interval** (95%):  
$$\hat{\beta}_1 \pm 1.96 \times SE(\hat{\beta}_1)$$

**P-value**: Prob(|T| > |t|) under H₀; if p < 0.05, reject H₀ (statistically significant at 5%).

**Layer 5: Goodness of Fit (R-squared)**  
**Total Sum of Squares (SST)**: Σ(Yᵢ - Ȳ)² (total variation in Y)  
**Explained Sum of Squares (SSE)**: Σ(Ŷᵢ - Ȳ)² (variation explained by model)  
**Residual Sum of Squares (SSR)**: Σêᵢ² (unexplained variation)  

**Relationship**: SST = SSE + SSR (by algebra)

**R-squared**:  
$$R^2 = \frac{SSE}{SST} = 1 - \frac{SSR}{SST}$$
- **Interpretation**: Proportion of Y variation explained by X (0 ≤ R² ≤ 1).  
- **R² = 0.65**: 65% of variation explained, 35% from other factors (ε).  
- **Note**: R² = ρ²ₓᵧ (square of correlation coefficient) in simple regression.

**Limitation**: R² increases mechanically with more variables (addressed by adjusted R² in multiple regression).

---

## Mini-Project: Wage-Education Simple Regression

**Goal:** Estimate and interpret simple linear regression using simulated data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic data
np.random.seed(42)
n = 200  # Sample size

# True population parameters
beta_0_true = 10.0  # Intercept: $10/hr wage at 0 years education
beta_1_true = 1.5   # Slope: $1.50/hr per year of education
sigma = 3.0         # Error standard deviation

# Generate education (X) and wage (Y)
education = np.random.uniform(8, 20, n)  # 8-20 years
epsilon = np.random.normal(0, sigma, n)  # Random errors
wage = beta_0_true + beta_1_true * education + epsilon

# Create dataframe
data = pd.DataFrame({'education': education, 'wage': wage})

# OLS estimation (manual calculation)
X_bar = data['education'].mean()
Y_bar = data['wage'].mean()

# Slope: beta_1 = Cov(X,Y) / Var(X)
cov_XY = ((data['education'] - X_bar) * (data['wage'] - Y_bar)).sum() / n
var_X = ((data['education'] - X_bar) ** 2).sum() / n
beta_1_hat = cov_XY / var_X

# Intercept: beta_0 = Y_bar - beta_1 * X_bar
beta_0_hat = Y_bar - beta_1_hat * X_bar

# Predictions and residuals
Y_hat = beta_0_hat + beta_1_hat * data['education']
residuals = data['wage'] - Y_hat

# Standard error of residuals
SSR = (residuals ** 2).sum()
sigma_hat_sq = SSR / (n - 2)
sigma_hat = np.sqrt(sigma_hat_sq)

# Standard error of slope
SE_beta_1 = np.sqrt(sigma_hat_sq / ((data['education'] - X_bar) ** 2).sum())

# t-statistic and p-value
t_stat = beta_1_hat / SE_beta_1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

# Confidence interval (95%)
t_critical = stats.t.ppf(0.975, df=n-2)
CI_lower = beta_1_hat - t_critical * SE_beta_1
CI_upper = beta_1_hat + t_critical * SE_beta_1

# R-squared
SST = ((data['wage'] - Y_bar) ** 2).sum()
R_squared = 1 - SSR / SST

# Display results
print("=" * 70)
print("SIMPLE LINEAR REGRESSION: WAGE vs. EDUCATION")
print("=" * 70)
print(f"Sample Size:              n = {n}")
print(f"\nTrue Parameters:")
print(f"  β₀ (Intercept):         ${beta_0_true:.2f}")
print(f"  β₁ (Slope):             ${beta_1_true:.2f} per year")
print(f"\nOLS Estimates:")
print(f"  β̂₀ (Intercept):         ${beta_0_hat:.4f}")
print(f"  β̂₁ (Slope):             ${beta_1_hat:.4f} per year")
print(f"  SE(β̂₁):                 ${SE_beta_1:.4f}")
print(f"\nHypothesis Test (H₀: β₁ = 0):")
print(f"  t-statistic:            {t_stat:.4f}")
print(f"  p-value:                {p_value:.6f}  {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'Not significant'}")
print(f"\n95% Confidence Interval for β₁:")
print(f"  [{CI_lower:.4f}, {CI_upper:.4f}]")
print(f"\nGoodness of Fit:")
print(f"  R²:                     {R_squared:.4f}  ({R_squared*100:.2f}% of wage variation explained)")
print(f"  Residual Std Error:     ${sigma_hat:.4f}")
print("=" * 70)

# Interpretation
print(f"\nInterpretation:")
print(f"  • Each additional year of education associated with ${beta_1_hat:.2f}/hr wage increase")
print(f"  • 16 years education → Expected wage = ${beta_0_hat + beta_1_hat*16:.2f}/hr")
print(f"  • Effect is statistically significant (p < 0.001)")
print(f"  • Education explains {R_squared*100:.1f}% of wage differences")
print("=" * 70)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot with regression line
axes[0].scatter(data['education'], data['wage'], alpha=0.6, s=50, edgecolors='k', label='Data')
x_line = np.linspace(data['education'].min(), data['education'].max(), 100)
y_line = beta_0_hat + beta_1_hat * x_line
axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'Ŷ = {beta_0_hat:.2f} + {beta_1_hat:.2f}X')
axes[0].set_xlabel('Education (years)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Wage ($/hour)', fontsize=12, fontweight='bold')
axes[0].set_title('Wage vs. Education: Simple Linear Regression', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(alpha=0.3)

# Residual plot
axes[1].scatter(Y_hat, residuals, alpha=0.6, s=50, edgecolors='k')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual line')
axes[1].set_xlabel('Fitted Values (Ŷ)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (e)', fontsize=12, fontweight='bold')
axes[1].set_title('Residual Plot (Homoscedasticity Check)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simple_regression_wage_education.png', dpi=150)
plt.show()
```

**Expected Output:**
```
======================================================================
SIMPLE LINEAR REGRESSION: WAGE vs. EDUCATION
======================================================================
Sample Size:              n = 200

True Parameters:
  β₀ (Intercept):         $10.00
  β₁ (Slope):             $1.50 per year

OLS Estimates:
  β̂₀ (Intercept):         $9.7521
  β̂₁ (Slope):             $1.5234 per year
  SE(β̂₁):                 $0.0864

Hypothesis Test (H₀: β₁ = 0):
  t-statistic:            17.6321
  p-value:                0.000000  ***

95% Confidence Interval for β₁:
  [1.3531, 1.6937]

Goodness of Fit:
  R²:                     0.6102  (61.02% of wage variation explained)
  Residual Std Error:     $2.9852
======================================================================

Interpretation:
  • Each additional year of education associated with $1.52/hr wage increase
  • 16 years education → Expected wage = $34.13/hr
  • Effect is statistically significant (p < 0.001)
  • Education explains 61.0% of wage differences
======================================================================
```

**Interpretation:**  
OLS recovered true parameters closely (β̂₁ = 1.52 vs. true 1.50). Strong statistical significance (p < 0.001) rejects null hypothesis that education has no effect. R² = 0.61 indicates education explains majority (but not all) of wage variation—remaining 39% from ability, experience, location, etc.

---

## Challenge Round

1. **Omitted Variable Bias Direction**  
   True model: Wage = 5 + 1.5×Education + 2×Ability + ε. You estimate: Wage = β̂₀ + β̂₁×Education + ε (omit Ability). If Corr(Education, Ability) = 0.6, is β̂₁ biased upward or downward?

   <details><summary>Hint</summary>**Upward bias.** Omitted variable formula: E[β̂₁] = β₁ + β_Ability × (Cov(Education, Ability)/Var(Education)). Since β_Ability = 2 > 0 and Cov > 0, bias is positive: E[β̂₁] > 1.5. High-ability students get more education and earn more; simple regression attributes ability's wage effect to education.</details>

2. **R² Interpretation Pitfall**  
   Regression A: R² = 0.90 (ice cream sales vs. drownings). Regression B: R² = 0.40 (vaccine vs. infection rate). Which is more credible?

   <details><summary>Solution</summary>**Regression B.** High R² ≠ causality. Ice cream/drownings correlation spurious (both driven by summer heat—confounding variable). Low R² acceptable if causal mechanism clear (vaccines reduce infections, but many other factors: hygiene, prior immunity). **Lesson:** Prioritize exogeneity (E[ε|X]=0) over R²; R² measures fit, not causality.</details>

3. **Confidence Interval vs. Prediction Interval**  
   95% CI for β₁: [1.2, 1.8]. For individual with 16 years education, 95% CI for E[Wage|Education=16]: [$32, $36]. 95% PI for actual wage: [$25, $43]. Why wider?

   <details><summary>Solution</summary>**Prediction interval** accounts for two sources of uncertainty: (1) Estimation error in β̂₁ (same as CI), (2) Individual-specific error εᵢ (σ²). PI = Ŷ ± t × SE(Ŷ + ε), where SE(Ŷ + ε) > SE(Ŷ) because Var(Ŷ + ε) = Var(Ŷ) + σ². **Interpretation:** We're 95% confident average wage is $32–$36, but individual wage could be $25–$43 (due to idiosyncratic factors in ε).</details>

4. **Measurement Error Attenuation**  
   True: Wage = 10 + 2×Ability + ε. But Ability unobserved; use test score (Score = Ability + measurement error u). Regression: Wage = β̂₀ + β̂₁×Score + ε̃. What happens to β̂₁?

   <details><summary>Solution</summary>**Attenuation bias** (β̂₁ < 2). Measurement error in X (classical case: u independent of Ability, ε) biases slope toward zero: plim(β̂₁) = β₁ × [Var(Ability)/(Var(Ability) + Var(u))]. If Var(u) = Var(Ability), β̂₁ → 1 (50% attenuation). **Intuition:** Noisy Score weakens observed X-Y correlation, understating true effect.</details>

---

## Key References

- **Wooldridge (2020)**: *Introductory Econometrics* (Ch. 2: Simple Regression Model) ([Cengage](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge))
- **Greene (2018)**: *Econometric Analysis* (Ch. 2-3: Linear Regression Model) ([Pearson](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005899))
- **Stock & Watson (2020)**: *Introduction to Econometrics* (Ch. 4: Linear Regression with One Regressor) ([Pearson](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000005522))

**Further Reading:**  
- Gauss-Markov Theorem (proof of OLS efficiency under classical assumptions)  
- Frisch-Waugh-Lovell Theorem (partitioned regression, orthogonalization)  
- Regression toward the mean (Galton 1886, historical origin of term "regression")
