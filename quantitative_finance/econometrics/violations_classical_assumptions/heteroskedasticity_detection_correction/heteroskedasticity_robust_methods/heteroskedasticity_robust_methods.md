# Heteroskedasticity: Detection, Diagnosis & Correction Methods

## I. Concept Skeleton

**Definition:** Heteroskedasticity refers to non-constant error variance across observations. In a regression model $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$, the variance of errors $\text{Var}(\epsilon_i | X_i) = \sigma_i^2$ varies with observation $i$ instead of being constant $\sigma^2$ (homoskedasticity assumption).

**Purpose:** Detect heteroskedasticity violations in regression models, understand sources of unequal variance, apply appropriate corrections (robust standard errors, weighted least squares, feasible GLS), and ensure valid statistical inference.

**Prerequisites:** OLS regression theory, variance estimation, statistical testing, matrix algebra, hypothesis testing.

---

## II. Comparative Framing

| **Aspect** | **OLS (Homoskedastic)** | **OLS + Robust SE** | **Weighted Least Squares (WLS)** | **Feasible GLS (FGLS)** | **Quantile Regression** |
|-----------|------------------------|-------------------|----------------------------------|------------------------|----------------------|
| **Assumption** | $\text{Var}(\epsilon_i\|X_i) = \sigma^2$ | Heteroskedasticity allowed | Known variance function | Estimate variance model | Robust to outliers |
| **β Coefficient** | Unbiased if homoskedastic | Unbiased (consistent) | Unbiased & efficient | Unbiased & efficient | Conditional median |
| **Efficiency** | BLUE if homoskedastic | Inefficient | Efficient if weights correct | Efficient asymptotically | Inefficient for mean |
| **Standard Errors** | Biased under hetero | Valid & consistent | Correct if model right | Valid for large N | Resistant to outliers |
| **Computational** | Simple | Simple (one-step) | Moderate (iterative) | Iterative | Computational burden |
| **Finite-Sample** | Valid only if homo | Poor if N<50 | Best if weights known | Depends on model fit | Robust always |
| **Example Application** | Balanced data | Large N surveys | Grouped experiments | Financial returns | Income inequality |

---

## III. Examples & Counterexamples

### Example 1: Increasing Variance (Classic Financial Case)
**Setup:**
- Model: Daily stock returns vs market returns
- $R_{\text{stock},t} = \alpha + \beta R_{\text{market},t} + \epsilon_t$
- Observation: High-volatility periods (crisis) show larger residual variance than calm periods
- Variance pattern: $\text{Var}(\epsilon_t | R_{\text{market},t}) \propto |R_{\text{market},t}|$ (increases with market stress)

**Problem:**
- OLS beta coefficient: $\hat{\beta} = 1.20$ (unbiased)
- OLS standard error: SE = 0.08 (biased, too narrow)
- 95% confidence interval: [1.04, 1.36] (too tight, true coverage ~85%)
- Robust SE: 0.15 (correct)
- 95% confidence interval: [0.90, 1.50] (correct coverage)

**Key Insight:** OLS underestimates uncertainty in high-volatility regimes. Robust SEs widen intervals, preventing false precision.

### Example 2: Grouped Heteroskedasticity (Labor Economics)
**Setup:**
- Wage regression by education level
- $\text{log(Wage)}_i = \beta_0 + \beta_1 \text{Education}_i + \epsilon_i$
- Groups: High school (σ² = 0.10), Bachelor (σ² = 0.25), Graduate (σ² = 0.40)
- Unequal variance by education level (skill dispersion increases with education)

**Analysis:**
- Sample sizes: n₁=500 (HS), n₂=300 (BA), n₃=100 (Grad)
- WLS weights: $w_i = 1/\sigma_i^2$ (inverse variance weights)
- Weight ratios: HS:BA:Grad = 1:0.40:0.25
- Interpretation: Graduate degree observations get lower weight (higher noise) in estimation
- Result: More efficient β̂ than OLS; lower standard error

**Edge Case Issue:** If true variance structure is $\sigma_i^2 \propto e^{2\gamma X_i}$ but we assume $\sigma_i^2 \propto X_i$, WLS is still better than OLS but not optimal.

### Example 3: Variance Increasing with Fitted Values (Regression Diagnostics)
**Setup:**
- Cross-sectional price model: $\text{Price}_i = \beta_0 + \beta_1 \text{Size}_i + \beta_2 \text{Location}_i + \epsilon_i$
- Scatter plot of residuals vs fitted values shows "cone shape" (variance expands as predicted price increases)
- Interpretation: Expensive properties (high fitted value) have more uncertain pricing

**Diagnostic Tests:**
- Breusch-Pagan test: $p < 0.05$ (reject homoskedasticity)
- White test: $p < 0.01$ (strong evidence of heteroskedasticity)
- Action: Apply HC robust standard errors or transform model

**Solutions:**
- Robust SE: Fast, asymptotically valid, valid inference immediately
- WLS with $\text{Var}(\epsilon_i) \propto \hat{Y}_i^2$: More efficient but requires variance model specification

---

## IV. Layer Breakdown

```
HETEROSKEDASTICITY FRAMEWORK

┌──────────────────────────────────────────────────────────┐
│         CLASSICAL LINEAR REGRESSION MODEL                │
│    yᵢ = β₀ + β₁x₁ᵢ + ... + βₖxₖᵢ + εᵢ                     │
│                                                          │
│    Homoskedasticity Assumption:                         │
│    Var(εᵢ|Xᵢ) = σ² for all i (CONSTANT)               │
│    ↓                                                     │
│    E[εᵢ²|Xᵢ] = σ²                                       │
└────────────────┬─────────────────────────────────────────┘
                 │
    ┌────────────▼─────────────┐
    │ HETEROSKEDASTICITY CHECK │
    │ (Var(εᵢ|Xᵢ) = σᵢ²)     │
    └────────┬─────────────────┘
             │
    ┌────────▼─────────────────────────────────────┐
    │ DIAGNOSTIC METHODS (Early Detection)         │
    │                                              │
    │ 1. GRAPHICAL:                               │
    │    ├─ Plot ε̂ᵢ vs X (look for pattern)      │
    │    ├─ Plot ε̂ᵢ² vs Ŷᵢ (fan/cone shape)     │
    │    ├─ Scale-location: √|ε̂ᵢ| vs Ŷᵢ         │
    │    └─ Pattern = heteroskedasticity          │
    │                                              │
    │ 2. STATISTICAL TESTS:                       │
    │    ├─ Breusch-Pagan:                       │
    │    │  └─ Regress ε̂ᵢ² on Xᵢ                │
    │    │  └─ H₀: coefficients = 0              │
    │    │  └─ LM statistic ~ χ²(k)             │
    │    ├─ White Test:                          │
    │    │  └─ Regress ε̂ᵢ² on Xᵢ, Xᵢ², Xᵢ·Xⱼ  │
    │    │  └─ More general (no functional form) │
    │    ├─ Goldfeld-Quandt:                      │
    │    │  └─ Split sample, compare variances   │
    │    │  └─ F-test for equal variances        │
    │    └─ Koenker Test:                        │
    │       └─ Robust version of BP               │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │ CONSEQUENCES IF NOT CORRECTED             │
    │                                            │
    │ ✓ β̂ still UNBIASED & CONSISTENT          │
    │   (point estimates valid)                 │
    │                                            │
    │ ✗ SE(β̂) BIASED:                         │
    │   └─ Usually too small (optimistic)      │
    │   └─ t-statistics inflated                │
    │   └─ p-values underestimated              │
    │   └─ Confidence intervals too narrow      │
    │                                            │
    │ ✗ β̂ NO LONGER BLUE:                      │
    │   └─ Not minimum variance                 │
    │   └─ Inefficient relative to WLS/FGLS    │
    │   └─ Precision loss if hetero severe     │
    │                                            │
    │ ✗ F-tests, Wald tests invalid            │
    └────────┬─────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────────┐
    │ CORRECTION METHODS                            │
    │                                               │
    │ 1. ROBUST STANDARD ERRORS (Easiest)         │
    │    ├─ White (HC0): Consistent estimator    │
    │    │  └─ Formula: Var(β̂) = (X'X)⁻¹ X'ΩX(X'X)⁻¹
    │    │  └─ Ω = diagonal matrix of ε̂ᵢ²      │
    │    ├─ HC1, HC2, HC3: Finite-sample adj.   │
    │    │  └─ HC1: Divide by (n-k)             │
    │    │  └─ HC3: More robust to leverage     │
    │    └─ Implementation: 1 line in Python    │
    │                                             │
    │ 2. WEIGHTED LEAST SQUARES (Efficient)      │
    │    ├─ If variance known: σᵢ² = f(Xᵢ)     │
    │    ├─ Weight observations: wᵢ = 1/σᵢ²   │
    │    ├─ Minimize: Σ wᵢ(yᵢ - ŷᵢ)²         │
    │    └─ Result: BLUE estimator              │
    │                                             │
    │ 3. FEASIBLE GLS (Two-stage)                │
    │    ├─ Stage 1: OLS estimate ε̂ᵢ          │
    │    ├─ Stage 2: Model variance            │
    │    │  └─ ln(ε̂ᵢ²) = α + γ·Xᵢ + vᵢ       │
    │    ├─ Extract ĥᵢ = exp(α̂ + γ̂·Xᵢ)      │
    │    ├─ Stage 3: WLS with wᵢ = 1/ĥᵢ      │
    │    └─ Efficient asymptotically            │
    │                                             │
    │ 4. TRANSFORMATION (Stabilize Variance)    │
    │    ├─ Log transformation: ln(y) not y    │
    │    ├─ Square root: √y stabilizes var    │
    │    └─ Combine with heteroskedastic       │
    │       error structure                     │
    └──────────────────────────────────────────┘

TYPES OF HETEROSKEDASTICITY:

├─ Proportional to X:
│  └─ σᵢ² = σ² · Xᵢ  (variance increases with regressor)
│  └─ Example: Spending variance increases with income
│
├─ Proportional to X²:
│  └─ σᵢ² = σ² · Xᵢ²  (stronger relationship)
│  └─ Example: Asset price uncertainty rises with value
│
├─ Proportional to E[Y]:
│  └─ σᵢ² = σ² · E[Yᵢ]²  (relative variance constant)
│  └─ Example: Count data, Poisson regression
│
├─ Grouped Heteroskedasticity:
│  └─ σᵢ² = σ_g² for i in group g
│  └─ Example: Different variance by region/industry
│
├─ ARCH Effects (Time Series):
│  └─ σₜ² = ω + α·εₜ₋₁²
│  └─ Example: Volatility clusters in financial returns
│
└─ Complex (Multi-dimensional):
   └─ σᵢ² depends on multiple X's and interactions
   └─ Example: Real estate prices (size, location, age)
```

---

## V. Mathematical Framework

### Heteroskedasticity Definition

In the classical model:
$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad i = 1, \ldots, n$$

**Homoskedasticity Assumption:**
$$\text{Var}(\epsilon_i | X_i) = \sigma^2 \quad \text{for all } i \text{ (constant)}$$

**Heteroskedasticity Violation:**
$$\text{Var}(\epsilon_i | X_i) = \sigma_i^2 \quad \text{(varies with } i \text{)}$$

### Consequences for OLS

Let $\Omega = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2)$ be the diagonal variance-covariance matrix.

**OLS estimator variance (true):**
$$\text{Var}(\hat{\beta}) = (X'X)^{-1} X' \Omega X (X'X)^{-1}$$

**OLS variance (what Gauss-Markov assumes, incorrect under hetero):**
$$\text{Var}_{\text{assumed}}(\hat{\beta}) = \sigma^2 (X'X)^{-1}$$

The bias arises because $\sigma^2 (X'X)^{-1} \neq (X'X)^{-1} X' \Omega X (X'X)^{-1}$ when $\Omega \neq \sigma^2 I$.

### White's Heteroskedasticity-Consistent Covariance Matrix

**HC0 (Asymptotically unbiased):**
$$\hat{\text{Var}}_{\text{HC0}}(\hat{\beta}) = (X'X)^{-1} \left( \sum_{i=1}^n \hat{\epsilon}_i^2 x_i x_i' \right) (X'X)^{-1}$$

where $\hat{\epsilon}_i = y_i - \hat{y}_i$ are OLS residuals and $x_i$ is the $i$-th row of $X$.

**HC1 (Finite-sample adjustment):**
$$\hat{\text{Var}}_{\text{HC1}}(\hat{\beta}) = \frac{n}{n-k} \hat{\text{Var}}_{\text{HC0}}(\hat{\beta})$$

where $k$ = number of regressors.

### Breusch-Pagan Test

**Null hypothesis:** $H_0: \text{Heteroskedasticity is absent}$

**Procedure:**
1. Estimate OLS: $\hat{\epsilon}_i = y_i - \hat{y}_i$
2. Estimate auxiliary regression: $\hat{\epsilon}_i^2 = \alpha_0 + \alpha_1 x_i + v_i$
3. Compute $R^2$ from auxiliary regression
4. Test statistic: $LM = n \cdot R^2 \sim \chi^2(k)$ under $H_0$

**Rejection Region:** $LM > \chi^2_{\alpha}(k)$ rejects homoskedasticity.

### Weighted Least Squares (WLS)

If variance function is known: $\sigma_i^2 = h(X_i)$

Transform model by dividing by $\sqrt{\sigma_i^2}$:
$$\frac{y_i}{\sigma_i} = \beta_0 \frac{1}{\sigma_i} + \beta_1 \frac{x_i}{\sigma_i} + \frac{\epsilon_i}{\sigma_i}$$

The transformed errors have constant variance = 1.

**WLS estimator:**
$$\hat{\beta}_{\text{WLS}} = \left( \sum_{i=1}^n \frac{1}{\sigma_i^2} x_i x_i' \right)^{-1} \sum_{i=1}^n \frac{1}{\sigma_i^2} x_i y_i$$

**Result:** $\hat{\beta}_{\text{WLS}}$ is BLUE (minimum variance) if weights are correct.

---

## VI. Python Mini-Project: Heteroskedasticity Detection & Correction

### Objective
Demonstrate:
1. Detecting heteroskedasticity via graphical and statistical tests
2. Comparing OLS, robust SE, WLS, and FGLS methods
3. Showing efficiency gains and confidence interval improvements
4. Practical application to simulated and real data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# HETEROSKEDASTICITY DETECTION AND CORRECTION
# ============================================================================

class HeteroskedasticityAnalysis:
    """
    Comprehensive heteroskedasticity detection and correction
    """
    
    def __init__(self, n_obs=200):
        """
        Generate data with heteroskedasticity
        y = 5 + 2*X + ε, where Var(ε) = σ²·X (variance increases with X)
        """
        self.n = n_obs
        self.X = np.random.uniform(1, 10, n_obs)
        
        # Heteroskedastic errors: variance proportional to X
        self.sigma = 0.5 * self.X  # std dev increases with X
        self.epsilon = np.random.normal(0, self.sigma)
        
        self.y = 5 + 2 * self.X + self.epsilon
        
        # Add constant term for regression
        self.X_design = np.column_stack([np.ones(n_obs), self.X])
        
    def fit_ols(self):
        """Fit OLS and compute naive (incorrect) standard errors"""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X_design, self.y)
        
        self.beta_ols = model.coef_
        self.y_pred = model.predict(self.X_design)
        self.residuals = self.y - self.y_pred
        
        # Naive variance (assumes homoskedasticity)
        n, k = self.X_design.shape
        mse = np.sum(self.residuals**2) / (n - k)
        self.X_prime_X_inv = np.linalg.inv(self.X_design.T @ self.X_design)
        
        # OLS standard errors (BIASED under heteroskedasticity)
        self.var_ols_naive = mse * self.X_prime_X_inv
        self.se_ols_naive = np.sqrt(np.diag(self.var_ols_naive))
        
        return self.beta_ols, self.se_ols_naive
    
    def white_robust_se(self):
        """
        Calculate White's heteroskedasticity-consistent standard errors
        Var(β̂) = (X'X)⁻¹ X'ΩX (X'X)⁻¹
        where Ω = diag(ε̂₁², ε̂₂², ..., ε̂ₙ²)
        """
        # White HC0 formula
        omega = np.diag(self.residuals**2)
        var_white = self.X_prime_X_inv @ (self.X_design.T @ omega @ self.X_design) @ self.X_prime_X_inv
        
        # HC1 finite-sample correction
        n, k = self.X_design.shape
        var_white_hc1 = (n / (n - k)) * var_white
        
        self.se_white = np.sqrt(np.diag(var_white))
        self.se_white_hc1 = np.sqrt(np.diag(var_white_hc1))
        
        return self.se_white, self.se_white_hc1
    
    def breusch_pagan_test(self):
        """
        Test for heteroskedasticity
        H₀: Homoskedasticity (errors have constant variance)
        """
        # Auxiliary regression: ε̂² on X
        aux_model = LinearRegression()
        aux_model.fit(self.X_design, self.residuals**2)
        
        # Compute R² of auxiliary regression
        y_aux_pred = aux_model.predict(self.X_design)
        ss_res = np.sum((self.residuals**2 - y_aux_pred)**2)
        ss_tot = np.sum((self.residuals**2 - np.mean(self.residuals**2))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        # Test statistic: LM = n·R²
        lm_stat = self.n * r2_aux
        
        # p-value from chi-square distribution
        dof = self.X_design.shape[1] - 1  # k regressors in auxiliary model
        p_value = 1 - stats.chi2.cdf(lm_stat, dof)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'r2_auxiliary': r2_aux,
            'reject_homoskedasticity': p_value < 0.05
        }
    
    def white_test(self):
        """
        More general White test including squared and interaction terms
        """
        # Create augmented design matrix with X, X², and interaction
        X_aug = np.column_stack([
            np.ones(self.n),
            self.X,
            self.X**2
        ])
        
        # Auxiliary regression: ε̂² on augmented X
        aux_model = LinearRegression(fit_intercept=False)
        aux_model.fit(X_aug, self.residuals**2)
        
        y_aux_pred = aux_model.predict(X_aug)
        ss_res = np.sum((self.residuals**2 - y_aux_pred)**2)
        ss_tot = np.sum((self.residuals**2 - np.mean(self.residuals**2))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        lm_stat = self.n * r2_aux
        dof = X_aug.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(lm_stat, dof)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'reject_homoskedasticity': p_value < 0.05
        }
    
    def weighted_least_squares(self):
        """
        WLS with weights proportional to 1/X (true variance structure is σ² ∝ X)
        """
        # True weights: since Var(ε) = σ²·X, weight = 1/σ²·X = 1/X
        weights = 1 / self.X
        
        # Transform by sqrt(weights)
        sqrt_w = np.sqrt(weights)
        X_weighted = self.X_design * sqrt_w[:, np.newaxis]
        y_weighted = self.y * sqrt_w
        
        # Fit OLS on transformed data
        model = LinearRegression(fit_intercept=False)
        model.fit(X_weighted, y_weighted)
        
        self.beta_wls = model.coef_
        y_pred_weighted = model.predict(X_weighted)
        residuals_weighted = y_weighted - y_pred_weighted
        
        # Compute SE for WLS
        n, k = X_weighted.shape
        mse_weighted = np.sum(residuals_weighted**2) / (n - k)
        X_weighted_prime_X = X_weighted.T @ X_weighted
        var_wls = mse_weighted * np.linalg.inv(X_weighted_prime_X)
        
        self.se_wls = np.sqrt(np.diag(var_wls))
        
        return self.beta_wls, self.se_wls
    
    def feasible_gls(self):
        """
        Two-stage FGLS estimation
        Stage 1: OLS to get residuals
        Stage 2: Model log(ε̂²) to estimate variance
        Stage 3: WLS using estimated variance
        """
        # Stage 1: OLS already computed (self.residuals)
        
        # Stage 2: Model variance
        # ln(ε̂²) = α + γ·X + v
        log_sq_res = np.log(self.residuals**2 + 1e-6)
        
        aux_model = LinearRegression()
        aux_model.fit(self.X_design, log_sq_res)
        
        # Predicted log variance
        log_h_hat = aux_model.predict(self.X_design)
        h_hat = np.exp(log_h_hat)
        
        # Stage 3: WLS with estimated weights
        weights_fgls = 1 / h_hat
        sqrt_w_fgls = np.sqrt(weights_fgls)
        
        X_fgls = self.X_design * sqrt_w_fgls[:, np.newaxis]
        y_fgls = self.y * sqrt_w_fgls
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_fgls, y_fgls)
        
        self.beta_fgls = model.coef_
        y_pred_fgls = model.predict(X_fgls)
        residuals_fgls = y_fgls - y_pred_fgls
        
        # SE for FGLS
        n, k = X_fgls.shape
        mse_fgls = np.sum(residuals_fgls**2) / (n - k)
        X_fgls_prime_X = X_fgls.T @ X_fgls
        var_fgls = mse_fgls * np.linalg.inv(X_fgls_prime_X)
        
        self.se_fgls = np.sqrt(np.diag(var_fgls))
        
        return self.beta_fgls, self.se_fgls
    
    def summary_table(self):
        """Create comparison table of all methods"""
        comparison = pd.DataFrame({
            'Method': ['OLS (Naive SE)', 'OLS (White HC0)', 'OLS (White HC1)', 'WLS', 'FGLS'],
            'β₀ Estimate': [
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_ols[0]:.4f}",
                f"{self.beta_wls[0]:.4f}",
                f"{self.beta_fgls[0]:.4f}"
            ],
            'SE(β₀)': [
                f"{self.se_ols_naive[0]:.4f}",
                f"{self.se_white[0]:.4f}",
                f"{self.se_white_hc1[0]:.4f}",
                f"{self.se_wls[0]:.4f}",
                f"{self.se_fgls[0]:.4f}"
            ],
            'β₁ Estimate': [
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_wls[1]:.4f}",
                f"{self.beta_fgls[1]:.4f}"
            ],
            'SE(β₁)': [
                f"{self.se_ols_naive[1]:.4f}",
                f"{self.se_white[1]:.4f}",
                f"{self.se_white_hc1[1]:.4f}",
                f"{self.se_wls[1]:.4f}",
                f"{self.se_fgls[1]:.4f}"
            ],
            '95% CI for β₁': [
                f"[{self.beta_ols[1]-1.96*self.se_ols_naive[1]:.4f}, {self.beta_ols[1]+1.96*self.se_ols_naive[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_white[1]:.4f}, {self.beta_ols[1]+1.96*self.se_white[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_white_hc1[1]:.4f}, {self.beta_ols[1]+1.96*self.se_white_hc1[1]:.4f}]",
                f"[{self.beta_wls[1]-1.96*self.se_wls[1]:.4f}, {self.beta_wls[1]+1.96*self.se_wls[1]:.4f}]",
                f"[{self.beta_fgls[1]-1.96*self.se_fgls[1]:.4f}, {self.beta_fgls[1]+1.96*self.se_fgls[1]:.4f}]"
            ]
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("HETEROSKEDASTICITY ANALYSIS: DETECTION & CORRECTION")
print("="*80)

# Initialize and fit all models
analysis = HeteroskedasticityAnalysis(n_obs=200)

# Fit OLS
analysis.fit_ols()
print(f"\n1. OLS ESTIMATION (with heteroskedasticity in data)")
print(f"   β₀ (true=5): {analysis.beta_ols[0]:.4f}")
print(f"   β₁ (true=2): {analysis.beta_ols[1]:.4f}")

# Get robust SEs
analysis.white_robust_se()
print(f"\n2. STANDARD ERRORS COMPARISON")
print(f"   {'':30} Naive OLS  White HC0  White HC1")
print(f"   {'SE(β₀)':30} {analysis.se_ols_naive[0]:9.4f}  {analysis.se_white[0]:9.4f}  {analysis.se_white_hc1[0]:9.4f}")
print(f"   {'SE(β₁)':30} {analysis.se_ols_naive[1]:9.4f}  {analysis.se_white[1]:9.4f}  {analysis.se_white_hc1[1]:9.4f}")
print(f"   SE Inflation Factor (β₁):     1.00x      {analysis.se_white[1]/analysis.se_ols_naive[1]:.2f}x      {analysis.se_white_hc1[1]/analysis.se_ols_naive[1]:.2f}x")

# Heteroskedasticity tests
bp_test = analysis.breusch_pagan_test()
white_test = analysis.white_test()

print(f"\n3. HETEROSKEDASTICITY TESTS")
print(f"   Breusch-Pagan Test:")
print(f"     LM Statistic: {bp_test['test_statistic']:.4f}")
print(f"     p-value: {bp_test['p_value']:.6f}")
print(f"     Conclusion: {'Reject H₀ (Heteroskedasticity detected)' if bp_test['reject_homoskedasticity'] else 'Fail to reject H₀'}")
print(f"\n   White Test:")
print(f"     LM Statistic: {white_test['test_statistic']:.4f}")
print(f"     p-value: {white_test['p_value']:.6f}")
print(f"     Conclusion: {'Reject H₀ (Heteroskedasticity detected)' if white_test['reject_homoskedasticity'] else 'Fail to reject H₀'}")

# WLS estimation
analysis.weighted_least_squares()
print(f"\n4. WEIGHTED LEAST SQUARES (WLS)")
print(f"   β₀: {analysis.beta_wls[0]:.4f} (SE: {analysis.se_wls[0]:.4f})")
print(f"   β₁: {analysis.beta_wls[1]:.4f} (SE: {analysis.se_wls[1]:.4f})")
print(f"   Efficiency gain over OLS: {(analysis.se_ols_naive[1]/analysis.se_wls[1] - 1)*100:.1f}% (lower SE)")

# FGLS estimation
analysis.feasible_gls()
print(f"\n5. FEASIBLE GLS (FGLS)")
print(f"   β₀: {analysis.beta_fgls[0]:.4f} (SE: {analysis.se_fgls[0]:.4f})")
print(f"   β₁: {analysis.beta_fgls[1]:.4f} (SE: {analysis.se_fgls[1]:.4f})")
print(f"   Efficiency gain over OLS: {(analysis.se_ols_naive[1]/analysis.se_fgls[1] - 1)*100:.1f}% (lower SE)")

# Comparison table
print(f"\n6. COMPREHENSIVE COMPARISON TABLE")
print(analysis.summary_table().to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Residuals vs Fitted (Heteroskedasticity Pattern)
ax1 = axes[0, 0]
ax1.scatter(analysis.y_pred, analysis.residuals, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
# Add reference lines showing variance pattern
ax1.fill_between(np.sort(analysis.y_pred), 
                 -1.96 * np.sort(analysis.sigma),
                 1.96 * np.sort(analysis.sigma),
                 alpha=0.2, color='red', label='True ±1.96σ (heteroskedastic)')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Panel 1: Residual Plot (Cone Shape Indicates Heteroskedasticity)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: |Residuals| vs X (Scale-Location)
ax2 = axes[0, 1]
abs_resid_sqrt = np.sqrt(np.abs(analysis.residuals))
ax2.scatter(analysis.X, abs_resid_sqrt, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
# Smooth trend line
sort_idx = np.argsort(analysis.X)
ax2.plot(analysis.X[sort_idx], analysis.sigma[sort_idx], 'r-', linewidth=2, label='True σ(X)')
ax2.set_xlabel('X')
ax2.set_ylabel('√|Residuals|')
ax2.set_title('Panel 2: Scale-Location Plot\n(Upward trend = heteroskedasticity)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Confidence Intervals Comparison
ax3 = axes[1, 0]
methods = ['OLS\n(Naive)', 'OLS\n(White)', 'OLS\n(HC1)', 'WLS', 'FGLS']
se_values = [analysis.se_ols_naive[1], analysis.se_white[1], 
             analysis.se_white_hc1[1], analysis.se_wls[1], analysis.se_fgls[1]]
ci_lower = [analysis.beta_ols[1] - 1.96*se for se in se_values[:3]] + \
           [analysis.beta_wls[1] - 1.96*analysis.se_wls[1], analysis.beta_fgls[1] - 1.96*analysis.se_fgls[1]]
ci_upper = [analysis.beta_ols[1] + 1.96*se for se in se_values[:3]] + \
           [analysis.beta_wls[1] + 1.96*analysis.se_wls[1], analysis.beta_fgls[1] + 1.96*analysis.se_fgls[1]]

y_pos = np.arange(len(methods))
ax3.errorbar(y_pos, [analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]],
             yerr=[np.array(ci_lower) - np.array([analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]]),
                   np.array(ci_upper) - np.array([analysis.beta_ols[1]]*3 + [analysis.beta_wls[1], analysis.beta_fgls[1]])],
             fmt='o', markersize=8, capsize=5, capthick=2, color='blue')
ax3.axhline(y=2.0, color='green', linestyle='--', linewidth=2, label='True β₁ = 2')
ax3.set_xticks(y_pos)
ax3.set_xticklabels(methods)
ax3.set_ylabel('β₁ Estimate with 95% CI')
ax3.set_title('Panel 3: Confidence Interval Widths\n(OLS Naive too narrow; White/WLS/FGLS appropriate)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Residuals Squared vs X (For Auxiliary Regression)
ax4 = axes[1, 1]
ax4.scatter(analysis.X, analysis.residuals**2, alpha=0.6, s=30, edgecolor='k', linewidth=0.5, label='Observed ε̂²')
# Fit auxiliary regression for visualization
aux_model = LinearRegression()
aux_model.fit(np.column_stack([np.ones(len(analysis.X)), analysis.X]), analysis.residuals**2)
X_plot = np.linspace(analysis.X.min(), analysis.X.max(), 100)
y_aux = aux_model.predict(np.column_stack([np.ones(len(X_plot)), X_plot]))
ax4.plot(X_plot, y_aux, 'r-', linewidth=2, label='Fitted: ε̂² = α + γ·X')
ax4.set_xlabel('X')
ax4.set_ylabel('Squared Residuals (ε̂²)')
ax4.set_title('Panel 4: Breusch-Pagan Auxiliary Regression\n(Upward trend confirms heteroskedasticity)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heteroskedasticity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"• Naive OLS SE underestimate true uncertainty by ~{(1 - analysis.se_ols_naive[1]/analysis.se_white_hc1[1])*100:.0f}%")
print(f"• White HC1 SE are ~{analysis.se_white_hc1[1]/analysis.se_ols_naive[1]:.2f}x wider (correct for inference)")
print(f"• WLS achieves ~{(1 - analysis.se_wls[1]/analysis.se_ols_naive[1])*100:.0f}% efficiency gain (lowest SE)")
print(f"• FGLS achieves ~{(1 - analysis.se_fgls[1]/analysis.se_ols_naive[1])*100:.0f}% efficiency gain (asymptotically optimal)")
print(f"• Breusch-Pagan p-value: {bp_test['p_value']:.4f} (strongly rejects homoskedasticity)")
print("="*80 + "\n")
```

### Output Explanation
- **Panel 1:** "Cone shape" shows clear heteroskedasticity; residual spread increases with fitted values.
- **Panel 2:** Scale-location plot confirms linear trend: residual variance proportional to X.
- **Panel 3:** OLS naive intervals too narrow (false precision). White/WLS/FGLS corrected widths appropriate.
- **Panel 4:** Breusch-Pagan auxiliary regression shows clear linear relationship (ε̂² increases with X).

**Efficiency Comparison:**
- Naive OLS SE: misleadingly narrow (15-25% too tight)
- WLS SE: lowest (most efficient) when weights correct
- FGLS SE: intermediate efficiency (depends on variance model fit)

---

## VII. References & Key Design Insights

1. **White, H. (1980).** "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." Econometrica, 48(4), 817-838.
   - Foundational robust SE method; HC0, HC1 variants; empirically widely used

2. **Breusch, T. S., & Pagan, A. R. (1979).** "A simple test for heteroskedasticity and random coefficient variation." Econometrica, 47(5), 1287-1294.
   - Standard hypothesis test; auxiliary regression approach; parametric test

3. **Wooldridge, J. M. (2019).** "Introductory Econometrics: A Modern Approach" (7th ed.). Cengage Learning.
   - Comprehensive coverage of heteroskedasticity, WLS, FGLS, finite-sample properties

4. **Long, J. S., & Ervin, L. H. (2000).** "Using heteroskedasticity consistent standard errors in the linear regression model." American Statistician, 54(3), 217-224.
   - Finite-sample comparison of HC0, HC1, HC2, HC3; recommendations for practice

**Key Design Concepts:**
- **Robustness:** White SE valid asymptotically regardless of true variance structure; no model specification needed
- **Efficiency:** WLS optimal if true weights known; FGLS asymptotically optimal; but both require variance model specification
- **Diagnostics:** Graphical tests (residual plots) immediately reveal heteroskedasticity; statistical tests quantify strength with p-values
- **Trade-off:** Simplicity (robust SE) vs efficiency (WLS/FGLS); robust SE preferred when variance structure unknown

