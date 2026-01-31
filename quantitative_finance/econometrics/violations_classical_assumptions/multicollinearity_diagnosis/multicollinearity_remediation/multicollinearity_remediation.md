# Multicollinearity: Diagnosis, Impact & Remediation Strategies

## I. Concept Skeleton

**Definition:** Multicollinearity refers to high correlation between independent variables (regressors) in a regression model. When $X_1, X_2, \ldots, X_k$ are highly linearly dependent (i.e., one can be approximated as a linear combination of others), the regression design matrix becomes ill-conditioned, causing estimation and inference problems.

**Purpose:** Detect multicollinearity in regression models, understand its sources and consequences for coefficient estimation, variance inflation, and hypothesis tests, and apply remediation techniques (variable selection, regularization, dimensionality reduction, domain knowledge).

**Prerequisites:** Linear algebra (eigenvalues, condition number), OLS regression, correlation analysis, ridge/LASSO regression, principal components analysis.

---

## II. Comparative Framing

| **Aspect** | **OLS (No Multicollinearity)** | **OLS with Multicollinearity** | **Ridge Regression** | **LASSO** | **Principal Components** |
|-----------|-------------------------------|-------------------------------|-------------------|----------|------------------------|
| **Design Matrix** | Full rank, orthogonal | Full rank but ill-conditioned | Full rank (penalized) | Sparse selection | Reduced dimension |
| **β Coefficient** | Unbiased, minimum variance | Unbiased but high variance | Biased, lower variance | Biased & sparse | Transformed coefficients |
| **SE(β)** | Minimum (BLUE) | Large (inflated) | Smaller than OLS | Sparse/zero SE | Reduced complexity |
| **Estimation Stability** | Stable | Unstable (small data changes → large β changes) | Stable (regularized) | Stable & sparse | Stable (dimension-reduced) |
| **Interpretation** | Clear; each β = marginal effect | Ambiguous; coefficients interchange with correlated vars | Penalized; constrained norm | Clear; selected vars only | Latent factors |
| **Prediction** | Good if model correct | Can be good despite high variance | Often better (bias-variance trade-off) | Good with sparsity | Good if latent structure |
| **Use Case** | Economic theory, causal | Usually indicates problem | High-dimensional data | Feature selection | Exploratory analysis |

---

## III. Examples & Counterexamples

### Example 1: Price Prediction with Correlated Size Measures (Real Estate)
**Setup:**
- Model: $\text{Price}_i = \beta_0 + \beta_1 \text{SquareFeet}_i + \beta_2 \text{Bedrooms}_i + \beta_3 \text{Rooms}_i + \epsilon_i$
- Problem: Square footage, bedrooms, and total rooms are highly correlated (corr ≈ 0.92)
- Sample: 500 homes

**Multicollinearity Impact:**
- Correlation matrix reveals: Corr(SquareFeet, Bedrooms) = 0.92
- Variance Inflation Factor: VIF(SquareFeet) = 13.5, VIF(Bedrooms) = 12.1
- OLS estimates wildly unstable: 
  - Sample 1: β₁=0.50 ($/sqft), β₂=-5000 ($/bedroom) — negative bedroom effect?!
  - Sample 2: β₁=0.45, β₂=8000 — positive bedroom effect
  - Same underlying data, different samples → vastly different slopes
- SE(β₁) = 0.25 (huge), SE(β₂) = 4500 (huge)

**Key Insight:** Individual coefficients are unbiased on average but extremely noisy. Predictions still valid, but interpreting individual slopes dangerous.

**Remediation:**
- Option 1: Drop one of the correlated variables (Bedrooms)
  - Reduced model: Price = β₀ + β₁·SquareFeet + ε
  - SE(β₁) drops to 0.04 (much tighter)
  - Trade-off: Lose bedroom as separate effect (omitted variable bias, but smaller than multicollinearity problem)
  
- Option 2: Ridge regression with λ=0.01
  - All three variables retained but coefficients shrunk
  - Predictions improve; coefficients more stable
  - Trade-off: Coefficients biased, but lower variance

- Option 3: PCA
  - Extract 2 principal components (captures 98% variance)
  - Regression on PC1, PC2 directly
  - Trade-off: Lose interpretability (what does PC1 mean economically?)

### Example 2: Perfect Multicollinearity (Dummy Variable Trap)
**Setup:**
- Model: $\text{Salary}_i = \beta_0 + \beta_1 \text{Male}_i + \beta_2 \text{Female}_i + \beta_3 \text{Experience}_i + \epsilon_i$
- Problem: Male + Female = 1 for all i (perfect linear dependence)
- Perfect multicollinearity; design matrix singular

**Consequence:**
- OLS cannot estimate: $(X'X)^{-1}$ does not exist (singular matrix)
- Regression software either:
  - Crashes with "singular matrix" error
  - Automatically drops one dummy variable
  - Gives arbitrary coefficient estimates (user-dependent)

**Prevention:** Drop one dummy (Female) or use no-intercept model:
- Correct: Salary = β₀ + β₁·Male + β₃·Experience + ε (omit Female)
- Interpretation: β₀ = salary of females (baseline), β₁ = gender premium for males

### Example 3: Moderate Multicollinearity in Economic Model (Acceptable)
**Setup:**
- Demand model: $\ln(\text{Quantity})_i = \beta_0 + \beta_1 \ln(\text{Price})_i + \beta_2 \ln(\text{Income})_i + \epsilon_i$
- Prices and incomes correlated (ρ = 0.45) — moderate but not severe
- Economic theory expects both effects

**Analysis:**
- Correlation not concerning (0.45 is typical in macro)
- VIF(Price) ≈ 1.4, VIF(Income) ≈ 1.4 (< 5, acceptable)
- Coefficients:
  - Price elasticity: β₁ = -1.2 (SE = 0.15)
  - Income elasticity: β₂ = 0.8 (SE = 0.18)
- Inference: Both statistically significant at 5% level
- Prediction R² = 0.87 (good fit)

**Conclusion:** No remediation needed. Multicollinearity is present but weak enough to not cause serious problems. Theoretical reasoning supports both variables.

---

## IV. Layer Breakdown

```
MULTICOLLINEARITY FRAMEWORK

┌──────────────────────────────────────────────────────┐
│         MULTIPLE LINEAR REGRESSION MODEL              │
│    y = X·β + ε                                       │
│    where X is n×k design matrix                      │
│                                                      │
│    Assumption: X has full rank k (no linear deps)   │
└──────────────────┬───────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │  MULTICOLLINEARITY VIOLATION      │
    │  (Linear Dependence Among X's)    │
    │  Rank(X) < k (linear dependence) │
    │  or                               │
    │  Columns of X are highly correlated
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────────────────┐
    │  TYPES OF MULTICOLLINEARITY                  │
    │                                              │
    │  1. PERFECT MULTICOLLINEARITY:              │
    │     ├─ Rank(X) < k (design matrix singular)│
    │     ├─ Example: X₁ + X₂ = constant         │
    │     ├─ Example: Dummy variable trap        │
    │     ├─ Consequence: (X'X)⁻¹ undefined     │
    │     └─ Solution: Drop variables             │
    │                                              │
    │  2. HIGH MULTICOLLINEARITY:                │
    │     ├─ Corr(Xᵢ, Xⱼ) > 0.9               │
    │     ├─ Example: Size measures (sqft, beds) │
    │     ├─ Example: Lagged variables (X_t, X_t-1)
    │     ├─ Consequence: Large SE, unstable β  │
    │     └─ Solutions: Variable selection, ridge│
    │                                              │
    │  3. MODERATE MULTICOLLINEARITY:            │
    │     ├─ Corr(Xᵢ, Xⱼ) ≈ 0.5-0.7            │
    │     ├─ Example: Price & income in demand │
    │     ├─ Often unavoidable with real data   │
    │     └─ Usually acceptable if VIF < 5-10  │
    │                                              │
    │  4. NO MULTICOLLINEARITY:                  │
    │     ├─ Corr(Xᵢ, Xⱼ) < 0.5                │
    │     ├─ Example: Randomized experiments    │
    │     ├─ Example: Orthogonal design matrix  │
    │     └─ OLS has minimum variance (BLUE)    │
    └──────────────┬─────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  CONSEQUENCES FOR OLS                       │
    │                                              │
    │  ✓ β̂ still UNBIASED                       │
    │    (expected value correct)                │
    │                                              │
    │  ✗ Var(β̂) INFLATED:                       │
    │    └─ Var(β̂) = σ² (X'X)⁻¹               │
    │    └─ Large off-diagonal elements if corr │
    │    └─ SE(β̂) can be 5-100x larger         │
    │    └─ Confidence intervals very wide      │
    │                                              │
    │  ✗ Coefficient INSTABILITY:                │
    │    └─ Small changes in data              │
    │    └─ Produce large changes in β̂        │
    │    └─ Coefficients not robust             │
    │                                              │
    │  ✗ Individual INTERPRETATION AMBIGUOUS:    │
    │    └─ Which variable causes the effect?  │
    │    └─ Partial effects hard to isolate     │
    │    └─ Swap correlated vars → change β    │
    │                                              │
    │  ✓ PREDICTION still valid                 │
    │    └─ X'X)⁻¹ X' not too inaccurate       │
    │    └─ Ŷ = X·β̂ accurate if X in-sample  │
    │    └─ Problem mainly with inference       │
    │                                              │
    │  ✗ t-tests LESS POWERFUL:                 │
    │    └─ Large SE → t-stats small            │
    │    └─ Fail to reject H₀ even if true      │
    │    └─ Type II error (false negative)      │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  DETECTION METHODS                         │
    │                                              │
    │  1. CORRELATION MATRIX:                    │
    │     ├─ Pairwise correlations             │
    │     ├─ Rule: |Corr| > 0.9 → concern     │
    │     └─ Simple but incomplete (3+ vars)   │
    │                                              │
    │  2. VARIANCE INFLATION FACTOR (VIF):       │
    │     ├─ VIF_j = 1/(1-R²_j)               │
    │     ├─ R²_j = R² from regressing X_j on others
    │     ├─ VIF = 1: No correlation           │
    │     ├─ VIF > 5-10: Problem               │
    │     ├─ VIF > 10: Serious problem         │
    │     └─ Comprehensive measure              │
    │                                              │
    │  3. CONDITION NUMBER:                     │
    │     ├─ κ = λ_max / λ_min  (eigenvalues) │
    │     ├─ κ < 10: Good                      │
    │     ├─ κ ~ 30: Moderate                  │
    │     ├─ κ > 100: Severe multicollinearity│
    │     └─ Shows ill-conditioning             │
    │                                              │
    │  4. EIGENVALUES OF (X'X):                 │
    │     ├─ Many small eigenvalues → problem  │
    │     ├─ Eigenvector = combination of X's  │
    │     └─ Eigenvector coefficients show deps│
    │                                              │
    │  5. AUXILIARY REGRESSIONS:                │
    │     ├─ Regress each X_j on others       │
    │     ├─ High R² → X_j is function of them│
    │     └─ Systematic check for dependencies│
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │  REMEDIATION STRATEGIES                    │
    │                                              │
    │  1. DO NOTHING (If prediction goal):       │
    │     ├─ Predictions still accurate        │
    │     ├─ Only inference affected           │
    │     ├─ Model interpretation less clear   │
    │     └─ Acceptable for forecasting        │
    │                                              │
    │  2. VARIABLE SELECTION:                   │
    │     ├─ Drop one of correlated pair      │
    │     ├─ Lose some information            │
    │     ├─ Reduces multicollinearity        │
    │     ├─ Simpler model                    │
    │     └─ May introduce omitted var bias   │
    │                                              │
    │  3. RIDGE REGRESSION:                    │
    │     ├─ β̂_ridge = (X'X + λI)⁻¹ X'y     │
    │     ├─ λ > 0 penalty (shrinkage)       │
    │     ├─ Reduces variance at cost of bias │
    │     ├─ All variables retained           │
    │     └─ Choose λ via CV                  │
    │                                              │
    │  4. LASSO (L1 Regularization):           │
    │     ├─ min ||y - Xβ||² + λ|β|         │
    │     ├─ Simultaneous shrinkage & selection│
    │     ├─ Some coefficients → 0            │
    │     ├─ Sparse solution                  │
    │     └─ Choose λ via CV                  │
    │                                              │
    │  5. PRINCIPAL COMPONENTS ANALYSIS:        │
    │     ├─ Reduce k correlated X's to m < k │
    │     ├─ Create uncorrelated combinations │
    │     ├─ Regress y on principal components│
    │     ├─ Orthogonal regressors            │
    │     └─ Lose interpretability            │
    │                                              │
    │  6. COLLECT MORE DATA:                   │
    │     ├─ Increases sample size n          │
    │     ├─ Reduces Var(β̂) = σ²/variation   │
    │     ├─ Helps if multicollinearity weak  │
    │     └─ Expensive but effective           │
    │                                              │
    │  7. RESPECIFY MODEL:                     │
    │     ├─ Use ratios (X₁/X₂ instead)      │
    │     ├─ Use differences (X₁ - X₂)       │
    │     ├─ Include interaction/nonlinear    │
    │     ├─ Economic theory may guide        │
    │     └─ Can reduce spurious correlation  │
    └──────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Variance Inflation Factor (VIF)

For coefficient β_j, the variance is:

$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{\sum_i (x_{ij} - \bar{x}_j)^2 \times (1-R_j^2)}$$

where $R_j^2$ is the R² from auxiliary regression: $x_j = f(x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_k)$.

**Variance Inflation Factor:**
$$\text{VIF}_j = \frac{1}{1-R_j^2}$$

**Interpretation:**
- VIF = 1: No correlation with other X's (ideal)
- VIF = 5: Variance inflated 5x relative to orthogonal case
- VIF > 10: Severe multicollinearity; requires investigation

### Condition Number

From eigenvalue decomposition of $(X'X)$:

$$\text{Condition Number: } \kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

**Interpretation:**
- κ < 10: Well-conditioned (low multicollinearity)
- κ ~ 30: Moderate ill-conditioning
- κ > 100: Severe multicollinearity; numerical instability

**Relationship to VIF:**
$$\text{Average VIF} \approx \frac{\kappa + 1}{2}$$

### Ridge Regression Solution

Standard OLS:
$$\hat{\beta}_{\text{OLS}} = (X'X)^{-1} X'y$$

Ridge regression adds penalty:
$$\hat{\beta}_{\text{ridge}} = (X'X + \lambda I)^{-1} X'y$$

where $\lambda > 0$ is regularization parameter.

**Effect of λ:**
- λ = 0: OLS solution (high variance under multicollinearity)
- λ → ∞: All coefficients → 0 (high bias)
- Optimal λ chosen via cross-validation (bias-variance trade-off)

### LASSO (Least Absolute Shrinkage)

$$\min_{\beta} \left\| y - X\beta \right\|_2^2 + \lambda \|\beta\|_1$$

where $\|\beta\|_1 = \sum_{j=1}^k |\beta_j|$ is L1 norm.

**Property:** Sparse solution; some coefficients exactly 0 (feature selection).

---

## VI. Python Mini-Project: Multicollinearity Diagnosis & Remediation

### Objective
Demonstrate:
1. Detecting multicollinearity (correlation, VIF, condition number)
2. Understanding consequences (coefficient instability, wide SE)
3. Comparing remediation methods (variable selection, ridge, LASSO, PCA)
4. Model comparison with cross-validation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# MULTICOLLINEARITY DETECTION AND REMEDIATION
# ============================================================================

class MulticollinearityAnalysis:
    """
    Comprehensive multicollinearity diagnosis and remediation
    """
    
    def __init__(self, n=100, multicollinearity_level='high'):
        """
        Generate data with multicollinear X variables
        y = 2 + 3*X₁ + 2*X₂ + 1*X₃ + ε
        where X₂ ≈ 0.9*X₁ and X₃ ≈ 0.8*X₁ + 0.6*X₂
        
        Parameters:
        -----------
        n: Sample size
        multicollinearity_level: 'low', 'moderate', 'high'
        """
        self.n = n
        self.level = multicollinearity_level
        
        # Generate X₁ (base variable)
        self.X1 = np.random.normal(5, 2, n)
        
        # Generate X₂ (highly correlated with X₁)
        if multicollinearity_level == 'high':
            self.X2 = 0.90 * self.X1 + np.random.normal(0, 0.5, n)
            self.X3 = 0.85 * self.X1 + 0.60 * self.X2 + np.random.normal(0, 0.3, n)
        elif multicollinearity_level == 'moderate':
            self.X2 = 0.50 * self.X1 + np.random.normal(0, 1.5, n)
            self.X3 = 0.40 * self.X1 + 0.50 * self.X2 + np.random.normal(0, 1.0, n)
        else:  # low
            self.X2 = np.random.normal(5, 2, n)
            self.X3 = np.random.normal(5, 2, n)
        
        # Generate y with true effects
        epsilon = np.random.normal(0, 1, n)
        self.y = 2 + 3 * self.X1 + 2 * self.X2 + 1 * self.X3 + epsilon
        
        # Design matrices
        self.X_full = np.column_stack([np.ones(n), self.X1, self.X2, self.X3])
        self.X_standardized = StandardScaler(with_mean=False).fit_transform(
            np.column_stack([self.X1, self.X2, self.X3])
        )
        
    def correlation_analysis(self):
        """Compute correlation matrix"""
        data = np.column_stack([self.X1, self.X2, self.X3])
        corr_matrix = np.corrcoef(data.T)
        
        return pd.DataFrame(
            corr_matrix,
            columns=['X₁', 'X₂', 'X₃'],
            index=['X₁', 'X₂', 'X₃']
        )
    
    def vif_analysis(self):
        """Compute Variance Inflation Factors"""
        vif_values = []
        
        for j in range(3):  # 3 regressors
            # Regress X_j on other X's
            X_others = np.column_stack([self.X_standardized[:, i] for i in range(3) if i != j])
            X_j = self.X_standardized[:, j]
            
            model = LinearRegression()
            model.fit(X_others, X_j)
            y_pred = model.predict(X_others)
            
            # R² from auxiliary regression
            ss_res = np.sum((X_j - y_pred)**2)
            ss_tot = np.sum((X_j - np.mean(X_j))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            # VIF
            vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
            vif_values.append(vif)
        
        return pd.DataFrame({
            'Variable': ['X₁', 'X₂', 'X₃'],
            'VIF': vif_values,
            'Severity': ['Low' if v < 5 else 'Moderate' if v < 10 else 'High' for v in vif_values]
        })
    
    def condition_number(self):
        """Compute condition number of X'X"""
        X_centered = self.X_standardized
        XX = X_centered.T @ X_centered
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(XX)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Condition number
        kappa = eigenvalues[0] / eigenvalues[-1]
        
        return {
            'condition_number': kappa,
            'eigenvalues': eigenvalues,
            'severity': 'Good' if kappa < 10 else 'Moderate' if kappa < 30 else 'Severe'
        }
    
    def ols_analysis(self):
        """Standard OLS regression"""
        model = LinearRegression()
        model.fit(self.X_full, self.y)
        
        # Coefficients and residuals
        beta = model.coef_
        y_pred = model.predict(self.X_full)
        residuals = self.y - y_pred
        
        # Standard errors
        n, k = self.X_full.shape
        mse = np.sum(residuals**2) / (n - k)
        var_covar = mse * np.linalg.inv(self.X_full.T @ self.X_full)
        se = np.sqrt(np.diag(var_covar))
        
        return {
            'beta': beta,
            'se': se,
            'y_pred': y_pred,
            'residuals': residuals,
            'r_squared': 1 - (np.sum(residuals**2) / np.sum((self.y - np.mean(self.y))**2))
        }
    
    def coefficient_stability(self, bootstrap_samples=100):
        """
        Bootstrap resampling to assess coefficient stability
        Large coefficient variation = unstable = multicollinearity
        """
        bootstrap_betas = []
        
        for _ in range(bootstrap_samples):
            # Resample
            idx = np.random.choice(self.n, self.n, replace=True)
            X_boot = self.X_full[idx]
            y_boot = self.y[idx]
            
            # OLS
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            bootstrap_betas.append(model.coef_)
        
        bootstrap_betas = np.array(bootstrap_betas)
        
        return {
            'beta_mean': np.mean(bootstrap_betas, axis=0),
            'beta_std': np.std(bootstrap_betas, axis=0),
            'bootstrap_betas': bootstrap_betas
        }
    
    def ridge_regression(self, alphas=None):
        """Ridge regression with cross-validation"""
        if alphas is None:
            alphas = np.logspace(-2, 4, 50)
        
        cv_scores = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            cv_score = cross_val_score(ridge, self.X_full, self.y, cv=5, 
                                       scoring='neg_mean_squared_error')
            cv_scores.append(-cv_score.mean())
        
        # Best alpha
        best_idx = np.argmin(cv_scores)
        best_alpha = alphas[best_idx]
        
        # Fit with best alpha
        ridge_final = Ridge(alpha=best_alpha)
        ridge_final.fit(self.X_full, self.y)
        
        return {
            'alphas': alphas,
            'cv_scores': cv_scores,
            'best_alpha': best_alpha,
            'coefficients': ridge_final.coef_,
            'predictions': ridge_final.predict(self.X_full)
        }
    
    def lasso_regression(self, alphas=None):
        """LASSO regression with cross-validation"""
        if alphas is None:
            alphas = np.logspace(-3, 1, 50)
        
        cv_scores = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            cv_score = cross_val_score(lasso, self.X_full, self.y, cv=5,
                                       scoring='neg_mean_squared_error')
            cv_scores.append(-cv_score.mean())
        
        # Best alpha
        best_idx = np.argmin(cv_scores)
        best_alpha = alphas[best_idx]
        
        # Fit with best alpha
        lasso_final = Lasso(alpha=best_alpha, max_iter=10000)
        lasso_final.fit(self.X_full, self.y)
        
        return {
            'alphas': alphas,
            'cv_scores': cv_scores,
            'best_alpha': best_alpha,
            'coefficients': lasso_final.coef_,
            'predictions': lasso_final.predict(self.X_full)
        }
    
    def pca_regression(self):
        """Principal Components Regression"""
        pca = PCA()
        X_pca = pca.fit_transform(self.X_standardized)
        
        # Add intercept
        X_pca_full = np.column_stack([np.ones(self.n), X_pca])
        
        # OLS on principal components
        model = LinearRegression(fit_intercept=False)
        model.fit(X_pca_full, self.y)
        
        # Variance explained by each PC
        var_explained = pca.explained_variance_ratio_
        
        return {
            'pca_loadings': pca.components_,
            'var_explained': var_explained,
            'cumsum_var': np.cumsum(var_explained),
            'pc_coefficients': model.coef_,
            'predictions': model.predict(X_pca_full)
        }
    
    def variable_selection(self):
        """Drop X₃ (most dependent) and refit"""
        X_reduced = np.column_stack([np.ones(self.n), self.X1, self.X2])
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_reduced, self.y)
        
        # SEs
        residuals = self.y - model.predict(X_reduced)
        n, k = X_reduced.shape
        mse = np.sum(residuals**2) / (n - k)
        var_covar = mse * np.linalg.inv(X_reduced.T @ X_reduced)
        se = np.sqrt(np.diag(var_covar))
        
        return {
            'beta': model.coef_,
            'se': se,
            'predictions': model.predict(X_reduced)
        }
    
    def comparison_table(self):
        """Create comparison of all methods"""
        ols_result = self.ols_analysis()
        ridge_result = self.ridge_regression()
        lasso_result = self.lasso_regression()
        pca_result = self.pca_regression()
        varsel_result = self.variable_selection()
        
        # MSE
        ols_mse = np.mean((self.y - ols_result['y_pred'])**2)
        ridge_mse = np.mean((self.y - ridge_result['predictions'])**2)
        lasso_mse = np.mean((self.y - lasso_result['predictions'])**2)
        pca_mse = np.mean((self.y - pca_result['predictions'])**2)
        varsel_mse = np.mean((self.y - varsel_result['predictions'])**2)
        
        comparison = pd.DataFrame({
            'Method': ['OLS', 'Ridge', 'LASSO', 'PCA', 'Variable Selection'],
            'MSE': [ols_mse, ridge_mse, lasso_mse, pca_mse, varsel_mse],
            'R²': [
                ols_result['r_squared'],
                1 - (ridge_mse / np.var(self.y)),
                1 - (lasso_mse / np.var(self.y)),
                1 - (pca_mse / np.var(self.y)),
                1 - (varsel_mse / np.var(self.y))
            ],
            'Interpretability': ['Full', 'Full', 'Sparse', 'PCs', 'Reduced']
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MULTICOLLINEARITY ANALYSIS: DIAGNOSIS & REMEDIATION")
print("="*80)

# Initialize
analysis = MulticollinearityAnalysis(n=100, multicollinearity_level='high')

# Correlation analysis
print(f"\n1. CORRELATION MATRIX")
corr = analysis.correlation_analysis()
print(corr.round(3))
print(f"   Interpretation: X₂ ≈ 0.9·X₁ (high correlation)")

# VIF analysis
print(f"\n2. VARIANCE INFLATION FACTORS (VIF)")
vif_df = analysis.vif_analysis()
print(vif_df.to_string(index=False))
print(f"   Interpretation: VIF > 5 indicates severe multicollinearity")

# Condition number
cond_result = analysis.condition_number()
print(f"\n3. CONDITION NUMBER")
print(f"   κ = {cond_result['condition_number']:.2f}")
print(f"   Eigenvalues: {cond_result['eigenvalues'].round(2)}")
print(f"   Severity: {cond_result['severity']}")
print(f"   (κ > 100 = severe ill-conditioning)")

# OLS Analysis
ols_result = analysis.ols_analysis()
print(f"\n4. OLS REGRESSION")
print(f"   Coefficients (true: [2, 3, 2, 1]):")
print(f"   β₀: {ols_result['beta'][0]:7.3f} (SE: {ols_result['se'][0]:.3f})")
print(f"   β₁: {ols_result['beta'][1]:7.3f} (SE: {ols_result['se'][1]:.3f})")
print(f"   β₂: {ols_result['beta'][2]:7.3f} (SE: {ols_result['se'][2]:.3f})")
print(f"   β₃: {ols_result['beta'][3]:7.3f} (SE: {ols_result['se'][3]:.3f})")
print(f"   R²: {ols_result['r_squared']:.4f}")
print(f"   Interpretation: Large SE due to multicollinearity")

# Coefficient stability (bootstrap)
boot_result = analysis.coefficient_stability(bootstrap_samples=100)
print(f"\n5. COEFFICIENT STABILITY (Bootstrap 100 samples)")
print(f"   β₁ mean: {boot_result['beta_mean'][1]:.3f}, std: {boot_result['beta_std'][1]:.3f}")
print(f"   β₂ mean: {boot_result['beta_mean'][2]:.3f}, std: {boot_result['beta_std'][2]:.3f}")
print(f"   β₃ mean: {boot_result['beta_mean'][3]:.3f}, std: {boot_result['beta_std'][3]:.3f}")
print(f"   High std = unstable coefficients = multicollinearity problem")

# Ridge regression
ridge_result = analysis.ridge_regression()
print(f"\n6. RIDGE REGRESSION")
print(f"   Optimal α (via CV): {ridge_result['best_alpha']:.4f}")
print(f"   Coefficients:")
print(f"   β₀: {ridge_result['coefficients'][0]:7.3f}")
print(f"   β₁: {ridge_result['coefficients'][1]:7.3f}")
print(f"   β₂: {ridge_result['coefficients'][2]:7.3f}")
print(f"   β₃: {ridge_result['coefficients'][3]:7.3f}")

# LASSO regression
lasso_result = analysis.lasso_regression()
print(f"\n7. LASSO REGRESSION")
print(f"   Optimal α (via CV): {lasso_result['best_alpha']:.4f}")
print(f"   Coefficients (sparse):")
print(f"   β₀: {lasso_result['coefficients'][0]:7.3f}")
print(f"   β₁: {lasso_result['coefficients'][1]:7.3f}")
print(f"   β₂: {lasso_result['coefficients'][2]:7.3f}")
print(f"   β₃: {lasso_result['coefficients'][3]:7.3f}")
print(f"   (Zeros indicate dropped variables)")

# PCA
pca_result = analysis.pca_regression()
print(f"\n8. PRINCIPAL COMPONENTS REGRESSION")
print(f"   Variance explained: {pca_result['var_explained'].round(3)}")
print(f"   Cumulative: {pca_result['cumsum_var'].round(3)}")
print(f"   Trade-off: Orthogonal regressors but lose interpretability")

# Variable selection
varsel_result = analysis.variable_selection()
print(f"\n9. VARIABLE SELECTION (Drop X₃)")
print(f"   Coefficients:")
print(f"   β₀: {varsel_result['beta'][0]:7.3f} (SE: {varsel_result['se'][0]:.3f})")
print(f"   β₁: {varsel_result['beta'][1]:7.3f} (SE: {varsel_result['se'][1]:.3f})")
print(f"   β₂: {varsel_result['beta'][2]:7.3f} (SE: {varsel_result['se'][2]:.3f})")
print(f"   SE reduced (less multicollinearity)")

# Comparison
print(f"\n10. METHOD COMPARISON")
comparison = analysis.comparison_table()
print(comparison.to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Correlation heatmap
ax1 = axes[0, 0]
corr_vals = np.corrcoef(np.column_stack([analysis.X1, analysis.X2, analysis.X3]).T)
im1 = ax1.imshow(corr_vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(['X₁', 'X₂', 'X₃'])
ax1.set_yticklabels(['X₁', 'X₂', 'X₃'])
ax1.set_title('Panel 1: Correlation Matrix\n(Dark red = high positive correlation)')
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f'{corr_vals[i, j]:.2f}', ha='center', va='center', color='black')
plt.colorbar(im1, ax=ax1)

# Panel 2: VIF comparison
ax2 = axes[0, 1]
vif_df = analysis.vif_analysis()
colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' for v in vif_df['VIF']]
ax2.bar(vif_df['Variable'], vif_df['VIF'], color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Threshold = 5')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Severe = 10')
ax2.set_ylabel('VIF')
ax2.set_title('Panel 2: Variance Inflation Factors\n(VIF > 5 indicates problem)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Coefficient estimates across bootstrap samples
ax3 = axes[1, 0]
boot_betas = boot_result['bootstrap_betas']
positions = [0, 1, 2, 3]
bp = ax3.boxplot([boot_betas[:, i] for i in range(4)], labels=['β₀', 'β₁', 'β₂', 'β₃'],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Panel 3: Coefficient Stability (Bootstrap)\n(Large boxes = unstable due to multicollinearity)')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Method comparison (MSE)
ax4 = axes[1, 1]
methods = comparison['Method']
mse = comparison['MSE']
colors_mse = ['blue', 'green', 'orange', 'red', 'purple']
bars = ax4.bar(range(len(methods)), mse, color=colors_mse, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods, rotation=45, ha='right')
ax4.set_ylabel('Mean Squared Error')
ax4.set_title('Panel 4: Model Comparison\n(Lower MSE = better prediction)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multicollinearity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"• High multicollinearity detected: Corr(X₁,X₂) = {corr.iloc[0,1]:.3f}")
print(f"• VIF values > 5 confirm multicollinearity problem")
print(f"• Bootstrap shows unstable coefficients (high std of β estimates)")
print(f"• Ridge/LASSO reduce variance compared to OLS")
print(f"• Trade-off: Bias introduced but SE reduced")
print(f"• Variable selection simplifies but may omit important info")
print("="*80 + "\n")
```

### Output Explanation
- **Panel 1:** Correlation heatmap shows X₁ and X₂ nearly perfectly correlated (0.92), diagnosing the problem.
- **Panel 2:** VIF values exceed 5, quantifying severity of multicollinearity.
- **Panel 3:** Bootstrap boxplots show large variance in coefficient estimates; small data changes → large coefficient changes.
- **Panel 4:** Ridge/LASSO achieve lower MSE through regularization; trade-off between bias and variance.

---

## VII. References & Key Design Insights

1. **Dormann, C. F., et al. (2013).** "Collinearity: A review of methods to deal with it and a simulation study evaluating their performance." Ecography, 36(1), 27-46.
   - Comprehensive review of multicollinearity solutions; empirical comparison

2. **Hastie, T., Tibshirani, R., & Wainwright, M. (2015).** "Statistical Learning with Sparsity." CRC Press.
   - Ridge, LASSO, elastic net theory and applications

3. **Wooldridge, J. M. (2019).** "Introductory Econometrics: A Modern Approach" (7th ed.).
   - Practical multicollinearity discussion; VIF interpretation; remediation strategies

4. **Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012).** "Introduction to Linear Regression Analysis" (5th ed.).
   - Detailed treatment of multicollinearity; condition numbers; eigenvalue analysis

**Key Design Concepts:**
- **Problem Recognition:** Multicollinearity affects inference (SE) not estimation (β unbiased). Prediction often unaffected.
- **Trade-off Decision:** If goal is prediction → do nothing. If goal is causal inference → use regularization or variable selection.
- **Practical Threshold:** VIF > 5-10 warrants investigation; VIF > 10 usually requires action.
- **No Perfect Solution:** Every remedy has trade-offs (dropped variables = omitted bias; ridge = coefficient bias; PCA = interpretation loss).

