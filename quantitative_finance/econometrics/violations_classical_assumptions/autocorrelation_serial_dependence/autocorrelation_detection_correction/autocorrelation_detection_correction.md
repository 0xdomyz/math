# Autocorrelation: Detection, Testing & Serial Dependence Correction

## I. Concept Skeleton

**Definition:** Autocorrelation (serial correlation) occurs when error terms in a regression model are correlated across observations (typically across time). In $y_t = \beta_0 + \beta_1 x_t + \epsilon_t$, the assumption $\text{Cov}(\epsilon_t, \epsilon_{t-s}) = 0$ for $s \neq 0$ is violated, creating dependence between consecutive or lagged errors.

**Purpose:** Detect autocorrelation violations in time series and panel models, understand sources of serial dependence (omitted variables, model misspecification, persistence), apply appropriate corrections (HAC standard errors, dynamic models, generalized differencing), and ensure valid statistical inference.

**Prerequisites:** Time series analysis, OLS regression, lag structures, ARMA processes, hypothesis testing, matrix algebra.

---

## II. Comparative Framing

| **Aspect** | **OLS (No Autocorr)** | **OLS + HAC SE** | **First Differences** | **Dynamic (Lagged Y)** | **Feasible GLSAR** |
|-----------|----------------------|----------------|-----------------------|----------------------|-------------------|
| **Assumption** | $\text{Cov}(\epsilon_t, \epsilon_{t-s})=0$ | Allows autocorr | ∆yₜ serially indep | Captures dynamics | Model AR structure |
| **β Coefficient** | Unbiased if no autocorr | Unbiased | Different model | Biased if X endogenous | Efficient |
| **Efficiency** | BLUE if no autocorr | Inefficient | Loses level info | Varies | Efficient asymptotically |
| **Standard Errors** | Biased under autocorr | Consistent (robust) | Unbiased | Biased (endogeneity) | Correct if AR right |
| **Inference** | Invalid if autocorr | Valid (large N) | Valid but different | Problematic | Valid for large N |
| **Sample Size Req** | Any | Large N (>100) | Large N | Instrumental vars | Moderate |
| **Use Case** | Cross-section | Time series low freq | Time series | Dynamic panel | Quarterly/annual data |

---

## III. Examples & Counterexamples

### Example 1: Macroeconomic Forecasting (Persistent Shocks)
**Setup:**
- Quarterly GDP growth model: $\Delta \text{GDP}_t = 1.2 + 0.5 \Delta \text{GDP}_{t-1} + \epsilon_t$
- Observation: Oil price shocks have persistent effects (positive shock this quarter affects next quarter)
- Errors show autocorrelation: $\text{Cov}(\epsilon_t, \epsilon_{t-1}) = 0.65$ (strong positive)
- Sample: 80 quarters (20 years)

**Problem:**
- OLS β₁ estimate: 0.50 (unbiased)
- OLS SE: 0.04 (biased, too narrow; ignores autocorrelation)
- Implied 95% CI: [0.42, 0.58] (too tight, true coverage ~70%)
- Durbin-Watson: DW = 0.72 (< 2, confirms positive autocorrelation)

**Correction:**
- Newey-West HAC SE: 0.08 (twice as wide as OLS SE)
- Corrected 95% CI: [0.34, 0.66] (appropriate coverage)
- Alternative: Add lagged GDP as regressor to model dynamics directly

**Key Insight:** Autocorrelated errors cause OLS SE to be biased downward (usually). HAC SEs widen confidence intervals, preventing false precision claims.

### Example 2: Cross-Sectional Spillovers (Spatial/Network Autocorrelation)
**Setup:**
- House price model across 500 neighborhoods: $\text{Price}_i = \beta_0 + \beta_1 \text{Income}_i + \epsilon_i$
- Spatial structure: Nearby neighborhoods have correlated unobservables (local demand, amenities)
- Observation: Price residuals show spatial correlation; homes in wealthy clusters have similar unexplained prices

**Problems:**
- Model violates "independence across observations" assumption
- OLS SE underestimate spatial clustering effects
- Standard inference (t-tests) invalid
- Degree of freedom artificially inflated (effective sample size = 500, but effective n_eff ~ 250 due to clustering)

**Correction Options:**
- Cluster-robust SE: Group observations by location/network, adjust SE for within-cluster correlation
- Spatial AR model: Model cross-sectional autocorrelation explicitly
- Include spatial lags: Add weighted average of neighbors' prices as regressor

**Result:** Spatial autocorrelation corrections widen SE by 30-50%, making inferences more conservative.

### Example 3: Omitted Variable Bias (Indirect Autocorrelation)
**Setup:**
- Stock returns model: $R_t = 0.001 + 0.80 \times \text{Market}_t + \epsilon_t$
- True model should include momentum: $R_t = 0.001 + 0.80 \times \text{Market}_t + 0.30 \times R_{t-1} + u_t$
- Momentum (true effect 0.30) is omitted
- Observation: Residuals show strong positive autocorrelation (AR(1) ≈ 0.40)

**Mechanism:**
- When $R_{t-1}$ is high, $R_t$ tends to be high (momentum)
- Model without momentum captures this via positive residual autocorrelation
- Residuals are NOT truly autocorrelated; they're correlated because of omitted variable

**Consequence:**
- OLS β₁ biased (market beta captures momentum effect)
- SE biased downward (autocorrelated residuals)
- Inference doubly wrong (wrong point estimate, wrong uncertainty)

**Solution:**
- Include lagged dependent variable (dynamic model): $R_t = 0.80 \times \text{Market}_t + 0.30 \times R_{t-1} + u_t$
- Now residuals become white noise (uncorrelated)
- Both β and SE correct

**Key Insight:** Sometimes autocorrelation signals model misspecification (omitted variables), not just a nuisance to correct.

---

## IV. Layer Breakdown

```
AUTOCORRELATION FRAMEWORK

┌─────────────────────────────────────────────────────────┐
│         TIME SERIES REGRESSION MODEL                    │
│    yₜ = β₀ + β₁x₁,ₜ + ... + βₖxₖ,ₜ + εₜ               │
│                                                         │
│    Classical Independence Assumption:                  │
│    Cov(εₜ, εₛ) = 0 for all t ≠ s (NO autocorrelation) │
│    ↓                                                    │
│    Ε[εₜ²|Xₜ] = σ² (constant variance, independent)   │
└────────────────┬──────────────────────────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │  AUTOCORRELATION VIOLATION     │
    │  (Serial Dependence: AR, MA)   │
    │  Cov(εₜ, εₜ₋ₛ) ≠ 0 (s>0)      │
    └────────────┬──────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │  TYPES OF AUTOCORRELATION                     │
    │                                               │
    │  1. AR(1) - Autoregressive:                  │
    │     εₜ = ρ·εₜ₋₁ + νₜ                        │
    │     ├─ Positive ρ>0: Shocks persist         │
    │     ├─ Negative ρ<0: Oscillating            │
    │     └─ |ρ| close to 1: Highly persistent   │
    │                                              │
    │  2. MA(1) - Moving Average:                 │
    │     εₜ = νₜ + θ·νₜ₋₁                        │
    │     └─ Shocks have lagged effects           │
    │                                              │
    │  3. ARMA(p,q):                              │
    │     Combination of AR and MA components     │
    │                                              │
    │  4. Spatial Autocorrelation:                │
    │     εᵢ correlated with εⱼ if i,j nearby    │
    │     (not just temporal)                     │
    └────────────┬────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │  CONSEQUENCES FOR OLS                         │
    │                                               │
    │  ✓ β̂ still UNBIASED (conditionally on X)    │
    │    (point estimates valid if X exogenous)   │
    │                                               │
    │  ✗ SE(β̂) BIASED:                           │
    │    └─ Usually too small (optimistic)        │
    │    └─ More severe if ρ close to 1          │
    │    └─ Positive ρ → SE understated more      │
    │    └─ Magnitude: SE_true ≈ SE_OLS × √(bias_factor)
    │                                              │
    │  ✗ β̂ NO LONGER BLUE:                        │
    │    └─ Not minimum variance                   │
    │    └─ Inefficient relative to GLS/GLSAR    │
    │                                              │
    │  ✗ F-tests, Wald tests invalid             │
    │                                              │
    │  ✗ If X contains lagged Y:                  │
    │    └─ β̂ becomes BIASED (Nickell bias)      │
    │    └─ Problem in dynamic panels             │
    └────────────┬────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │  DETECTION METHODS                           │
    │                                               │
    │  1. GRAPHICAL:                               │
    │     ├─ Plot εₜ over time (visual pattern)    │
    │     ├─ ACF (Autocorrelation Function)       │
    │     │  └─ Shows correlation at lags 1,2,... │
    │     └─ PACF (Partial ACF)                   │
    │        └─ Shows lag structure AR(p)/MA(q)  │
    │                                              │
    │  2. DURBIN-WATSON TEST:                     │
    │     DW = Σ(εₜ - εₜ₋₁)² / Σεₜ²              │
    │     ├─ DW ≈ 2: No autocorrelation           │
    │     ├─ DW < 2: Positive autocorr (ρ>0)    │
    │     ├─ DW > 2: Negative autocorr (ρ<0)    │
    │     ├─ DW ≈ 0: ρ ≈ 1 (unit root)           │
    │     └─ Approximation: DW ≈ 2(1-ρ̂)        │
    │                                              │
    │  3. LJUNG-BOX TEST:                         │
    │     Q-statistic = n(n+2)Σ(ρ̂²ₖ/(n-k))      │
    │     H₀: No autocorrelation up to lag K      │
    │     Q ~ χ²(K) under H₀                      │
    │                                              │
    │  4. BREUSCH-GODFREY TEST:                   │
    │     Auxiliary regression: εₜ on Xₜ, εₜ₋₁... │
    │     LM = n·R² ~ χ²(p) under H₀             │
    │     └─ More powerful than DW, handles MA   │
    │                                              │
    │  5. VISUAL ACF/PACF:                        │
    │     ├─ ACF decays slowly → AR process      │
    │     ├─ PACF cuts off at lag p → AR(p)      │
    │     ├─ ACF cuts off at lag q → MA(q)       │
    │     └─ Both decay → ARMA(p,q)              │
    └────────────┬────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────┐
    │  CORRECTION METHODS                          │
    │                                               │
    │  1. HAC (Heteroskedasticity & Autocorr       │
    │     Consistent) SE:                          │
    │     ├─ Newey-West: Weights decrease with lag│
    │     ├─ Asymptotically consistent            │
    │     ├─ No model specification needed        │
    │     └─ Robust to unknown AR(∞) structure    │
    │                                              │
    │  2. FIRST DIFFERENCES:                      │
    │     ├─ Model: ∆yₜ = ∆xₜ·β + ∆εₜ           │
    │     ├─ Removes level autocorr (if I(1))    │
    │     ├─ Loses long-run info                 │
    │     └─ Useful for cointegration testing    │
    │                                              │
    │  3. DYNAMIC MODEL (Add Lags):               │
    │     ├─ yₜ = β₀ + ρ·yₜ₋₁ + γ·xₜ + εₜ      │
    │     ├─ Captures serial dependence directly │
    │     ├─ Requires lagged dependent var       │
    │     └─ Nickell bias if fixed effects + lag│
    │                                              │
    │  4. GENERALIZED LEAST SQUARES (GLS):        │
    │     ├─ If AR(1) known: ρ given            │
    │     ├─ Transform: yₜ* = yₜ - ρ·yₜ₋₁       │
    │     ├─ Fit OLS on transformed model       │
    │     └─ Efficient (BLUE)                    │
    │                                              │
    │  5. FEASIBLE GLS (GLSAR):                   │
    │     ├─ Stage 1: Estimate ρ̂ from residuals │
    │     ├─ Stage 2: Transform using ρ̂        │
    │     ├─ Stage 3: OLS on transformed        │
    │     └─ Efficient asymptotically            │
    │                                              │
    │  6. INSTRUMENTAL VARIABLES (Dynamic):       │
    │     ├─ If yₜ₋₁ on RHS, endogenous        │
    │     ├─ Use lags as instruments: yₜ₋₂,yₜ₋₃ │
    │     ├─ Arellano-Bond estimator            │
    │     └─ Consistent in dynamic panel data   │
    └──────────────────────────────────────────────┘
```

---

## V. Mathematical Framework

### Autocorrelation Definition

In time series model:
$$y_t = \beta_0 + \beta_1 x_t + \epsilon_t, \quad t = 1, \ldots, T$$

**Classical Independence Assumption:**
$$\text{Cov}(\epsilon_t, \epsilon_s) = 0 \quad \forall t \neq s$$

**Autocorrelation Violation - AR(1) Process:**
$$\epsilon_t = \rho \epsilon_{t-1} + \nu_t, \quad |\rho| < 1$$

where $\nu_t \sim \text{iid}(0, \sigma_\nu^2)$.

### Durbin-Watson Statistic

$$DW = \frac{\sum_{t=2}^T (\hat{\epsilon}_t - \hat{\epsilon}_{t-1})^2}{\sum_{t=1}^T \hat{\epsilon}_t^2}$$

**Relationship to AR(1) coefficient:**
$$DW \approx 2(1 - \rho)$$

- $\rho = 0$ (no autocorr): DW ≈ 2
- $\rho = 1$ (unit root): DW ≈ 0
- $\rho = -1$ (perfect negative): DW ≈ 4

**Limitations:** DW test inconclusive if X includes lagged Y or if higher-order AR processes present.

### Newey-West HAC Standard Errors

To account for autocorrelation and heteroskedasticity:

$$\text{Var}_{\text{NW}}(\hat{\beta}) = (X'X)^{-1} \left[ \sum_{t=1}^T \hat{\epsilon}_t^2 x_t x_t' + 2 \sum_{j=1}^{L} w_{L}(j) \sum_{t=j+1}^T \hat{\epsilon}_t \hat{\epsilon}_{t-j} x_t x_{t-j}' \right] (X'X)^{-1}$$

where:
- $w_L(j)$ = weight function (decreases with lag $j$, e.g., Bartlett: $w(j) = 1 - \frac{j}{L+1}$)
- $L$ = lag truncation (typically $L = \lfloor 4(T/100)^{2/9} \rfloor$)
- First term captures contemporaneous variance
- Second term captures autocovariances at lags 1 to $L$

### Breusch-Godfrey Test

**Null hypothesis:** $H_0: \text{No autocorrelation up to lag } p$

**Procedure:**
1. Estimate OLS: $\hat{\epsilon}_t = y_t - \hat{y}_t$
2. Auxiliary regression: $\hat{\epsilon}_t = \alpha_0 + \sum_{j=1}^p \phi_j \hat{\epsilon}_{t-j} + X_t \gamma + u_t$
3. Compute $R^2$ from auxiliary
4. Test statistic: $LM = T \cdot R^2 \sim \chi^2(p)$ under $H_0$

This test is more general than Durbin-Watson (handles MA, higher-order AR).

### Generalized Least Squares (GLS) for AR(1)

If true autocorrelation structure is AR(1) with known $\rho$:

**Transform model** by quasi-differencing:
$$y_t^* = y_t - \rho y_{t-1}$$
$$x_t^* = x_t - \rho x_{t-1}$$

**OLS on transformed model** gives BLUE estimator:
$$\hat{\beta}_{\text{GLS}} = \left( \sum_{t=2}^T x_t^* (x_t^*)' \right)^{-1} \sum_{t=2}^T x_t^* y_t^*$$

**Result:** Var$(\hat{\beta}_{\text{GLS}}) < $ Var$(\hat{\beta}_{\text{OLS}})$ (more efficient).

---

## VI. Python Mini-Project: Autocorrelation Detection & Correction

### Objective
Demonstrate:
1. Generating time series with autocorrelated errors
2. Detecting autocorrelation via DW, Breusch-Godfrey, ACF/PACF
3. Comparing OLS vs HAC vs GLS vs Dynamic methods
4. Showing efficiency gains and confidence interval improvements

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# AUTOCORRELATION DETECTION AND CORRECTION
# ============================================================================

class AutocorrelationAnalysis:
    """
    Comprehensive autocorrelation detection and correction
    """
    
    def __init__(self, T=150, rho=0.65):
        """
        Generate time series data with AR(1) errors
        y_t = 5 + 2*X_t + ε_t, where ε_t = ρ*ε_{t-1} + ν_t
        
        Parameters:
        -----------
        T: Number of time periods
        rho: AR(1) coefficient (autocorrelation strength)
        """
        self.T = T
        self.rho = rho
        
        # Generate X (exogenous regressor)
        self.X = np.random.normal(5, 2, T)
        
        # Generate autocorrelated errors
        nu = np.random.normal(0, 1.0, T)
        epsilon = np.zeros(T)
        epsilon[0] = nu[0]
        for t in range(1, T):
            epsilon[t] = rho * epsilon[t-1] + nu[t]
        
        self.epsilon = epsilon
        
        # Generate dependent variable
        self.y = 5 + 2 * self.X + epsilon
        
        # Design matrix
        self.X_design = np.column_stack([np.ones(T), self.X])
        
    def fit_ols(self):
        """Fit OLS and compute naive (incorrect) standard errors"""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X_design, self.y)
        
        self.beta_ols = model.coef_
        self.y_pred = model.predict(self.X_design)
        self.residuals = self.y - self.y_pred
        
        # Naive variance (assumes independence)
        n, k = self.X_design.shape
        mse = np.sum(self.residuals**2) / (n - k)
        self.X_prime_X_inv = np.linalg.inv(self.X_design.T @ self.X_design)
        
        # OLS standard errors (BIASED under autocorrelation)
        self.var_ols_naive = mse * self.X_prime_X_inv
        self.se_ols_naive = np.sqrt(np.diag(self.var_ols_naive))
        
        return self.beta_ols, self.se_ols_naive
    
    def durbin_watson_test(self):
        """Compute Durbin-Watson statistic"""
        dw = np.sum(np.diff(self.residuals)**2) / np.sum(self.residuals**2)
        
        # Approximate AR(1) coefficient
        rho_est = 1 - (dw / 2)
        
        # Approximate standard error of rho
        rho_se = np.sqrt((1 - rho_est**2) / self.T)
        
        return {
            'dw_statistic': dw,
            'rho_estimated': rho_est,
            'rho_se': rho_se,
            'conclusion': 'Positive autocorr' if dw < 2 else 'Negative autocorr' if dw > 2 else 'No autocorr'
        }
    
    def breusch_godfrey_test(self, p=1):
        """
        Breusch-Godfrey test for autocorrelation
        More powerful than DW; handles MA terms
        """
        # Auxiliary regression: ε̂_t = α + φ*ε̂_{t-1} + X*γ + u_t
        y_aux = self.residuals[p:]
        X_aux = []
        
        for t in range(p, self.T):
            row = [1] + list(self.X_design[t, :]) + list(self.residuals[t-1:t-p:-1])
            X_aux.append(row)
        
        X_aux = np.array(X_aux)
        
        # Fit auxiliary regression
        aux_model = LinearRegression(fit_intercept=False)
        aux_model.fit(X_aux, y_aux)
        
        y_aux_pred = aux_model.predict(X_aux)
        ss_res = np.sum((y_aux - y_aux_pred)**2)
        ss_tot = np.sum((y_aux - np.mean(y_aux))**2)
        r2_aux = 1 - (ss_res / ss_tot)
        
        # LM statistic
        lm_stat = (self.T - p) * r2_aux
        
        # p-value
        p_value = 1 - stats.chi2.cdf(lm_stat, p)
        
        return {
            'test_statistic': lm_stat,
            'p_value': p_value,
            'r2_auxiliary': r2_aux,
            'reject_independence': p_value < 0.05
        }
    
    def newey_west_se(self, lags=None):
        """
        Newey-West HAC standard errors
        Robust to autocorrelation and heteroskedasticity
        """
        if lags is None:
            # Automatic lag selection
            lags = int(np.ceil(4 * (self.T / 100)**(2/9)))
        
        # Bartlett weights
        def bartlett_weights(lag, max_lag):
            return 1 - lag / (max_lag + 1)
        
        # Initialize long-run variance
        omega = np.zeros_like(self.X_prime_X_inv)
        
        # Contemporaneous term
        for t in range(self.T):
            omega += self.residuals[t]**2 * np.outer(self.X_design[t], self.X_design[t])
        
        # Autocovariance terms
        for lag in range(1, lags + 1):
            weight = bartlett_weights(lag, lags)
            for t in range(lag, self.T):
                cross_prod = self.residuals[t] * self.residuals[t-lag]
                omega += 2 * weight * cross_prod * np.outer(self.X_design[t], self.X_design[t-lag])
        
        # Variance-covariance matrix
        var_nw = self.X_prime_X_inv @ omega @ self.X_prime_X_inv / self.T
        self.se_nw = np.sqrt(np.diag(var_nw))
        
        return self.se_nw
    
    def feasible_gls_ar1(self):
        """
        Feasible GLS for AR(1) process
        Stage 1: Estimate ρ from residuals
        Stage 2: Quasi-difference and apply OLS
        """
        # Stage 1: Estimate ρ using DW approximation or direct estimation
        dw_result = self.durbin_watson_test()
        rho_est = dw_result['rho_estimated']
        
        # Ensure stationarity
        rho_est = np.clip(rho_est, -0.99, 0.99)
        
        # Stage 2: Quasi-difference (GLS transformation)
        # First observation: use original
        y_gls = np.zeros(self.T)
        X_gls = np.zeros_like(self.X_design)
        
        y_gls[0] = self.y[0]
        X_gls[0] = self.X_design[0]
        
        # Remaining observations: quasi-differenced
        for t in range(1, self.T):
            y_gls[t] = self.y[t] - rho_est * self.y[t-1]
            X_gls[t] = self.X_design[t] - rho_est * self.X_design[t-1]
        
        # Stage 3: OLS on transformed model
        model_gls = LinearRegression(fit_intercept=False)
        model_gls.fit(X_gls, y_gls)
        
        self.beta_gls = model_gls.coef_
        y_pred_gls = model_gls.predict(X_gls)
        residuals_gls = y_gls - y_pred_gls
        
        # SE for GLS
        n, k = X_gls.shape
        mse_gls = np.sum(residuals_gls**2) / (n - k)
        X_gls_prime_X = X_gls.T @ X_gls
        var_gls = mse_gls * np.linalg.inv(X_gls_prime_X)
        
        self.se_gls = np.sqrt(np.diag(var_gls))
        
        return self.beta_gls, self.se_gls
    
    def dynamic_model(self):
        """
        Add lagged dependent variable to capture dynamics
        y_t = β_0 + β_1*X_t + ρ*y_{t-1} + ε_t
        """
        # Create lagged y
        y_lag = np.concatenate([[self.y[0]], self.y[:-1]])
        
        # Design matrix with lagged y
        X_dynamic = np.column_stack([np.ones(self.T), self.X, y_lag])
        
        # OLS
        model_dyn = LinearRegression(fit_intercept=False)
        model_dyn.fit(X_dynamic, self.y)
        
        self.beta_dynamic = model_dyn.coef_
        y_pred_dyn = model_dyn.predict(X_dynamic)
        residuals_dyn = self.y - y_pred_dyn
        
        # SE for dynamic model
        n, k = X_dynamic.shape
        mse_dyn = np.sum(residuals_dyn**2) / (n - k)
        X_dyn_prime_X = X_dynamic.T @ X_dynamic
        var_dyn = mse_dyn * np.linalg.inv(X_dyn_prime_X)
        
        self.se_dynamic = np.sqrt(np.diag(var_dyn))
        
        return self.beta_dynamic, self.se_dynamic
    
    def summary_table(self):
        """Create comparison table"""
        comparison = pd.DataFrame({
            'Method': ['OLS (Naive)', 'OLS (Newey-West)', 'FGLS (AR1)', 'Dynamic (Lagged Y)'],
            'β₁ Estimate': [
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_ols[1]:.4f}",
                f"{self.beta_gls[1]:.4f}",
                f"{self.beta_dynamic[1]:.4f}"
            ],
            'SE(β₁)': [
                f"{self.se_ols_naive[1]:.4f}",
                f"{self.se_nw[1]:.4f}",
                f"{self.se_gls[1]:.4f}",
                f"{self.se_dynamic[1]:.4f}"
            ],
            '95% CI for β₁': [
                f"[{self.beta_ols[1]-1.96*self.se_ols_naive[1]:.4f}, {self.beta_ols[1]+1.96*self.se_ols_naive[1]:.4f}]",
                f"[{self.beta_ols[1]-1.96*self.se_nw[1]:.4f}, {self.beta_ols[1]+1.96*self.se_nw[1]:.4f}]",
                f"[{self.beta_gls[1]-1.96*self.se_gls[1]:.4f}, {self.beta_gls[1]+1.96*self.se_gls[1]:.4f}]",
                f"[{self.beta_dynamic[1]-1.96*self.se_dynamic[1]:.4f}, {self.beta_dynamic[1]+1.96*self.se_dynamic[1]:.4f}]"
            ]
        })
        
        return comparison


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("AUTOCORRELATION ANALYSIS: DETECTION & CORRECTION")
print("="*80)

# Initialize
analysis = AutocorrelationAnalysis(T=150, rho=0.65)

# Fit OLS
analysis.fit_ols()
print(f"\n1. OLS ESTIMATION (with AR(1) errors, ρ=0.65)")
print(f"   β₀ (true=5): {analysis.beta_ols[0]:.4f}")
print(f"   β₁ (true=2): {analysis.beta_ols[1]:.4f}")

# Durbin-Watson Test
dw_result = analysis.durbin_watson_test()
print(f"\n2. DURBIN-WATSON TEST")
print(f"   DW Statistic: {dw_result['dw_statistic']:.4f}")
print(f"   (DW ≈ 2 if no autocorr, < 2 if positive)")
print(f"   Estimated ρ: {dw_result['rho_estimated']:.4f} (true ρ = 0.65)")
print(f"   Conclusion: {dw_result['conclusion']}")

# Breusch-Godfrey Test
bg_result = analysis.breusch_godfrey_test(p=1)
print(f"\n3. BREUSCH-GODFREY TEST")
print(f"   LM Statistic: {bg_result['test_statistic']:.4f}")
print(f"   p-value: {bg_result['p_value']:.6f}")
print(f"   Conclusion: {'Reject independence (autocorr detected)' if bg_result['reject_independence'] else 'No autocorr detected'}")

# Standard Errors Comparison
analysis.newey_west_se()
print(f"\n4. STANDARD ERRORS COMPARISON")
print(f"   {'':30} OLS Naive  Newey-West")
print(f"   {'SE(β₀)':30} {analysis.se_ols_naive[0]:9.4f}  {analysis.se_nw[0]:9.4f}")
print(f"   {'SE(β₁)':30} {analysis.se_ols_naive[1]:9.4f}  {analysis.se_nw[1]:9.4f}")
print(f"   SE Inflation Factor (β₁):     1.00x      {analysis.se_nw[1]/analysis.se_ols_naive[1]:.2f}x")

# FGLS
analysis.feasible_gls_ar1()
print(f"\n5. FEASIBLE GLS (AR(1))")
print(f"   β₀: {analysis.beta_gls[0]:.4f} (SE: {analysis.se_gls[0]:.4f})")
print(f"   β₁: {analysis.beta_gls[1]:.4f} (SE: {analysis.se_gls[1]:.4f})")
print(f"   Efficiency gain: {(analysis.se_ols_naive[1]/analysis.se_gls[1] - 1)*100:.1f}%")

# Dynamic model
analysis.dynamic_model()
print(f"\n6. DYNAMIC MODEL (Lagged Dependent Variable)")
print(f"   β₀: {analysis.beta_dynamic[0]:.4f}")
print(f"   β₁ (X coeff): {analysis.beta_dynamic[1]:.4f}")
print(f"   ρ (lagged Y coeff): {analysis.beta_dynamic[2]:.4f}")

# Comparison table
print(f"\n7. COMPREHENSIVE COMPARISON")
print(analysis.summary_table().to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Time series of residuals (visual autocorrelation)
ax1 = axes[0, 0]
ax1.plot(analysis.residuals, 'b-', linewidth=1, alpha=0.7, label='OLS Residuals')
ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax1.fill_between(range(len(analysis.residuals)), 
                 -2*np.std(analysis.residuals),
                 2*np.std(analysis.residuals),
                 alpha=0.1, color='blue')
ax1.set_xlabel('Time Period (t)')
ax1.set_ylabel('Residuals (ε̂ₜ)')
ax1.set_title('Panel 1: Residual Time Series\n(Persistent patterns indicate autocorrelation)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: ACF (Autocorrelation Function)
ax2 = axes[0, 1]
plot_acf(analysis.residuals, lags=20, ax=ax2, title='Panel 2: ACF of Residuals\n(Slow decay = strong autocorr)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

# Panel 3: Lagged residual scatter (ε̂ₜ vs ε̂ₜ₋₁)
ax3 = axes[1, 0]
ax3.scatter(analysis.residuals[:-1], analysis.residuals[1:], alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
# Fit line to show correlation
z = np.polyfit(analysis.residuals[:-1], analysis.residuals[1:], 1)
p = np.poly1d(z)
x_line = np.linspace(analysis.residuals[:-1].min(), analysis.residuals[:-1].max(), 100)
ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'ρ̂ = {z[0]:.3f}')
ax3.set_xlabel('ε̂ₜ₋₁ (Lagged Residual)')
ax3.set_ylabel('ε̂ₜ (Current Residual)')
ax3.set_title('Panel 3: Residual Autocorrelation\n(Positive slope = positive autocorr)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Confidence interval comparison
ax4 = axes[1, 1]
methods = ['OLS\n(Naive)', 'OLS\n(Newey-West)', 'FGLS', 'Dynamic']
se_vals = [analysis.se_ols_naive[1], analysis.se_nw[1], analysis.se_gls[1], analysis.se_dynamic[1]]
betas = [analysis.beta_ols[1], analysis.beta_ols[1], analysis.beta_gls[1], analysis.beta_dynamic[1]]

ci_lower = [b - 1.96*se for b, se in zip(betas, se_vals)]
ci_upper = [b + 1.96*se for b, se in zip(betas, se_vals)]

y_pos = np.arange(len(methods))
ax4.errorbar(y_pos, betas,
             yerr=[np.array(betas) - np.array(ci_lower),
                   np.array(ci_upper) - np.array(betas)],
             fmt='o', markersize=8, capsize=5, capthick=2, color='blue')
ax4.axvline(x=2.0, color='green', linestyle='--', linewidth=2, label='True β₁ = 2')
ax4.set_xticks(y_pos)
ax4.set_xticklabels(methods)
ax4.set_ylabel('β₁ Estimate with 95% CI')
ax4.set_title('Panel 4: Confidence Intervals Comparison\n(OLS too narrow; Newey-West/FGLS appropriate)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"• Naive OLS SE underestimate by ~{(1 - analysis.se_ols_naive[1]/analysis.se_nw[1])*100:.0f}%")
print(f"• Newey-West SE are ~{analysis.se_nw[1]/analysis.se_ols_naive[1]:.2f}x wider (correct)")
print(f"• FGLS achieves ~{(analysis.se_ols_naive[1]/analysis.se_gls[1] - 1)*100:.0f}% efficiency gain")
print(f"• DW = {dw_result['dw_statistic']:.3f} (< 2, confirms positive autocorr)")
print(f"• Breusch-Godfrey p-value: {bg_result['p_value']:.4f} (strongly rejects independence)")
print("="*80 + "\n")
```

### Output Explanation
- **Panel 1:** Residual time series shows persistent clustering (positive values persist, then negative values persist).
- **Panel 2:** ACF decays slowly (doesn't cut off sharply), indicating strong AR(1) process.
- **Panel 3:** Lagged scatter shows strong positive relationship; OLS β̂ᵢₜ and β̂ᵢₜ₋₁ correlated.
- **Panel 4:** OLS CI too narrow. Newey-West/FGLS widen intervals, preventing false significance.

---

## VII. References & Key Design Insights

1. **Newey, W. K., & West, K. D. (1987).** "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix." Econometrica, 55(3), 703-708.
   - HAC standard errors; Bartlett weights; robust to unknown lag structure

2. **Breusch, T. S., & Godfrey, L. G. (1986).** "Data transformation tests." Economic Reviews, 4(2), 171-233.
   - Generalized test for autocorrelation; handles MA terms unlike Durbin-Watson

3. **Wooldridge, J. M. (2015).** "Introductory Econometrics: A Modern Approach" (6th ed.).
   - Comprehensive coverage; dynamic models; Nickell bias; AR vs MA

4. **Andrews, D. W. (1991).** "Heteroskedasticity and autocorrelation consistent covariance matrix estimation." Econometrica, 59(3), 817-858.
   - Optimal bandwidth selection; data-dependent lag truncation

**Key Design Concepts:**
- **Persistence:** AR(1) with ρ close to 1 causes severe SE bias (proportional to $\sqrt{1+\rho}$)
- **Trade-off:** HAC robust but inefficient; FGLS efficient but requires correct model specification
- **Model Misspecification:** Often autocorrelation signals omitted dynamics, not just a nuisance (adding lagged Y may eliminate autocorrelation entirely)
- **Nickell Bias:** If Y_lagged on RHS with fixed effects, OLS biased (worse in short panels); requires IV methods

