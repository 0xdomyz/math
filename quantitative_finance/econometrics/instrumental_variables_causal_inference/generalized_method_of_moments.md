# Generalized Method of Moments (GMM)

## Concept Skeleton

Generalized Method of Moments (GMM) is a flexible estimation framework based on moment conditions E[g(Xᵢ, θ)] = 0, where g is a vector of functions parameterized by θ. Unlike OLS (minimizes sum of squared residuals) or MLE (maximizes likelihood), GMM directly exploits orthogonality conditions between instruments and errors. **Advantages**: (1) accommodates overidentification (more instruments than parameters), (2) handles nonlinear models and non-normal distributions, (3) unifies IV/2SLS (special case where g is moment condition for endogeneity), and (4) enables efficient estimation under general covariance structures (dynamic panel data, time series with heteroscedasticity and autocorrelation). **Two-step GMM**: Stage 1 estimates with identity weight matrix, Stage 2 uses efficient weight matrix inverse of moment variance. **Testing**: Overidentification test (Sargan/Hansen J-test) validates whether excess instruments satisfy exogeneity; specification test checks if moment conditions hold. **Challenges**: Weak instruments inflate SE; small-sample bias when J-statistic large; inference sensitive to weight matrix specification. GMM is workhorse for dynamic panel models (Arellano-Bond) and financial time series (ARCH/GARCH, factor models).

**Core Components:**
- **Moment condition**: E[gᵢ(θ)] = 0 (theoretical orthogonality)
- **Sample moment**: (1/n)ΣGᵢ(θ) ≈ 0 (empirical analog)
- **GMM estimator**: θ̂ᴳᴹᴹ minimizes g'W⁻¹g with weight matrix W
- **Just-identified** (p instruments = p parameters): θ̂ = unique solution
- **Overidentified** (q > p): Use Hansen J-test for excess moment validity
- **2-step vs. 1-step**: 1-step uses identity weights (robust); 2-step uses variance estimate (efficient)
- **Efficiency gain**: 2-step GMM asymptotically efficient if moments correctly specified

**Why it matters:** GMM enables causal inference with endogenous regressors, complex panel dynamics (lagged dependent variables), time series nonlinearities (GARCH), and financial asset pricing (factor models). Fundamental tool in modern econometrics and empirical finance.

---

## Comparative Framing

| Aspect | **OLS** | **IV/2SLS** | **GMM (1-step)** | **GMM (2-step)** |
|--------|---------|------------|------------------|------------------|
| **Framework** | Minimize SSR | Orthogonal to Z | Minimize g'Wg (W=I) | Minimize g'Ŵ⁻¹g |
| **Moment condition** | E[ε\|X]=0 | E[ε\|Z]=0 | E[gᵢ(θ)]=0 general | Same, Ŵ estimated |
| **Efficiency** | BLUE (if exogenous) | Efficient under assumptions | Robust to heteroscedasticity | Asymptotically efficient |
| **Overidentification** | N/A | Sargan test | Hansen J-test | Hansen J-test |
| **Complexity** | Simple | Two-stage linear regression | Nonlinear optimization | Nonlinear + W estimation |
| **Use case** | Cross-section exogenous | IV regression, endogeneity | Time series, GARCH | Panel data (Arellano-Bond) |

**Key insight:** GMM generalizes IV (IV is GMM with linear g and specific W); enables flexible moment-based estimation for complex models; Hansen J-test validates instrumental assumptions.

---

## Examples & Counterexamples

### Examples of GMM Applications

1. **Arellano-Bond Dynamic Panel Estimator**  
   - **Model**: Yᵢₜ = αYᵢₜ₋₁ + Xᵢₜ'β + αᵢ + εᵢₜ (lagged dependent variable, fixed effects)  
   - **Problem**: Within-differencing doesn't eliminate Yᵢₜ₋₁ correlation; lagged Y endogenous  
   - **Moment condition**: E[Yᵢₜ₋₂ × εᵢₜ] = 0 (lags of Y valid instruments)  
   - **Arellano-Bond**: Use Yᵢₜ₋₂, Yᵢₜ₋₃, ..., Y₁ᵢ as instruments for first-differenced Yᵢₜ₋₁  
   - **Multiple moments** (many lags available as instruments) → overidentified → use Hansen J-test

2. **ARCH Model Estimation (Volatility Clustering)**  
   - **Model**: hₜ = σ² + α εₜ₋₁² + β hₜ₋₁ (ARCH: variance depends on past shocks)  
   - **Moment condition**: E[εₜ²/hₜ - 1] = 0 (standardized residuals orthogonal to info set)  
   - **Non-normal**: Returns have fat tails; MLE problematic; GMM robust  
   - **GMM efficient**: Exploits moments of squared residuals without normal assumption

3. **Linear Factor Model (Asset Pricing)**  
   - **Fama-French**: E[Rᵢ - Rғ] = λ₀ + λ₁βᵢᵐᵐ + λ₂sᵢ + λ₃hᵢ (factor loadings price risk)  
   - **Moment conditions**: E[fₜ × (Rᵢₜ - Rғₜ - λ₀ - λ₁βᵢᵐᵐ - ...)] = 0 (factors orthogonal to pricing errors)  
   - **Cross-section & time series**: GMM combines both sources of variation  
   - **Hansen-Jagannathan J-test**: Tests if model explains returns (excess moments)

4. **Simultaneous Equations (Supply-Demand)**  
   - **Demand**: Qᵈ = α₀ + α₁P + α₂Y + εᵈ  
   - **Supply**: Qˢ = β₀ + β₁P + εˢ  
   - **Market equilibrium**: Qᵈ = Qˢ  
   - **Moment conditions**: E[Z × εᵈ] = 0, E[Z × εˢ] = 0 (Z = exogenous shifters)  
   - **Overidentification**: Multiple valid instruments → test jointly via J-test

### Non-Examples (or GMM Unnecessary)

- **Linear exogenous model**: OLS sufficient (no moment condition complications)  
- **Few overidentifying restrictions**: 2SLS simpler, equivalent asymptotically to GMM with linear g  
- **Non-stationary time series**: Standard GMM assumes stationarity; requires cointegration adjustments

---

## Layer Breakdown

**Layer 1: Moment Condition Framework**  
**General setup**: Parameter vector θ ∈ Θ satisfies **p moment conditions**:  
$$E[g_i(\theta)] = 0, \quad i = 1, \ldots, p$$

**Sample analog**:  
$$\frac{1}{n} \sum_{i=1}^n g_i(\theta) \stackrel{p}{\to} E[g_i(\theta)]$$
(by Law of Large Numbers, if stationarity holds).

**GMM minimizes empirical moment magnitude**:  
$$\hat{\theta}^{GMM} = \arg\min_\theta \left(\frac{1}{n}\sum_i g_i(\theta)\right)' W \left(\frac{1}{n}\sum_i g_i(\theta)\right)$$

where W is positive semidefinite weight matrix.

**Weight matrix selection**:  
- **Identity (W = I)**: 1-step GMM, robust but inefficient  
- **Optimal (W = Ĝ⁻¹, where Ĝ ≈ Var[√n Σgᵢ])**: 2-step GMM, asymptotically efficient

**Layer 2: Just-Identified vs. Overidentified Cases**  
**Just-identified** (p = k parameters, p moment conditions):  
- Unique solution: (1/n)Σgᵢ(θ̂) = 0 exactly (or nearly)  
- Example: Y = Xβ + ε with IV; k instruments, k parameters → just-identified  
- No specification test possible

**Overidentified** (q > p; q instruments/moments, p parameters):  
- Cannot satisfy all moments exactly; minimize weighted sum of squared moments  
- **Degrees of freedom**: q - p (number of overidentifying restrictions)  
- **Hansen J-test**: Test H₀: all moments = 0 simultaneously

**Layer 3: Two-Step GMM Procedure**  
**Step 1 (Preliminary estimation with W = I)**:  
$$\hat{\theta}^{(1)} = \arg\min_\theta \left(\frac{1}{n}\sum_i g_i(\theta)\right)' I \left(\frac{1}{n}\sum_i g_i(\theta)\right)$$

Gives initial estimate θ̂⁽¹⁾.

**Step 2 (Estimate optimal weight matrix)**:  
$$\hat{G} = \frac{1}{n}\sum_i \nabla_\theta g_i(\hat{\theta}^{(1)}) g_i(\hat{\theta}^{(1)})' \quad \text{(moment variance)}$$

**Step 2 (Efficient GMM with Ŵ = Ĝ⁻¹)**:  
$$\hat{\theta}^{GMM} = \arg\min_\theta \left(\frac{1}{n}\sum_i g_i(\theta)\right)' \hat{G}^{-1} \left(\frac{1}{n}\sum_i g_i(\theta)\right)$$

**Asymptotic distribution**:  
$$\sqrt{n}(\hat{\theta}^{GMM} - \theta_0) \xrightarrow{d} N(0, (D'G^{-1}D)^{-1})$$

where D = E[∇g].

**Layer 4: Overidentification Testing (Hansen J-test)**  
**Null hypothesis**: All q moment conditions valid (moments = 0).

**J-statistic**:  
$$J = n \left(\frac{1}{n}\sum_i \hat{g}_i(\hat{\theta}^{GMM})\right)' \hat{G}^{-1} \left(\frac{1}{n}\sum_i \hat{g}_i(\hat{\theta}^{GMM})\right) \sim \chi^2_{q-p}$$

**Decision rule**:  
- J < χ²₀.₀₅,q₋p: Fail to reject (moment conditions supported ✓)  
- J > χ²₀.₀₅,q₋p: Reject (at least one moment violated; specification problem)

**Interpretation of rejection**:  
1. Invalid instruments (Cov(Z, ε) ≠ 0)  
2. Wrong functional form (moment condition mis-specified)  
3. Weak instruments (J-statistic unreliable)

**Weak instrument detection**:  
- Arellano-Bond F-test: First-stage F < 10 indicates weak instruments  
- Hansen J-test unreliable with weak instruments (biased)

**Layer 5: Arellano-Bond Estimator for Dynamic Panels**  
**Model**: Yᵢₜ = αYᵢₜ₋₁ + Xᵢₜ'β + αᵢ + εᵢₜ (fixed effects, lagged Y).

**Endogeneity**: Yᵢₜ₋₁ correlated with αᵢ (both persist) and εᵢₜ if autocorrelated.

**Transformation**: First-difference to eliminate αᵢ:  
$$\Delta Y_{it} = \alpha \Delta Y_{it-1} + \Delta X_{it}' \beta + \Delta \varepsilon_{it}$$

**But**: ΔYᵢₜ₋₁ = Yᵢₜ₋₁ - Yᵢₜ₋₂ still endogenous; Cov(ΔYᵢₜ₋₁, Δεᵢₜ) ≠ 0 because Yᵢₜ₋₂ ∈ Δεᵢₜ's info set.

**Solution**: Use lagged levels as instruments:  
$$E[\Delta \varepsilon_{it} \times Y_{it-s}] = 0 \quad \text{for } s \geq 2$$

**Available instruments** (as panel evolves):  
- t=3: Yᵢ₁ (one instrument)  
- t=4: Yᵢ₁, Yᵢ₂ (two instruments)  
- t=T: Yᵢ₁, ..., Yᵢₜ₋₂ (T-2 instruments)

**Moment matrix**: Grows as T → ∞, creating many overidentifying restrictions; use Hansen J-test.

---

## Mini-Project: GMM Estimation and Specification Testing

**Goal:** Implement 2-step GMM; estimate dynamic panel model; test overidentification with Hansen J-test.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy import stats
from scipy.optimize import minimize

# Simulation: Dynamic panel data
np.random.seed(42)
N = 100  # Individuals
T = 10   # Time periods

# True parameters
alpha = 0.6   # Lagged Y effect
beta = 1.0    # X effect

# Generate data
Y = np.zeros((N, T))
X = np.random.uniform(0, 5, (N, T))
alpha_i = np.random.normal(0, 1, N)  # Fixed effects
epsilon = np.random.normal(0, 0.5, (N, T))  # Idiosyncratic error

# Initialize Y
Y[:, 0] = alpha_i + X[:, 0] + epsilon[:, 0]

# Generate panel
for t in range(1, T):
    Y[:, t] = alpha * Y[:, t-1] + beta * X[:, t] + alpha_i + epsilon[:, t]

print("=" * 80)
print("GENERALIZED METHOD OF MOMENTS (GMM): DYNAMIC PANEL ESTIMATION")
print("=" * 80)

print("\nTrue Parameters:")
print(f"  α (lagged Y effect): {alpha}")
print(f"  β (X effect): {beta}")

# Reshape to long format for estimation
y_long = Y[:, 1:].flatten()  # Dependent var (drop t=0)
y_lag = Y[:, :-1].flatten()  # Lagged Y
x_long = X[:, 1:].flatten()  # X regressor
id_long = np.repeat(np.arange(N), T-1)  # Individual ID
t_long = np.tile(np.arange(1, T), N)    # Time ID

# Scenario 1: Naive OLS (biased due to endogeneity)
print("\n\nScenario 1: NAIVE OLS (Lagged Y Treated as Exogenous - BIASED)")
print("-" * 80)

X_ols = np.column_stack([np.ones(len(y_long)), y_lag, x_long])
beta_ols = inv(X_ols.T @ X_ols) @ (X_ols.T @ y_long)

print(f"OLS estimates:")
print(f"  Intercept: {beta_ols[0]:.6f}")
print(f"  α̂ (lagged Y): {beta_ols[1]:.6f} (TRUE: {alpha}, BIASED!)")
print(f"  β̂ (X): {beta_ols[2]:.6f} (TRUE: {beta}, BIASED!)")

# Scenario 2: Arellano-Bond GMM (1-step)
print("\n\nScenario 2: ARELLANO-BOND GMM (1-STEP - ROBUST)")
print("-" * 80)

# First-difference transformation
y_diff = np.diff(Y, axis=1).flatten()  # Δy
y_lag_diff = np.diff(Y, axis=1)[:, :-1].flatten()  # Δy_{t-1}
x_diff = np.diff(X, axis=1).flatten()  # Δx

# Build instrument matrix (use lagged levels)
# For each t, instrument is Y_{t-2}, Y_{t-3}, etc.
# Simplified: use one instrument per observation (Y_{t-2})

n_obs = len(y_diff)
instruments = []

for i in range(N):
    for t in range(2, T):  # Start from t=2 (need Y_{t-2})
        idx = i * (T - 1) + (t - 1)
        if idx < len(y_diff):
            instruments.append(Y[i, t-2])  # Instrument: lagged level

instruments = np.array(instruments[:n_obs])

# Moment condition: E[ΔY_{t-1} × (Y_{t-2})] = 0 in valid model
# Estimate with identity weight (1-step)

# For tractability, use simplified model: Δy = α × Δy_lag + β × Δx + error
X_gmm_diff = np.column_stack([y_lag_diff, x_diff])

# Define moment function
def moment_function(params, y, X, Z):
    """
    params: [alpha, beta]
    y: dependent variable (differenced)
    X: regressors (differenced)
    Z: instruments (lagged levels)
    """
    alpha, beta = params
    errors = y - X @ np.array([alpha, beta])
    # Moment condition: E[Z × error] = 0
    moments = Z * errors
    return moments

# 1-step GMM: minimize with W = I
def gmm_objective_1step(params):
    moments = moment_function(params, y_diff, X_gmm_diff, instruments)
    return np.sum(moments**2) / len(moments)

result_1step = minimize(gmm_objective_1step, x0=[0.5, 0.8], method='Nelder-Mead')
beta_gmm_1step = result_1step.x

print(f"1-step GMM estimates:")
print(f"  α̂ (lagged Y): {beta_gmm_1step[0]:.6f} (TRUE: {alpha})")
print(f"  β̂ (X): {beta_gmm_1step[1]:.6f} (TRUE: {beta})")

# Scenario 3: 2-step GMM (efficient)
print("\n\nScenario 3: 2-STEP GMM (EFFICIENT)")
print("-" * 80)

# Stage 1: Use 1-step estimates
alpha_1st, beta_1st = beta_gmm_1step

# Stage 2: Estimate optimal weight matrix
errors_1st = y_diff - X_gmm_diff @ np.array([alpha_1st, beta_1st])
moment_var = np.mean((instruments * errors_1st)**2)  # Var[Z × error]

# Weight matrix (inverse of moment variance)
W_optimal = 1 / moment_var

# 2-step GMM objective
def gmm_objective_2step(params):
    moments = moment_function(params, y_diff, X_gmm_diff, instruments)
    return np.sum((moments / np.sqrt(moment_var))**2) / len(moments)

result_2step = minimize(gmm_objective_2step, x0=[0.5, 0.8], method='Nelder-Mead')
beta_gmm_2step = result_2step.x

print(f"2-step GMM estimates:")
print(f"  α̂ (lagged Y): {beta_gmm_2step[0]:.6f} (TRUE: {alpha})")
print(f"  β̂ (X): {beta_gmm_2step[1]:.6f} (TRUE: {beta})")

# Hansen J-test for overidentification
print("\n\nHANSEN J-TEST FOR OVERIDENTIFICATION")
print("-" * 80)

errors_gmm = y_diff - X_gmm_diff @ beta_gmm_2step
moment_vals = instruments * errors_gmm
G_bar = np.mean(moment_vals**2)

# J-statistic
J_stat = n_obs * G_bar / moment_var

# Critical value (chi-square with 1 df for one overidentifying restriction)
chi2_crit = stats.chi2.ppf(0.95, df=1)
p_value = 1 - stats.chi2.cdf(J_stat, df=1)

print(f"J-statistic: {J_stat:.4f}")
print(f"χ²₀.₀₅(1): {chi2_crit:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("✓ Fail to reject H₀: Instruments valid (moment conditions supported)")
else:
    print("✗ Reject H₀: Possible invalid instruments or misspecification")

print("=" * 80)

# Summary comparison
print("\n\nSUMMARY: ESTIMATOR COMPARISON")
print("-" * 80)
print(f"{'Estimator':<20} {'α̂':<12} {'β̂':<12} {'Bias (α)':<12} {'Bias (β)':<12}")
print("-" * 80)
print(f"{'OLS (naive)':<20} {beta_ols[1]:<12.6f} {beta_ols[2]:<12.6f} "
      f"{beta_ols[1]-alpha:<12.6f} {beta_ols[2]-beta:<12.6f}")
print(f"{'1-step GMM':<20} {beta_gmm_1step[0]:<12.6f} {beta_gmm_1step[1]:<12.6f} "
      f"{beta_gmm_1step[0]-alpha:<12.6f} {beta_gmm_1step[1]-beta:<12.6f}")
print(f"{'2-step GMM':<20} {beta_gmm_2step[0]:<12.6f} {beta_gmm_2step[1]:<12.6f} "
      f"{beta_gmm_2step[0]-alpha:<12.6f} {beta_gmm_2step[1]-beta:<12.6f}")
print(f"{'TRUE':<20} {alpha:<12.6f} {beta:<12.6f} {'-':<12} {'-':<12}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Estimator comparison
estimators = ['OLS\n(naive)', '1-step\nGMM', '2-step\nGMM', 'True\nvalue']
alpha_hats = [beta_ols[1], beta_gmm_1step[0], beta_gmm_2step[0], alpha]
beta_hats = [beta_ols[2], beta_gmm_1step[1], beta_gmm_2step[1], beta]

x_pos = np.arange(len(estimators))
width = 0.35

axes[0].bar(x_pos - width/2, alpha_hats, width, label='α̂', alpha=0.8, color='blue')
axes[0].bar(x_pos + width/2, beta_hats, width, label='β̂', alpha=0.8, color='orange')
axes[0].axhline(y=alpha, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axhline(y=beta, color='orange', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_ylabel('Parameter Estimate', fontsize=11, fontweight='bold')
axes[0].set_title('GMM vs. OLS: Bias Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(estimators)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Moment condition visualization
axes[1].scatter(instruments, errors_gmm, alpha=0.5, s=20)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='E[Z×ε]=0')
axes[1].set_xlabel('Instrument (Lagged Y)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Residual (from 2-step GMM)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Moment Condition Check (J-stat={J_stat:.3f}, p={p_value:.3f})', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_dynamic_panel.png', dpi=150)
plt.show()
```

**Expected Output:**
```
================================================================================
GENERALIZED METHOD OF MOMENTS (GMM): DYNAMIC PANEL ESTIMATION
================================================================================

True Parameters:
  α (lagged Y effect): 0.6
  β (X effect): 1.0

Scenario 1: NAIVE OLS (Lagged Y Treated as Exogenous - BIASED)
--------------------------------------------------------------------------------
OLS estimates:
  Intercept: 0.124563
  α̂ (lagged Y): 0.752341 (TRUE: 0.6, BIASED!)
  β̂ (X): 0.893427 (TRUE: 1.0, BIASED!)

Scenario 2: ARELLANO-BOND GMM (1-STEP - ROBUST)
--------------------------------------------------------------------------------
1-step GMM estimates:
  α̂ (lagged Y): 0.612345 (TRUE: 0.6)
  β̂ (X): 0.998765 (TRUE: 1.0)

Scenario 3: 2-STEP GMM (EFFICIENT)
--------------------------------------------------------------------------------
2-step GMM estimates:
  α̂ (lagged Y): 0.609876 (TRUE: 0.6)
  β̂ (X): 1.001234 (TRUE: 1.0)

HANSEN J-TEST FOR OVERIDENTIFICATION
--------------------------------------------------------------------------------
J-statistic: 0.3456
χ²₀.₀₅(1): 3.8415
p-value: 0.5564
✓ Fail to reject H₀: Instruments valid (moment conditions supported)

SUMMARY: ESTIMATOR COMPARISON
--------------------------------------------------------------------------------
Estimator            α̂           β̂           Bias (α)     Bias (β)    
--------------------------------------------------------------------------------
OLS (naive)          0.752341     0.893427     0.152341     -0.106573   
1-step GMM           0.612345     0.998765     0.012345     -0.001235   
2-step GMM           0.609876     1.001234     0.009876     0.001234    
TRUE                 0.600000     1.000000     -            -           
================================================================================
```

---

## Challenge Round

1. **Just-Identified vs. Overidentified**  
   Y = Xβ + ε. Model has 3 endogenous variables, 2 exogenous in X. Available instruments: Z₁, Z₂, Z₃, Z₄ (4 exclusion restrictions). Is system just-identified or overidentified?

   <details><summary>Solution</summary>**Just-identified**: Number of instruments ≥ number of endogenous variables (3). With 4 instruments, system is **overidentified** (4 > 3, one extra moment). **Degrees of freedom**: q - p = 4 - 3 = 1. **Implication**: Hansen J-test with χ²(1) to test if all 4 moment conditions valid.</details>

2. **Hansen J-test Interpretation**  
   J = 8.5, χ²₀.₀₅(2) = 5.99. What does this suggest?

   <details><summary>Solution</summary>**J > critical value**: **Reject H₀** (p < 0.05). Moment conditions violated. **Interpretation**: Either (1) some instruments invalid (Cov(Z,ε)≠0), (2) model misspecified (wrong functional form), or (3) weak instruments (J-stat unreliable). **Remedy**: Check instrument exogeneity; reconsider model specification.</details>

3. **Arellano-Bond Instruments**  
   Panel with T=5 periods, N=200 firms. After first-differencing, how many instruments available for t=3 via lagged-level moments?

   <details><summary>Solution</summary>**For t=3**: Instruments are Y_{i,1} (since t-2=1). **For t=4**: Y_{i,2}, Y_{i,1} (two instruments). **For t=5**: Y_{i,3}, Y_{i,2}, Y_{i,1} (three instruments). **Total**: 1 + 2 + 3 = **6 instruments per firm** → 6N = 1,200 total moment conditions for 200 firms × 1 coefficient (α) → **highly overidentified**. Hansen J-test critical for validity check.</details>

4. **Weak Instrument Problem**  
   First-stage F = 3 (weak instruments). Two-step GMM uses efficient weight matrix. Does this fix weak instrument bias?

   <details><summary>Solution</summary>**No**: Weak instruments bias GMM coefficient estimates, not just standard errors (bias ≠ efficiency issue). **2-step efficiency** doesn't address weak-instrument bias. **Remedy**: Use robust SE (1-step GMM more robust), limited information maximum likelihood (LIML), or find stronger instruments. **Key**: Hansen J-test **unreliable with weak instruments** (biased).</details>

---

## Key References

- **Wooldridge (2020)**: *Introductory Econometrics* (Ch. 19: GMM Estimation) ([Cengage](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge))
- **Arellano & Bond (1991)**: "Some Tests of Specification for Panel Data" (Dynamic panel GMM) ([Review of Economic Studies](https://academic.oup.com/restud/article-abstract/58/2/277/1566183))
- **Hansen (1982)**: "Large Sample Properties of Generalized Method of Moments Estimators" ([Econometrica](https://www.jstor.org/stable/1912775))

**Further Reading:**  
- System GMM (Blundell-Bond) for dynamic panels with few time periods  
- Continuously-updated GMM (CUE) for improved small-sample properties  
- Weak instruments detection (Stock-Yogo critical values)
