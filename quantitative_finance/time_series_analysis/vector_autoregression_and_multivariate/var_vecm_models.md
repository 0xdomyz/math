# Vector Autoregression (VAR) and Vector Error Correction Models (VECM)

## 1. Concept Skeleton
**Definition:** VAR models multiple time series jointly where each variable depends on its own lags and lags of all other variables; VECM extends VAR for cointegrated systems with long-run equilibrium relationships  
**Purpose:** Capture dynamic interdependencies, forecast multiple series simultaneously, analyze impulse responses (shocks propagation), decompose forecast variance, model long-run equilibrium with short-run adjustment dynamics  
**Prerequisites:** ARIMA models, stationarity/unit root testing (ADF), matrix algebra, cointegration (Johansen test), Granger causality, lag selection (AIC/BIC), structural identification

## 2. Comparative Framing
| Model | VAR(p) | VECM(p-1) | Structural VAR (SVAR) | Bayesian VAR (BVAR) | Panel VAR |
|-------|--------|-----------|----------------------|---------------------|-----------|
| **Specification** | Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + εₜ | ΔYₜ = αβ'Yₜ₋₁ + Γ₁ΔYₜ₋₁ + ... + εₜ | A₀Yₜ = A₁Yₜ₋₁ + ... + Bεₜ | VAR + prior (Minnesota/Normal-Wishart) | VAR with cross-section dimension |
| **Stationarity** | All series I(0) | Series I(1) cointegrated | I(0) or cointegrated | I(0) typically | I(0) typically |
| **Long-Run** | No explicit equilibrium | β'Yₜ = 0 (cointegration vectors) | Can impose long-run restrictions | Shrinkage toward RW/equilibrium | Fixed effects capture heterogeneity |
| **Parameters** | K²p + K(K+1)/2 (many) | Fewer than VAR (common trends) | Same as VAR + restrictions | Same but regularized | K²p per unit + fixed effects |
| **Identification** | Reduced-form (Cholesky) | Cointegration rank r < K | Theory-driven (AB-model, sign) | Priors (shrinkage) | Fixed/random effects |
| **Use Case** | Stationary macro/finance | Exchange rates, interest rates | Monetary policy shocks | Small samples, many variables | Cross-country panel data |

| Concept | VAR Property | VECM Property | Interpretation | Example |
|---------|-------------|---------------|----------------|---------|
| **Granger Causality** | X Granger-causes Y if lags of X improve Y forecast | Same as VAR | Predictive precedence (not true causality) | Oil prices → inflation |
| **Impulse Response Function (IRF)** | Trace effect of 1 SD shock in εᵢ on all variables over time | Decompose into transitory + permanent | Dynamic multipliers | Interest rate shock → GDP path |
| **Forecast Error Variance Decomposition (FEVD)** | % of h-step forecast variance from each shock | Shows importance of shocks | Attribution analysis | How much GDP variance from oil shocks? |
| **Cointegration Rank (r)** | N/A (VAR assumes stationarity) | r = # of cointegrating relationships | # of long-run equilibria | r=1: Single attractor (PPP, UIP) |
| **Adjustment Speed (α)** | N/A | How fast deviations from equilibrium corrected | Error-correction strength | Half-life = log(0.5)/log(1-α) |
| **Common Trends (K-r)** | N/A | # of stochastic trends driving system | Unobservable permanent shocks | K=3, r=1: 2 common trends |

## 3. Examples + Counterexamples

**Simple Example:**  
Bivariate VAR(1) for GDP and unemployment: GDPₜ = a₁GDPₜ₋₁ + a₂Uₜ₋₁ + ε¹ₜ, Uₜ = b₁GDPₜ₋₁ + b₂Uₜ₋₁ + ε²ₜ. Granger causality: Does unemployment lag predict GDP? IRF: 1% GDP shock → unemployment response over 12 quarters.

**Perfect Fit:**  
Interest rate term structure: Short rate (3m) and long rate (10y) cointegrated (r=1). VECM captures mean reversion to yield spread equilibrium. Speed of adjustment α ≈ 0.15 (half-life ~4.3 months). Out-of-sample forecasts beat VAR in differences.

**Failure Case:**  
High-dimensional VAR (K=50 variables, p=4): 10,000+ parameters with T=200 observations. Severe overfitting, poor forecasts, unstable estimates. BVAR with Minnesota prior or factor models (FAVAR) needed instead.

**Edge Case:**  
Near-cointegration: Variables appear cointegrated in sample, but Johansen test barely rejects at 5%. VECM vs VAR in differences similar forecasts. Structural breaks can create false cointegration—check parameter stability.

**Common Mistake:**  
Estimating VAR on non-stationary I(1) series without checking cointegration. Spurious dynamics, invalid inference. Must difference (if no cointegration) or use VECM (if cointegrated). Always test stationarity first.

**Counterexample:**  
Daily stock returns (10 stocks): No cointegration expected (prices drift independently). VAR appropriate for returns (stationary). VECM would impose spurious long-run relationships. Portfolio weights from VECM misleading.

## 4. Layer Breakdown
```
VAR and VECM Framework:

├─ VAR(p) - Vector Autoregression:
│   ├─ Model Structure:
│   │   │ K-dimensional system:
│   │   │ Yₜ = A₁Yₜ₋₁ + A₂Yₜ₋₂ + ... + AₚYₜ₋ₚ + c + εₜ
│   │   │
│   │   │ Components:
│   │   │   - Yₜ: K×1 vector (Y₁ₜ, Y₂ₜ, ..., Yₖₜ)' of variables
│   │   │   - Aᵢ: K×K coefficient matrices (i=1,...,p)
│   │   │   - c: K×1 constant/deterministic terms
│   │   │   - εₜ: K×1 white noise, εₜ ~ N(0, Σ)
│   │   │   - Σ: K×K covariance matrix (contemporaneous correlations)
│   │   │
│   │   │ Total parameters: K²p + K (intercepts) + K(K+1)/2 (Σ)
│   │   │ Example: K=5, p=4 → 100 + 5 + 15 = 120 parameters
│   │   │
│   │   └─ Compact form:
│   │       Yₜ = AXₜ₋₁ + εₜ, where Xₜ₋₁ = (Yₜ₋₁', Yₜ₋₂', ..., Yₜ₋ₚ', 1)'
│   │       A: K×(Kp+1) full parameter matrix
│   ├─ Stationarity Condition:
│   │   │ Companion form: Eigenvalues of companion matrix < 1 (modulus)
│   │   │ Companion matrix F (Kp×Kp):
│   │   │   ┌ A₁  A₂  ...  Aₚ₋₁  Aₚ ┐
│   │   │   │ I   0   ...   0    0  │
│   │   │   │ 0   I   ...   0    0  │
│   │   │   │ ..  ..  ...  ..   .. │
│   │   │   └ 0   0   ...   I    0  ┘
│   │   │
│   │   │ All eigenvalues λᵢ: |λᵢ| < 1 for stability
│   │   │ If some |λᵢ| = 1: Unit root (non-stationary)
│   │   └─ Check: det(I - A₁z - A₂z² - ... - Aₚzᵖ) = 0 roots outside unit circle
│   ├─ Estimation (Equation-by-Equation OLS):
│   │   │ Each equation estimated separately:
│   │   │   y₁ₜ = β₁Xₜ₋₁ + ε₁ₜ  (OLS on equation 1)
│   │   │   y₂ₜ = β₂Xₜ₋₁ + ε₂ₜ  (OLS on equation 2)
│   │   │   ...
│   │   │
│   │   │ Why OLS works:
│   │   │   - Same regressors in all equations (balanced design)
│   │   │   - OLS = MLE under normality
│   │   │   - Efficient even with correlated errors across equations
│   │   │
│   │   ├─ Residual Covariance:
│   │   │   Σ̂ = (1/T) Σₜ εₜεₜ' (sample covariance of residuals)
│   │   │   Used for joint inference, IRF confidence bands
│   │   │
│   │   └─ Standard Errors:
│   │       Asymptotic: Var(vec(Â)) = Σ ⊗ (X'X)⁻¹
│   │       Bootstrap for small samples or non-normal errors
│   ├─ Lag Selection:
│   │   ├─ Information Criteria:
│   │   │   │ AIC(p) = log|Σ̂(p)| + (2/T) × K²p
│   │   │   │ BIC(p) = log|Σ̂(p)| + (log(T)/T) × K²p
│   │   │   │ HQ(p) = log|Σ̂(p)| + (2log(log(T))/T) × K²p
│   │   │   │
│   │   │   │ Choose p minimizing criterion
│   │   │   │ BIC more parsimonious (stronger penalty)
│   │   │   └─ Typically p ≤ 4 for quarterly data, p ≤ 12 for monthly
│   │   ├─ Sequential Testing:
│   │   │   LR test: H0: p = p₀ vs H1: p = p₀ + 1
│   │   │   LR = T(log|Σ̂(p₀)| - log|Σ̂(p₀+1)|) ~ χ²(K²)
│   │   │   Start with large p, test down until fail to reject
│   │   └─ Trade-off:
│   │       Too small p: Omitted lag bias (autocorrelated residuals)
│   │       Too large p: Overfitting (poor forecasts, many parameters)
│   ├─ Granger Causality Testing:
│   │   │ Definition: X Granger-causes Y if past X helps predict Y
│   │   │ (beyond Y's own past)
│   │   │
│   │   │ Test: H0: X does not Granger-cause Y
│   │   │   Restricted model: Yₜ = f(Yₜ₋₁, ..., Yₜ₋ₚ) + εₜ
│   │   │   Unrestricted: Yₜ = f(Yₜ₋₁, ..., Yₜ₋ₚ, Xₜ₋₁, ..., Xₜ₋ₚ) + εₜ
│   │   │
│   │   │ F-test: F = [(SSR_R - SSR_U)/p] / [SSR_U/(T-2p-1)]
│   │   │ Or Wald test: χ²(p) under H0
│   │   │
│   │   ├─ Interpretation:
│   │   │   Rejection: X contains useful info for forecasting Y
│   │   │   Failure: X doesn't improve Y forecast (given Y's history)
│   │   │   NOT true causality (correlation in time)
│   │   │
│   │   ├─ Bidirectional:
│   │   │   X→Y and Y→X both possible (feedback)
│   │   │   Example: GDP ↔ unemployment (bidirectional causality)
│   │   │
│   │   └─ Instantaneous Causality:
│   │       Test if contemporaneous εₓₜ affects Yₜ
│   │       Check correlation in Σ̂ (off-diagonal)
│   │       Requires structural identification (ordering)
│   └─ Diagnostic Checks:
│       ├─ Residual Autocorrelation:
│       │   Portmanteau test (multivariate Ljung-Box)
│       │   H0: No autocorrelation in residuals up to lag h
│       │   Should not reject (p > 0.05)
│       ├─ Residual Normality:
│       │   Jarque-Bera multivariate test
│       │   Not critical (asymptotic theory robust)
│       │   But matters for small-sample inference
│       ├─ Stability:
│       │   Recursive estimation, CUSUM test
│       │   Eigenvalues stable over time?
│       │   Structural breaks → split sample or regime-switching VAR
│       └─ Heteroscedasticity:
│           ARCH-LM test on residuals
│           If present: Use robust SE or multivariate GARCH
│
├─ Impulse Response Functions (IRF):
│   ├─ Definition:
│   │   │ IRF traces dynamic effect of one variable's shock on all variables
│   │   │ ψᵢⱼ(h): Response of variable i at horizon h to 1-unit shock in variable j
│   │   │
│   │   │ Moving Average Representation (Wold):
│   │   │   Yₜ = μ + Σₕ₌₀^∞ Ψₕεₜ₋ₕ
│   │   │   Ψₕ: K×K impulse response matrices
│   │   │
│   │   │ Computation:
│   │   │   Ψ₀ = Iₖ (identity)
│   │   │   Ψₕ = A₁Ψₕ₋₁ + A₂Ψₕ₋₂ + ... + AₚΨₕ₋ₚ (h ≥ 1)
│   │   │
│   │   └─ Horizon h: Typically h = 12-24 (quarters) or 36-60 (months)
│   ├─ Orthogonalization (Cholesky):
│   │   │ Problem: εₜ components contemporaneously correlated
│   │   │ Solution: Transform to orthogonal shocks ηₜ = P⁻¹εₜ
│   │   │
│   │   │ Cholesky decomposition: Σ = PP'
│   │   │ P: Lower triangular matrix
│   │   │ ηₜ ~ N(0, I) uncorrelated
│   │   │
│   │   │ Ordering matters:
│   │   │   Variable ordered first affected by all contemporaneous shocks
│   │   │   Variable ordered last only affects itself contemporaneously
│   │   │
│   │   │ Example: [GDP, Interest Rate, Inflation]
│   │   │   GDP affected by all three shocks contemporaneously
│   │   │   Interest Rate by interest + inflation (not GDP)
│   │   │   Inflation only by itself (slowest moving)
│   │   │
│   │   │ Orthogonal IRF: θₕ = ΨₕP
│   │   │ Each column = response to 1 SD orthogonal shock
│   │   │
│   │   └─ Sensitivity:
│   │       Different orderings → different IRFs
│   │       Robustness check: Try multiple orderings
│   │       Alternative: Structural VAR (theory-based identification)
│   ├─ Confidence Intervals:
│   │   ├─ Analytical (Delta Method):
│   │   │   Asymptotic variance of vec(Ψₕ)
│   │   │   Complex formula (recursive)
│   │   │   Assumes normality
│   │   ├─ Bootstrap (Preferred):
│   │   │   1. Resample residuals εₜ* (with replacement)
│   │   │   2. Generate Y*ₜ using estimated Â and resampled ε*
│   │   │   3. Re-estimate VAR on Y*ₜ, compute IRF*
│   │   │   4. Repeat B times (B=1000-5000)
│   │   │   5. Percentile CI: [2.5%, 97.5%] of IRF* distribution
│   │   │
│   │   │   Advantages: No normality, captures parameter uncertainty
│   │   │   Commonly reported: Median + 68%/90%/95% bands
│   │   └─ Bayesian:
│   │       Posterior distribution of IRF from BVAR
│   │       Credible intervals from Gibbs sampler draws
│   ├─ Interpretation:
│   │   │ Peak response: When does maximum effect occur?
│   │   │ Persistence: How long until effect dies out? (half-life)
│   │   │ Sign: Positive/negative transmission?
│   │   │ Magnitude: Economic significance (% or pp)
│   │   │
│   │   │ Example: Monetary policy shock (25bp rate hike)
│   │   │   - GDP response peaks at -0.5% after 6 quarters
│   │   │   - Inflation falls by -0.2pp, peaks at 8 quarters
│   │   │   - Effect dissipates after ~3 years (12 quarters)
│   │   │
│   │   └─ Cumulative IRF:
│   │       Σₕ₌₀^H ψᵢⱼ(h): Total effect up to horizon H
│   │       Long-run multiplier: Σₕ₌₀^∞ ψᵢⱼ(h)
│   └─ Structural IRF (SVAR):
│       Impose economic theory restrictions
│       Short-run: AB-model (contemporaneous restrictions)
│       Long-run: Blanchard-Quah (permanent vs transitory shocks)
│       Sign restrictions: IRF must have certain signs
│
├─ Forecast Error Variance Decomposition (FEVD):
│   ├─ Definition:
│   │   │ FEVD: % of h-step forecast error variance for variable i
│   │   │         due to shocks in variable j
│   │   │
│   │   │ h-step forecast error: eₜ(h) = Yₜ₊ₕ - Ŷₜ₊ₕ|ₜ
│   │   │                               = Σₛ₌₀^{h-1} Ψₛεₜ₊ₕ₋ₛ
│   │   │
│   │   │ Variance: MSE(h) = Σₛ₌₀^{h-1} ΨₛΣΨₛ'
│   │   │
│   │   │ Contribution of shock j to variance of variable i:
│   │   │   ω_{ij}(h) = [Σₛ₌₀^{h-1} (ψᵢⱼ,ₛ)²] / MSEᵢ(h)
│   │   │
│   │   └─ Properties:
│   │       Σⱼ ω_{ij}(h) = 1 (sum to 100% for each variable)
│   │       ω_{ij}(h) ∈ [0,1] (non-negative)
│   │       Depends on Cholesky ordering (orthogonalization)
│   ├─ Interpretation:
│   │   │ High ω_{ij}: Shock j important for explaining variable i
│   │   │ Low ω_{ij}: Shock j minor contributor
│   │   │
│   │   │ Example: GDP forecast variance at h=12 quarters:
│   │   │   - 60% from GDP shocks (own innovations)
│   │   │   - 25% from monetary policy shocks
│   │   │   - 15% from oil price shocks
│   │   │
│   │   │ Inference: Monetary policy explains 1/4 of GDP uncertainty
│   │   │
│   │   └─ Horizon effects:
│   │       Short h: Own shocks dominate (diagonal)
│   │       Long h: Cross-variable shocks more important
│   │       As h→∞: Depends on system persistence
│   └─ Applications:
│       Policy analysis: Which shocks drive business cycles?
│       Risk management: Decompose portfolio variance
│       Macro forecasting: Understand forecast uncertainty sources
│
├─ Cointegration and VECM:
│   ├─ Cointegration Concept:
│   │   │ Definition: I(1) series Yₜ cointegrated if ∃ linear combo
│   │   │            β'Yₜ ~ I(0) (stationary)
│   │   │
│   │   │ Interpretation: Long-run equilibrium relationship
│   │   │   β'Yₜ = 0 is attractor (mean-reverting)
│   │   │   Deviations temporary (transitory shocks)
│   │   │
│   │   │ Example: Log prices of two stocks
│   │   │   p₁ₜ, p₂ₜ both I(1) (random walks)
│   │   │   If cointegrated: p₁ₜ - β p₂ₜ ~ I(0)
│   │   │   Pairs trading: Spread mean-reverts
│   │   │
│   │   ├─ Cointegration Rank (r):
│   │   │   r = # of linearly independent cointegrating vectors
│   │   │   0 ≤ r ≤ K-1 (at most K-1 for K variables)
│   │   │   r = 0: No cointegration (VAR in differences)
│   │   │   r = K: All stationary (VAR in levels)
│   │   │   0 < r < K: Cointegrated system (VECM)
│   │   │
│   │   ├─ Common Trends (K-r):
│   │   │   # of stochastic trends driving system
│   │   │   Example: K=3, r=1 → 2 common trends
│   │   │   System has 1 equilibrium, 2 permanent shocks
│   │   │
│   │   └─ Economic Examples:
│   │       PPP: Exchange rate ~ price differential (r=1)
│   │       UIP: Forward rate ~ future spot (r=1)
│   │       Term structure: Yields cointegrated (r=K-1)
│   │       Consumption/Income: Long-run proportionality (r=1)
│   ├─ Johansen Test (Maximum Likelihood):
│   │   │ Test H0: rank = r vs H1: rank > r
│   │   │
│   │   │ Two test statistics:
│   │   │   1. Trace: λ_trace(r) = -T Σᵢ₌ᵣ₊₁^K log(1-λ̂ᵢ)
│   │   │   2. Max eigenvalue: λ_max(r) = -T log(1-λ̂ᵣ₊₁)
│   │   │
│   │   │ λ̂ᵢ: Eigenvalues from canonical correlation
│   │   │ Critical values: Johansen tables (depend on deterministic terms)
│   │   │
│   │   │ Sequential testing:
│   │   │   Start with r=0, test if r≥1
│   │   │   If reject, test r=1 vs r≥2
│   │   │   Continue until first non-rejection
│   │   │
│   │   ├─ Deterministic Terms:
│   │   │   Case 1: No intercept, no trend
│   │   │   Case 2: Intercept in cointegrating relation
│   │   │   Case 3: Intercept in levels (most common)
│   │   │   Case 4: Trend in cointegrating relation
│   │   │   Case 5: Trend in levels
│   │   │
│   │   │   Choice affects critical values and interpretation
│   │   │   Typically: Case 3 (restricted constant)
│   │   │
│   │   └─ Small-Sample Issues:
│   │       Test over-rejects in small samples (size distortion)
│   │       Bartlett correction or bootstrap
│   │       Sensitive to lag length p
│   ├─ VECM Specification:
│   │   │ General form:
│   │   │ ΔYₜ = αβ'Yₜ₋₁ + Γ₁ΔYₜ₋₁ + Γ₂ΔYₜ₋₂ + ... + Γₚ₋₁ΔYₜ₋ₚ₊₁ + c + εₜ
│   │   │
│   │   │ Components:
│   │   │   - ΔYₜ: First differences (K×1)
│   │   │   - β: K×r cointegrating vectors (long-run parameters)
│   │   │   - α: K×r adjustment speeds (loading matrix)
│   │   │   - Γᵢ: K×K short-run dynamics (i=1,...,p-1)
│   │   │   - εₜ: K×1 white noise
│   │   │
│   │   │ Error Correction Term: αβ'Yₜ₋₁
│   │   │   - β'Yₜ₋₁: Deviation from equilibrium (r×1)
│   │   │   - α: How fast each variable adjusts to restore equilibrium
│   │   │
│   │   ├─ Example (r=1, K=2):
│   │   │   β' = [1, -β₂] → equilibrium: Y₁ₜ = β₂Y₂ₜ
│   │   │   Error correction: ectₜ₋₁ = Y₁ₜ₋₁ - β₂Y₂ₜ₋₁
│   │   │
│   │   │   ΔY₁ₜ = α₁ · ectₜ₋₁ + γ₁₁ΔY₁ₜ₋₁ + γ₁₂ΔY₂ₜ₋₁ + ε₁ₜ
│   │   │   ΔY₂ₜ = α₂ · ectₜ₋₁ + γ₂₁ΔY₁ₜ₋₁ + γ₂₂ΔY₂ₜ₋₁ + ε₂ₜ
│   │   │
│   │   │   Interpretation:
│   │   │     - α₁ < 0: Y₁ adjusts down if above equilibrium
│   │   │     - α₂ > 0: Y₂ adjusts up if Y₁ too high
│   │   │     - Speed: |αᵢ| large → fast adjustment
│   │   │
│   │   └─ Half-Life of Disequilibrium:
│   │       h = log(0.5) / log(1 + α̂)
│   │       Example: α = -0.15 → h ≈ 4.3 periods
│   │       How long to close half the gap to equilibrium
│   ├─ Estimation (Johansen ML):
│   │   │ Two-step procedure:
│   │   │   1. Estimate β (cointegrating vectors) via ML
│   │   │      Solve eigenvalue problem in canonical correlations
│   │   │   2. Conditional on β̂, estimate α, Γᵢ by OLS
│   │   │
│   │   │ Identification:
│   │   │   β not unique: β and βH give same space (H nonsingular)
│   │   │   Normalization: Set one element of β to 1
│   │   │   Example: β' = [1, β₂, β₃] → Y₁ = β₂Y₂ + β₃Y₃
│   │   │
│   │   │ Testing restrictions on β:
│   │   │   LR test: χ²(# restrictions)
│   │   │   Example: Test β₂ = 1 (one-for-one relationship)
│   │   │
│   │   └─ Weak Exogeneity:
│   │       Test H0: αᵢ = 0 (variable i doesn't adjust to disequilibrium)
│   │       If αᵢ = 0, variable i weakly exogenous
│   │       Can condition on it, estimate partial system
│   ├─ VECM vs VAR in Differences:
│   │   │ If cointegrated, VAR in differences (ΔVAR):
│   │   │   - Misspecified (omits error correction)
│   │   │   - Loses long-run information
│   │   │   - Forecasts inferior (especially medium/long horizon)
│   │   │
│   │   │ VECM advantages:
│   │   │   - Correct specification (uses equilibrium)
│   │   │   - Better forecasts (pulls toward equilibrium)
│   │   │   - Interpretable long-run relationships
│   │   │   - Fewer parameters (common trends restriction)
│   │   │
│   │   │ When VAR in levels appropriate:
│   │   │   - All series I(0) (stationary)
│   │   │   - No cointegration (r=0)
│   │   │   - Short horizon forecasts only
│   │   │
│   │   └─ Granger Representation Theorem:
│   │       Cointegration ⇔ Error Correction Representation exists
│   │       Long-run equilibrium ⇔ ECM with αβ' term
│   └─ VECM Impulse Responses:
│       IRF decomposition: Transitory vs Permanent shocks
│       Permanent shocks: Push system to new equilibrium (K-r shocks)
│       Transitory shocks: Mean-reverting deviations (r shocks)
│       Identification: Gonzalo-Granger or Johansen decomposition
│
├─ Forecasting with VAR/VECM:
│   ├─ Direct Multi-Step Forecasting:
│   │   │ h-step ahead: Ŷₜ₊ₕ|ₜ = Âʰ Yₜ + Σⱼ₌₀^{h-1} Âʲ ĉ
│   │   │ where Â = companion matrix
│   │   │
│   │   │ Forecast error covariance grows with h:
│   │   │   Σ(h) = Σⱼ₌₀^{h-1} Ψⱼ Σ Ψⱼ'
│   │   │
│   │   │ Confidence intervals: Ŷₜ₊ₕ|ₜ ± 1.96 × √Σᵢᵢ(h)
│   │   │
│   │   └─ VECM forecasting:
│   │       Forecast pulls toward long-run equilibrium
│   │       Short horizon: Dominated by short-run dynamics
│   │       Long horizon: Converges to equilibrium relationship
│   ├─ Scenario Analysis:
│   │   Conditional forecasts: Fix some variables, forecast others
│   │   Example: Given oil price path, forecast GDP/inflation
│   │   Requires inverting VAR system
│   ├─ Forecast Evaluation:
│   │   │ Out-of-sample metrics (h-step ahead):
│   │   │   - RMSE: √(1/H Σ(Yₜ₊ₕ - Ŷₜ₊ₕ|ₜ)²)
│   │   │   - MAE: (1/H) Σ|Yₜ₊ₕ - Ŷₜ₊ₕ|ₜ|
│   │   │   - MAPE: (100/H) Σ|eₜ/Yₜ₊ₕ|
│   │   │
│   │   │ Multivariate:
│   │   │   - Trace RMSE: Sum of individual RMSEs
│   │   │   - Log determinant: log|Σ̂ₑ|
│   │   │
│   │   └─ Diebold-Mariano Test:
│   │       Compare two models' forecast accuracy
│   │       H0: Equal forecast accuracy
│   │       Test statistic from loss differential
│   └─ Combination Forecasts:
│       Average VAR, VECM, BVAR, univariate forecasts
│       Often outperforms single model
│       Weights: Equal, BIC-based, or optimal (minimize MSE)
│
└─ Advanced Extensions:
    ├─ Bayesian VAR (BVAR):
    │   ├─ Minnesota Prior:
    │   │   Own lags more important than others
    │   │   Recent lags more important than distant
    │   │   Shrink toward random walk (unit root)
    │   │   Hyperparameters: λ (overall tightness), θ (cross-variable)
    │   ├─ Advantages:
    │   │   Handles many variables (regularization)
    │   │   Better forecasts than OLS VAR (especially K large)
    │   │   Incorporates prior beliefs (economic theory)
    │   └─ Estimation:
    │       Posterior: Prior × Likelihood
    │       Gibbs sampler or direct (Normal-Wishart conjugate)
    ├─ Structural VAR (SVAR):
    │   Identify structural shocks (policy, technology, demand)
    │   Short-run restrictions: AB-model (contemporaneous)
    │   Long-run restrictions: Blanchard-Quah (permanent/transitory)
    │   Sign restrictions: Impose IRF signs (theory-based)
    │   Instrument variables: External instruments (proxy SVAR)
    ├─ Time-Varying VAR:
    │   Parameters change over time (structural change)
    │   TVP-VAR: Stochastic volatility + time-varying coefficients
    │   Estimation: Kalman filter, particle filter
    │   Applications: Monetary policy regime changes
    ├─ Factor-Augmented VAR (FAVAR):
    │   Summarize many variables (100+) via factors
    │   VAR on factors + key observed variables
    │   Overcomes curse of dimensionality
    │   Applications: Macro forecasting with big data
    └─ Panel VAR:
        Cross-section + time series (countries, firms)
        Fixed effects (country-specific intercepts)
        Estimation: GMM (Arellano-Bond) for dynamic panel
        Applications: Cross-country policy analysis
```

**Interaction:** Test stationarity (ADF) → If I(0): VAR, if I(1): Test cointegration (Johansen) → If r>0: VECM, if r=0: VAR in differences → Select lag order (BIC) → Estimate by OLS/ML → Diagnose residuals → Compute IRF (Cholesky or structural) → FEVD for shock attribution → Granger causality tests → Forecast out-of-sample → Evaluate vs benchmarks

## 5. Mini-Project
Comprehensive VAR and VECM analysis with IRF, FEVD, Granger causality, cointegration testing, and forecasting:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.stattools import adfuller, kpss, coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("VECTOR AUTOREGRESSION (VAR) AND VECTOR ERROR CORRECTION MODELS (VECM)")
print("="*80)

# Generate synthetic data for 3-variable system
np.random.seed(42)
n = 500

# System 1: Stationary VAR(2) for GDP, Unemployment, Interest Rate
print("\n" + "="*80)
print("PART 1: VAR MODEL (STATIONARY SYSTEM)")
print("="*80)

# True VAR(2) parameters
# Y1t = 0.6*Y1t-1 - 0.2*Y1t-2 + 0.1*Y2t-1 - 0.1*Y3t-1 + e1t
# Y2t = -0.3*Y1t-1 + 0.5*Y2t-2 + 0.2*Y3t-1 + e2t  
# Y3t = 0.2*Y1t-1 - 0.1*Y2t-1 + 0.7*Y3t-1 + e3t

A1 = np.array([[0.6, 0.1, -0.1],
               [-0.3, 0.0, 0.2],
               [0.2, -0.1, 0.7]])

A2 = np.array([[-0.2, 0.0, 0.0],
               [0.0, 0.5, 0.0],
               [0.0, 0.0, 0.0]])

# Covariance matrix
Sigma = np.array([[1.0, 0.3, -0.2],
                  [0.3, 0.8, 0.1],
                  [-0.2, 0.1, 0.6]])

# Simulate
Y = np.zeros((n, 3))
for t in range(2, n):
    Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + np.random.multivariate_normal([0, 0, 0], Sigma)

# Create DataFrame
dates = pd.date_range('2000-01-01', periods=n, freq='Q')
df_var = pd.DataFrame(Y, index=dates, columns=['GDP_Growth', 'Unemployment_Rate', 'Interest_Rate'])

print("\nSimulated VAR Data (First 10 observations):")
print(df_var.head(10))

print("\nDescriptive Statistics:")
print(df_var.describe())

# Check stationarity
print("\n" + "="*80)
print("STATIONARITY TESTING")
print("="*80)

def adf_test(series, name):
    result = adfuller(series, autolag='AIC')
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    print(f"  Result: {'Stationary' if result[1] < 0.05 else 'Non-stationary (unit root)'}")
    return result[1] < 0.05

stationary_checks = {}
for col in df_var.columns:
    stationary_checks[col] = adf_test(df_var[col], col)

all_stationary = all(stationary_checks.values())
print(f"\nConclusion: {'All series stationary - proceed with VAR' if all_stationary else 'Check cointegration or difference'}")

# Lag order selection
print("\n" + "="*80)
print("VAR LAG ORDER SELECTION")
print("="*80)

model = VAR(df_var)
lag_order = model.select_order(maxlags=8)
print("\nLag Order Selection:")
print(lag_order.summary())

optimal_lag = lag_order.aic
print(f"\nOptimal lag (AIC): {optimal_lag}")

# Estimate VAR
print("\n" + "="*80)
print(f"VAR({optimal_lag}) ESTIMATION")
print("="*80)

var_model = model.fit(optimal_lag)
print(var_model.summary())

# Extract parameters
params = var_model.params
print("\nEstimated Coefficient Matrix A1:")
print(params.iloc[:3, :])

# Diagnostics
print("\n" + "="*80)
print("VAR DIAGNOSTICS")
print("="*80)

# Residual autocorrelation
print("\n1. RESIDUAL SERIAL CORRELATION")
print("-" * 40)
residuals = var_model.resid
for col in residuals.columns:
    lb_result = acorr_ljungbox(residuals[col], lags=10, return_df=True)
    print(f"\n{col}:")
    print(f"  Ljung-Box p-value (lag 10): {lb_result['lb_pvalue'].iloc[-1]:.4f}")
    print(f"  Result: {'PASS' if lb_result['lb_pvalue'].iloc[-1] > 0.05 else 'FAIL (autocorrelation)'}")

# Normality
print("\n2. RESIDUAL NORMALITY")
print("-" * 40)
for col in residuals.columns:
    jb_stat, jb_pval = stats.jarque_bera(residuals[col])
    print(f"{col}: JB={jb_stat:.2f}, p-value={jb_pval:.4f} - {'Normal' if jb_pval > 0.05 else 'Non-normal'}")

# Stability (eigenvalues)
print("\n3. STABILITY CHECK (Eigenvalues)")
print("-" * 40)
eigenvalues = np.linalg.eigvals(var_model.coefs[0])
if optimal_lag > 1:
    # Companion matrix for higher lags
    companion = var_model.get_eq_index('companion_matrix', 0)
    
print("Eigenvalues (modulus):")
for i, ev in enumerate(eigenvalues):
    print(f"  λ{i+1} = {np.abs(ev):.4f}")
    
stable = np.all(np.abs(eigenvalues) < 1)
print(f"\nStability: {'STABLE (all |λ| < 1)' if stable else 'UNSTABLE'}")

# Granger Causality
print("\n" + "="*80)
print("GRANGER CAUSALITY TESTS")
print("="*80)

causality_results = {}
for caused in df_var.columns:
    print(f"\nVariable: {caused}")
    print("-" * 40)
    for causing in df_var.columns:
        if caused != causing:
            test = var_model.test_causality(caused, causing, kind='f')
            causality_results[(causing, caused)] = test.pvalue
            print(f"  {causing} → {caused}: p-value = {test.pvalue:.4f} {'***' if test.pvalue < 0.01 else '**' if test.pvalue < 0.05 else '*' if test.pvalue < 0.10 else 'NS'}")

print("\nInterpretation: p < 0.05 indicates Granger causality")

# Impulse Response Functions
print("\n" + "="*80)
print("IMPULSE RESPONSE FUNCTIONS (IRF)")
print("="*80)

irf = var_model.irf(periods=20)
print("\nComputing IRFs with Cholesky orthogonalization...")
print(f"Ordering: {list(df_var.columns)}")

# Summary of IRFs at selected horizons
print("\nIRF Summary (selected horizons):")
for h in [1, 4, 8, 20]:
    print(f"\nHorizon h={h}:")
    irfs_at_h = irf.irfs[h-1]
    for i, target in enumerate(df_var.columns):
        print(f"  {target}:")
        for j, shock in enumerate(df_var.columns):
            print(f"    Shock from {shock}: {irfs_at_h[i, j]:.4f}")

# Forecast Error Variance Decomposition
print("\n" + "="*80)
print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("="*80)

fevd = var_model.fevd(periods=20)
print("\nFEVD at horizon 20:")
for i, var_name in enumerate(df_var.columns):
    print(f"\n{var_name} variance explained by:")
    for j, shock_name in enumerate(df_var.columns):
        pct = fevd.decomp[19, i, j] * 100  # h=20 (index 19)
        print(f"  {shock_name}: {pct:.2f}%")

# Forecasting
print("\n" + "="*80)
print("OUT-OF-SAMPLE FORECASTING")
print("="*80)

train_size = 450
test_size = n - train_size

df_train = df_var.iloc[:train_size]
df_test = df_var.iloc[train_size:]

# Fit on training data
var_train = VAR(df_train).fit(optimal_lag)

# Forecast
forecast_steps = test_size
forecast = var_train.forecast(df_train.values[-optimal_lag:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=df_test.index, columns=df_var.columns)

# Evaluation
mae = np.mean(np.abs(df_test - forecast_df), axis=0)
rmse = np.sqrt(np.mean((df_test - forecast_df)**2, axis=0))

print(f"\nForecast Evaluation ({forecast_steps}-step ahead):")
for i, col in enumerate(df_var.columns):
    print(f"\n{col}:")
    print(f"  MAE: {mae[i]:.4f}")
    print(f"  RMSE: {rmse[i]:.4f}")

# PART 2: VECM with Cointegration
print("\n" + "="*80)
print("PART 2: VECM MODEL (COINTEGRATED SYSTEM)")
print("="*80)

# Generate cointegrated data
# Two I(1) series with r=1 cointegrating relationship
print("\nGenerating cointegrated system...")

# Common trend (random walk)
common_trend = np.cumsum(np.random.normal(0, 1, n))

# Two series with cointegrating relationship: Y1 - 2*Y2 ~ I(0)
beta = 2.0  # Cointegration coefficient
transitory = np.random.normal(0, 0.5, n)

Y1_coint = common_trend + transitory
Y2_coint = (common_trend - transitory) / beta

# Add short-run dynamics
for t in range(2, n):
    ect = Y1_coint[t-1] - beta * Y2_coint[t-1]  # Error correction term
    Y1_coint[t] += -0.15 * ect  # Adjustment speed
    Y2_coint[t] += 0.08 * ect

df_coint = pd.DataFrame({'Y1': Y1_coint, 'Y2': Y2_coint}, index=dates)

print("\nCointegrated Data (First 10 observations):")
print(df_coint.head(10))

# Test for unit roots
print("\n" + "="*80)
print("UNIT ROOT TESTS (Should be I(1))")
print("="*80)

for col in df_coint.columns:
    adf_test(df_coint[col], col)

# Test cointegration
print("\n" + "="*80)
print("JOHANSEN COINTEGRATION TEST")
print("="*80)

# Johansen test
johansen_result = coint_johansen(df_coint, det_order=0, k_ar_diff=2)

print("\nJohansen Test Results:")
print("\nTrace Statistic:")
for i in range(len(johansen_result.lr1)):
    print(f"  r ≤ {i}: Statistic = {johansen_result.lr1[i]:.2f}, "
          f"Critical Value (5%) = {johansen_result.cvt[i, 1]:.2f}, "
          f"Result: {'Reject' if johansen_result.lr1[i] > johansen_result.cvt[i, 1] else 'Fail to reject'}")

print("\nMax Eigenvalue Statistic:")
for i in range(len(johansen_result.lr2)):
    print(f"  r = {i}: Statistic = {johansen_result.lr2[i]:.2f}, "
          f"Critical Value (5%) = {johansen_result.cvm[i, 1]:.2f}, "
          f"Result: {'Reject' if johansen_result.lr2[i] > johansen_result.cvm[i, 1] else 'Fail to reject'}")

# Determine cointegration rank
coint_rank = 0
for i in range(len(johansen_result.lr1)):
    if johansen_result.lr1[i] > johansen_result.cvt[i, 1]:
        coint_rank = i + 1
        
print(f"\n*** Cointegration Rank: r = {coint_rank} ***")
print(f"Number of cointegrating relationships: {coint_rank}")
print(f"Number of common trends: {len(df_coint.columns) - coint_rank}")

if coint_rank > 0:
    print("\nEstimated Cointegrating Vector(s):")
    for i in range(coint_rank):
        print(f"  β{i+1}': {johansen_result.evec[:, i]}")

# Estimate VECM
print("\n" + "="*80)
print("VECM ESTIMATION")
print("="*80)

if coint_rank > 0:
    # Select lag order for VECM
    vecm_lag = select_order(df_coint, maxlags=8, deterministic='ci')
    print(f"\nOptimal VECM lag: {vecm_lag.aic}")
    
    # Fit VECM
    vecm_model = VECM(df_coint, k_ar_diff=vecm_lag.aic, coint_rank=coint_rank, deterministic='ci')
    vecm_fitted = vecm_model.fit()
    
    print(vecm_fitted.summary())
    
    # Extract parameters
    print("\n" + "="*80)
    print("VECM PARAMETER INTERPRETATION")
    print("="*80)
    
    alpha = vecm_fitted.alpha
    beta = vecm_fitted.beta
    
    print("\nCointegrating Vector (β):")
    print(beta)
    print(f"\nEquilibrium relationship: {beta[0,0]:.3f}*Y1 + {beta[1,0]:.3f}*Y2 = 0")
    print(f"Normalized: Y1 ≈ {-beta[1,0]/beta[0,0]:.3f} * Y2")
    
    print("\nAdjustment Speeds (α - Loading Matrix):")
    print(alpha)
    for i, col in enumerate(df_coint.columns):
        print(f"\n{col}:")
        print(f"  α = {alpha[i,0]:.4f}")
        if alpha[i,0] != 0:
            half_life = np.log(0.5) / np.log(1 + alpha[i,0]) if (1 + alpha[i,0]) > 0 and (1 + alpha[i,0]) < 1 else np.inf
            print(f"  Half-life: {half_life:.2f} periods")
            print(f"  Interpretation: {'Adjusts to restore equilibrium' if alpha[i,0] < 0 else 'Moves away from equilibrium (check)'}")
    
    # VECM forecasting
    print("\n" + "="*80)
    print("VECM FORECASTING")
    print("="*80)
    
    df_coint_train = df_coint.iloc[:train_size]
    df_coint_test = df_coint.iloc[train_size:]
    
    vecm_train = VECM(df_coint_train, k_ar_diff=vecm_lag.aic, coint_rank=coint_rank, deterministic='ci')
    vecm_train_fitted = vecm_train.fit()
    
    # Forecast
    vecm_forecast = vecm_train_fitted.predict(steps=test_size)
    vecm_forecast_df = pd.DataFrame(vecm_forecast, index=df_coint_test.index, columns=df_coint.columns)
    
    # Compare to naive VAR in differences
    df_coint_diff = df_coint_train.diff().dropna()
    var_diff = VAR(df_coint_diff).fit(vecm_lag.aic)
    
    # Cumulative sum of forecasted differences
    last_level = df_coint_train.iloc[-1].values
    var_diff_forecast = var_diff.forecast(df_coint_diff.values[-vecm_lag.aic:], steps=test_size)
    var_levels_forecast = last_level + np.cumsum(var_diff_forecast, axis=0)
    var_levels_df = pd.DataFrame(var_levels_forecast, index=df_coint_test.index, columns=df_coint.columns)
    
    # Evaluation
    vecm_rmse = np.sqrt(np.mean((df_coint_test - vecm_forecast_df)**2, axis=0))
    var_rmse = np.sqrt(np.mean((df_coint_test - var_levels_df)**2, axis=0))
    
    print(f"\nForecast RMSE Comparison ({test_size}-step ahead):")
    for i, col in enumerate(df_coint.columns):
        print(f"\n{col}:")
        print(f"  VECM: {vecm_rmse[i]:.4f}")
        print(f"  VAR(diff): {var_rmse[i]:.4f}")
        print(f"  Improvement: {(1 - vecm_rmse[i]/var_rmse[i])*100:.2f}%")

# Visualizations
fig = plt.figure(figsize=(20, 16))

# VAR plots
# Plot 1: Time series
ax1 = plt.subplot(4, 3, 1)
df_var.plot(ax=ax1, alpha=0.8)
ax1.set_title('Simulated VAR Data', fontweight='bold')
ax1.set_xlabel('Date')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# Plot 2: IRF
ax2 = plt.subplot(4, 3, 2)
irf.plot(impulse='GDP_Growth', response='Unemployment_Rate', ax=ax2)
ax2.set_title('IRF: GDP → Unemployment', fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: IRF
ax3 = plt.subplot(4, 3, 3)
irf.plot(impulse='Interest_Rate', response='GDP_Growth', ax=ax3)
ax3.set_title('IRF: Interest Rate → GDP', fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: FEVD for GDP
ax4 = plt.subplot(4, 3, 4)
horizons = range(1, 21)
for j, shock in enumerate(df_var.columns):
    fevd_series = [fevd.decomp[h-1, 0, j] * 100 for h in horizons]
    ax4.plot(horizons, fevd_series, label=shock, marker='o', markersize=4)
ax4.set_title('FEVD: GDP Growth Variance', fontweight='bold')
ax4.set_xlabel('Horizon')
ax4.set_ylabel('% Variance Explained')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: Forecast vs Actual
ax5 = plt.subplot(4, 3, 5)
ax5.plot(df_test.index, df_test['GDP_Growth'], label='Actual', linewidth=2)
ax5.plot(forecast_df.index, forecast_df['GDP_Growth'], label='Forecast', linestyle='--', linewidth=2)
ax5.set_title('VAR Forecast: GDP Growth', fontweight='bold')
ax5.set_xlabel('Date')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Residuals
ax6 = plt.subplot(4, 3, 6)
residuals['GDP_Growth'].plot(ax=ax6, alpha=0.7)
ax6.axhline(0, color='red', linestyle='--')
ax6.set_title('VAR Residuals: GDP Growth', fontweight='bold')
ax6.set_ylabel('Residual')
ax6.grid(alpha=0.3)

# VECM plots
# Plot 7: Cointegrated series
ax7 = plt.subplot(4, 3, 7)
df_coint.plot(ax=ax7, alpha=0.8)
ax7.set_title('Cointegrated Series (I(1))', fontweight='bold')
ax7.set_xlabel('Date')
ax7.legend()
ax7.grid(alpha=0.3)

# Plot 8: Spread (equilibrium relationship)
ax8 = plt.subplot(4, 3, 8)
spread = df_coint['Y1'] - (-beta[1,0]/beta[0,0]) * df_coint['Y2']
spread.plot(ax=ax8, alpha=0.8, color='green')
ax8.axhline(0, color='red', linestyle='--')
ax8.set_title(f'Equilibrium Error (Y1 - {-beta[1,0]/beta[0,0]:.2f}*Y2)', fontweight='bold')
ax8.set_ylabel('Deviation')
ax8.grid(alpha=0.3)

# Plot 9: ACF of spread
ax9 = plt.subplot(4, 3, 9)
plot_acf(spread, lags=30, ax=ax9, alpha=0.05)
ax9.set_title('ACF of Equilibrium Error (Should decay)', fontweight='bold')
ax9.grid(alpha=0.3)

# Plot 10: VECM forecast
ax10 = plt.subplot(4, 3, 10)
ax10.plot(df_coint_test.index, df_coint_test['Y1'], label='Actual', linewidth=2)
ax10.plot(vecm_forecast_df.index, vecm_forecast_df['Y1'], label='VECM', linestyle='--', linewidth=2)
ax10.plot(var_levels_df.index, var_levels_df['Y1'], label='VAR(diff)', linestyle=':', linewidth=2, alpha=0.7)
ax10.set_title('VECM vs VAR Forecast: Y1', fontweight='bold')
ax10.set_xlabel('Date')
ax10.legend()
ax10.grid(alpha=0.3)

# Plot 11: Forecast errors
ax11 = plt.subplot(4, 3, 11)
vecm_errors = df_coint_test['Y1'] - vecm_forecast_df['Y1']
var_errors = df_coint_test['Y1'] - var_levels_df['Y1']
ax11.plot(df_coint_test.index, vecm_errors, label='VECM', alpha=0.7)
ax11.plot(df_coint_test.index, var_errors, label='VAR(diff)', alpha=0.7)
ax11.axhline(0, color='red', linestyle='--')
ax11.set_title('Forecast Errors: Y1', fontweight='bold')
ax11.set_ylabel('Error')
ax11.legend()
ax11.grid(alpha=0.3)

# Plot 12: Heatmap of Granger causality p-values
ax12 = plt.subplot(4, 3, 12)
causality_matrix = np.ones((3, 3))
for (causing, caused), pval in causality_results.items():
    i = list(df_var.columns).index(caused)
    j = list(df_var.columns).index(causing)
    causality_matrix[i, j] = pval

sns.heatmap(causality_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            xticklabels=df_var.columns, yticklabels=df_var.columns, 
            vmin=0, vmax=0.1, ax=ax12, cbar_kws={'label': 'p-value'})
ax12.set_title('Granger Causality P-values\n(Row ← Column)', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. VAR({optimal_lag}) captures dynamic interdependencies among {len(df_var.columns)} variables")
print(f"2. Granger causality: {sum(p < 0.05 for p in causality_results.values())} significant relationships detected")
print(f"3. IRF shows shock transmission over 20 periods with Cholesky ordering")
print(f"4. FEVD: GDP variance explained {fevd.decomp[19, 0, 0]*100:.1f}% by own shocks at h=20")
print(f"5. Cointegration rank: r={coint_rank} equilibrium relationship(s) in system")
print(f"6. VECM adjustment speed: α={alpha[0,0]:.3f} for Y1 (half-life ~{np.log(0.5)/np.log(1+alpha[0,0]):.1f} periods)")
print(f"7. VECM outperforms VAR in differences by ~{np.mean((1 - vecm_rmse/var_rmse)*100):.1f}% RMSE")
print(f"8. Stability check: All eigenvalues |λ| < 1 → {'STABLE' if stable else 'CHECK REQUIRED'}")
print(f"9. Long-run equilibrium: Y1 ≈ {-beta[1,0]/beta[0,0]:.2f} * Y2")
print(f"10. Forecast horizon: {test_size} steps out-of-sample evaluation")

print("\n" + "="*80)
print("WORKFLOW SUMMARY")
print("="*80)
print("VAR:")
print("  1. Test stationarity (ADF) → All I(0) required")
print("  2. Select lag order (AIC/BIC)")
print("  3. Estimate by OLS equation-by-equation")
print("  4. Diagnose residuals (autocorrelation, normality)")
print("  5. Granger causality tests")
print("  6. Compute IRF (Cholesky or structural identification)")
print("  7. FEVD for variance attribution")
print("  8. Out-of-sample forecasting")
print("\nVECM:")
print("  1. Test unit roots → Series should be I(1)")
print("  2. Johansen test for cointegration rank r")
print("  3. If r > 0: Estimate VECM (error correction)")
print("  4. Interpret β (long-run equilibrium) and α (adjustment speeds)")
print("  5. Forecast with equilibrium constraint")
print("  6. Compare to VAR in differences (should outperform)")
```

## 6. Challenge Round
Advanced VAR/VECM applications and extensions:

1. **Structural VAR with Sign Restrictions:** Identify monetary policy shock (interest rate) using sign restrictions: GDP↓, Inflation↓, Rate↑ on impact. Compare to Cholesky ordering. How sensitive are IRFs to identification scheme?

2. **Bayesian VAR for High Dimensions:** Estimate BVAR with K=20 macroeconomic variables using Minnesota prior. Tune hyperparameters (λ, θ) via cross-validation. Compare forecast accuracy to VAR (likely overfits) and factor models (FAVAR).

3. **Time-Varying Parameter VAR:** Allow coefficients to evolve (TVP-VAR with stochastic volatility). Estimate via Kalman filter or Gibbs sampling. How do monetary policy transmission mechanisms change over time (1980s vs 2000s vs post-2008)?

4. **Panel VAR:** Multi-country dataset (GDP, interest rates, inflation for 10 countries). Estimate with fixed effects. Test for cross-country spillovers. How do US shocks affect Europe?

5. **Cointegration with Structural Breaks:** Test for cointegration allowing for regime shifts (Gregory-Hansen test). Exchange rate data with policy regime changes. Does cointegration break down during crises?

6. **Long-Run Restrictions (Blanchard-Quah):** Decompose shocks into supply (permanent GDP effect) vs demand (transitory). Estimate using long-run identifying restrictions. What fraction of GDP variance from supply shocks?

7. **Forecast Combination:** Average forecasts from VAR, VECM, BVAR, univariate ARIMA, exponential smoothing. Optimal weights via inverse MSE. Does combination beat best single model on rolling window?

## 7. Key References
- [Sims, "Macroeconomics and Reality" (Econometrica, 1980)](https://www.jstor.org/stable/1912017) - foundational VAR paper, Nobel Prize work for empirical macro methodology
- [Johansen, "Statistical Analysis of Cointegration Vectors" (Journal of Economic Dynamics and Control, 1988)](https://www.sciencedirect.com/science/article/abs/pii/0165188988900413) - maximum likelihood cointegration testing, VECM framework
- [Lütkepohl, "New Introduction to Multiple Time Series Analysis" (2005)](https://link.springer.com/book/10.1007/978-3-540-27752-1) - comprehensive textbook covering VAR/VECM theory, estimation, testing

---
**Status:** Core multivariate time series methodology | **Complements:** ARIMA Models, Cointegration Theory, Structural Econometrics, Macroeconomic Forecasting, Monetary Policy Analysis, Portfolio Risk Management
