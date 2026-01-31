# Bayesian Model Comparison

## 1. Concept Skeleton
**Definition:** Framework comparing competing models using posterior probability of each model given data; Bayes factors quantify relative evidence  
**Purpose:** Select best-fitting model while accounting for complexity; avoid overfitting (automatic via penalty for parameters); unified probabilistic approach  
**Prerequisites:** Marginal likelihood computation, Bayes factor interpretation, model specification, prior elicitation for model space

## 2. Comparative Framing
| Method | Framework | Information Used | Penalty for Complexity | Computational Cost | Interpretation |
|--------|-----------|------------------|----------------------|-------------------|-----------------|
| **Bayes Factor** | P(y\|M₁)/P(y\|M₂) | Marginal likelihood | Implicit (prior integration) | Very high | Posterior odds |
| **AIC** | 2k - 2log(L̂) | MLE + k parameters | Linear: 2k | Very low | Relative KL divergence |
| **BIC** | -2log(L̂) + k·log(n) | MLE + sample size | Log(n)·k (stronger) | Very low | Asymptotic Bayes factor |
| **Posterior Predictive** | Compare p(y_new\|y) | Full posterior | Implicit | High | Cross-validated predictions |
| **WAIC** | -2LPPD + 2p_waic | Pointwise likelihood | Data-based | High | Leave-one-out approximation |

## 3. Examples + Counterexamples

**Simple Example:**  
Linear vs quadratic: M₁: y = β₀+β₁x; M₂: y = β₀+β₁x+β₂x² → BF₁₂ = 8 (8× evidence for M₁); simpler model preferred despite M₂ slightly higher R²

**Failure Case:**  
Misspecified priors: Model 1 prior too loose (wide β); Model 2 prior tight → Bayes factor biased toward Model 2 (artificial penalty on Model 1); solution: use same prior info for both

**Edge Case:**  
Non-nested models (different likelihoods; hard to compare BF); use information criteria (AIC, BIC) instead of Bayes factors; or posterior predictive cross-validation

## 4. Layer Breakdown
```
Bayesian Model Comparison Structure:
├─ Bayes Theorem for Model Comparison:
│   ├─ Posterior model probabilities:
│   │   ├─ P(M_j|y) = P(y|M_j)·P(M_j) / Σ_k P(y|M_k)·P(M_k)
│   │   ├─ P(y|M_j): Marginal likelihood (evidence) for model j
│   │   ├─ P(M_j): Prior probability of model j (often equal)
│   │   └─ Δ P(M_j|y) vs P(M_j): Shift due to data
│   ├─ Marginal likelihood (evidence):
│   │   ├─ P(y|M) = ∫ P(y|θ,M)·P(θ|M) dθ
│   │   ├─ Integration over all parameters θ weighted by prior
│   │   ├─ High likelihood under M AND prior believes θ plausible → Large P(y|M)
│   │   ├─ Automatic complexity penalty:
│   │   │   ├─ More parameters → Posterior narrower (around MLE)
│   │   │   ├─ Likelihood peak higher but volume under integral reduced (Occam's razor)
│   │   │   └─ Trade-off: Fit vs complexity implicit in P(y|M)
│   │   └─ Computational burden: Intractable for most models
│   ├─ Example - Gaussian with unknown variance:
│   │   ├─ Model 1: σ² known = 1
│   │   ├─ Model 2: σ² unknown; prior σ² ~ IG(α, δ)
│   │   ├─ Marginal likelihood (Model 1): Integrates over μ
│   │   ├─ Marginal likelihood (Model 2): Integrates over μ, σ²
│   │   ├─ M2 integrand higher variance → Lower integral (penalty for extra parameter)
│   │   └─ If data supports unknown σ² → Penalty small; M2 preferred
│   └─ Equal priors: P(M₁) = P(M₂) → Odds = BF (factor between evidences)
├─ Bayes Factor:
│   ├─ Definition: Ratio of marginal likelihoods
│   │   ├─ BF₁₂ = P(y|M₁) / P(y|M₂)
│   │   ├─ BF₁₂ > 1: Evidence for M₁; BF₁₂ < 1: Evidence for M₂
│   │   ├─ Interpretation scale (Kass & Raftery 1995):
│   │   │   ├─ BF 1-3: Weak evidence
│   │   │   ├─ BF 3-10: Moderate evidence
│   │   │   ├─ BF 10-30: Strong evidence
│   │   │   ├─ BF 30-100: Very strong evidence
│   │   │   └─ BF >100: Decisive evidence
│   │   └─ Opposite (BF₂₁ = 1/BF₁₂): Evidence for model 2
│   ├─ Properties:
│   │   ├─ Asymmetric: BF₁₂ ≠ BF₂₁ (not ±1; ratios instead)
│   │   ├─ Depends critically on priors:
│   │   │   ├─ Wide prior on Model 1 → Lower evidence (volume penalty)
│   │   │   ├─ Narrow prior → Higher evidence (concentrated mass)
│   │   │   └─ Different priors → Different BF (not "objective")
│   │   ├─ Sensitive to sample size:
│   │   │   ├─ Truth model likelihood grows ~n; false model grows <n
│   │   │   ├─ Asymptotically: BF → ∞ for true model (consistency)
│   │   │   └─ Small n: BF variable; prior effects large
│   │   └─ Violates likelihood principle (frequentist criticism):
│   │       ├─ Depends on sampling distribution P(y|θ), not just observed y
│   │       └─ Different stopping rules → Different BF (with same data)
│   ├─ Computation:
│   │   ├─ Challenge: P(y|M) integral high-dimensional, intractable
│   │   ├─ Methods:
│   │   │   ├─ Laplace approximation: Approximate posterior as Normal around mode
│   │   │   │   ├─ BF̂ ∝ P(ŷ|θ̂)·P(θ̂) × (2π)^{k/2}/√|H|
│   │   │   │   ├─ Fast; reasonably accurate for large n
│   │   │   │   └─ Fails if posterior non-normal (multimodal, heavy-tailed)
│   │   │   ├─ Importance sampling: Sample from proposal; weight by posterior/proposal
│   │   │   │   ├─ Accurate but requires good proposal
│   │   │   │   └─ Computationally intensive
│   │   │   ├─ Harmonic mean estimator:
│   │   │   │   ├─ P(y|M) ≈ 1/(1/M)·Σ 1/P(y|θ^m) [θ^m from MCMC]
│   │   │   │   ├─ Easy from posterior samples
│   │   │   │   └─ High variance; unreliable with importance weights
│   │   │   ├─ Thermodynamic integration:
│   │   │   │   ├─ Use tempering (β ∈ [0,1]): P_β(θ|y) ∝ P(y|θ)^β P(θ)
│   │   │   │   ├─ Numerically integrate over β
│   │   │   │   ├─ Accurate; computationally expensive (multiple runs)
│   │   │   │   └─ Stan, PyMC: Sometimes available
│   │   │   └─ Nested sampling: Specific to high-dimensional integrals
│   │   │       ├─ Sophisticated; research algorithms
│   │   │       └─ Software: MultiNest, dynesty (Python)
│   │   └─ Pragmatic: Use approximations; check sensitivity to priors
│   ├─ Sensitivity to priors:
│   │   ├─ Example: Compare normal-linear vs normal-quadratic
│   │   ├─ Varying priors on β₂ (quadratic coeff):
│   │   │   ├─ Prior 1: β₂ ~ N(0, 10²) → BF₁₂ = 5 (slight preference M1)
│   │   │   ├─ Prior 2: β₂ ~ N(0, 1²) → BF₁₂ = 3 (weaker preference M1)
│   │   │   └─ Prior 3: β₂ ~ N(0, 0.1²) → BF₁₂ = 1 (indifferent)
│   │   ├─ Interpretation: Wider prior = lower BF (volume penalty)
│   │   └─ Mitigation: Report BF range over reasonable priors; use default priors
│   └─ Example application - model selection:
│       ├─ Nested: M₁ (intercept); M₂ (intercept + slope)
│       ├─ Compute BF₁₂ using Laplace approximation
│       ├─ If BF > 10 → Strong evidence M₁
│       └─ Decision: Drop slope from model
├─ Information Criteria (Frequentist Approximations):
│   ├─ Akaike Information Criterion (AIC):
│   │   ├─ AIC = -2log(L̂) + 2k where k = # parameters
│   │   ├─ Lower AIC = better model
│   │   ├─ Interpretation: Expected KL divergence (model to true distribution)
│   │   ├─ Approximates Bayes factor asymptotically (under conditions)
│   │   ├─ Less penalty for complexity than BIC (tends toward more parameters)
│   │   └─ Use: Model selection when n moderate; not too large penalty
│   ├─ Bayesian Information Criterion (BIC):
│   │   ├─ BIC = -2log(L̂) + k·log(n)
│   │   ├─ Log(n) weight: Increases with sample size (stronger penalty)
│   │   ├─ Asymptotic approximation: log(BF₁₂) ≈ (BIC₂ - BIC₁)/2
│   │   ├─ Consistency: As n→∞, BIC → true model (frequency theoretically)
│   │   └─ Use: Model selection when n large; prefer simpler models
│   ├─ Comparison:
│   │   ├─ Example: Linear (k=2), quadratic (k=3), cubic (k=4) models
│   │   ├─ AIC: Might choose quadratic or cubic (low penalty)
│   │   ├─ BIC (n=100): Sparser (log(100)≈4.6); prefers linear
│   │   ├─ Bayes Factor: Similar to BIC for large n
│   │   └─ Recommendation: BIC for model selection; AIC for prediction
│   └─ Limitations:
│       ├─ Based on MLE (point estimate); ignores uncertainty
│       ├─ AIC not consistent (selects too complex asymptotically)
│       ├─ BIC assumes single true model exists
│       └─ Neither incorporates posterior model probabilities directly
├─ Posterior Predictive Checks:
│   ├─ Idea: Compare predictions from each model to new data
│   │   ├─ Model 1: p(y_new|y, M₁) = ∫ p(y_new|θ₁, M₁)p(θ₁|y, M₁) dθ₁
│   │   ├─ Model 2: Similar (integrate over θ₂)
│   │   ├─ Compare predictive distributions (Bayesian cross-validation)
│   │   └─ Better prediction → Better model
│   ├─ Implementation (leave-one-out cross-validation):
│   │   ├─ For i = 1 to n:
│   │   │   ├─ Posterior p(θ|y_{-i}, M) [leaving out i-th observation]
│   │   │   ├─ Predictive p(y_i|y_{-i}, M) = ∫ p(y_i|θ, M)p(θ|y_{-i}, M) dθ
│   │   │   └─ Sum: LPPD = Σ log p(y_i|y_{-i}, M) [log posterior predictive density]
│   │   ├─ Compare LPPD across models (higher better)
│   │   ├─ Efficient approximation (WAIC): Avoid refitting n times
│   │   └─ Standard error: Can assess uncertainty in model comparison
│   ├─ Advantages:
│   │   ├─ Direct comparison of predictive performance (intuitive)
│   │   ├─ Not affected by prior specification (data-centric)
│   │   ├─ Naturally handles non-nested models
│   │   └─ Works with approximate posteriors (not exact integration)
│   ├─ Disadvantages:
│   │   ├─ Computationally intensive (refits n times)
│   │   ├─ WAIC approximation sometimes inaccurate
│   │   └─ May favor overfitting if priors weak (not explicit complexity penalty)
│   └─ Example:
│       ├─ Model 1: Logistic; LPPD = -120
│       ├─ Model 2: Probit; LPPD = -118
│       ├─ Difference: 2 log points (slight preference Model 2)
│       └─ SE of difference: 3 → Difference uncertain (overlapping CIs)
├─ Model Averaging:
│   ├─ Idea: Don't select one model; average predictions across models
│   │   ├─ Posterior model probabilities: P(M_j|y)
│   │   ├─ Bayesian model average prediction:
│   │   │   ├─ p(y_new|y) = Σ_j P(M_j|y)·p(y_new|y, M_j)
│   │   │   └─ Weighted average; accounts for model uncertainty
│   │   ├─ Advantage: Accounts for model specification uncertainty
│   │   └─ When to use: Multiple models plausible; avoid arbitrary selection
│   ├─ Computation:
│   │   ├─ Posterior samples from each model
│   │   ├─ Weight by posterior model probabilities
│   │   ├─ For predictions: Sample model from posterior; draw from that model's predictive
│   │   └─ Result: Posterior predictive mixture
│   ├─ Example - mixture regression:
│   │   ├─ Model 1: Single regression (all data same process)
│   │   ├─ Model 2: Mixture regression (two processes)
│   │   ├─ Posterior: 40% M₁, 60% M₂
│   │   ├─ New prediction: 40% from single regression + 60% from mixture
│   │   └─ More robust than single model choice
│   └─ Limitation:
│       ├─ If one model much better: Averaging doesn't help
│       ├─ Interpretation: Combined model (hard to communicate)
│       └─ Recommendation: Use when models genuinely competitive
├─ Practical Workflow:
│   ├─ Step 1: Specify competing models (informed by domain, theory)
│   ├─ Step 2: Choose priors (weakly informative, consistent across models)
│   ├─ Step 3: Fit models; compute posterior probabilities
│   │   ├─ Option A: Bayes Factors (exact but computation-heavy)
│   │   ├─ Option B: Information criteria (fast; reasonable approximation)
│   │   ├─ Option C: Cross-validation (practical; avoids priors)
│   │   └─ Recommendation: Use BIC if n>100; otherwise all three
│   ├─ Step 4: Sensitivity analysis
│   │   ├─ Vary priors on each model
│   │   ├─ Verify BF/AIC rankings stable
│   │   ├─ If sensitive: Report range; caution in conclusions
│   │   └─ If robust: Confidence in model selection
│   ├─ Step 5: Model averaging vs selection
│   │   ├─ If clear winner (BF>30): Use single model
│   │   ├─ If close (BF 1-10): Consider model average
│   │   └─ If tied: Need more data or refine models
│   └─ Documentation:
│       ├─ Report all models considered (not just winner)
│       ├─ Priors for each model
│       ├─ Comparison method & results (BF, AIC, cross-validation)
│       └─ Sensitivity analysis findings
└─ Software & Tools:
    ├─ Stan: Laplace approximation via `generated_quantities`; can export samples
    ├─ PyMC3: Thermodynamic integration (pm.step_logp); WAIC/LOO built-in
    ├─ Bayesian: bridgesampling (R) computes Bayes factors
    ├─ Loo package (R, Python): Efficient WAIC/LOO cross-validation
    └─ Comparative review: Often model-specific; no universal tool
```

**Key Insight:** Bayes factors elegant but computation-intensive; BIC simpler, reasonable for large n; cross-validation practical and priors-free; model averaging accounts for uncertainty; always check prior sensitivity

## 5. Mini-Project
Compare nested models using BIC and cross-validation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import LeaveOneOut

# Set seed
np.random.seed(42)

# Generate data from true quadratic process
n = 100
X = np.linspace(-3, 3, n)
y_true = 1 + 2*X - 0.5*X**2
y = y_true + np.random.normal(0, 0.5, n)

print("="*70)
print("Bayesian Model Comparison: Polynomial Regression")
print("="*70)
print(f"Data: n={n} observations")
print(f"True process: y = 1 + 2x - 0.5x²")
print(f"Noise: σ=0.5")
print("")

# Define models
models = {
    'Linear': 1,
    'Quadratic': 2,
    'Cubic': 3,
    'Quartic': 4
}

# Fit models and compute statistics
results = {}

for model_name, degree in models.items():
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    
    # RSS and likelihood
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    sigma2 = rss / n
    ll = -n/2 * np.log(sigma2) - rss / (2*sigma2)
    
    # AIC and BIC
    k = degree + 1  # coefficients
    aic = -2*ll + 2*k
    bic = -2*ll + k*np.log(n)
    
    # R-squared
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - rss / ss_tot
    
    # Adjusted R²
    r2_adj = 1 - (1-r2)*(n-1)/(n-k)
    
    # Cross-validation (leave-one-out)
    loo_ll = 0
    for i in range(n):
        # Fit without i-th point
        X_loo = np.delete(X, i)
        y_loo = np.delete(y, i)
        coeffs_loo = np.polyfit(X_loo, y_loo, degree)
        
        # Predict i-th point
        y_pred_i = np.polyval(coeffs_loo, X[i])
        residual_i = y[i] - y_pred_i
        
        # Estimate sigma from LOO fit
        y_pred_loo = np.polyval(coeffs_loo, X_loo)
        rss_loo = np.sum((y_loo - y_pred_loo)**2)
        sigma2_loo = rss_loo / (n-1)
        
        # Log-likelihood for i-th point
        loo_ll += -0.5*np.log(sigma2_loo) - residual_i**2 / (2*sigma2_loo)
    
    results[model_name] = {
        'degree': degree,
        'k': k,
        'rss': rss,
        'sigma2': sigma2,
        'll': ll,
        'aic': aic,
        'bic': bic,
        'r2': r2,
        'r2_adj': r2_adj,
        'loo_ll': loo_ll,
        'coeffs': coeffs,
        'y_pred': y_pred
    }

# Print results table
print("Model Comparison Statistics:")
print("-"*70)
print(f"{'Model':<12} {'k':<4} {'R²':<8} {'R²_adj':<8} {'AIC':<8} {'BIC':<8} {'LPPD_LOO':<12}")
print("-"*70)

for name in models.keys():
    res = results[name]
    print(f"{name:<12} {res['k']:<4} {res['r2']:>6.3f}  {res['r2_adj']:>6.3f}  "
          f"{res['aic']:>6.1f}  {res['bic']:>6.1f}  {res['loo_ll']:>10.1f}")

# Model comparison metrics (lower better for AIC/BIC, higher better for LPPD)
print("\n" + "="*70)
print("Model Selection Comparison:")
print("-"*70)

aic_vals = [results[m]['aic'] for m in models.keys()]
bic_vals = [results[m]['bic'] for m in models.keys()]
loo_vals = [results[m]['loo_ll'] for m in models.keys()]

aic_best = np.argmin(aic_vals)
bic_best = np.argmin(bic_vals)
loo_best = np.argmax(loo_vals)

best_names = list(models.keys())
print(f"AIC: Best = {best_names[aic_best]} (Δ from best: AIC weights proportional to exp(-Δ/2))")
for i, name in enumerate(best_names):
    delta_aic = aic_vals[i] - aic_vals[aic_best]
    weight_aic = np.exp(-delta_aic / 2) / np.sum(np.exp(-np.array(aic_vals) + aic_vals[aic_best]) / 2)
    print(f"    {name}: Δ={delta_aic:>6.1f}, weight={weight_aic:>6.1%}")

print(f"\nBIC: Best = {best_names[bic_best]} (Δ from best; interpret as log(BF))")
for i, name in enumerate(best_names):
    delta_bic = bic_vals[i] - bic_vals[bic_best]
    print(f"    {name}: Δ={delta_bic:>6.1f}")

print(f"\nLOO-CV: Best = {best_names[loo_best]} (higher LPPD better)")
for i, name in enumerate(best_names):
    delta_loo = loo_vals[i] - loo_vals[loo_best]
    print(f"    {name}: LPPD={loo_vals[i]:>7.1f}, Δ from best={delta_loo:>6.1f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fitted curves
ax = axes[0, 0]
ax.scatter(X, y, alpha=0.5, s=30, color='blue', label='Data')
for name, degree in models.items():
    y_pred = results[name]['y_pred']
    style = '-' if name in ['Linear', 'Quadratic'] else '--'
    ax.plot(X, y_pred, linestyle=style, linewidth=2, label=f'{name} (degree {degree})')
ax.plot(X, y_true, 'k:', linewidth=2, label='True')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Fitted Regression Models')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: AIC comparison
ax = axes[0, 1]
names_list = list(models.keys())
aic_vals = [results[m]['aic'] for m in names_list]
aic_delta = np.array(aic_vals) - min(aic_vals)
colors = ['red' if delta < 2 else 'orange' if delta < 7 else 'gray' for delta in aic_delta]
ax.bar(range(len(names_list)), aic_delta, color=colors, alpha=0.6, edgecolor='black')
ax.axhline(2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Δ<2 (reasonable)')
ax.axhline(7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Δ>7 (weak evidence)')
ax.set_ylabel('ΔAI = AIC - min(AIC)')
ax.set_title('AIC Model Comparison (lower better)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 3: BIC comparison
ax = axes[1, 0]
bic_vals = [results[m]['bic'] for m in names_list]
bic_delta = np.array(bic_vals) - min(bic_vals)
ax.bar(range(len(names_list)), bic_delta, color='purple', alpha=0.6, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(2.3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Δ≈log(3) moderate')
ax.axhline(4.6, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Δ≈log(100) strong')
ax.set_ylabel('ΔBIC = BIC - min(BIC)')
ax.set_title('BIC Model Comparison (log Bayes Factor scale)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cross-validation comparison
ax = axes[1, 1]
loo_vals = [results[m]['loo_ll'] for m in names_list]
loo_delta = np.array(loo_vals) - min(loo_vals)
colors_loo = ['green' if delta > 0 else 'red' for delta in loo_delta]
ax.bar(range(len(names_list)), loo_delta, color=colors_loo, alpha=0.6, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('ΔLPPD = LPPD - max(LPPD)')
ax.set_title('Leave-One-Out Cross-Validation (higher better)')
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Interpretation:")
print("="*70)
print(f"1. AIC favors {best_names[aic_best]} (lowest AIC)")
print(f"   → Balances fit vs parameters; prefers predictive performance")
print("")
print(f"2. BIC favors {best_names[bic_best]} (lowest BIC; stronger penalty)")
print(f"   → Emphasizes parsimony; prefers simpler model if comparable fit")
print("")
print(f"3. LOO-CV favors {best_names[loo_best]} (highest LPPD)")
print(f"   → Out-of-sample predictive ability (most practically relevant)")
print("")
print("4. True model: Quadratic")
print("   → All methods correctly identify quadratic as best/competitive")
```

## 6. Challenge Round
When model comparison misleads:
- **Prior sensitivity**: Different reasonable priors → Different Bayes factors; report range; don't overstate confidence
- **Non-nested models**: Can't directly compare Bayes factors; use cross-validation or information criteria instead
- **Overfitting (weak priors)**: Too-flexible model with weak regularization → Lower BIC/AIC; use stricter priors or information criteria
- **Small sample**: BIC inconsistent for n small; use AIC or cross-validation; wait for more data
- **Model misspecification**: True model not in candidate set → Best fit still wrong; use residual diagnostics; consider additional models
- **Data-dependent prior selection**: Choose prior after seeing data → Biased comparison; specify priors before analysis (pre-registration)

## 7. Key References
- [Kass & Raftery: Bayes Factors (1995)](https://www.jstor.org/stable/2291521) - Interpretation scale; computation methods
- [Gelman et al: Using Bayes Factors (2013)](https://arxiv.org/abs/1312.0906) - Practical guide; pitfalls
- [Vehtari et al: Practical Bayesian Model Evaluation (2017)](https://arxiv.org/abs/1507.04544) - LOO-CV, WAIC, model comparison

---
**Status:** Core model selection | **Complements:** Bayesian Inference, Prior Distributions, MCMC, Hierarchical Models
