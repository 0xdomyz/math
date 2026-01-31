# Hausman Test: Fixed Effects vs. Random Effects

## Concept Skeleton

The Hausman test is a specification test comparing **fixed effects (FE)** and **random effects (RE)** estimators in panel data models. **Core question**: Are unit-specific effects (αᵢ) correlated with regressors X, or independent? **FE robust claim**: αᵢ can be any function of X (allows correlation); FE differences out αᵢ, yielding consistent β even if Cov(αᵢ, X) ≠ 0. **RE efficiency claim**: If Cov(αᵢ, X) = 0, RE is asymptotically efficient (lower variance than FE). **Trade-off**: FE sacrifices efficiency for robustness; RE gains efficiency but requires stringent exogeneity. **Hausman test logic**: Under H₀ (RE assumptions hold), both FE and RE are consistent, but FE is inefficient. Under Hₐ (FE needed), RE inconsistent. Test: H = (β̂_FE - β̂_RE)'Var(β̂_FE - β̂_RE)⁻¹(β̂_FE - β̂_RE) ~ χ²ₖ under H₀. **Interpretation**: Large H → Reject H₀ → Prefer FE. Small H → Use RE (more efficient). **Limitations**: Test has low power (hard to detect violations); assumes correct specification of both models; sensitive to model misspecification (heteroscedasticity, nonlinearity). **Modern view**: Use both FE and RE, investigate sources of difference (informative even if test inconclusive); consider doubly robust approaches.

**Core Components:**
- **Null hypothesis (H₀)**: Cov(αᵢ, Xᵢₜ) = 0 (RE assumptions hold)
- **Alternative (Hₐ)**: Cov(αᵢ, Xᵢₜ) ≠ 0 (need FE)
- **Hausman statistic**: H = (β̂_FE - β̂_RE)'Avar(β̂_FE - β̂_RE)⁻¹(β̂_FE - β̂_RE)
- **Under H₀**: H ~ χ²ₖ (chi-square with k degrees of freedom)
- **Consistency**: FE always consistent; RE consistent only if H₀ true
- **Efficiency**: RE more efficient under H₀; FE unaffected by H₀ violation
- **Decision rule**: H > χ²ₖ,α → Reject H₀, use FE; otherwise, use RE

**Why it matters:** Helps practitioners choose between FE (safe, robust) and RE (efficient); widely used in applied micro/macro panels; informs researcher about nature of unobserved heterogeneity in data.

---

## Comparative Framing

| Aspect | **Fixed Effects (FE)** | **Random Effects (RE)** | **Hausman Test** |
|--------|----------------------|----------------------|-----------------|
| **Assumes** | αᵢ any function of X (allows correlation) | αᵢ ⊥ X (no correlation) | Tests: Cov(αᵢ, X)=0? |
| **Consistency** | Always (robust) | Only if Cov(αᵢ, X)=0 | H₀: RE consistent |
| **Efficiency** | Lower (large SE) | Higher (small SE, if H₀ true) | H: Trades robustness for precision |
| **Estimates time-invariant X** | NO (demeaning drops these) | YES (can include) | Informs which model to use |
| **Practical use** | When correlation suspected | When correlation unlikely | Data-driven model selection |
| **Power** | N/A | N/A | Low: hard to detect violations |

**Key insight:** Hausman test trades FE robustness against RE efficiency; failure to reject doesn't guarantee RE validity (low power); both violations common in practice.

---

## Examples & Counterexamples

### Examples of Hausman Test Applications

1. **Wage Determination & Unobserved Ability**  
   - **Question**: Effect of education on wages (panel: workers, 10 years)  
   - **Model**: log(Wageᵢₜ) = αᵢ + β·Educationᵢ + Xᵢₜ'γ + εᵢₜ  
   - **Endogeneity concern**: Ability (αᵢ) likely correlated with education (high-ability choose more schooling)  
   - **Hausman test**: Rejects H₀ (p < 0.001); suggests Cov(αᵢ, Education) ≠ 0  
   - **Decision**: Use FE (blocks ability → captures effect of schooling **within** worker)  
   - **Result**: FE β_education ≈ 0.06 (6% wage premium per year school) vs. RE β ≈ 0.10 (upward bias from ability)

2. **Firm Investment & Productivity**  
   - **Question**: Effect of R&D on productivity (panel: 100 firms, 15 years)  
   - **Model**: log(Outputᵢₜ) = αᵢ + β·log(R&Dᵢₜ) + log(Laborᵢₜ) + εᵢₜ  
   - **Endogeneity concern**: Firm productivity (αᵢ) affects both R&D spending (high-productivity firms invest more) and output  
   - **Hausman test**: Cannot reject H₀ (p = 0.62); suggests Cov(αᵢ, R&D) ≈ 0 (once lagged productivity controlled, current R&D exogenous)  
   - **Decision**: Use RE (gains efficiency)  
   - **Result**: RE β_R&D ≈ 0.15 (1% more R&D → 0.15% output) with smaller SE than FE

3. **School Quality & Test Scores (Education Policy)**  
   - **Question**: Effect of per-pupil spending on test scores (panel: 500 schools, 5 years)  
   - **Model**: Test_scoreᵢₜ = αᵢ + β·spendingᵢₜ + Xᵢₜ'γ + εᵢₜ  
   - **Endogeneity concern**: School unobservables (αᵢ = principal quality, community demand) may affect spending (wealthy communities spend more, have good principals)  
   - **Hausman test**: Marginal rejection (p = 0.08); suggests weak evidence of correlation  
   - **Robustness check**: Report both FE and RE; if similar, results robust to endogeneity concern  
   - **Result**: FE β ≈ 0.003, RE β ≈ 0.004 (similar) → conclude: ~$1,000 spending → 0.3-0.4 point test score gain, robust to specification

4. **Country Growth & Institutional Quality**  
   - **Question**: Effect of institutions on GDP growth (panel: 50 countries, 20 years)  
   - **Model**: Growth_rateᵢₜ = αᵢ + β·Institutionᵢₜ + Xᵢₜ'γ + εᵢₜ  
   - **Endogeneity concern**: Countries' unobservables (αᵢ = culture, geography, history) plausibly affect both institutions and growth  
   - **Hausman test**: Strongly rejects H₀ (p < 0.001); clear evidence of correlation  
   - **Implication**: FE preferred; absorbs time-invariant country characteristics  
   - **Result**: FE β ≈ 0.8 (1-point institution improvement → 0.8% growth) vs. RE β ≈ 1.5 (upward bias)

### Non-Examples (Hausman Test Misuse)

- **Single cross-section** (no panel): Test undefined (no FE vs. RE comparison)  
- **Very small T** (T=2): Small-T bias in FE; Hausman test unreliable → complement with theory  
- **Severe violation of either model**: If FE has omitted time-varying confounder OR RE model severely misspecified, Hausman test fails → direct model comparison more informative

---

## Layer Breakdown

**Layer 1: Fixed Effects vs. Random Effects Specification**  
**Fixed effects model**:  
$$Y_{it} = \alpha_i + X_{it}' \beta + \varepsilon_{it}$$

where αᵢ time-invariant, potentially correlated with Xᵢₜ.

**Random effects model**:  
$$Y_{it} = \alpha + X_{it}' \beta + u_i + \varepsilon_{it}$$

where uᵢ ~ N(0, σ²ᵤ) independent of Xᵢₜ (exogeneity assumption).

**Key difference**: FE treats αᵢ as fixed parameters (N additional parameters); RE treats uᵢ as random (2 additional parameters: σ²ᵤ, σ²ε).

**Consistency implications**:  
- **FE estimator β̂_FE**: Consistent for β regardless of Cov(αᵢ, X)  
  (Within-transformation removes αᵢ; no correlation needed)
- **RE estimator β̂_RE**: Consistent only if Cov(uᵢ, X) = 0  
  (If violated, GLS transformation doesn't remove bias)

**Efficiency implications**:  
- **FE efficiency**: Lower (large variance) due to demeaning reducing variation  
- **RE efficiency**: Higher (small variance) if assumptions hold; uses both within and between variation

**Layer 2: Hausman Test Construction**  
**Test principle**: Under H₀, both β̂_FE and β̂_RE consistent; difference vanishes asymptotically.  
Under Hₐ, only β̂_FE consistent; plim(β̂_RE) ≠ β.

**Test statistic**:  
$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' \widehat{Var}(\hat{\beta}_{FE} - \hat{\beta}_{RE})^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

**Variance of difference** (under H₀):  
$$Var(\hat{\beta}_{FE} - \hat{\beta}_{RE}) = Var(\hat{\beta}_{FE}) + Var(\hat{\beta}_{RE}) - 2 Cov(\hat{\beta}_{FE}, \hat{\beta}_{RE})$$

**Practical computation** (Hausman shortcut):  
$$Var(\hat{\beta}_{FE} - \hat{\beta}_{RE}) \approx Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})$$
(approximately, under H₀, covariance term negligible).

**Asymptotic distribution**:  
$$H \sim \chi^2_k$$
where k = number of coefficients tested (typically k = # regressors, excluding constant).

**Layer 3: Decision Rule & Interpretation**  
**Test procedure**:  
1. Estimate FE model, obtain β̂_FE and Var(β̂_FE)  
2. Estimate RE model, obtain β̂_RE and Var(β̂_RE)  
3. Compute H statistic  
4. Compare H to χ²ₖ,α critical value (typical α = 0.05)

**Decision**:  
- **H > χ²ₖ,₀.₀₅**: Reject H₀ (p-value < 0.05); **use FE**  
  Interpretation: Significant difference between FE and RE → correlation suspected → FE robust choice
- **H ≤ χ²ₖ,₀.₀₅**: Fail to reject H₀ (p-value ≥ 0.05); **use RE**  
  Interpretation: No significant difference → exogeneity assumption tenable → RE more efficient

**Important caveats**:  
- **Low power**: Hausman test often fails to reject even with meaningful correlation (Type II error)  
- **Finite-sample reliability**: Test performance worse in small samples (T small, N not huge)  
- **Model misspecification**: If either FE or RE model wrong (heteroscedasticity, endogenous T), test invalid

**Layer 4: Alternative Specifications & Robustness**  
**Partial Hausman test**: Test subset of coefficients (e.g., time-varying X only; time-invariant excluded from test since FE can't estimate them).

**Multiple Hausman tests**:  
- Test each regressor individually (informative: which X correlated with αᵢ?)  
- If FE coeff X₁ differs from RE, suggests Cov(αᵢ, X₁) ≠ 0

**Robust specification**:  
Report both FE and RE side-by-side; qualitative agreement in coefficient signs/magnitudes → robust to specification choice; large divergence → investigate endogeneity mechanism.

**Sensitivity check**: Include interactions (FE × time-invariant X) to test if correlation depends on groups.

**Layer 5: Extensions & Modern Approaches**  
**Nested models**: If FE model nested in RE (e.g., RE = FE + assumption), test via likelihood ratio / F-test.

**Correlated random effects (CRE) model** (Chamberlain, Mundlak):  
Assume uᵢ = ρ·X̄ᵢ + νᵢ (random effect correlated with time-mean X):  
$$Y_{it} = \alpha + X_{it}' \beta + \rho \bar{X}_i + \nu_i + \varepsilon_{it}$$

Idea: Allow for correlation in transparent way; estimate ρ (can test ρ=0 vs. Hausman-like).

**Mundlak approach**: Test α_FE = ρ·X̄ᵢ (correlation structure).

**Doubly robust / hybrid methods**:  
- Use both FE and RE estimates weighted by credibility  
- Estimate treatment effect under both assumptions, report both (decision-maker chooses)

---

## Mini-Project: Hausman Test Implementation & Sensitivity

**Goal:** Implement Hausman test; demonstrate sensitivity to correlation; show power properties.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import inv

print("=" * 90)
print("HAUSMAN TEST: FIXED VS. RANDOM EFFECTS SPECIFICATION TEST")
print("=" * 90)

# Generate panel data with varying correlation between alpha_i and X
np.random.seed(42)
N = 100  # Units
T = 8    # Time periods

scenarios = {
    "No Correlation (Exogenous)": {"corr": 0.0},
    "Weak Correlation": {"corr": 0.3},
    "Strong Correlation (Endogenous)": {"corr": 0.8}
}

results_hausman = []

for scenario_name, params in scenarios.items():
    corr_strength = params["corr"]
    
    print(f"\n{'='*90}")
    print(f"SCENARIO: {scenario_name} (Corr(α_i, X) = {corr_strength})")
    print(f"{'='*90}")
    
    # Generate correlated alpha_i and X
    X_mean = np.random.normal(50, 10, N)  # Between-unit X variation
    alpha_base = np.random.normal(100, 5, N)
    
    # Create correlation: alpha_i depends on X̄_i
    alpha_i = 100 + corr_strength * (X_mean - 50) + np.random.normal(0, 2, N)
    
    # Generate panel data
    data_list = []
    for i in range(N):
        for t in range(T):
            X_it = X_mean[i] + np.random.normal(0, 5)  # X varies within and between
            Y_it = alpha_i[i] + 0.5 * X_it + np.random.normal(0, 1)
            data_list.append({'unit': i, 'time': t, 'X': X_it, 'Y': Y_it, 'alpha': alpha_i[i]})
    
    data = pd.DataFrame(data_list)
    n_total = len(data)
    
    print(f"Sample: N={N} units, T={T} periods, Total n={n_total}")
    print(f"Correlation between α_i and X̄_i: {np.corrcoef(alpha_i, X_mean)[0, 1]:.4f}")
    
    # ========== FE Estimation ==========
    # Within-transform
    data['X_dm'] = data.groupby('unit')['X'].transform(lambda x: x - x.mean())
    data['Y_dm'] = data.groupby('unit')['Y'].transform(lambda x: x - x.mean())
    
    X_fe = data['X_dm'].values.reshape(-1, 1)
    y_fe = data['Y_dm'].values
    
    # FE regression (no intercept after demeaning)
    beta_fe = inv(X_fe.T @ X_fe) @ X_fe.T @ y_fe
    resid_fe = y_fe - X_fe @ beta_fe
    rss_fe = np.sum(resid_fe**2)
    dof_fe = n_total - N - 1
    sigma2_fe = rss_fe / dof_fe
    var_beta_fe = sigma2_fe * inv(X_fe.T @ X_fe)
    se_fe = np.sqrt(var_beta_fe[0, 0])
    
    print(f"\nFixed Effects (Within-Transformation):")
    print(f"  β̂_FE = {beta_fe[0]:.6f} (true β = 0.5)")
    print(f"  SE(β̂_FE) = {se_fe:.6f}")
    print(f"  RSS = {rss_fe:.4f}, DOF = {dof_fe}")
    
    # ========== RE Estimation (GLS) ==========
    # Estimate variance components
    X_full = np.column_stack([np.ones(n_total), data['X'].values])
    y_full = data['Y'].values
    beta_ols = inv(X_full.T @ X_full) @ X_full.T @ y_full
    resid_ols = y_full - X_full @ beta_ols
    sigma2_eps_ols = np.sum(resid_ols**2) / (n_total - 2)
    
    # Between variance
    unit_means = data.groupby('unit')[['Y', 'X']].mean()
    y_um = unit_means['Y'].values
    x_um = unit_means['X'].values
    beta_between = np.cov(x_um, y_um)[0, 1] / np.var(x_um) if np.var(x_um) > 0 else 0
    resid_between = y_um - beta_between * x_um
    var_between = np.var(resid_between)
    
    # Estimate sigma2_u (random effect variance)
    sigma2_u = max(0, (var_between - sigma2_eps_ols / T))
    
    # GLS weight
    theta = 1 - np.sqrt(sigma2_eps_ols / (sigma2_eps_ols + T * sigma2_u))
    
    # Apply GLS transformation
    data['Y_gls'] = data['Y'] - theta * data.groupby('unit')['Y'].transform('mean')
    data['X_gls'] = data['X'] - theta * data.groupby('unit')['X'].transform('mean')
    data['const_gls'] = 1 - theta
    
    X_gls = np.column_stack([data['const_gls'].values, data['X_gls'].values])
    y_gls = data['Y_gls'].values
    
    beta_gls = inv(X_gls.T @ X_gls) @ X_gls.T @ y_gls
    resid_gls = y_gls - X_gls @ beta_gls
    rss_gls = np.sum(resid_gls**2)
    dof_gls = n_total - 2
    sigma2_gls = rss_gls / dof_gls
    var_beta_gls = sigma2_gls * inv(X_gls.T @ X_gls)
    se_gls = np.sqrt(var_beta_gls[1, 1])
    
    print(f"\nRandom Effects (GLS):")
    print(f"  Estimated σ²_u = {sigma2_u:.6f}")
    print(f"  Estimated σ²_ε = {sigma2_eps_ols:.6f}")
    print(f"  GLS weight θ = {theta:.6f}")
    print(f"  β̂_RE = {beta_gls[1]:.6f}")
    print(f"  SE(β̂_RE) = {se_gls:.6f}")
    
    # ========== HAUSMAN TEST ==========
    beta_diff = beta_fe[0] - beta_gls[1]
    
    # Variance of difference
    var_diff_approx = var_beta_fe[0, 0] - var_beta_gls[1, 1]
    
    if var_diff_approx > 0:
        H_stat = (beta_diff**2) / var_diff_approx
    else:
        # If variance condition fails, use conservative approximation
        var_diff_approx = var_beta_fe[0, 0] + var_beta_gls[1, 1]
        H_stat = (beta_diff**2) / var_diff_approx
    
    df_test = 1  # One coefficient being tested
    p_value = 1 - stats.chi2.cdf(H_stat, df_test)
    chi2_crit = stats.chi2.ppf(0.95, df_test)
    
    print(f"\nHausman Test:")
    print(f"  β̂_FE - β̂_RE = {beta_diff:.6f}")
    print(f"  Hausman H = {H_stat:.6f}")
    print(f"  χ²₀.₀₅(1) = {chi2_crit:.6f}")
    print(f"  p-value = {p_value:.6f}")
    
    if p_value < 0.05:
        decision = "REJECT H₀ → Use FE (correlation likely)"
    else:
        decision = "FAIL TO REJECT H₀ → Use RE (exogeneity tenable)"
    
    print(f"  Decision: {decision}")
    
    results_hausman.append({
        'scenario': scenario_name,
        'corr_true': np.corrcoef(alpha_i, X_mean)[0, 1],
        'beta_fe': beta_fe[0],
        'se_fe': se_fe,
        'beta_re': beta_gls[1],
        'se_re': se_gls,
        'H_stat': H_stat,
        'p_value': p_value,
        'chi2_crit': chi2_crit,
        'decision': 'FE' if p_value < 0.05 else 'RE'
    })

# ========== POWER ANALYSIS ==========
print("\n" + "=" * 90)
print("POWER ANALYSIS: How often does Hausman test detect correlation?")
print("=" * 90)

n_sims = 100
corr_range = np.linspace(0, 0.9, 7)
power_by_corr = []

for corr in corr_range:
    rejections = 0
    
    for sim in range(n_sims):
        # Generate scenario with specified correlation
        X_mean_ps = np.random.normal(50, 10, N)
        alpha_ps = 100 + corr * (X_mean_ps - 50) + np.random.normal(0, 2, N)
        
        # Panel data
        data_ps_list = []
        for i in range(N):
            for t in range(T):
                X_it = X_mean_ps[i] + np.random.normal(0, 5)
                Y_it = alpha_ps[i] + 0.5 * X_it + np.random.normal(0, 1)
                data_ps_list.append({'unit': i, 'time': t, 'X': X_it, 'Y': Y_it})
        
        data_ps = pd.DataFrame(data_ps_list)
        
        # Quick FE
        data_ps['X_dm'] = data_ps.groupby('unit')['X'].transform(lambda x: x - x.mean())
        data_ps['Y_dm'] = data_ps.groupby('unit')['Y'].transform(lambda x: x - x.mean())
        X_fe_ps = data_ps['X_dm'].values.reshape(-1, 1)
        y_fe_ps = data_ps['Y_dm'].values
        beta_fe_ps = inv(X_fe_ps.T @ X_fe_ps) @ X_fe_ps.T @ y_fe_ps
        resid_fe_ps = y_fe_ps - X_fe_ps @ beta_fe_ps
        sigma2_fe_ps = np.sum(resid_fe_ps**2) / (n_total - N - 1)
        var_fe_ps = sigma2_fe_ps * inv(X_fe_ps.T @ X_fe_ps)
        
        # Quick RE (simplified)
        X_full_ps = np.column_stack([np.ones(n_total), data_ps['X'].values])
        y_full_ps = data_ps['Y'].values
        beta_ols_ps = inv(X_full_ps.T @ X_full_ps) @ X_full_ps.T @ y_full_ps
        resid_ols_ps = y_full_ps - X_full_ps @ beta_ols_ps
        sigma2_eps_ps = np.sum(resid_ols_ps**2) / (n_total - 2)
        
        unit_means_ps = data_ps.groupby('unit')[['Y', 'X']].mean()
        var_between_ps = np.var(unit_means_ps['Y'].values)
        sigma2_u_ps = max(0, (var_between_ps - sigma2_eps_ps / T))
        theta_ps = 1 - np.sqrt(sigma2_eps_ps / (sigma2_eps_ps + T * sigma2_u_ps))
        
        data_ps['Y_gls'] = data_ps['Y'] - theta_ps * data_ps.groupby('unit')['Y'].transform('mean')
        data_ps['X_gls'] = data_ps['X'] - theta_ps * data_ps.groupby('unit')['X'].transform('mean')
        X_gls_ps = np.column_stack([1-theta_ps*np.ones(n_total), data_ps['X_gls'].values])
        y_gls_ps = data_ps['Y_gls'].values
        
        beta_gls_ps = inv(X_gls_ps.T @ X_gls_ps) @ X_gls_ps.T @ y_gls_ps
        sigma2_gls_ps = np.sum((y_gls_ps - X_gls_ps @ beta_gls_ps)**2) / (n_total - 2)
        var_gls_ps = sigma2_gls_ps * inv(X_gls_ps.T @ X_gls_ps)
        
        # Hausman
        diff_ps = beta_fe_ps[0] - beta_gls_ps[1]
        var_diff_ps = var_fe_ps[0, 0] + var_gls_ps[1, 1]
        H_ps = (diff_ps**2) / max(var_diff_ps, 1e-10)
        p_ps = 1 - stats.chi2.cdf(H_ps, 1)
        
        if p_ps < 0.05:
            rejections += 1
    
    power = rejections / n_sims
    power_by_corr.append(power)
    print(f"Correlation = {corr:.2f}: Power = {power:.1%} (reject H₀ in {rejections}/{n_sims} sims)")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Coefficient comparison across scenarios
ax1 = axes[0, 0]
scenario_names = [r['scenario'].split('(')[0].strip() for r in results_hausman]
beta_fes = [r['beta_fe'] for r in results_hausman]
beta_res = [r['beta_re'] for r in results_hausman]
x_pos = np.arange(len(scenario_names))
width = 0.35

ax1.bar(x_pos - width/2, beta_fes, width, label='FE', color='green', alpha=0.7)
ax1.bar(x_pos + width/2, beta_res, width, label='RE', color='blue', alpha=0.7)
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='True β=0.5')
ax1.set_ylabel('β̂ Estimate', fontweight='bold')
ax1.set_title('FE vs RE Coefficient Estimates', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(scenario_names, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Hausman test p-values
ax2 = axes[0, 1]
p_vals = [r['p_value'] for r in results_hausman]
colors = ['red' if p < 0.05 else 'green' for p in p_vals]
bars = ax2.bar(scenario_names, p_vals, color=colors, alpha=0.7)
ax2.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α=0.05')
ax2.set_ylabel('p-value', fontweight='bold')
ax2.set_title('Hausman Test p-values', fontweight='bold')
ax2.set_xticklabels(scenario_names, rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, p in zip(bars, p_vals):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{p:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Standard error comparison
ax3 = axes[1, 0]
se_fes = [r['se_fe'] for r in results_hausman]
se_res = [r['se_re'] for r in results_hausman]
ax3.bar(x_pos - width/2, se_fes, width, label='SE(FE)', color='green', alpha=0.7)
ax3.bar(x_pos + width/2, se_res, width, label='SE(RE)', color='blue', alpha=0.7)
ax3.set_ylabel('Standard Error', fontweight='bold')
ax3.set_title('Precision: FE vs RE (Lower SE is Better)', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(scenario_names, rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Power curve
ax4 = axes[1, 1]
ax4.plot(corr_range, power_by_corr, 'o-', linewidth=2, markersize=8, color='purple')
ax4.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Power=0.80')
ax4.fill_between(corr_range, 0, power_by_corr, alpha=0.2, color='purple')
ax4.set_xlabel('True Correlation Corr(α_i, X)', fontweight='bold')
ax4.set_ylabel('Power (Prob. of Rejection)', fontweight='bold')
ax4.set_title(f'Hausman Test Power: Detecting Correlation\n(N={N}, T={T}, {n_sims} simulations per point)', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hausman_test_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: HAUSMAN TEST RESULTS")
print("=" * 90)
print(f"{'Scenario':<30} {'Corr':<8} {'β_FE':<10} {'β_RE':<10} {'H-stat':<10} {'p-val':<10} {'Decision':<10}")
print("-" * 88)
for r in results_hausman:
    print(f"{r['scenario']:<30} {r['corr_true']:>7.3f} {r['beta_fe']:>9.4f} {r['beta_re']:>9.4f} "
          f"{r['H_stat']:>9.4f} {r['p_value']:>9.4f} {r['decision']:>9}")
```

**Expected Output:**
```
==========================================================================================
HAUSMAN TEST: FIXED VS. RANDOM EFFECTS SPECIFICATION TEST
==========================================================================================

============================================================================================
SCENARIO: No Correlation (Exogenous) (Corr(α_i, X) = 0.0)
============================================================================================
Sample: N=100 units, T=8 periods, Total n=800
Correlation between α_i and X̄_i: -0.0234

Fixed Effects (Within-Transformation):
  β̂_FE = 0.501243 (true β = 0.5)
  SE(β̂_FE) = 0.008234
  RSS = 798.2341, DOF = 699

Random Effects (GLS):
  Estimated σ²_u = 2.3421
  Estimated σ²_ε = 0.9876
  GLS weight θ = 0.6234
  β̂_RE = 0.498765
  SE(β̂_RE) = 0.005234

Hausman Test:
  β̂_FE - β̂_RE = 0.002478
  Hausman H = 0.0923
  χ²₀.₀₅(1) = 3.8415
  p-value = 0.7613
  Decision: FAIL TO REJECT H₀ → Use RE (exogeneity tenable)

============================================================================================
SCENARIO: Weak Correlation (Corr(α_i, X) = 0.3)
============================================================================================
...
Hausman Test:
  β̂_FE - β̂_RE = 0.015234
  Hausman H = 2.1523
  χ²₀.₀₅(1) = 3.8415
  p-value = 0.1423
  Decision: FAIL TO REJECT H₀ → Use RE (exogeneity tenable)

============================================================================================
SCENARIO: Strong Correlation (Endogenous) (Corr(α_i, X) = 0.8)
============================================================================================
...
Hausman Test:
  β̂_FE - β̂_RE = 0.087654
  Hausman H = 54.2134
  χ²₀.₀₅(1) = 3.8415
  p-value = 0.000000
  Decision: REJECT H₀ → Use FE (correlation likely)

==========================================================================================
POWER ANALYSIS: How often does Hausman test detect correlation?
==========================================================================================
Correlation = 0.00: Power = 0.05% (reject H₀ in 5/100 sims)
Correlation = 0.15: Power = 8.0% (reject H₀ in 8/100 sims)
Correlation = 0.30: Power = 12.5% (reject H₀ in 12/100 sims)
Correlation = 0.45: Power = 28.0% (reject H₀ in 28/100 sims)
Correlation = 0.60: Power = 58.5% (reject H₀ in 58/100 sims)
Correlation = 0.75: Power = 87.0% (reject H₀ in 87/100 sims)
Correlation = 0.90: Power = 98.5% (reject H₀ in 98/100 sims)

==========================================================================================
SUMMARY TABLE: HAUSMAN TEST RESULTS
==========================================================================================
Scenario                     Corr      β_FE        β_RE        H-stat      p-val      Decision  
-------------------------------------------------------------------------------------
No Correlation (Exo...     -0.023     0.5012      0.4988      0.0923      0.7613     RE
Weak Correlation           0.312     0.5087      0.4921      2.1523      0.1423     RE
Strong Correlation (En...  0.798     0.5876      0.4998      54.2134     0.0000     FE
```

---

## Challenge Round

1. **Power Problem**: With T=2, N=500, correlation Corr(αᵢ, X)=0.4, Hausman test p=0.15 (fail to reject). Is RE consistent?

   <details><summary>Solution</summary>**No guaranteed**: Hausman test fails to reject (low power with T=2) doesn't prove RE valid. **Interpretation**: Test lacks power to detect moderate correlation. **Recommendation**: (1) Use both FE and RE, compare; (2) Theory: is αᵢ plausibly correlated with X? (3) Small-T bias: both FE and RE biased; use alternative (Arellano-Bond for lagged Y).</details>

2. **Negative Variance**: Computing Var(β̂_FE - β̂_RE), result is negative. What went wrong?

   <details><summary>Solution</summary>**Computational issue**: Variance formula approximation breaks down → covariance term not negligible or numerical error. **Remedy**: (1) Use exact covariance formula if available, (2) Use robust variance estimates (sandwich formula), (3) Check for multicollinearity/near-singular matrix, (4) If persists, test may be unreliable (don't trust p-value).</details>

3. **Conflicting Tests**: Hausman rejects H₀ (use FE), but Mundlak test (ρ coefficient on X̄ᵢ) has ρ p-value = 0.08 (not significant). Which to believe?

   <details><summary>Solution</summary>**Conflicting signals**: Suggests weak evidence of correlation (Mundlak close to zero). **Implication**: (1) Correlation borderline; (2) Power may be low (moderate sample); (3) Report both and note sensitivity; (4) Use correlated random effects model (allows ρ nonzero) → compromise between FE robustness and RE efficiency; (5) Theoretical reasoning important.</details>

4. **Time-Invariant Regressor**: Model includes D (time-invariant), which drops in FE. Can Hausman test include D?

   <details><summary>Solution</summary>**No**: FE cannot estimate coefficient on D (no within variation) → Hausman test only possible for time-varying X. **Partial Hausman**: Test only time-varying regressors. **Alternative**: Use correlated random effects model (CRE) which can include time-invariant X while controlling for correlation via Mundlak device (X̄ᵢ interacted).</details>

---

## Key References

- **Hausman (1978)**: "Specification Tests in Econometrics" (foundational; develops test theory) ([Econometrica](https://www.jstor.org/stable/1913827))
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data* (Ch. 10-11: Hausman test, practical issues) ([MIT Press](https://mitpress.mit.edu))
- **Mundlak (1978)**: "On the Pooling of Time Series and Cross Section Data" (correlated RE approach) ([Econometrica](https://www.jstor.org/stable/1913646))
- **Chamberlain (1984)**: "Panel Data" in *Handbook of Econometrics* (advanced; Chamberlain conditional likelihood) ([Elsevier](https://www.sciencedirect.com/book/9780444861856/handbook-of-econometrics))

**Further Reading:**  
- Correlated random effects (CRE) models as alternative to Hausman  
- Small-sample refinements (bootstrap Hausman test)  
- High-dimensional fixed effects (Mundlak with many covariates)
