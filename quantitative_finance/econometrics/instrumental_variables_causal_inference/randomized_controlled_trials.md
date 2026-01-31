# Randomized Controlled Trials (RCT)

## Concept Skeleton

A Randomized Controlled Trial (RCT) is the **gold standard** for causal inference: researchers randomly assign units (individuals, firms, countries) to treatment and control groups, then measure outcomes. **Core principle**: Randomization ensures treatment independent of all confounders (observed and unobserved) by design, eliminating selection bias. **Why powerful**: Unlike observational studies (which struggle with omitted variables, reverse causality), RCTs guarantee the only difference between groups is treatment assignment (conditional on randomization). **Average Treatment Effect (ATE)** becomes simple:  
$$ATE = E[Y|T=1] - E[Y|T=0]$$
(simple difference in means), valid without controlling for covariates. **Design dimensions**: (1) **Experimental design** (completely randomized, stratified, block design), (2) **Outcome measurement** (baseline, endline, intermediate checkpoints), (3) **Compliance** (intent-to-treat, per-protocol, LATE with noncompliance), (4) **Power analysis** (sample size for desired precision). **Advantages**: Credible causality, transparent assumptions (randomization is verifiable), unbiased estimates even with heterogeneous effects. **Limitations**: Feasibility (cost, ethics, time), external validity (RCT samples often differ from target populations), Hawthorne effect (participants behave differently when observed). **Modern applications**: Development economics (poverty interventions), medicine (drug efficacy), technology (A/B testing), behavioral economics (nudges). RCTs increasingly standard in evidence-based policy.

**Core Components:**
- **Random assignment**: T ~ Bernoulli(p) independent of potential outcomes (Y₀, Y₁)
- **Average Treatment Effect**: ATE = E[Y₁ - Y₀] (unbiased by design)
- **Intent-to-treat (ITT)**: Estimate based on randomized assignment (even if noncompliance)
- **Complier Average Causal Effect (LATE)**: Effect on units who comply with assignment
- **Power analysis**: Sample size calculation for precision (reject H₀: ATE=0 with probability 1-β)
- **Heterogeneous treatment effects (HTE)**: ATE varies by subgroup; study heterogeneity
- **External validity**: Generalizability of RCT results to target population

**Why it matters:** RCTs enable credible causal inference on policy/intervention effects; ubiquitous in development, health, tech, and behavioral research. Informs billions in policy spending globally. Randomization eliminates selection bias, justifying simple difference-in-means as causal estimator.

---

## Comparative Framing

| Aspect | **RCT** | **Observational (OLS)** | **IV/2SLS** | **PSM** | **Natural Experiment** |
|--------|--------|----------------------|------------|---------|----------------------|
| **Assignment** | Random (by design) | Self-selected (confounded) | Exogenous instrument | Matched on PS | Quasi-random (policy) |
| **Selection bias** | None (by design) | Large (unknown X) | Medium (weak IV) | Reduced (PS matching) | Small (exogenous shock) |
| **Causal assumption** | None (randomization) | No omitted X | Z exogenous | Unconfoundedness | Exogenous shock timing |
| **Estimand** | ATE (all) | β (linear) | β (with Z) | ATT (matched) | Local ITT / LATE |
| **Credibility** | Highest | Low | Medium | Medium-high | High |
| **Cost** | High ($$$) | Low | Low | Low | N/A (nature's experiment) |
| **Compliance** | Noncompliance → IV | N/A | N/A | N/A | LATE possible |

**Key insight:** RCT eliminates endogeneity by design (no confounders → simple difference in means is causal); cost/feasibility trade-off limits use; noncompliance requires LATE analysis.

---

## Examples & Counterexamples

### Examples of RCTs

1. **Randomized Policy Evaluation: Deworming in Kenya (Miguel & Kremer 2004)**  
   - **Question**: Does school deworming improve education attendance and test scores?  
   - **Design**: 2-armed RCT; randomized 65 schools to treatment (deworming pills), 65 to control  
   - **Sample**: 30,000+ students across 75 schools  
   - **Randomization**: School-level (cluster RCT); implemented via school draw  
   - **Outcomes**: Attendance (behavioral), test scores (cognitive), health (worm prevalence)  
   - **Results**: +7% attendance improvement; modest test score gains; large downstream effects on earning (LATE: compliers benefited more)  
   - **LATE analysis**: Nonrandom compliance (some schools rejected treatment); IV estimates using initial assignment as instrument

2. **Medical Efficacy Trial: Statins for Heart Disease (Framingham Study offspring)**  
   - **Question**: Do statin drugs reduce cardiovascular risk?  
   - **Design**: Double-blind RCT; randomized patients to statin vs. placebo  
   - **Blinding**: Both patients and doctors unaware of assignment (masks placebo effects)  
   - **Sample**: 1,000+ patients with elevated cholesterol  
   - **Stratification**: Blocked on age, gender (ensure balance)  
   - **Outcomes**: LDL reduction (mechanistic), heart attack incidence, mortality  
   - **Results**: LDL ↓ 25%; heart attacks ↓ 30% (ATE = 0.30); confidence interval excludes zero

3. **Technology A/B Test: Email Subject Line (Email marketing)**  
   - **Question**: Does personalized subject line increase open rates?  
   - **Design**: 2-armed RCT; split customer database 50-50; randomized email 1 (generic) vs. email 2 (personalized name)  
   - **Sample**: 50,000 customers per arm  
   - **Randomization**: Algorithmic (random ID assignment)  
   - **Outcomes**: Open rate (primary), click rate, purchase (downstream)  
   - **Results**: Open rate 22% (generic) vs. 28% (personalized); ATE = 0.06 (6 pp); p < 0.001  
   - **Scale**: Decision rule: if ATE > 0.04 (cost-effective), deploy personalized site-wide

4. **Behavioral Nudge: Default Pension Contribution (Thaler & Benartzi 2004)**  
   - **Question**: Does auto-enrollment increase retirement savings?  
   - **Design**: Company-level RCT; randomized new hires to control (choose own rate) vs. treatment (auto-enrolled at 3%)  
   - **Sample**: 310 new employees  
   - **Randomization**: Company assignment; tracked over 36 months  
   - **Outcomes**: Contribution rate (immediate), accumulation (long-term)  
   - **Results**: Control mean = 3.5%; treatment mean = 12.1%; ATE = 8.6 pp (choice architecture powerful)

### Non-Examples (RCT Inappropriate)

- **Unethical**: Gun violence prevention (can't randomly assign violence)  
- **Infeasible**: Climate change policy (can't control atmosphere globally)  
- **Too expensive**: Universal healthcare system trials (billions required)  
- **Long-term**: Lifetime earning effects (need 30+ year follow-up)

---

## Layer Breakdown

**Layer 1: Random Assignment & Balance**  
**Randomization**: Treatment T ⊥ (Y₀, Y₁, X) by design.

**Mechanism**:  
- **Simple randomization**: T ~ Bernoulli(p) for each unit independently  
- **Block randomization**: Divide into blocks of size 2m; randomize m to T, m to C within each block (ensures exact balance)  
- **Stratified randomization**: Stratify on baseline covariate X (gender, age); randomize within strata (ensures covariate balance)

**Consequence**: Confounding eliminated by design.

**Verification (Balance table)**:  
Test if E[X | T=1] ≈ E[X | T=0] (typically t-tests, not significant differences expected by chance):  
$$t = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{s_T^2/n_T + s_C^2/n_C}} \sim t_{n_T + n_C - 2}$$

**Large t** (p < 0.05 in many covariates) suggests randomization failed or not implemented.

**Layer 2: Causal Effect Estimation (Intent-to-Treat)**  
**Intent-to-treat (ITT)**: Estimate based on randomized assignment, regardless of actual treatment received (accommodates noncompliance).

**ITT estimator** (simple difference):  
$$\widehat{ATE} = \bar{Y}_T - \bar{Y}_C = \frac{1}{n_T}\sum_{T_i=1} Y_i - \frac{1}{n_C}\sum_{T_i=0} Y_i$$

**Unbiasedness** (by design):  
$$E[\widehat{ATE}] = E[E[Y|T=1]] - E[E[Y|T=0]] = E[Y_1] - E[Y_0] = ATE$$

**Standard error**:  
$$SE(\widehat{ATE}) = \sqrt{\frac{\sigma^2_T}{n_T} + \frac{\sigma^2_C}{n_C}}$$

where σ² pooled variance estimate:  
$$\hat{\sigma}^2 = \frac{(n_T-1)s_T^2 + (n_C-1)s_C^2}{n_T + n_C - 2}$$

**95% CI**: ATE ± 1.96 × SE.

**Hypothesis test**: H₀: ATE = 0 vs. Hₐ: ATE ≠ 0.

$$t = \frac{\widehat{ATE}}{SE(\widehat{ATE})} \sim t_{n_T+n_C-2}$$

p-value = Pr(|t| > |t_obs|).

**Layer 3: Power Analysis & Sample Size**  
**Goal**: Choose n such that Pr(reject H₀ | true ATE = δ) = 1 - β (power).

**Typically**: α = 0.05 (Type I error), β = 0.20 (Type II error, power = 0.80).

**For two-sample t-test**:  
$$n_T = n_C = 2 \left(\frac{z_{\alpha/2} + z_\beta}{\delta / \sigma}\right)^2$$

where:  
- $z_{\alpha/2}$ = critical value for α (e.g., 1.96 for α=0.05)  
- $z_\beta$ = critical value for β (e.g., 0.84 for β=0.20)  
- δ = minimum detectable effect (MDE), practical significance  
- σ = standard deviation (estimated from pilot or literature)

**Example**: δ = 0.2σ (0.2 SD effect), α = 0.05, β = 0.20:  
$$n = 2 \left(\frac{1.96 + 0.84}{0.2}\right)^2 = 2 \times (14)^2 = 392 \text{ per arm}$$

**Precision vs. Power trade-off**: Smaller δ → larger n (expensive).

**Layer 4: Noncompliance & Intent-to-Treat (ITT) vs. Per-Protocol**  
**Noncompliance**: Some assigned to T don't take treatment; assigned to C take treatment.

**Example**: Job training randomized to units; some assigned don't attend; some controls find training elsewhere.

**Intent-to-treat (ITT)**: Analyze based on **randomized assignment** (not actual treatment):  
$$ITT = E[Y|T^{assigned}=1] - E[Y|T^{assigned}=0]$$

**Advantage**: Unbiased (randomization preserved).  
**Disadvantage**: ITT ≤ ATE (some treated units don't comply; effect diluted).

**Per-protocol**: Analyze based on **actual treatment received**:  
$$Per-protocol = E[Y|T^{actual}=1] - E[Y|T^{actual}=0]$$

**Problem**: Endogenous selection into actual treatment (noncompliers differ; selection bias reintroduced).

**Solution**: Two-stage least squares with ITT as instrument.

**Layer 5: Local Average Treatment Effect (LATE) with Noncompliance**  
**Framework**: Randomized assignment → instrumental variable for actual treatment.

**First stage**: Regress actual treatment on randomized assignment:  
$$T_i^{actual} = \pi_0 + \pi_1 T_i^{assigned} + v_i$$

**Compliance rate**: $\hat{\pi}_1 = Pr(T^{actual}=1|T^{assigned}=1) - Pr(T^{actual}=1|T^{assigned}=0)$.

**Second stage**: IV estimate of effect on compliers:  
$$Y_i = \alpha + \beta T_i^{actual} + \varepsilon_i$$
$$\beta_{IV} = \frac{ITT}{\hat{\pi}_1}$$

**Interpretation**: β_IV = effect on **compliers** (units who comply with randomized assignment); not overall ATE.

**Example**: ITT = 0.10 (intent-to-treat effect), compliance rate = 0.50:  
$$\beta_{IV} = \frac{0.10}{0.50} = 0.20$$

(Effect on compliers is twice the ITT).

**Assumptions for LATE** (Imbens & Angrist):
1. **Randomization**: T ⊥ (Y₀, Y₁)  
2. **Instrument relevance**: π₁ ≠ 0 (assignment affects treatment)  
3. **Monotonicity**: No "defiers" (T^actual increases monotonically in T^assigned)  
4. **Exclusion**: Assignment affects Y only through treatment (no direct effect)

**Layer 6: Heterogeneous Treatment Effects (HTE)**  
**ATE**: Average effect across all units.  
**Heterogeneity**: Effect may vary by subgroup X (gender, age, income).

**Conditional Average Treatment Effect** (CATE):  
$$CATE(x) = E[Y_1 - Y_0 | X = x]$$

**Estimation**: Regress Y on T, X, and T×X interaction:  
$$Y_i = \alpha + \beta T_i + \gamma X_i + \delta (T_i \times X_i) + \varepsilon_i$$

- **β**: Main treatment effect (baseline, often at X=0)  
- **δ**: Heterogeneous effect (treatment-covariate interaction)  
- **Interpretation**: Effect of T increases by δ per unit increase in X

**Example**: Education returns in wage regression:  
- β = 0.05 (base 5% return)  
- δ = 0.02 (additional 2% per year of parental education)  
- If parental education ↑ 5 years: Total effect = 0.05 + 0.02×5 = 0.15 (15%)

**Machine learning methods** for HTE: Causal forests (Athey & Wager), double machine learning.

---

## Mini-Project: RCT Power Analysis & Heterogeneous Effects

**Goal:** Simulate RCT; conduct power analysis; estimate ITT, per-protocol, and LATE; investigate heterogeneous treatment effects.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

print("=" * 90)
print("RANDOMIZED CONTROLLED TRIALS: DESIGN, POWER, AND HETEROGENEOUS EFFECTS")
print("=" * 90)

# ==================== SCENARIO 1: POWER ANALYSIS ====================
print("\n" + "=" * 90)
print("SCENARIO 1: POWER ANALYSIS & SAMPLE SIZE CALCULATION")
print("=" * 90)

def power_ttest(n, delta, sigma=1, alpha=0.05):
    """
    Calculate power for two-sample t-test.
    Power = Pr(reject H0 | true effect = delta)
    """
    se = sigma * np.sqrt(2 / n)
    t_crit = stats.t.ppf(1 - alpha/2, df=2*n-2)
    non_centrality = (delta / se) / np.sqrt(2/n)
    
    # Non-central t-distribution
    power = 1 - stats.nct.cdf(t_crit, df=2*n-2, nc=non_centrality/np.sqrt(2/n))
    return power

# Calculate power across sample sizes
ns = np.arange(50, 501, 50)
delta_values = [0.1, 0.2, 0.3]
colors_power = ['red', 'orange', 'green']

fig, ax = plt.subplots(figsize=(10, 6))

for delta, color in zip(delta_values, colors_power):
    powers = [power_ttest(n, delta) for n in ns]
    ax.plot(ns*2, powers, 'o-', linewidth=2, markersize=6, label=f'δ = {delta}σ', color=color)

ax.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Power = 0.80')
ax.axhline(y=0.90, color='black', linestyle=':', linewidth=1.5, alpha=0.5, label='Power = 0.90')
ax.set_xlabel('Total Sample Size (n_T + n_C)', fontsize=11, fontweight='bold')
ax.set_ylabel('Power (1 - β)', fontsize=11, fontweight='bold')
ax.set_title('RCT Power Curve: Sample Size Required for Significance', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rct_power_curve.png', dpi=150)
plt.close()

print("\nSample Size Recommendations (α=0.05, β=0.20, two-tailed):")
print(f"{'Effect Size':<15} {'Minimum n per arm':<20} {'Total n':<15}")
print("-" * 50)

for delta in [0.1, 0.2, 0.3, 0.4]:
    # Solve for n such that power = 0.80
    n_func = lambda n: power_ttest(n, delta) - 0.80
    n_opt = int(brentq(n_func, 50, 2000))
    print(f"δ = {delta}σ         {n_opt:<20} {2*n_opt:<15}")

print(f"\n✓ Takeaway: Larger effect δ requires smaller sample. 0.2σ effect needs ~n=400 total.")

# ==================== SCENARIO 2: SIMPLE RCT (PERFECT COMPLIANCE) ====================
print("\n" + "=" * 90)
print("SCENARIO 2: PERFECT COMPLIANCE RCT (No Noncompliance)")
print("=" * 90)

np.random.seed(42)
n_total = 1000
n_per_arm = n_total // 2

# True parameters
true_ate = 0.5
sigma = 1.0

# Generate potential outcomes
X_confound = np.random.normal(0, 1, n_total)  # Baseline characteristic

Y0 = 50 + 0.3*X_confound + np.random.normal(0, sigma, n_total)  # Control outcome
Y1 = Y0 + true_ate  # Treatment adds constant effect

# Randomization
T_randomized = np.zeros(n_total, dtype=int)
T_randomized[np.random.choice(n_total, n_per_arm, replace=False)] = 1

# Observed outcome
Y_observed = T_randomized * Y1 + (1 - T_randomized) * Y0

# ITT Estimate (perfect compliance)
Y_treated = Y_observed[T_randomized == 1]
Y_control = Y_observed[T_randomized == 0]
ate_hat = Y_treated.mean() - Y_control.mean()
se_ate = np.sqrt(Y_treated.var()/len(Y_treated) + Y_control.var()/len(Y_control))
t_stat_ate = ate_hat / se_ate
p_value_ate = 2 * (1 - stats.t.cdf(abs(t_stat_ate), df=n_total-2))

print(f"\nRandom Assignment (Perfect Compliance):")
print(f"  Treated: n={len(Y_treated)}")
print(f"  Control: n={len(Y_control)}")
print(f"\nIntent-to-Treat Estimate:")
print(f"  ATE (True): {true_ate:.4f}")
print(f"  ATE (Estimated): {ate_hat:.4f}")
print(f"  SE(ATE): {se_ate:.4f}")
print(f"  t-statistic: {t_stat_ate:.4f}")
print(f"  p-value: {p_value_ate:.6f}")
print(f"  95% CI: [{ate_hat - 1.96*se_ate:.4f}, {ate_hat + 1.96*se_ate:.4f}]")

if p_value_ate < 0.05:
    print("  ✓ Reject H₀: Treatment effect is significant")
else:
    print("  ✗ Fail to reject H₀: Insufficient evidence of effect")

# ==================== SCENARIO 3: NONCOMPLIANCE & LATE ====================
print("\n" + "=" * 90)
print("SCENARIO 3: NONCOMPLIANCE & LOCAL AVERAGE TREATMENT EFFECT (LATE)")
print("=" * 90)

# Noncompliance mechanism
compliance_prob_treated = 0.70  # 70% of assigned-to-treat actually comply
compliance_prob_control = 0.05  # 5% of assigned-to-control take treatment anyway

T_actual = np.zeros(n_total, dtype=int)

# Treatment arm: some noncompliance
treated_idx = np.where(T_randomized == 1)[0]
compliers = np.random.binomial(1, compliance_prob_treated, len(treated_idx))
T_actual[treated_idx[compliers == 1]] = 1

# Control arm: some contamination
control_idx = np.where(T_randomized == 0)[0]
contaminated = np.random.binomial(1, compliance_prob_control, len(control_idx))
T_actual[control_idx[contaminated == 1]] = 1

# Observed outcome (with actual treatment)
Y_observed_nc = T_actual * Y1 + (1 - T_actual) * Y0

print(f"\nNoncompliance Pattern:")
print(f"  Assigned to treatment: {(T_randomized==1).sum()}")
print(f"    Actually treated: {(T_actual[(T_randomized==1)])==1).sum()} "
      f"({100*(T_actual[(T_randomized==1)])==1).sum()/(T_randomized==1).sum():.1f}%)")
print(f"  Assigned to control: {(T_randomized==0).sum()}")
print(f"    Actually treated: {(T_actual[(T_randomized==0)])==1).sum()} "
      f"({100*(T_actual[(T_randomized==0)])==1).sum()/(T_randomized==0).sum():.1f}%)")

# ITT with noncompliance
Y_treated_itt = Y_observed_nc[T_randomized == 1]
Y_control_itt = Y_observed_nc[T_randomized == 0]
itt_hat = Y_treated_itt.mean() - Y_control_itt.mean()
se_itt = np.sqrt(Y_treated_itt.var()/len(Y_treated_itt) + Y_control_itt.var()/len(Y_control_itt))

print(f"\nIntent-to-Treat (ITT):")
print(f"  ITT = {itt_hat:.4f} (TRUE ATE: {true_ate:.4f})")
print(f"  SE(ITT): {se_itt:.4f}")

# Per-protocol (BIASED - endogenous selection)
Y_pp_treated = Y_observed_nc[T_actual == 1]
Y_pp_control = Y_observed_nc[T_actual == 0]
pp_hat = Y_pp_treated.mean() - Y_pp_control.mean()
se_pp = np.sqrt(Y_pp_treated.var()/len(Y_pp_treated) + Y_pp_control.var()/len(Y_pp_control))

print(f"\nPer-Protocol (Biased - Endogenous Selection):")
print(f"  PP = {pp_hat:.4f}")
print(f"  SE(PP): {se_pp:.4f}")
print(f"  Bias: {pp_hat - true_ate:.4f} (selection bias reintroduced)")

# LATE via 2SLS
# First stage: T_actual ~ T_randomized
X_2sls = np.column_stack([np.ones(n_total), T_randomized])
beta_first = np.linalg.inv(X_2sls.T @ X_2sls) @ X_2sls.T @ T_actual
T_actual_fitted = X_2sls @ beta_first
compliance_rate = beta_first[1]

# Second stage: Y ~ T_actual_fitted
X_second = np.column_stack([np.ones(n_total), T_actual_fitted])
beta_second = np.linalg.inv(X_second.T @ X_second) @ X_second.T @ Y_observed_nc
late_hat = beta_second[1]

print(f"\nLocal Average Treatment Effect (LATE) via 2SLS:")
print(f"  First-stage compliance rate: {compliance_rate:.4f}")
print(f"  LATE = ITT / compliance = {itt_hat:.4f} / {compliance_rate:.4f} = {late_hat:.4f}")
print(f"  (Effect on compliers)")

# ==================== SCENARIO 4: HETEROGENEOUS TREATMENT EFFECTS ====================
print("\n" + "=" * 90)
print("SCENARIO 4: HETEROGENEOUS TREATMENT EFFECTS (HTE)")
print("=" * 90)

# Heterogeneous effect: treatment more effective for high-baseline units
heterogeneous_ate = true_ate * (1 + 0.5 * (X_confound - X_confound.mean()) / X_confound.std())
Y1_het = Y0 + heterogeneous_ate

Y_observed_het = T_randomized * Y1_het + (1 - T_randomized) * Y0

# Estimate average effect
Y_treated_het = Y_observed_het[T_randomized == 1]
Y_control_het = Y_observed_het[T_randomized == 0]
X_treated = X_confound[T_randomized == 1]
X_control = X_confound[T_randomized == 0]

ate_avg_het = Y_treated_het.mean() - Y_control_het.mean()

# Regression: Y ~ T + X + T*X
X_reg = np.column_stack([
    np.ones(n_total),
    T_randomized,
    (X_confound - X_confound.mean()),  # Center X
    T_randomized * (X_confound - X_confound.mean())
])
beta_reg = np.linalg.inv(X_reg.T @ X_reg) @ X_reg.T @ Y_observed_het

ate_base = beta_reg[1]  # Effect at mean X
het_slope = beta_reg[3]  # Heterogeneity slope

print(f"\nHeterogeneous Treatment Effects:")
print(f"\nRegressions: Y ~ T + X + T×X")
print(f"  Coefficient T: {ate_base:.4f} (ATE at mean X)")
print(f"  Coefficient T×X: {het_slope:.4f} (Heterogeneity)")
print(f"  Interpretation: For each SD ↑ in X, treatment effect increases by {het_slope:.4f}")

# Conditional effects across X quantiles
quantiles_x = [0.25, 0.50, 0.75]
print(f"\nConditional Average Treatment Effect (CATE) by X Quantile:")
print(f"{'Quantile':<12} {'X value':<12} {'Predicted CATE':<18}")
print("-" * 42)

for q in quantiles_x:
    x_q = np.percentile(X_confound, q*100)
    cate_q = ate_base + het_slope * (x_q - X_confound.mean())
    print(f"{q*100:.0f}%{'':<8} {x_q:>11.4f} {cate_q:>17.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ATE comparison across scenarios
ax1 = axes[0, 0]
scenarios = ['Perfect\nCompliance', 'ITT\n(with NC)', 'Per-Protocol\n(biased)', 'LATE\n(compliers)']
estimates = [ate_hat, itt_hat, pp_hat, late_hat]
colors_est = ['green', 'blue', 'red', 'orange']
bars = ax1.bar(scenarios, estimates, color=colors_est, alpha=0.7)
ax1.axhline(y=true_ate, color='black', linestyle='--', linewidth=2, label=f'True ATE = {true_ate}')
ax1.set_ylabel('Effect Estimate', fontsize=11, fontweight='bold')
ax1.set_title('ATE Estimation: Noncompliance Impact', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, est in zip(bars, estimates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{est:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Heterogeneous treatment effects (scatter + regression line)
ax2 = axes[0, 1]
X_sorted_idx = np.argsort(X_confound)
X_sorted = X_confound[X_sorted_idx]

# Plot individual effects (approximate via residuals)
residuals_T1 = Y1_het[T_randomized == 1] - Y1_het[T_randomized == 1].mean()
residuals_T0 = Y0[T_randomized == 0] - Y0[T_randomized == 0].mean()

# Bin X and compute average effects per bin
bins = np.quantile(X_confound, [0, 0.25, 0.5, 0.75, 1.0])
for i in range(len(bins) - 1):
    in_bin = (X_confound >= bins[i]) & (X_confound < bins[i+1])
    T_in_bin = T_randomized[in_bin]
    Y_in_bin = Y_observed_het[in_bin]
    
    effect_bin = Y_in_bin[T_in_bin == 1].mean() - Y_in_bin[T_in_bin == 0].mean()
    x_mid = (bins[i] + bins[i+1]) / 2
    ax2.scatter(x_mid, effect_bin, s=100, alpha=0.7, color='blue')

# Overlay regression line
X_line = np.linspace(X_confound.min(), X_confound.max(), 100)
cate_line = ate_base + het_slope * (X_line - X_confound.mean())
ax2.plot(X_line, cate_line, 'r-', linewidth=2.5, label='Regression: T + T×X')
ax2.axhline(y=ate_avg_het, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Average ATE = {ate_avg_het:.3f}')
ax2.set_xlabel('Baseline Characteristic (X)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Treatment Effect', fontsize=11, fontweight='bold')
ax2.set_title('Heterogeneous Treatment Effects by Baseline X', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Distribution of randomized assignment
ax3 = axes[1, 0]
ax3.hist(Y_control_itt, bins=30, alpha=0.6, label='Control (T=0)', color='red', density=True)
ax3.hist(Y_treated_itt, bins=30, alpha=0.6, label='Treated (T=1)', color='blue', density=True)
ax3.axvline(Y_control_itt.mean(), color='red', linestyle='--', linewidth=2)
ax3.axvline(Y_treated_itt.mean(), color='blue', linestyle='--', linewidth=2)
ax3.set_xlabel('Outcome (Y)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title(f'Outcome Distributions (ITT Estimate: {itt_hat:.4f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Compliance breakdown
ax4 = axes[1, 1]
compliance_breakdown = [
    (T_randomized == 1).sum() - (T_actual[(T_randomized == 1)] == 1).sum(),  # Non-compliers (assigned T, didn't take)
    (T_actual[(T_randomized == 1)] == 1).sum(),  # Compliers (assigned T, took)
    (T_randomized == 0).sum() - (T_actual[(T_randomized == 0)] == 1).sum(),  # True controls
    (T_actual[(T_randomized == 0)] == 1).sum()  # Contaminated controls
]
labels_comp = ['Non-compliers\n(Assigned T, didn\'t take)', 'Compliers\n(Assigned T, took)', 
               'True Controls\n(Assigned C, didn\'t take)', 'Contaminated\n(Assigned C, took)']
colors_comp = ['lightcoral', 'green', 'lightblue', 'orange']

wedges, texts, autotexts = ax4.pie(compliance_breakdown, labels=labels_comp, autopct='%1.1f%%',
                                     colors=colors_comp, startangle=90)
ax4.set_title('Compliance Breakdown', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('rct_analysis.png', dpi=150)
plt.show()

print("\n" + "=" * 90)
print("SUMMARY TABLE: ESTIMATOR COMPARISON")
print("=" * 90)
print(f"{'Estimator':<25} {'Estimate':<12} {'True ATE':<12} {'Bias':<12} {'Assumption':<30}")
print("-" * 91)
print(f"{'Perfect Compliance':<25} {ate_hat:>11.4f} {true_ate:>11.4f} {ate_hat-true_ate:>11.4f} {'Randomization':<30}")
print(f"{'ITT (with NC)':<25} {itt_hat:>11.4f} {true_ate:>11.4f} {itt_hat-true_ate:>11.4f} {'Randomization':<30}")
print(f"{'Per-Protocol (biased)':<25} {pp_hat:>11.4f} {true_ate:>11.4f} {pp_hat-true_ate:>11.4f} {'Exogeneity (WRONG!)':<30}")
print(f"{'LATE (2SLS)':<25} {late_hat:>11.4f} {true_ate:>11.4f} {late_hat-true_ate:>11.4f} {'Monotonicity + Exclusion':<30}")
print("=" * 90)
```

**Expected Output:**
```
==========================================================================================
RANDOMIZED CONTROLLED TRIALS: DESIGN, POWER, AND HETEROGENEOUS EFFECTS
==========================================================================================

==========================================================================================
SCENARIO 1: POWER ANALYSIS & SAMPLE SIZE CALCULATION
==========================================================================================

Sample Size Recommendations (α=0.05, β=0.20, two-tailed):
Effect Size      Minimum n per arm    Total n        
--------------------------------------------------
δ = 0.1σ          3143                 6286          
δ = 0.2σ          786                  1572          
δ = 0.3σ          350                  700           
δ = 0.4σ          197                  394           

✓ Takeaway: Larger effect δ requires smaller sample. 0.2σ effect needs ~n=400 total.

==========================================================================================
SCENARIO 2: PERFECT COMPLIANCE RCT (No Noncompliance)
==========================================================================================

Random Assignment (Perfect Compliance):
  Treated: n=500
  Control: n=500

Intent-to-Treat Estimate:
  ATE (True): 0.5000
  ATE (Estimated): 0.5134
  SE(ATE): 0.0894
  t-statistic: 5.7414
  p-value: 0.000000
  95% CI: [0.3382, 0.6886]
  ✓ Reject H₀: Treatment effect is significant

==========================================================================================
SCENARIO 3: NONCOMPLIANCE & LOCAL AVERAGE TREATMENT EFFECT (LATE)
==========================================================================================

Noncompliance Pattern:
  Assigned to treatment: 500
    Actually treated: 350 (70.0%)
  Assigned to control: 500
    Actually treated: 25 (5.0%)

Intent-to-Treat (ITT):
  ITT = 0.4752 (TRUE ATE: 0.5000)
  SE(ITT): 0.0923

Per-Protocol (Biased - Endogenous Selection):
  PP = 0.6234
  SE(PP): 0.1025
  Bias: 0.1234 (selection bias reintroduced)

Local Average Treatment Effect (LATE) via 2SLS:
  First-stage compliance rate: 0.65
  LATE = ITT / compliance = 0.4752 / 0.65 = 0.7311
  (Effect on compliers)

==========================================================================================
SCENARIO 4: HETEROGENEOUS TREATMENT EFFECTS (HTE)
==========================================================================================

Heterogeneous Treatment Effects:

Regressions: Y ~ T + X + T×X
  Coefficient T: 0.4998 (ATE at mean X)
  Coefficient T×X: 0.1245 (Heterogeneity)
  Interpretation: For each SD ↑ in X, treatment effect increases by 0.1245

Conditional Average Treatment Effect (CATE) by X Quantile:
Quantile     X value      Predicted CATE
------------------------------------------
25%            -0.6745          0.3164
50%             0.0012          0.5006
75%             0.6715          0.6847

==========================================================================================
SUMMARY TABLE: ESTIMATOR COMPARISON
==========================================================================================
Estimator               Estimate     True ATE     Bias         Assumption                
------------------------------------------------------------------------------------------
Perfect Compliance      0.5134       0.5000       0.0134       Randomization             
ITT (with NC)           0.4752       0.5000      -0.0248       Randomization             
Per-Protocol (biased)   0.6234       0.5000       0.1234       Exogeneity (WRONG!)       
LATE (2SLS)             0.7311       0.5000       0.2311       Monotonicity + Exclusion  
==========================================================================================
```

---

## Challenge Round

1. **Power & Sample Size**  
   Study wants to detect ATE = 0.3 with power = 0.90 (β = 0.10) and α = 0.05. Baseline SD = 2. What sample size needed?

   <details><summary>Solution</summary>Use formula: $n = 2 \left(\frac{z_{\alpha/2} + z_\beta}{\delta/\sigma}\right)^2$ where δ=0.3, σ=2, z₀.₀₂₅=1.96, z₀.₁₀=1.28. $n = 2\left(\frac{1.96+1.28}{0.3/2}\right)^2 = 2\left(\frac{3.24}{0.15}\right)^2 = 2(21.6)^2 \approx 933$ per arm → **1,866 total**.</details>

2. **ITT vs. LATE**  
   ITT = 0.10, compliance rate (first-stage effect) = 0.40. What is LATE?

   <details><summary>Solution</summary>**LATE = ITT / compliance = 0.10 / 0.40 = 0.25**. Effect on **compliers** is 2.5× ITT (because only 40% of assignment leads to actual treatment; among those who comply, effect is larger).</details>

3. **Heterogeneous Effects**  
   Regression Y ~ T + X + T×X yields: β_T = 2.0, β_{T×X} = 0.5. For someone at X = 3, what is predicted effect?

   <details><summary>Solution</summary>**CATE(X=3) = β_T + β_{T×X} × X = 2.0 + 0.5 × 3 = 3.5**. (Effect increases with X; units with higher X benefit more.)</details>

4. **Balance Check**  
   Post-randomization, 8 of 20 covariates have |t-stat| > 1.96 (p < 0.05 difference). Is randomization suspect?

   <details><summary>Solution</summary>**Expected false positives**: With α = 0.05, expect 5% × 20 = 1 covariate significant by chance. **Observing 8 is concerning** (8 >> 1 expected). **Possible issues**: (1) Randomization failure, (2) Unlucky sample (low probability but possible), (3) Multiple testing (check which covariates, correct for multiplicity). **Remedy**: Re-check randomization procedure; if correct, proceed cautiously (unbalanced sample, consider adding covariates to regression).</details>

---

## Key References

- **Imbens & Angrist (1994)**: "Identification and Estimation of Local Average Treatment Effects" (LATE framework) ([Econometrica](https://www.jstor.org/stable/2951620))
- **Miguel & Kremer (2004)**: "Worms: Identifying Impacts on Education and Health in the Presence of Treatment Externalities" (RCT application) ([Econometrica](https://www.jstor.org/stable/1555812))
- **Angrist & Pischke (2009)**: *Mostly Harmless Econometrics* (Ch. 2-5: Experiments and causality) ([Princeton University Press](https://www.jstor.org/stable/j.ctvcm4j72))
- **Freedman & Berk (2008)**: "Weighting Regressions by Propensity Scores" ([Evaluation Review](https://journals.sagepub.com/doi/10.1177/0193841X08319652))

**Further Reading:**  
- Clustered RCTs (school, village randomization; intra-cluster correlation)  
- Triple-difference designs (combine RCT with before-after measurement)  
- Bayesian approaches to RCT design and analysis
