# Difference-in-Differences (DiD)

## 1. Concept Skeleton
**Definition:** Quasi-experimental method comparing outcome changes between treatment and control groups before/after intervention  
**Purpose:** Estimate causal treatment effect by differencing out time-invariant confounders and common time trends  
**Prerequisites:** Panel data structure, parallel trends assumption, fixed effects models, interaction terms

## 2. Comparative Framing
| Method | DiD | RCT | IV | Matching |
|--------|-----|-----|----|---------| 
| **Design** | Observational with timing variation | Experimental randomization | Instrument-based | Balance observables |
| **Key Assumption** | Parallel trends (counterfactual) | Random assignment | Valid instrument | No unobservables |
| **Data Requirement** | Pre/post for treat & control | Random assignment | Instrument availability | Rich covariates |
| **Bias from** | Differential trends | Violations rare if RCT | Weak/invalid instruments | Unobserved confounding |

## 3. Examples + Counterexamples

**Classic Example:**  
Card & Krueger (1994): Minimum wage increase in NJ (treated) vs PA (control). Compare employment changes before/after policy.

**Failure Case:**  
Pre-treatment trends diverging: Treated group already declining faster. DiD attributes natural divergence to treatment effect.

**Edge Case:**  
Treatment staggered across units/times: Bacon decomposition shows negative weights, 2WFE estimator biased under heterogeneous effects.

## 4. Layer Breakdown
```
Difference-in-Differences Framework:
├─ Setup:
│   ├─ Treatment Group: D=1 receives intervention at time T
│   ├─ Control Group: D=0 never treated (or not yet)
│   └─ Periods: Pre-treatment (t<T), Post-treatment (t≥T)
├─ Estimand:
│   ├─ First Difference (Time): E[Y_post - Y_pre | D=1] - E[Y_post - Y_pre | D=0]
│   ├─ Alternative: (E[Y_post|D=1] - E[Y_post|D=0]) - (E[Y_pre|D=1] - E[Y_pre|D=0])
│   └─ Causal Effect: τ^DiD = Average Treatment Effect on Treated (ATT)
├─ Regression Form (2x2 Design):
│   Y_it = β₀ + β₁·Treat_i + β₂·Post_t + β₃·(Treat×Post)_it + ε_it
│   ├─ β₁: Level difference between groups (time-invariant)
│   ├─ β₂: Common time trend (affects both groups)
│   ├─ β₃: DiD estimate (treatment effect) ← Key parameter
│   └─ Interaction term isolates treatment effect
├─ Assumptions:
│   ├─ Parallel Trends: E[Y₀_post - Y₀_pre | D=1] = E[Y₀_post - Y₀_pre | D=0]
│   │   └─ Counterfactual: Treated would have same trend as control absent treatment
│   ├─ No Anticipation: Treatment effect only post-intervention
│   ├─ Stable Composition: Same units in pre/post (or representative sampling)
│   └─ SUTVA: No spillovers between treated and control
├─ Extensions:
│   ├─ Multiple Periods: Y_it = α_i + λ_t + τ·Treat_it + ε_it (TWFE)
│   ├─ Covariates: X_it controls for time-varying confounders
│   ├─ Staggered Adoption: Callaway & Sant'Anna (2021), Sun & Abraham (2021)
│   └─ Continuous Treatment: Dose-response DiD
└─ Diagnostics:
    ├─ Event Study: τ_k for k periods relative to treatment
    ├─ Placebo Tests: Test τ_k = 0 for k < 0 (pre-treatment)
    └─ Robustness: Permutation tests, synthetic control comparison
```

**Interaction:** Group differences + Time differences cancel confounders → Isolate causal effect

## 5. Mini-Project
Simulate and estimate DiD with event study visualization:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

np.random.seed(42)

# ===== Data Generating Process =====
n_units = 100  # 50 treated, 50 control
n_periods = 20  # 10 pre-treatment, 10 post-treatment
treatment_period = 10

# Generate panel data
units = np.repeat(np.arange(n_units), n_periods)
periods = np.tile(np.arange(n_periods), n_units)
treated = np.repeat([1]*50 + [0]*50, n_periods)  # First 50 units treated

# Fixed effects
unit_fe = np.repeat(np.random.normal(50, 10, n_units), n_periods)
time_fe = np.tile(np.arange(n_periods) * 0.5, n_units)  # Common time trend

# Treatment indicator
post = (periods >= treatment_period).astype(int)
treat_post = treated * post

# True treatment effect: +5 units starting at period 10
true_effect = 5.0
treatment_effect = treat_post * true_effect

# Generate outcome with parallel pre-trends
epsilon = np.random.normal(0, 5, n_units * n_periods)
Y = unit_fe + time_fe + treatment_effect + epsilon

# Create DataFrame
df = pd.DataFrame({
    'unit': units,
    'period': periods,
    'treated': treated,
    'post': post,
    'treat_post': treat_post,
    'Y': Y
})

# ===== Basic 2x2 DiD =====
# Collapse to pre/post means
pre_treat = df[(df['treated']==1) & (df['period']<treatment_period)]['Y'].mean()
post_treat = df[(df['treated']==1) & (df['period']>=treatment_period)]['Y'].mean()
pre_control = df[(df['treated']==0) & (df['period']<treatment_period)]['Y'].mean()
post_control = df[(df['treated']==0) & (df['period']>=treatment_period)]['Y'].mean()

# Manual DiD calculation
did_manual = (post_treat - pre_treat) - (post_control - pre_control)

print("="*70)
print("2x2 DIFFERENCE-IN-DIFFERENCES")
print("="*70)
print("\nGroup Means:")
print(f"               Pre-Treatment    Post-Treatment    Difference")
print(f"Treated:       {pre_treat:8.2f}        {post_treat:8.2f}         {post_treat-pre_treat:8.2f}")
print(f"Control:       {pre_control:8.2f}        {post_control:8.2f}         {post_control-pre_control:8.2f}")
print(f"Difference:    {pre_treat-pre_control:8.2f}        {post_treat-post_control:8.2f}         {did_manual:8.2f} ← DiD")
print(f"\nTrue Treatment Effect: {true_effect:.2f}")
print(f"Estimated DiD Effect:  {did_manual:.2f}")
print(f"Estimation Error:      {did_manual - true_effect:.2f}")

# ===== Regression DiD (2x2) =====
did_reg = smf.ols('Y ~ treated + post + treat_post', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['unit']})

print("\n" + "="*70)
print("REGRESSION DiD (2x2 Design)")
print("="*70)
print(did_reg.summary().tables[1])
print(f"\nManual DiD: {did_manual:.4f}")
print(f"Regression DiD (treat_post): {did_reg.params['treat_post']:.4f}")
print(f"Difference: {abs(did_manual - did_reg.params['treat_post']):.6f}")

# ===== Two-Way Fixed Effects (TWFE) =====
# Include unit and time fixed effects
twfe_formula = 'Y ~ treat_post + C(unit) + C(period)'
twfe_model = smf.ols(twfe_formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['unit']})

print("\n" + "="*70)
print("TWO-WAY FIXED EFFECTS (Unit + Time FE)")
print("="*70)
print(f"Treatment Effect (τ): {twfe_model.params['treat_post']:.4f}")
print(f"Standard Error:       {twfe_model.bse['treat_post']:.4f}")
print(f"95% CI: [{twfe_model.params['treat_post'] - 1.96*twfe_model.bse['treat_post']:.4f}, "
      f"{twfe_model.params['treat_post'] + 1.96*twfe_model.bse['treat_post']:.4f}]")
print(f"t-statistic: {twfe_model.tvalues['treat_post']:.2f}")
print(f"p-value: {twfe_model.pvalues['treat_post']:.4f}")

# ===== Event Study Design =====
# Create event time indicators (relative to treatment)
df['event_time'] = df['period'] - treatment_period
df['event_time'] = df['event_time'].where(df['treated']==1, -999)  # Only for treated

# Create event time dummies (omit t=-1 as reference)
event_dummies = pd.get_dummies(df['event_time'], prefix='event')
event_cols = [col for col in event_dummies.columns if col != 'event_-1.0' and col != 'event_-999.0']
df = pd.concat([df, event_dummies[event_cols]], axis=1)

# Event study regression
event_formula = 'Y ~ ' + ' + '.join(event_cols) + ' + C(unit) + C(period)'
event_model = smf.ols(event_formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['unit']})

# Extract event study coefficients
event_coefs = []
event_ses = []
event_times = []

for t in range(-9, 10):
    if t == -1:  # Reference period
        event_coefs.append(0)
        event_ses.append(0)
    else:
        col_name = f'event_{float(t)}'
        if col_name in event_model.params.index:
            event_coefs.append(event_model.params[col_name])
            event_ses.append(event_model.bse[col_name])
        else:
            event_coefs.append(np.nan)
            event_ses.append(np.nan)
    event_times.append(t)

event_results = pd.DataFrame({
    'event_time': event_times,
    'coef': event_coefs,
    'se': event_ses
})

print("\n" + "="*70)
print("EVENT STUDY COEFFICIENTS")
print("="*70)
print(event_results.round(4))

# Test parallel trends (pre-treatment coefficients jointly zero)
pre_event_cols = [col for col in event_cols if int(col.split('_')[1].split('.')[0]) < -1]
if pre_event_cols:
    hypothesis = ' = '.join(pre_event_cols) + ' = 0'
    f_test = event_model.f_test(hypothesis)
    print(f"\nPre-treatment F-test (parallel trends):")
    print(f"H₀: All pre-treatment coefficients = 0")
    print(f"F-statistic: {f_test.fvalue[0][0]:.2f}")
    print(f"p-value: {f_test.pvalue:.4f}")
    if f_test.pvalue < 0.05:
        print("✗ Reject H₀: Evidence against parallel trends")
    else:
        print("✓ Fail to reject: Parallel trends assumption supported")

# ===== Visualizations =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean outcomes over time by group
treat_means = df[df['treated']==1].groupby('period')['Y'].mean()
control_means = df[df['treated']==0].groupby('period')['Y'].mean()

axes[0, 0].plot(treat_means.index, treat_means.values, 'o-', 
                linewidth=2, markersize=6, label='Treated', color='blue')
axes[0, 0].plot(control_means.index, control_means.values, 's-', 
                linewidth=2, markersize=6, label='Control', color='red')
axes[0, 0].axvline(treatment_period, color='black', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='Treatment Start')
axes[0, 0].set_xlabel('Period')
axes[0, 0].set_ylabel('Mean Outcome (Y)')
axes[0, 0].set_title('Parallel Trends Visualization')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Add parallel trends line (counterfactual)
post_periods = np.arange(treatment_period, n_periods)
pre_treat_trend = (treat_means.iloc[:treatment_period].values[-1] - 
                   treat_means.iloc[:treatment_period].values[0]) / (treatment_period - 1)
pre_control_trend = (control_means.iloc[:treatment_period].values[-1] - 
                     control_means.iloc[:treatment_period].values[0]) / (treatment_period - 1)
counterfactual = (treat_means.iloc[treatment_period-1] + 
                  (post_periods - treatment_period + 1) * 
                  (control_means.iloc[treatment_period:].values - control_means.iloc[treatment_period-1]) /
                  (control_means.iloc[treatment_period:].index - treatment_period + 1))
axes[0, 0].plot(post_periods, 
                treat_means.iloc[treatment_period-1] + 
                (control_means.iloc[treatment_period:].values - control_means.iloc[treatment_period-1]),
                ':', linewidth=2, color='blue', alpha=0.5, label='Counterfactual (Treated)')

# Plot 2: Event study coefficients
event_valid = event_results.dropna()
axes[0, 1].errorbar(event_valid['event_time'], event_valid['coef'],
                    yerr=1.96*event_valid['se'], fmt='o-', capsize=5,
                    linewidth=2, markersize=6, color='darkgreen')
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[0, 1].axvline(-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Treatment Start')
axes[0, 1].axhline(true_effect, color='blue', linestyle=':', linewidth=2,
                   label=f'True Effect ({true_effect})')
axes[0, 1].fill_between([-10, -1], -20, 20, alpha=0.1, color='gray',
                        label='Pre-Treatment')
axes[0, 1].set_xlabel('Event Time (Relative to Treatment)')
axes[0, 1].set_ylabel('Treatment Effect Estimate')
axes[0, 1].set_title('Event Study Plot (95% CI)')
axes[0, 1].legend()
axes[0, 1].set_ylim(-5, 10)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Distribution of estimates across methods
estimates_df = pd.DataFrame({
    'Method': ['Manual\n2x2', 'Regression\n2x2', 'TWFE', 'Event Study\nAvg Post', 'True Effect'],
    'Estimate': [
        did_manual,
        did_reg.params['treat_post'],
        twfe_model.params['treat_post'],
        event_results[event_results['event_time']>=0]['coef'].mean(),
        true_effect
    ],
    'SE': [
        0,  # Not calculated for manual
        did_reg.bse['treat_post'],
        twfe_model.bse['treat_post'],
        event_results[event_results['event_time']>=0]['se'].mean(),
        0
    ]
})

x_pos = np.arange(len(estimates_df)-1)
axes[1, 0].bar(x_pos, estimates_df['Estimate'].iloc[:-1],
               yerr=1.96*estimates_df['SE'].iloc[:-1], capsize=5,
               color=['skyblue', 'lightgreen', 'coral', 'plum'], alpha=0.7)
axes[1, 0].axhline(true_effect, color='black', linestyle='--', linewidth=2,
                   label=f'True Effect ({true_effect})')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(estimates_df['Method'].iloc[:-1])
axes[1, 0].set_ylabel('Treatment Effect Estimate')
axes[1, 0].set_title('Comparison Across Methods (95% CI)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Summary table
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.95, 'DIFFERENCE-IN-DIFFERENCES SUMMARY', 
                ha='center', fontsize=13, weight='bold',
                transform=axes[1, 1].transAxes)

summary_text = f"""
Design:
  • Units: {n_units} ({n_units//2} treated, {n_units//2} control)
  • Periods: {n_periods} ({treatment_period} pre, {n_periods-treatment_period} post)
  • Treatment Period: {treatment_period}

True Treatment Effect: {true_effect:.2f}

Estimates:
  • Manual 2x2:     {did_manual:.3f}
  • Regression 2x2: {did_reg.params['treat_post']:.3f} ({did_reg.bse['treat_post']:.3f})
  • TWFE:           {twfe_model.params['treat_post']:.3f} ({twfe_model.bse['treat_post']:.3f})

Assumption Tests:
  • Parallel Trends (F-test p-value): {f_test.pvalue:.4f}
    {"✓ Supported" if f_test.pvalue >= 0.05 else "✗ Violated"}

Standard Errors:
  • Clustering: By unit
  • Robust to within-unit correlation
"""

axes[1, 1].text(0.05, 0.8, summary_text,
                transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does DiD fail or mislead?
- **Non-parallel trends**: Treatment group diverging pre-intervention invalidates counterfactual
- **Anticipation effects**: Units change behavior before treatment (reform announcements, threat effects)
- **Composition changes**: Different units in pre/post periods (attrition, sample selection)
- **Spillovers**: Control group affected by treatment (SUTVA violation, market equilibrium effects)
- **Heterogeneous treatment effects + staggered timing**: 2WFE estimator uses negative weights, biased
- **Ashenfelter's dip**: Temporary pre-treatment decline (regression to mean) inflates effect

## 7. Key References
- [Card & Krueger (1994) - Minimum Wages and Employment](https://davidcard.berkeley.edu/papers/njmin-aer.pdf)
- [Angrist & Pischke - Mostly Harmless Econometrics (Ch 5)](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- [Goodman-Bacon (2021) - Difference-in-differences with variation in treatment timing](https://doi.org/10.1016/j.jeconom.2021.03.014)

---
**Status:** Core quasi-experimental method | **Complements:** Event Studies, Synthetic Control, Panel Data Methods
