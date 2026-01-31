# Survival Analysis

## 1. Concept Skeleton
**Definition:** Statistical methods analyzing time-to-event data with censoring, estimating hazard and survival functions over time  
**Purpose:** Model time until event (death, failure, recovery), handle incomplete observations, compare survival across groups  
**Prerequisites:** Probability distributions, hazard concepts, censoring types, non-parametric methods

## 2. Comparative Framing
| Method | Kaplan-Meier | Cox Proportional Hazards | Parametric (Weibull) |
|--------|--------------|--------------------------|---------------------|
| **Type** | Non-parametric | Semi-parametric | Fully parametric |
| **Use** | Survival curves | Covariate effects | Model entire distribution |
| **Assumptions** | None on distribution | Proportional hazards | Specific distribution |
| **Output** | Survival function | Hazard ratios | Full probability model |

## 3. Examples + Counterexamples

**Simple Example:**  
Clinical trial: Track patient survival 5 years. Some patients lost to follow-up (censored) at 3 years → survival analysis handles incomplete data

**Failure Case:**  
Hazards cross over time: Violates proportional hazards assumption, Cox model coefficients misleading → use time-varying coefficients

**Edge Case:**  
No events observed: All censored, cannot estimate survival curve reliably → need longer follow-up or different design

## 4. Layer Breakdown
```
Survival Analysis Components:
├─ Key Functions:
│   ├─ Survival Function: S(t) = P(T > t) [probability surviving past t]
│   ├─ Hazard Function: h(t) = lim[P(t ≤ T < t+Δt | T ≥ t) / Δt]
│   │   └─ Instantaneous failure rate at time t
│   └─ Cumulative Hazard: H(t) = ∫h(u)du = -ln(S(t))
├─ Censoring Types:
│   ├─ Right Censoring: Event not yet observed (most common)
│   ├─ Left Censoring: Event occurred before observation start
│   └─ Interval Censoring: Event in time window
├─ Kaplan-Meier Estimator:
│   └─ Ŝ(t) = Π(1 - dᵢ/nᵢ) for all tᵢ ≤ t
│   └─ dᵢ: events at time tᵢ, nᵢ: at risk at tᵢ
├─ Log-Rank Test:
│   └─ Compare survival curves between groups
│   └─ H₀: No difference in survival distributions
├─ Cox Proportional Hazards:
│   └─ h(t|X) = h₀(t) × exp(β₁X₁ + ... + βₚXₚ)
│   └─ h₀(t): baseline hazard (unspecified)
│   └─ exp(βⱼ): hazard ratio for Xⱼ
├─ Assumptions:
│   ├─ Independent censoring: Censoring unrelated to event risk
│   ├─ Proportional hazards: Hazard ratios constant over time
│   └─ Correct functional form: Covariates enter linearly
└─ Applications:
    ├─ Clinical trials (patient survival)
    ├─ Engineering (component failure)
    ├─ Criminology (recidivism)
    └─ Economics (unemployment duration)
```

**Interaction:** Define event → Handle censoring → Estimate survival → Test differences → Model covariates

## 5. Mini-Project
Kaplan-Meier curves and Cox regression:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Generate simulated survival data
np.random.seed(42)
n = 200

# Two treatment groups
treatment = np.random.binomial(1, 0.5, n)  # 0=Control, 1=Treatment
age = np.random.normal(60, 10, n)
age = np.clip(age, 30, 90)

# Generate survival times (treatment improves survival)
# Baseline hazard higher for older patients and control group
baseline_hazard = 0.001 * np.exp(0.03 * (age - 60) - 0.5 * treatment)
time = np.random.exponential(1 / baseline_hazard)

# Add censoring (30% randomly censored)
censored = np.random.binomial(1, 0.3, n)
observed_time = np.where(censored, np.random.uniform(0, time), time)
event_observed = 1 - censored

# Create DataFrame
df = pd.DataFrame({
    'time': observed_time,
    'event': event_observed,
    'treatment': treatment,
    'age': age
})

print("Survival Data Summary:")
print(f"Total observations: {n}")
print(f"Events observed: {event_observed.sum()} ({event_observed.sum()/n*100:.1f}%)")
print(f"Censored: {censored.sum()} ({censored.sum()/n*100:.1f}%)")
print(f"Control group: {(treatment==0).sum()}, Treatment group: {(treatment==1).sum()}")

# Kaplan-Meier Estimation by Group
kmf_control = KaplanMeierFitter()
kmf_treatment = KaplanMeierFitter()

# Fit each group
mask_control = df['treatment'] == 0
mask_treatment = df['treatment'] == 1

kmf_control.fit(df[mask_control]['time'], 
                df[mask_control]['event'], 
                label='Control')
kmf_treatment.fit(df[mask_treatment]['time'], 
                  df[mask_treatment]['event'], 
                  label='Treatment')

print("\nMedian Survival Times:")
print(f"Control: {kmf_control.median_survival_time_:.2f} time units")
print(f"Treatment: {kmf_treatment.median_survival_time_:.2f} time units")

# Log-Rank Test
results = logrank_test(df[mask_control]['time'], 
                       df[mask_treatment]['time'],
                       df[mask_control]['event'], 
                       df[mask_treatment]['event'])

print("\nLog-Rank Test (Control vs Treatment):")
print(f"Test statistic: {results.test_statistic:.3f}")
print(f"P-value: {results.p_value:.4f}")
print(f"Conclusion: {'Significant difference' if results.p_value < 0.05 else 'No significant difference'}")

# Cox Proportional Hazards Model
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event')

print("\nCox Proportional Hazards Results:")
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']])

print("\nInterpretation:")
for var in ['treatment', 'age']:
    hr = np.exp(cph.params_[var])
    print(f"{var}: HR={hr:.3f} (", end="")
    if var == 'treatment':
        print(f"{(1-hr)*100:.1f}% lower hazard in treatment group)")
    else:
        print(f"{(hr-1)*100:.1f}% higher hazard per 1-year increase)")

# Check proportional hazards assumption
cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Kaplan-Meier curves
ax = axes[0, 0]
kmf_control.plot_survival_function(ax=ax, ci_show=True)
kmf_treatment.plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel('Time')
ax.set_ylabel('Survival Probability')
ax.set_title('Kaplan-Meier Survival Curves')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Cumulative events
ax = axes[0, 1]
kmf_control.plot_cumulative_density(ax=ax, ci_show=False)
kmf_treatment.plot_cumulative_density(ax=ax, ci_show=False)
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative Event Probability')
ax.set_title('Cumulative Event Probability')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: At-risk table
ax = axes[0, 2]
kmf_control.plot_survival_function(ax=ax, ci_show=False, alpha=0.7)
kmf_treatment.plot_survival_function(ax=ax, ci_show=False, alpha=0.7)
add_at_risk_counts(kmf_control, kmf_treatment, ax=ax)
ax.set_xlabel('Time')
ax.set_ylabel('Survival Probability')
ax.set_title('Survival Curves with At-Risk Counts')

# Plot 4: Histogram of event times
ax = axes[1, 0]
ax.hist(df[df['event']==1]['time'], bins=30, alpha=0.6, label='Events', edgecolor='black')
ax.hist(df[df['event']==0]['time'], bins=30, alpha=0.6, label='Censored', edgecolor='black')
ax.set_xlabel('Time')
ax.set_ylabel('Count')
ax.set_title('Distribution of Event and Censoring Times')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Survival by age quartiles
ax = axes[1, 1]
age_quartiles = pd.qcut(df['age'], q=4, labels=['Q1 (youngest)', 'Q2', 'Q3', 'Q4 (oldest)'])
for quartile in age_quartiles.unique():
    mask = age_quartiles == quartile
    kmf_temp = KaplanMeierFitter()
    kmf_temp.fit(df[mask]['time'], df[mask]['event'], label=str(quartile))
    kmf_temp.plot_survival_function(ax=ax, ci_show=False)

ax.set_xlabel('Time')
ax.set_ylabel('Survival Probability')
ax.set_title('Survival by Age Quartile')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Cox model - Partial effects plot
ax = axes[1, 2]
# Show survival for average age, comparing treatments
age_mean = df['age'].mean()
synthetic_data = pd.DataFrame({
    'treatment': [0, 1],
    'age': [age_mean, age_mean]
})

for idx, row in synthetic_data.iterrows():
    survival_func = cph.predict_survival_function(row.to_frame().T)
    label = 'Control' if row['treatment'] == 0 else 'Treatment'
    ax.plot(survival_func.index, survival_func.values[:, 0], 
            label=f"{label} (age={age_mean:.0f})", linewidth=2)

ax.set_xlabel('Time')
ax.set_ylabel('Survival Probability')
ax.set_title(f'Cox Model: Predicted Survival\n(at mean age={age_mean:.0f})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Additional diagnostics
print("\nModel Concordance Index (C-index):")
print(f"C-index: {cph.concordance_index_:.3f}")
print("(>0.5 indicates predictive power; 0.5=random, 1.0=perfect)")
```

## 6. Challenge Round
When is survival analysis the wrong tool?
- No censoring: Use standard regression or time series
- Repeated events: Use recurrent event models or frailty models
- Competing risks: Standard methods biased, use competing risk analysis
- Non-proportional hazards: Use stratified Cox or time-varying coefficients
- Interval censoring dominates: Use specialized interval-censored methods

## 7. Key References
- [Survival Analysis Overview (Wikipedia)](https://en.wikipedia.org/wiki/Survival_analysis)
- [lifelines Python Library](https://lifelines.readthedocs.io/)
- [Kaplan-Meier Estimator Explained](https://www.statisticshowto.com/kaplan-meier-survival-estimator/)

---
**Status:** Essential time-to-event analysis | **Complements:** Regression, Clinical Trials, Reliability Engineering
