# Claim Duration & Termination Patterns

## 1. Concept Skeleton
**Definition:** Length of time an STD claim remains active (paying benefits) from start to termination via recovery, death, or shift to LTD  
**Purpose:** Predict cost per claim, forecast reserve requirements, identify population health trends  
**Prerequisites:** Disability basics, survival analysis, claims management process

## 2. Comparative Framing
| Factor | **Short Illness** | **Musculoskeletal** | **Mental Health** | **Surgical Recovery** |
|--------|-------------------|---------------------|-------------------|----------------------|
| **Median Duration** | 5–7 days | 14–30 days | 30–90 days | 21–42 days |
| **Recovery Rate (6 mo)** | 95%+ | 80–90% | 40–70% | 90–95% |
| **% Transitioning to LTD** | <1% | 2–5% | 15–25% | 1–3% |
| **Cost driver** | High frequency | Moderate frequency/duration | Duration | Frequency |

## 3. Examples + Counterexamples

**Short Duration (Typical STD):**  
Flu or minor surgery: 10-day average claim → 60% recovery within 2 weeks → benefits end

**Extended Duration (Stacking):**  
Back injury: 45-day STD benefit → extends toward 90-day limit → potential transition to LTD

**No Recovery (LTD Transition):**  
Severe depression: 60-day STD claim active; shows no improvement → transitions to LTD at 90-day mark with own-occupation protection

**Counter-Example - Quick Return:**  
Broken arm in cast: 14-day STD → removed from work restriction earlier than expected → employee returns part-time (residual benefit)

## 4. Layer Breakdown
```
Claim Duration & Termination Framework:
├─ Termination Events (Mutually Exclusive):
│   ├─ Recovery: Employee returns to work, fully capable
│   │   ├─ Full recovery: Returns to pre-disability job at 100% capacity
│   │   ├─ Partial recovery: Returns but with restrictions (residual benefit may apply)
│   │   └─ Median: 50–75% of claims recover within 30–60 days
│   │
│   ├─ Transition to LTD: STD benefit ends, LTD benefit begins
│   │   ├─ Timing: At elimination period end or benefit expiration
│   │   ├─ Rate: 5–15% of STD claims (varies by cause)
│   │   └─ Severity: Indicates prolonged disability beyond STD scope
│   │
│   ├─ Death: Employee dies while on STD
│   │   ├─ Rate: 0.1–0.5% (very low in working population)
│   │   └─ Impact: Benefit stops; may trigger life insurance payout instead
│   │
│   ├─ Voluntary Termination: Employee quits while disabled
│   │   ├─ Rate: 10–20% (depends on job market, job satisfaction)
│   │   └─ Benefit: Stops upon employment end
│   │
│   ├─ Benefit Exhaustion: Reaches maximum benefit period
│   │   ├─ Typical STD max: 13–26 weeks
│   │   └─ Outcome: Must transition to LTD or end (if LTD not available)
│   │
│   └─ Administrative Closure: Policy non-compliance, fraud detection
│       ├─ Non-compliance: Missed medical appointments, unapproved work
│       └─ Fraud: Intentional misrepresentation of condition
│
├─ Duration Variation by Cause:
│   ├─ Acute (flu, broken limb): 7–14 days median
│   ├─ Surgical: 21–42 days (depends on invasiveness)
│   ├─ Musculoskeletal (back, shoulder): 14–60 days
│   ├─ Respiratory (asthma, pneumonia): 10–30 days
│   ├─ Mental Health (depression, anxiety): 30–180 days
│   ├─ Pregnancy-related: 42–56 days (maternity disability)
│   └─ Cancer/Serious Illness: 90+ days (often LTD)
│
├─ Covariates Affecting Duration:
│   ├─ Age: Older age → longer duration (diminished recovery)
│   ├─ Occupation: Sedentary → often shorter; physically demanding → longer
│   ├─ Earnings Level: Higher earner → slightly longer (higher motivation to maintain)
│   ├─ Diagnosis: Mental health/cancer → longest; acute viral → shortest
│   ├─ Treatment Compliance: Better compliance → faster recovery
│   ├─ Vocational Rehab: Access to retraining → shorter duration, better outcomes
│   └─ Workplace Accommodation: Available accommodations → faster return to work
│
└─ Survival Analysis:
    ├─ Kaplan-Meier Curves: Proportion of claims "surviving" (still active) by month
    ├─ Hazard Rate: Probability of termination in month t given still active at t-1
    ├─ Cumulative Incidence: Cause-specific termination (recovery vs LTD vs death)
    └─ Predictive Modeling: Estimate time-to-termination by individual characteristics
```

## 5. Mini-Project: Claim Duration Analysis & Forecasting

**Goal:** Analyze claim duration by cause and predict termination probability.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import median_survival_times

# Simulated STD claims data
np.random.seed(42)
n_claims = 500

# Cause-specific duration distributions (Weibull parameters are realistic)
causes = ['Flu/Acute', 'Surgical', 'Musculoskeletal', 'Mental Health']
weibull_params = {
    'Flu/Acute': (2.0, 10),  # (shape, scale) → median ~9 days
    'Surgical': (1.8, 30),    # median ~27 days
    'Musculoskeletal': (1.5, 35),  # median ~33 days
    'Mental Health': (1.3, 80)   # median ~74 days
}

claims_list = []

for cause in causes:
    shape, scale = weibull_params[cause]
    n = int(n_claims / len(causes))
    
    # Generate durations from Weibull
    durations = np.random.weibull(shape, n) * scale
    
    # Termination reasons (event type)
    recovery_rate = {'Flu/Acute': 0.95, 'Surgical': 0.92, 
                     'Musculoskeletal': 0.80, 'Mental Health': 0.50}[cause]
    
    events = np.random.choice(
        ['Recovery', 'LTD Transition', 'Other'],
        size=n,
        p=[recovery_rate, 1 - recovery_rate - 0.05, 0.05]
    )
    
    for i in range(n):
        claims_list.append({
            'Cause': cause,
            'Duration_Days': durations[i],
            'Event': events[i],
            'Event_Indicator': 1 if events[i] != 'Censored' else 0
        })

df = pd.DataFrame(claims_list)

# Summary statistics
print("Claim Duration Summary by Cause:")
print(df.groupby('Cause')['Duration_Days'].describe())

print("\nTermination Reason Distribution:")
print(df['Event'].value_counts(normalize=True))

# Kaplan-Meier survival curves by cause
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: KM Curves by cause
ax = axes[0, 0]
kmf = KaplanMeierFitter()

for cause in causes:
    mask = df['Cause'] == cause
    T = df.loc[mask, 'Duration_Days']
    E = df.loc[mask, 'Event_Indicator']
    kmf.fit(T, E, label=cause)
    kmf.plot_survival_function(ax=ax, linewidth=2)

ax.set_xlabel('Days Since Claim Start')
ax.set_ylabel('Proportion of Claims Still Active')
ax.set_title('Kaplan-Meier Survival Curves by Cause')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: Duration histogram by cause
ax = axes[0, 1]
for cause in causes:
    mask = df['Cause'] == cause
    ax.hist(df.loc[mask, 'Duration_Days'], bins=30, alpha=0.5, label=cause)
ax.set_xlabel('Claim Duration (days)')
ax.set_ylabel('Number of Claims')
ax.set_title('Duration Distribution by Cause')
ax.legend()
ax.set_xlim(0, 200)

# Plot 3: Median duration vs termination type
ax = axes[1, 0]
median_by_event = df.groupby('Event')['Duration_Days'].median()
colors = ['green', 'orange', 'red']
median_by_event.plot(kind='bar', ax=ax, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Median Duration (days)')
ax.set_title('Median Duration by Termination Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Proportion terminating by duration threshold
ax = axes[1, 1]
thresholds = np.arange(0, 150, 10)
terminated = []
for threshold in thresholds:
    pct = (df['Duration_Days'] <= threshold).sum() / len(df) * 100
    terminated.append(pct)

ax.plot(thresholds, terminated, 'o-', linewidth=2, markersize=6, color='darkblue')
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% Claims')
ax.set_xlabel('Duration (days)')
ax.set_ylabel('% of Claims Terminated By This Duration')
ax.set_title('Cumulative Claim Termination')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Weibull AFT model: predict duration by cause
print("\n\nWeibull Accelerated Failure Time Model:")
print("Estimating log(Duration) by cause...")

# Create dummy variables for cause
cause_dummies = pd.get_dummies(df['Cause'], drop_first=True)
X = cause_dummies

aft = WeibullAFTFitter()
aft.fit(df['Duration_Days'], df['Event_Indicator'], X, show_progress=False)

print("\nModel Summary:")
print(aft.summary)

# Predicted median durations
print("\n\nMedian Duration by Cause (from model):")
for cause in causes:
    cause_vec = pd.DataFrame([cause_dummies.columns == c for c in cause_dummies.columns]).T
    # Simplified prediction (using reference level)
    print(f"{cause}: ~{df[df['Cause'] == cause]['Duration_Days'].median():.0f} days")
```

**Key Insights:**
- Duration highly skewed: Most claims short (< 30 days), few very long (> 90 days)
- Cause is primary driver: Flu/acute illness → ~10 days; mental health → ~75 days
- LTD transition rate: 5–15% overall; much higher for mental health (~25%)
- Median duration often quoted to underwriters (varies 10–75 days by population mix)

## 6. Relationships & Dependencies
- **To Reserving:** Duration models determine claim reserve calculations
- **To Premium Pricing:** Expected cost per claim = (daily benefit) × (expected duration)
- **To Benefit Design:** Longer max benefit period → lower LTD transition rate
- **To Return-to-Work Programs:** Effective rehab reduces duration by 10–20%

## References
- [Milliman Disability Benchmarks](https://www.milliman.com) - Duration data by cause
- [Society of Actuaries (SOA)](https://www.soa.org) - Experience studies on claim duration
- [LIMRA Disability Insurance Study](https://www.limra.com) - Termination rates and trends

