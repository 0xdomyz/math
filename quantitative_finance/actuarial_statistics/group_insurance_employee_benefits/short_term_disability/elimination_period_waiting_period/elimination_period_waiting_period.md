# Elimination Period & Waiting Period

## 1. Concept Skeleton
**Definition:** Waiting period before short-term disability (STD) benefits commence; employee bears loss during this window  
**Purpose:** Reduce moral hazard, align with emergency savings expectations, control claim frequency and insurer costs  
**Prerequisites:** Disability insurance fundamentals, claim adjudication process, benefit triggers

## 2. Comparative Framing
| Factor | 0-Day | 7-Day | 14-Day |
|--------|-------|--------|--------|
| **Employee out-of-pocket cost** | None | ~1 week salary loss | ~2 week salary loss |
| **Claim frequency impact** | High (all absences) | Moderate | Lower (weeds out brief absences) |
| **Insurer cost** | Highest premiums | Mid-range | Lower premiums |
| **Market prevalence** | Rare (union/government) | Common | Standard for group STD |

## 3. Examples + Counterexamples

**Simple Example:**  
Employee disability begins Monday; 7-day elimination period → benefits start following Monday (1 week salary loss by employee)

**Failure Case:**  
0-day elimination with no definition of disability → claims for minor colds or 1-day absences drain pool, premium spikes

**Edge Case:**  
Stacked STD after Long-Term Disability: LTD 90-day elimination means STD covers first 90 days; overlap period uses STD benefit

## 4. Layer Breakdown
```
Elimination Period Structure:
├─ Calendar Days vs Business Days:
│   ├─ Calendar: 7, 14, 21, 30 days from disability onset
│   ├─ Business Days: Only weekdays count; weekends/holidays pause clock
│   └─ Impact: Calendar-day periods are shorter effective wait
├─ Continuous vs Non-Continuous Disability:
│   ├─ Continuous: Single unbroken absence; elimination period runs once
│   ├─ Non-Continuous: Multiple short absences for same condition
│   │   ├─ If gap < 30 days: Treated as continuous (period doesn't reset)
│   │   └─ If gap ≥ 30 days: New elimination period applies (reset)
│   └─ Recurrence rule: Prevents re-triggering via small gaps
├─ Elimination Period vs Benefit Waiting:
│   ├─ Elimination: Time before benefits paid (insurer waits)
│   ├─ Integration with STD: No retroactive payment for elimination period
│   └─ Integration with LTD: STD covers gap, then LTD begins
└─ Exceptions & Modifications:
    ├─ Accident vs Sickness: Some plans waive for accidents (0-day)
    ├─ Recurrent claims: May honor prior elimination if same cause
    └─ Return-to-work incentives: Partial benefits during ramp-up
```

## 5. Mini-Project: Elimination Period Impact on Claims

**Goal:** Model frequency impact of varying elimination periods on a group STD plan.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulated population: 1000 employees, 1 year
np.random.seed(42)
n_employees = 1000
claims_data = []

# Generate claim events: rate ~0.5 per employee per year
for emp_id in range(n_employees):
    # Poisson process: average 0.5 claims/year
    n_claims = np.random.poisson(0.5)
    
    for claim_idx in range(n_claims):
        # Random claim start date
        day_of_year = np.random.randint(1, 366)
        
        # Duration (log-normal: most short, some long)
        duration = np.random.lognormal(mean=2, sigma=1.2)  # ~7 days median
        duration = max(1, int(duration))
        
        claims_data.append({
            'emp_id': emp_id,
            'start_day': day_of_year,
            'duration_days': duration
        })

df = pd.DataFrame(claims_data)

# Calculate claims paid under different elimination periods
elimination_periods = [0, 7, 14, 21]
results = []

for elim_days in elimination_periods:
    # Days benefit paid = max(0, duration - elimination)
    df['days_paid'] = df['duration_days'].apply(
        lambda d: max(0, d - elim_days)
    )
    
    # Count non-zero claims (those that trigger any payment)
    claims_paid = (df['days_paid'] > 0).sum()
    
    # Total benefit days (proxy for cost)
    total_benefit_days = df['days_paid'].sum()
    
    results.append({
        'Elimination Days': elim_days,
        'Claims Paid': claims_paid,
        'Total Benefit Days': total_benefit_days,
        'Avg Days per Claim': total_benefit_days / max(1, claims_paid)
    })

results_df = pd.DataFrame(results)
print("Elimination Period Impact on Claims:\n")
print(results_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Claims paid vs elimination period
axes[0].bar(results_df['Elimination Days'], results_df['Claims Paid'], 
            color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Elimination Period (days)')
axes[0].set_ylabel('Number of Claims Paid')
axes[0].set_title('Claims Frequency vs Elimination Period')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Total benefit days vs elimination period
axes[1].plot(results_df['Elimination Days'], results_df['Total Benefit Days'], 
             'o-', linewidth=2, markersize=8, color='darkred')
axes[1].set_xlabel('Elimination Period (days)')
axes[1].set_ylabel('Total Benefit Days Paid')
axes[1].set_title('Total Claims Cost vs Elimination Period')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity: show claim duration distribution
print("\n\nClaim Duration Distribution:")
print(df['duration_days'].describe())
print(f"Median duration: {df['duration_days'].median():.1f} days")
print(f"% claims < 7 days: {(df['duration_days'] < 7).sum() / len(df) * 100:.1f}%")
print(f"% claims < 14 days: {(df['duration_days'] < 14).sum() / len(df) * 100:.1f}%")
```

**Key Insights:**
- 7-day elimination eliminates ~30–50% of very short claims (< 7 days)
- 14-day elimination reduces claim count ~50%, but few claims shorter than 14 days
- Longer periods improve insurer costs but shift burden entirely to employees
- Business day vs calendar day choice affects effective waiting period

## 6. Relationships & Dependencies
- **To Claim Adjudication:** Elimination period is trigger; determines when evaluation begins
- **To Benefit Amount:** Once triggered, benefit amount calculated independent of elimination length
- **To Offset Provisions:** Elimination period doesn't affect offsets (e.g., SSDI offset still applies)
- **To Return-to-Work Riders:** Some plans reduce or waive elimination if employee returns part-time

## References
- [Group Disability Insurance Standards](https://www.acli.com) - American Council of Life Insurers
- [UNUM STD Plan Design](https://www.unum.com) - Common market parameters
- [Actuarial Standards Board: ASB #1](https://www.actuarialstandardsboard.org/) - Valuation standards

