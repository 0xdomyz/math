# Benefit Percentage & Replacement Ratio

## 1. Concept Skeleton
**Definition:** Fraction of pre-disability earnings that STD insurance replaces; expressed as percentage (e.g., 60%, 66⅔%)  
**Purpose:** Balance income protection with return-to-work incentives; control claim duration (too high % reduces motivation to return)  
**Prerequisites:** Disability insurance fundamentals, earnings definitions, benefit calculation

## 2. Comparative Framing
| Replacement Ratio | **50%** | **60%** | **66⅔%** | **70%** |
|-------------------|---------|---------|----------|---------|
| **Return-to-work incentive** | Strong | Moderate | Weak | Weakest |
| **Affordability (employee)** | Lower premiums | Standard | Mid-range | Higher premiums |
| **Market prevalence** | Less common | Very common | Common (public sector) | Rare (union only) |
| **Benefit cap impact** | Higher % need cap | Cap typically binding | Cap often restrictive | Cap very restrictive |

## 3. Examples + Counterexamples

**Simple Example:**  
Pre-disability salary: $60,000/year ($5,000/month)  
Benefit percentage: 60%  
Monthly STD benefit: $5,000 × 60% = $3,000/month (before taxes/offsets)

**Failure Case:**  
Replacement ratio 100%: Zero financial incentive to return to work → claims extend artificially; premium spirals upward

**Edge Case:**  
High earner with $250,000 salary + 66⅔% ratio = $16,667/month benefit → often capped at $10,000 or $15,000/month maximum

## 4. Layer Breakdown
```
Benefit Percentage Structure:
├─ Earnings Definition:
│   ├─ Base Salary: Only regular pay (most common)
│   ├─ Average Earnings: Last 12 months / last 3 months (smooths bonus variation)
│   ├─ Includes Bonuses: Adds complexity; requires documentation
│   └─ Excludes: Overtime, commissions (often excluded or averaged separately)
├─ Benefit Calculation:
│   ├─ Gross Benefit: Pre-disability earnings × Benefit percentage
│   ├─ Less Offsets: SSDI, pension, other group DI (if applicable)
│   ├─ Less Taxes: STD benefits taxable if employer-paid (federal, FICA)
│   └─ Net Benefit: What employee receives monthly
├─ Replacement Ratio Impact on Behavior:
│   ├─ High % (70%+): Longer claims, higher cost, premium increases
│   ├─ Standard (60%): Balanced: protection + incentive to recover
│   ├─ Low % (50%): Strong return-to-work incentive but less protection
│   └─ Nonlinear effect: Even small change (60% → 66%) increases claim duration 5–10%
├─ Benefit Maximums & Minimums:
│   ├─ Maximum monthly: $10,000–$20,000 (depends on plan design)
│   ├─ Minimum monthly: Often $100–$250 (prevents tiny payments)
│   └─ Excess income offset: High earners disproportionately affected by max
└─ Effective Replacement (After Taxes & Offsets):
    ├─ Tax impact: If employer-paid, ~25% reduction (rough average)
    ├─ SSDI offset: Reduces benefit by SSDI amount (pending or awarded)
    └─ Effective ratio: Often 30–50% net after all deductions
```

## 5. Mini-Project: Replacement Ratio Impact on Claim Duration

**Goal:** Model claim duration elasticity with respect to benefit percentage using behavioral economics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Empirical observation: Claim duration increases with benefit percentage
# Model: Duration = base_duration * exp(elasticity * (replacement% - baseline%))

# Historical data from actuarial studies
replacement_ratios = np.array([50, 55, 60, 65, 66.67, 70])
avg_claim_duration = np.array([18, 20, 23, 25, 26, 28])  # days (illustrative)

# Fit exponential model
def duration_model(pct, base_duration, elasticity):
    """Duration increases exponentially with replacement percentage"""
    baseline_pct = 60  # Reference point
    return base_duration * np.exp(elasticity * (pct - baseline_pct) / 100)

# Fit to data
popt, pcov = curve_fit(duration_model, replacement_ratios, avg_claim_duration, 
                       p0=[23, 0.1], maxfev=5000)
base_duration, elasticity = popt

print(f"Model Fit:")
print(f"  Base duration at 60%: {base_duration:.2f} days")
print(f"  Elasticity: {elasticity:.4f} (% change per 1% increase in benefit%)")

# Predict across range
pct_range = np.linspace(40, 75, 50)
duration_predicted = duration_model(pct_range, base_duration, elasticity)

# Cost projection: assume daily cost = average benefit amount
# Higher % → higher daily cost AND longer duration (compounding effect)
daily_benefit = 100  # Illustrative
total_cost = (replacement_ratios / 100) * daily_benefit * avg_claim_duration

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Duration vs Replacement Ratio
axes[0].scatter(replacement_ratios, avg_claim_duration, s=100, 
                alpha=0.7, color='steelblue', label='Observed')
axes[0].plot(pct_range, duration_predicted, 'r-', linewidth=2, label='Fitted Model')
axes[0].axvline(60, color='gray', linestyle='--', alpha=0.5, label='Standard 60%')
axes[0].set_xlabel('Replacement Ratio (%)')
axes[0].set_ylabel('Average Claim Duration (days)')
axes[0].set_title('Claim Duration vs Benefit Percentage')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Total Expected Cost (Duration × Benefit)
daily_benefit_range = (pct_range / 100) * 200  # Daily benefit increases with %
total_cost_range = daily_benefit_range * duration_model(pct_range, base_duration, elasticity)

axes[1].plot(pct_range, total_cost_range, 'o-', linewidth=2, 
             markersize=6, color='darkred')
axes[1].set_xlabel('Replacement Ratio (%)')
axes[1].set_ylabel('Total Expected Cost per Claim ($)')
axes[1].set_title('Total Claim Cost: Duration × Benefit')
axes[1].grid(alpha=0.3)

# Plot 3: Cost difference vs 60% baseline
baseline_cost = total_cost_range[np.argmin(np.abs(pct_range - 60))]
cost_increase = ((total_cost_range - baseline_cost) / baseline_cost) * 100

axes[2].fill_between(pct_range, 0, cost_increase, alpha=0.5, color='orange')
axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2].axvline(60, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Replacement Ratio (%)')
axes[2].set_ylabel('% Increase in Total Cost vs 60%')
axes[2].set_title('Cost Sensitivity to Replacement Ratio')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity table
print("\n\nCost Sensitivity Analysis:")
print("Replacement % | Avg Duration (days) | Daily Benefit | Total Cost | % vs 60%")
print("-" * 75)
for pct in [50, 55, 60, 65, 66.67, 70, 75]:
    dur = duration_model(pct, base_duration, elasticity)
    daily = (pct / 100) * 200
    total = daily * dur
    pct_diff = ((total - baseline_cost) / baseline_cost) * 100
    print(f"{pct:6.1f}% | {dur:18.1f} | ${daily:11.2f} | ${total:9.2f} | {pct_diff:+6.1f}%")
```

**Key Insights:**
- Small increases (60% → 66%) increase expected cost by 15–25% (duration elasticity)
- Nonlinear effect: cost compounds (benefit % up + duration up = double pressure)
- Optimal design balances protection (higher %) with moral hazard (lower %)
- Market standard 60% reflects empirical sweet spot

## 6. Relationships & Dependencies
- **To Elimination Period:** Longer elimination period → can justify higher % (employee bears initial loss)
- **To Offsets:** Higher base % more likely capped when SSDI or workers' comp applied
- **To Maximum Benefit:** Higher % earners hit max sooner; effective replacement lower
- **To Underwriting Criteria:** Higher % may trigger medical underwriting for high-risk occupations

## References
- [Milliman Disability Benchmarks](https://www.milliman.com) - Market practice data on replacement ratios
- [LIMRA Research: Disability Insurance Study](https://www.limra.com) - Behavioral elasticity estimates
- [Actuarial Research Clearing House: Disability Claim Duration](https://www.soa.org) - Society of Actuaries

