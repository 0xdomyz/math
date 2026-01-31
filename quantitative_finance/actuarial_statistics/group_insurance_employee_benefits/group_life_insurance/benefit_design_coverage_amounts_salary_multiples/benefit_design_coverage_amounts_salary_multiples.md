# Benefit Design: Coverage Amounts & Salary Multiples

## 1. Concept Skeleton
**Definition:** Determination of death benefit amounts offered under group life insurance plans; typically expressed as multiple of salary (1×, 2×, 3×) or fixed dollar amounts  
**Purpose:** Balance employee financial protection with employer cost management; align with market competitiveness  
**Prerequisites:** Group underwriting fundamentals, mortality pricing, employee demographics

## 2. Comparative Framing
| Benefit Design | **1× Salary** | **2× Salary** | **3× Salary** | **Fixed $100K** |
|----------------|---------------|---------------|---------------|-----------------|
| **Cost per employee** | Low (~$15-20/mo) | Medium (~$30-40/mo) | High (~$45-60/mo) | Fixed (~$25/mo) |
| **Market prevalence** | Traditional | Very common (standard) | Growing (tech, finance) | Smaller firms |
| **Adequacy (80% replacement)** | Insufficient for most | Good for mid-salary | Excellent | Varies by income |
| **Variation by salary** | Wide range | Moderate | Moderate | Same for all |
| **Admin complexity** | Simple | Simple | Simple | Simpler |

## 3. Examples + Counterexamples

**Simple Example:**  
Company with average salary $60,000; 2× salary design = $120,000 benefit per employee. Young employee (age 25, $40K salary) = $80K benefit; Senior employee (age 55, $100K salary) = $200K benefit

**Market Standard Example:**  
Tech company offering competitive package: 3× salary + $100K accidental death + spouse at 50% = total ~$280K for median ($85K) employee

**Fixed Amount Example:**  
Small business (50 employees, salary range $35K–$150K): Flat $100K per person. Low-wage worker (adequacy: 2–3× salary); high-wage worker (adequacy: <1× salary, inadequate)

**Failure Case:**  
Company offers 1× salary in inflationary environment; hasn't adjusted in 10 years. Effective purchasing power of benefit drops 25%; employee dissatisfaction; competitor with 2× salary attracts talent

**Edge Case:**  
Executive with $500K salary; 3× salary = $1.5M benefit, but policy maximum cap $750K → capped benefit applies (asymmetric protection for high earners)

## 4. Layer Breakdown
```
Benefit Amount Structure:
├─ Salary-Multiple Designs:
│   ├─ 1× Salary (Basic):
│   │   ├─ Calculation: Annual base salary × 1
│   │   ├─ Example: $50K salary = $50K death benefit
│   │   ├─ Cost: Lowest among multiples
│   │   ├─ Adequacy: ~12-month income replacement (basic security)
│   │   ├─ Use case: Budget-conscious employers; blue-collar workers
│   │   └─ Historical: Traditional design from 1970s-80s
│   │
│   ├─ 2× Salary (Market Standard):
│   │   ├─ Calculation: Annual base salary × 2
│   │   ├─ Example: $50K salary = $100K death benefit
│   │   ├─ Cost: ~60-70% more than 1× (but nonlinear; lower per $1K as total rises)
│   │   ├─ Adequacy: ~24-month income replacement (reasonable buffer)
│   │   ├─ Use case: Most common; middle market; Fortune 500
│   │   ├─ Market prevalence: ~50-60% of group plans
│   │   └─ Competitive positioning: Baseline for retention
│   │
│   ├─ 3× Salary (Generous/Competitive):
│   │   ├─ Calculation: Annual base salary × 3
│   │   ├─ Example: $50K salary = $150K death benefit
│   │   ├─ Cost: ~80-100% more than 2×
│   │   ├─ Adequacy: ~36-month income replacement (strong security)
│   │   ├─ Use case: Tech, finance, professional services (talent war)
│   │   ├─ Market prevalence: Growing (15-20% of plans, especially private equity)
│   │   └─ ROI: Attracts top talent; improves retention
│   │
│   ├─ Higher Multiples (4×, 5×):
│   │   ├─ Rare: <5% of plans
│   │   ├─ Use: Specific industries (offshore oil, high-hazard)
│   │   ├─ Cost: Very high
│   │   └─ Indication: Occupational hazard justifies extra protection
│   │
│   └─ Salary Definition Nuances:
│       ├─ Base salary only: Regular wages (most common)
│       ├─ Average earnings: Last 12 months / last 3 months (smooths variability)
│       ├─ Includes bonus: Base + annual incentive (more generous)
│       ├─ Includes commissions: Base + variable comp (less common; complex)
│       ├─ Excludes: Overtime (rarely included); benefits; allowances
│       └─ Locking mechanism: Benefit locked at enrollment or updates annually
│
├─ Fixed Dollar Amount Designs:
│   ├─ Flat coverage: All employees get same amount (e.g., $100K)
│   │   ├─ Pros: Simplicity; cost predictability; no salary admin overhead
│   │   ├─ Cons: Inadequate for high earners; overly generous for low-wage workers
│   │   ├─ Use: Small businesses; standardized industries
│   │   └─ Common levels: $50K, $75K, $100K, $150K
│   │
│   ├─ Tiered fixed amounts (by salary band or class):
│   │   ├─ Example: $75K (under $30K salary) / $100K ($30-50K) / $150K (over $50K)
│   │   ├─ Hybrid approach: Fixes for bands; improves targeting
│   │   ├─ Admin: Moderate; requires salary band maintenance
│   │   └─ Benefit: Better adequacy equity than flat; simpler than salary multiples
│   │
│   └─ Per-diem fixed amounts (by job class):
│       ├─ Hourly workers: $50K–$75K
│       ├─ Salaried: $100K–$150K
│       ├─ Managers: $150K–$200K
│       ├─ Executives: $250K–$500K
│       └─ Fairness: Class-based; occupational risk accounted
│
├─ Caps & Maximums:
│   ├─ Individual benefit cap:
│   │   ├─ Common caps: $250K, $500K, $1M (limits per-person liability)
│   │   ├─ Rationale: Risk management; prevent adverse selection (very high earners)
│   │   ├─ Example: 3× salary design with $500K cap
│   │   │   ├─ Employee salary $100K: Benefit = $300K (uncapped)
│   │   │   ├─ Employee salary $250K: Benefit = $500K (capped, not $750K)
│   │   │   └─ Impact: High earners not fully protected; may seek supplemental
│   │   └─ Optional: Some plans tier caps by age/class
│   │
│   ├─ Aggregate (plan) cap: Total benefits all employees <  maximum
│   │   ├─ Rare: Usually only in self-funded plans
│   │   ├─ Use: Protect employer from catastrophic claims
│   │   └─ Example: $50M cap for 1,000-person company
│   │
│   └─ Minimum benefit floor:
│       ├─ Ensures no employee gets benefit too low
│       ├─ Example: $50K floor (employee at very low salary still gets $50K)
│       └─ Fairness: Prevents inadequate coverage for entry-level workers
│
├─ Age-Based Reductions:
│   ├─ Age <25: Reduced benefit (e.g., 50% of normal)
│   │   ├─ Rationale: Lower mortality; cost savings
│   │   ├─ Example: Age 23, 2× salary $60K normally = $120K; reduced to $60K
│   │   └─ Market practice: Common for under-25
│   │
│   ├─ Age 25–64: Full benefit (100%)
│   │   └─ Prime years; standard rates apply
│   │
│   ├─ Age 65–69: Reduced benefit (e.g., 50% at age 65)
│   │   ├─ Rationale: Increasing mortality; cost control into retirement
│   │   ├─ Decline: Gradual or cliff; varies by plan
│   │   ├─ Example: Age 65 gets 50%, age 70 gets 25% or terminates
│   │   └─ Market practice: Very common (40-50% plans apply age reductions)
│   │
│   ├─ Age 70+: Terminated or minimal (e.g., $10K-$25K)
│   │   ├─ Rationale: High mortality; cost becomes prohibitive
│   │   ├─ Conversion option: Convert group to individual before expiration
│   │   └─ Retiree coverage: Separate, reduced benefit (if offered)
│   │
│   └─ Schedule example:
│       ├─ Age 25-64: 100% benefit
│       ├─ Age 65: 75% benefit
│       ├─ Age 66: 50% benefit
│       ├─ Age 70: Terminates
│       └─ Formula: 100% − 5% per year after 64
│
├─ Occupational Rating Modifications:
│   ├─ Standard risk (office, professional): 100% of normal benefit
│   ├─ Hazardous (construction, mining): 75-90% benefit (reduced for exposure)
│   ├─ High-hazard (offshore, law enforcement): 50% benefit or exclusion
│   └─ Rationale: Reflect increased mortality for hazardous roles
│
└─ Comparisons to Social Security:
    ├─ Social Security survivor benefit (SSNB):
    │   ├─ Widow/widower: ~75% of deceased's PIA (Primary Insurance Amount)
    │   ├─ Children: 75% each (multiple children share cap)
    │   ├─ Family cap: ~175-180% of deceased PIA
    │   ├─ Typical: $2,000-$3,000/month total for family
    │   └─ Adequacy: Modest; often insufficient alone
    │
    ├─ Group life + SSNB integration:
    │   ├─ Combined: $100K group life + $30K/year SSNB ($2,500/mo)
    │   ├─ Total resources: ~$6,000/month income replacement
    │   ├─ Planning horizon: Covers 10-15 year gap to age 18 (child support)
    │   └─ Spouse coverage: Must bridge until SS kicks in at age 60 (reduced) or 66 (full)
    │
    └─ Rule of thumb:
        ├─ Total death benefit (all sources) should be 5-10× annual income
        ├─ Group life covers 2-3×
        ├─ Supplemental (voluntary) covers 1-3×
        ├─ Social Security covers ~2× over 15 years
        └─ Gap: Often requires individual policy for high earners
```

## 5. Mini-Project: Optimal Benefit Design by Company Profile

**Goal:** Compare coverage designs across company types and salary distributions.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Company profiles
companies = {
    'Startup (Tech)': {'n': 150, 'salary_mean': 95000, 'salary_std': 35000},
    'Fortune 500': {'n': 5000, 'salary_mean': 65000, 'salary_std': 25000},
    'Manufacturing': {'n': 800, 'salary_mean': 45000, 'salary_std': 12000},
    'Professional Services': {'n': 250, 'salary_mean': 85000, 'salary_std': 40000},
}

designs = {
    '1x Salary': lambda s: s * 1,
    '2x Salary': lambda s: s * 2,
    '3x Salary': lambda s: s * 3,
    'Fixed $100K': lambda s: 100000,
}

# Generate salary distributions (lognormal is realistic)
np.random.seed(42)
salary_data = {}

for company, params in companies.items():
    n = params['n']
    mean = params['salary_mean']
    std = params['salary_std']
    
    # Fit lognormal (more realistic than normal for salary)
    sigma = np.sqrt(np.log(1 + (std/mean)**2))
    mu = np.log(mean) - sigma**2 / 2
    
    salaries = np.random.lognormal(mu, sigma, n)
    salary_data[company] = salaries

# Calculate benefit costs and statistics
results = []

for company, salaries in salary_data.items():
    for design_name, design_func in designs.items():
        benefits = np.array([design_func(s) for s in salaries])
        
        # Mortality assumption: simplified (age-averaged)
        mortality_rate = 0.003  # 0.3% annual mortality (group average)
        annual_claim_cost = (benefits * mortality_rate).sum()
        
        # Admin load
        admin_load = 0.12  # 12% administrative cost
        total_premium = annual_claim_cost * (1 + admin_load)
        
        monthly_per_employee = total_premium / len(salaries) / 12
        
        results.append({
            'Company': company,
            'Design': design_name,
            'Avg Benefit': benefits.mean(),
            'Max Benefit': benefits.max(),
            'Min Benefit': benefits.min(),
            'Std Dev Benefits': benefits.std(),
            'Annual Premium': total_premium,
            'Monthly per Employee': monthly_per_employee,
            'Premium as % of Payroll': (total_premium / salaries.sum()) * 100,
        })

results_df = pd.DataFrame(results)

# Display
print("BENEFIT DESIGN COMPARISON BY COMPANY\n")
for company in companies.keys():
    print(f"\n{company}:")
    subset = results_df[results_df['Company'] == company]
    print(subset[['Design', 'Avg Benefit', 'Monthly per Employee', 'Premium as % of Payroll']].to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Average benefit by design and company
pivot_avg = results_df.pivot_table(values='Avg Benefit', index='Company', columns='Design')
pivot_avg.plot(kind='bar', ax=axes[0, 0], color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
axes[0, 0].set_title('Average Benefit by Company & Design')
axes[0, 0].set_ylabel('Benefit ($)')
axes[0, 0].legend(title='Design', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Monthly cost per employee
pivot_cost = results_df.pivot_table(values='Monthly per Employee', index='Company', columns='Design')
pivot_cost.plot(kind='bar', ax=axes[0, 1], color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
axes[0, 1].set_title('Monthly Cost per Employee')
axes[0, 1].set_ylabel('Cost ($)')
axes[0, 1].legend(title='Design', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Salary distribution for Startup
ax = axes[1, 0]
salaries_startup = salary_data['Startup (Tech)']
ax.hist(salaries_startup, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(salaries_startup.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${salaries_startup.mean():.0f}')
ax.set_xlabel('Salary ($)')
ax.set_ylabel('Number of Employees')
ax.set_title('Salary Distribution: Startup (Tech)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Benefit vs salary scatter (Startup, all designs)
ax = axes[1, 1]
colors_scatter = {'1x Salary': 'steelblue', '2x Salary': 'orange', '3x Salary': 'green', 'Fixed $100K': 'red'}
for design in designs.keys():
    benefits = np.array([designs[design](s) for s in salaries_startup])
    ax.scatter(salaries_startup, benefits, alpha=0.5, s=30, label=design, color=colors_scatter[design])

ax.set_xlabel('Salary ($)')
ax.set_ylabel('Death Benefit ($)')
ax.set_title('Benefit vs Salary: Startup (Tech)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Adequacy analysis: % of employees with benefit < 3× salary (80% replacement)
print("\n\nADEQUACY ANALYSIS (% with benefit < 3× salary):")
for company, salaries in salary_data.items():
    for design_name, design_func in designs.items():
        benefits = np.array([design_func(s) for s in salaries])
        target = salaries * 3
        pct_adequate = (benefits >= target).mean() * 100
        results_df.loc[(results_df['Company'] == company) & (results_df['Design'] == design_name), 'Adequacy %'] = pct_adequate

# Summary table
adequacy_pivot = results_df.pivot_table(values='Adequacy %', index='Company', columns='Design')
print(adequacy_pivot.to_string())
```

**Key Insights:**
- 2× salary: Market standard; ~60% lower cost than 3×, good adequacy for most
- Startup tech: Often offers 3× or higher to compete for talent
- Fixed $100K: Works for low-wage cohorts; inadequate for high earners
- Age reductions: Save 15–25% on total premium cost
- Salary growth: Benefit increases with pay; protects top earners proportionally

## 6. Relationships & Dependencies
- **To Pricing:** Benefit amount primary cost driver; linear relationship to premium
- **To Participation:** Generous benefit (3×) improves enrollment; low cost may reduce
- **To Offsets:** Social Security survivor benefits interact; total household income estimated
- **To Underwriting:** High individual amounts may trigger medical underwriting

## References
- [Milliman Group Life Benchmarks](https://www.milliman.com) - Market benefit design data
- [Society of Actuaries (SOA) Experience Studies](https://www.soa.org) - Mortality assumptions
- [American Council of Life Insurers (ACLI)](https://www.acli.com) - Industry standards

