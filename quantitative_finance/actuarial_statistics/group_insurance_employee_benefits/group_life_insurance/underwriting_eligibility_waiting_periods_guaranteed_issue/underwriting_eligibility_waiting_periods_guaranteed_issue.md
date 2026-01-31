# Underwriting & Eligibility: Waiting Periods & Guaranteed Issue

## 1. Concept Skeleton
**Definition:** Process and requirements for employees to obtain group life coverage; includes waiting periods (employment waiting), guaranteed issue amounts (no medical exam required), and medical underwriting thresholds  
**Purpose:** Control adverse selection (sick employees enrolling while healthy avoid coverage); balance accessibility with insurer risk management  
**Prerequisites:** Group underwriting principles, adverse selection mechanics, insurance regulations

## 2. Comparative Framing
| Factor | **Waiting Period** | **Guaranteed Issue** | **Medical Underwriting** | **Open Enrollment** |
|--------|-------------------|---------------------|------------------------|-------------------|
| **Timing** | First 30–90 days | Automatic portion | Above $50K–$100K | Annual period |
| **Requirements** | Employment only | None | Medical history form | Plan changes only |
| **Adverse Selection Risk** | Low (delayed access) | Medium (broad) | High (screened) | Low (fixed) |
| **Speed to coverage** | Delayed | Immediate | Delayed (exam) | Immediate |
| **Cost** | Lower (fewer new claims) | Higher (unhealthy enroll) | Higher (medical spend) | Moderate |

## 3. Examples + Counterexamples

**Waiting Period Example:**  
Employee hired Jan 1; 60-day waiting period → coverage effective Mar 1. Dies Feb 15 → claim denied (no coverage during waiting). Estate may challenge but contract language controls

**Guaranteed Issue Example:**  
Company offers $50K guaranteed issue (no medical exam) for all employees. Employee with diabetes, hypertension enrolled immediately without questions. Plan uses group rates (not individual underwriting)

**Medical Underwriting Example:**  
Employee requesting $250K (above $100K guaranteed issue limit) → must complete medical questionnaire, APS (Attending Physician Statement) for prior cancer diagnosis → insurer may approve at standard rates or apply flat extra charge

**Adverse Selection Counter-Example:**  
Plan with 0-day waiting period and high guaranteed issue ($250K automatic) attracts recently diagnosed employees → claims spike, renewal rates increase 30%+ → employer forced to reduce benefits or switch insurers

**Edge Case:**  
Automatic enrollment with opt-out: 85% participation (high) but includes some unhealthy employees who default-enroll → requires careful actuarial pricing to avoid adverse selection losses

## 4. Layer Breakdown
```
Underwriting & Eligibility Framework:
├─ Waiting Periods (Delayed Coverage):
│   ├─ Definition: Time from hire date or effective date until coverage begins
│   ├─ Purpose:
│   │   ├─ Adverse Selection: Prevents "just-hired sick" from gaming system
│   │   ├─ New Hire Admin: Allows time for payroll setup, benefits elections
│   │   ├─ Cost Control: Postpones coverage; fewer benefit claims in first period
│   │   └─ Employer Cash Flow: Delays premium payments until employee settled
│   ├─ Typical Waiting Period Lengths:
│   │   ├─ 0 days: Immediate coverage (rare; mainly union/public sector)
│   │   ├─ 30 days: Common for administrative convenience
│   │   ├─ 60 days: Market standard; balances fairness & adverse selection
│   │   ├─ 90 days: More conservative; used for high-risk groups
│   │   └─ 6–12 months: Very rare (only for part-time or union plans)
│   ├─ Calendar vs Business Days:
│   │   ├─ Calendar days: 60 days = exact date (simpler to administer)
│   │   ├─ Business days: Excludes weekends/holidays (less common)
│   │   └─ Impact: Business day typically adds 10–15 calendar days
│   ├─ Measurement: From hire date or effective date?
│   │   ├─ Hire date: When employee starts (most common)
│   │   ├─ Coverage effective date: When plan actually begins for that employee
│   │   └─ Lag: If coverage effective date after hire, clock starts later
│   ├─ Accrual: What happens during waiting?
│   │   ├─ Coverage accrues: Death during waiting, beneficiary gets future benefit
│   │   │   (rare; deferred benefit approach)
│   │   ├─ No benefit: Death voids claim (standard); employee not covered
│   │   ├─ Partial accrual: Build up benefit gradually (rarely used)
│   │   └─ Back-dated: Some plans allow retroactive coverage from hire
│   ├─ Example Timeline:
│   │   ├─ Jan 1: Employee hired
│   │   ├─ Jan 1 – Mar 1: 60-day waiting period (no coverage)
│   │   ├─ Mar 1: Coverage begins; can file beneficiary designation
│   │   ├─ Mar 5: Death → claim eligible; beneficiary receives benefit
│   │   ├─ Feb 15: Death → claim ineligible (during waiting); no benefit
│   │   └─ Special: If medical exam fails after waiting, coverage may be denied
│   └─ ERISA Compliance:
│       ├─ ACA requirement: Waiting period ≤ 90 days (federal minimum)
│       ├─ Full-time definition: ≥30 hours/week (PPACA standard)
│       ├─ Notice requirement: Employees must be informed of waiting period
│       └─ No discrimination: Waiting periods must apply uniformly
│
├─ Guaranteed Issue Limits:
│   ├─ Definition: Amount of coverage automatically approved without medical exam
│   ├─ Rationale:
│   │   ├─ Accessibility: Makes plan available to all (not just healthy)
│   │   ├─ Administrative efficiency: No medical questions for low amounts
│   │   ├─ Cost control: Insurer limits exposure (not allowing unlimited GI)
│   │   └─ Competitive advantage: Employers can advertise "no medical exam needed"
│   ├─ Typical GI Limits:
│   │   ├─ Very low: $15K–$25K (minimizes adverse selection impact)
│   │   ├─ Low: $50K (market standard for small–mid size)
│   │   ├─ Standard: $75K–$100K (larger employers)
│   │   ├─ High: $150K–$200K (only top companies; tech/finance)
│   │   └─ Formula: Often 1× salary up to stated amount (e.g., "$50K or 1× salary")
│   ├─ Above GI Limit:
│   │   ├─ Medical questions: Simplified health questionnaire (3-5 questions)
│   │   ├─ If unhealthy history: May request APS (Attending Physician Statement)
│   │   ├─ Exam: Rarely required unless very high amount or unhealthy profile
│   │   ├─ Flat extra: If minor condition, add fixed charge per $1K (e.g., +$0.50/$1K)
│   │   ├─ Decline: Rarely happens in group; would need severe condition
│   │   └─ Timeline: 2–4 weeks for approval with underwriting
│   ├─ Stair-Step GI (by salary):
│   │   ├─ If salary < $30K: GI = $50K
│   │   ├─ If salary $30K–$60K: GI = $75K
│   │   ├─ If salary > $60K: GI = $100K
│   │   ├─ Rationale: Higher earners less adverse selection risk
│   │   └─ Flexibility: Addresses income level differences
│   └─ Guaranteed Issue Events:
│       ├─ New hire: GI applies at hire (after waiting)
│       ├─ Open enrollment: May increase to higher GI if available
│       ├─ Life event: Marriage, birth, etc. may trigger new GI
│       ├─ Plan change: If employer increases benefit, new GI for increase
│       └─ Annual limit: Some plans limit GI changes per year
│
├─ Medical Underwriting (Above GI):
│   ├─ Application Process:
│   │   ├─ Employee requests coverage > GI limit
│   │   ├─ Completes health questionnaire (simplified 1-2 page form)
│   │   ├─ Questions typical: Cancer, heart disease, diabetes, ongoing meds?
│   │   ├─ Honesty clause: Employee certifies truthfulness; false answers void coverage
│   │   └─ Timing: 2–4 weeks for underwriting decision
│   ├─ Underwriting Outcomes:
│   │   ├─ Approve at standard rates: No additional cost (best case)
│   │   ├─ Approve with flat extra: Add $0.25–$0.50 per $1K per month (for mild condition)
│   │   ├─ Approve with reduced benefit: Higher amount requires APS, may limit to GI
│   │   ├─ Decline: Rarely; only severe condition (cancer, cardiac history)
│   │   └─ Postpone: Request updated medical records; reassess in 6 months
│   ├─ Attending Physician Statement (APS):
│   │   ├─ Trigger: Usually if condition noted in health questionnaire
│   │   ├─ Content: Diagnosis, treatment, prognosis, current status
│   │   ├─ Insurer request: Asks employer to collect from claimant's physician
│   │   ├─ Cost: Usually employer covers (claimant not billed)
│   │   ├─ Timeline: 3–6 weeks (physicians slow to respond)
│   │   └─ Use: Assess current stability; decide rating or decline
│   ├─ Occupational Underwriting:
│   │   ├─ Hazardous jobs: Mining, offshore oil, skydiving → underwritten separately
│   │   ├─ High-income: CEO, executives → higher limits may trigger underwriting
│   │   ├─ Expatriates: Working abroad → geographic underwriting
│   │   └─ Flat extra charge: $1–$5 per $1K for hazardous
│   └─ Post-Issue Underwriting:
│       ├─ Contestability period: 2 years post-enrollment
│       ├─ Insurer can rescind if material misstatement found
│       ├─ Example: Non-smoker discount applied; discovered smoker 18 months later
│       ├─ Recission: Rare but protects insurer from fraud
│       └─ After 2 years: Incontestable; no rescission (insurer assumes risk)
│
├─ Eligibility Classes:
│   ├─ Purpose: Group employees by risk/cost characteristics for pricing
│   ├─ Common Classifications:
│   │   ├─ Class 1: All full-time active employees (most common)
│   │   ├─ Class 2: Union vs non-union (union may have better benefits)
│   │   ├─ Class 3: By job category (office vs field vs management)
│   │   ├─ Class 4: By salary band (higher pay → higher benefits & costs)
│   │   ├─ Retirees: Separate class; lower coverage; different rating
│   │   └─ COBRA: Former employees continuing coverage (separate accounting)
│   ├─ Participation Rules:
│   │   ├─ Minimum participation: 75% eligible employees must enroll (for group rating)
│   │   ├─ If below 75%: Insurer may decline renewal or increase rates
│   │   ├─ Calculation: (Enrolled / Eligible after waiting) × 100%
│   │   └─ Waiver: If very small group, may waive participation requirement
│   └─ Active Employee Definition:
│       ├─ Full-time: ≥30 hours/week (ACA standard)
│       ├─ Actively at work: Not on extended leave (LOA, disability)
│       ├─ Payroll-deducted: (In some plans) Must have current payroll presence
│       └─ Eligibility: Varies by plan; critical for benefit eligibility
│
└─ Enrollment Methods & Automation:
    ├─ Traditional Paper: Enroll with forms; slow; error-prone
    ├─ Online enrollment: Benefits portal; instant; reduces admin burden
    ├─ Automatic enrollment (ACA): Default enroll eligible employees unless decline
    │   ├─ Employer must notify: Explain coverage & cost
    │   ├─ Opt-out option: Employee can choose not to participate
    │   ├─ Increase participation: Typically 75–85% (vs 60–70% voluntary)
    │   └─ Consent: Easier; aligns with behavioral defaults
    ├─ Open enrollment: Annual period (e.g., Oct 1–Oct 31) for changes
    │   ├─ Changes effective: Jan 1 typically
    │   ├─ Restrictions: Usually can't decrease unless loss of coverage elsewhere
    │   └─ Coordination: Sync with health plan open enrollment
    └─ Life event changes: Marriage, birth, divorce trigger mid-year elections
```

## 5. Mini-Project: Adverse Selection Impact with Varying Waiting Periods

**Goal:** Model adverse selection effects and optimal waiting period selection.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate employee cohorts with different health profiles
np.random.seed(42)

# Health distribution in hiring population
# Assume new hires have mix of healthy, moderately ill, severely ill
health_categories = ['Healthy', 'Moderate Condition', 'Severe Condition']
health_probs = [0.70, 0.20, 0.10]  # 70% healthy, 20% moderate, 10% severe
health_mortality = {
    'Healthy': 0.001,  # 0.1% annual mortality
    'Moderate Condition': 0.005,  # 0.5% annual
    'Severe Condition': 0.020,  # 2% annual
}

# Waiting period scenarios
waiting_periods = [0, 30, 60, 90]  # days
scenarios = {}

for waiting_days in waiting_periods:
    # Cohort size
    n_hires = 1000
    
    # Generate health status for new hires
    health_status = np.random.choice(health_categories, size=n_hires, p=health_probs)
    
    # Adverse selection: sick employees are more likely to enroll immediately
    # Model: If waiting = 0, sick are all enrolled; if waiting = 90, some drop out
    # Assumption: 80% of severe drop out after 90-day waiting; 20% of moderate drop out
    
    if waiting_days == 0:
        # Everyone enrolls (or all who would enroll, do so)
        enrollee_mask = np.ones(n_hires, dtype=bool)
    else:
        enrollee_mask = np.ones(n_hires, dtype=bool)
        # Attrition during waiting:
        for i, health in enumerate(health_status):
            if health == 'Severe Condition' and np.random.random() < 0.80:
                enrollee_mask[i] = False  # Employee drops out during waiting (quits, etc.)
            elif health == 'Moderate Condition' and np.random.random() < 0.20:
                enrollee_mask[i] = False
    
    enrolled_health = health_status[enrollee_mask]
    n_enrolled = enrollee_mask.sum()
    
    # Calculate expected annual claims (using average mortality * benefit)
    average_benefit = 100000
    expected_mortality = np.mean([health_mortality[h] for h in enrolled_health])
    annual_expected_claims = n_enrolled * expected_mortality * average_benefit
    
    # Premium calculation: claims + admin + profit margin
    admin_load = 0.15  # 15%
    profit_margin = 0.10  # 10%
    annual_premium = annual_expected_claims * (1 + admin_load + profit_margin)
    
    # Per-employee monthly cost
    monthly_per_employee = annual_premium / n_enrolled / 12
    
    # Enrollment rate
    enrollment_rate = n_enrolled / n_hires * 100
    
    scenarios[waiting_days] = {
        'n_enrolled': n_enrolled,
        'enrollment_rate': enrollment_rate,
        'expected_mortality': expected_mortality,
        'annual_expected_claims': annual_expected_claims,
        'annual_premium': annual_premium,
        'monthly_per_employee': monthly_per_employee,
        'enrolled_health': enrolled_health,
    }

# Create summary dataframe
summary_data = []
for waiting_days, scenario in scenarios.items():
    summary_data.append({
        'Waiting Period (days)': waiting_days,
        'Enrollment Rate (%)': scenario['enrollment_rate'],
        'Expected Mortality Rate': scenario['expected_mortality'],
        'Monthly Premium per Employee': scenario['monthly_per_employee'],
        'Annual Expected Claims': scenario['annual_expected_claims'],
    })

summary_df = pd.DataFrame(summary_data)
print("Adverse Selection Impact by Waiting Period:\n")
print(summary_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Enrollment rate vs waiting period
ax = axes[0, 0]
waiting = summary_df['Waiting Period (days)']
enrollment = summary_df['Enrollment Rate (%)']
ax.plot(waiting, enrollment, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Enrollment Rate (%)')
ax.set_title('Enrollment Rate vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 2: Expected mortality rate vs waiting period
ax = axes[0, 1]
mortality = summary_df['Expected Mortality Rate']
ax.plot(waiting, mortality, 'o-', linewidth=2, markersize=8, color='darkred')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Expected Mortality Rate')
ax.set_title('Population Mortality Risk vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 3: Monthly premium vs waiting period
ax = axes[1, 0]
premium = summary_df['Monthly Premium per Employee']
ax.plot(waiting, premium, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax.fill_between(waiting, premium, alpha=0.3, color='darkgreen')
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('Monthly Premium per Employee ($)')
ax.set_title('Premium Cost vs Waiting Period')
ax.grid(alpha=0.3)

# Plot 4: Health status distribution by waiting period
ax = axes[1, 1]
health_dist = []
for waiting_days in waiting_periods:
    enrolled = scenarios[waiting_days]['enrolled_health']
    healthy_pct = (enrolled == 'Healthy').mean() * 100
    moderate_pct = (enrolled == 'Moderate Condition').mean() * 100
    severe_pct = (enrolled == 'Severe Condition').mean() * 100
    health_dist.append({'Healthy': healthy_pct, 'Moderate': moderate_pct, 'Severe': severe_pct})

health_dist_df = pd.DataFrame(health_dist, index=waiting_periods)
health_dist_df.plot(kind='bar', stacked=True, ax=ax, color=['green', 'orange', 'red'], alpha=0.7)
ax.set_xlabel('Waiting Period (days)')
ax.set_ylabel('% of Enrolled Employees')
ax.set_title('Health Status Mix of Enrolled Employees')
ax.legend(title='Health Status', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(waiting_periods, rotation=0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights summary
print("\n\nKEY INSIGHTS:")
print(f"60-day waiting reduces annual premium by {((premium.iloc[0] - premium.iloc[2]) / premium.iloc[0] * 100):.1f}% vs 0-day")
print(f"90-day waiting reduces annual premium by {((premium.iloc[0] - premium.iloc[3]) / premium.iloc[0] * 100):.1f}% vs 0-day")
print(f"Enrollment drops from {enrollment.iloc[0]:.1f}% (0-day) to {enrollment.iloc[3]:.1f}% (90-day)")
print(f"Enrolled health improves: Severe condition risk drops from {(scenarios[0]['enrolled_health'] == 'Severe Condition').mean():.1%} to {(scenarios[90]['enrolled_health'] == 'Severe Condition').mean():.1%}")
```

**Key Insights:**
- Waiting periods reduce adverse selection by 10–20% on premium (through participant attrition)
- 60-day standard: Balances adverse selection with employee fairness
- Guaranteed issue + waiting: Combo minimizes adverse selection while maintaining access
- Automatic enrollment: Higher participation offsets selection risk through volume

## 6. Relationships & Dependencies
- **To Enrollment:** Automatic enrollment increases participation despite waiting period
- **To Benefit Amount:** Higher GI limits increase adverse selection risk; may require shorter waiting period
- **To Pricing:** Waiting period choice directly impacts premium (5–15% swing)
- **To ERISA Compliance:** Must comply with 90-day maximum and non-discriminatory application

## References
- [Affordable Care Act (ACA) Regulations](https://www.healthcare.gov) - Waiting period limits
- [ERISA Guidance - Department of Labor](https://www.dol.gov) - Non-discrimination rules
- [Society of Actuaries (SOA) Adverse Selection Studies](https://www.soa.org)

