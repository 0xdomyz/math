# Term Insurance

## 1. Concept Skeleton
**Definition:** Fixed death benefit B payable only if death occurs within n-year term; pure risk protection without savings component  
**Purpose:** Lowest-cost insurance protection; risk management for income replacement, mortgage protection, family security  
**Prerequisites:** Present value of benefits (Aₓ:n̄|), discount factor (v), survival probability (ₚₓ)

## 2. Comparative Framing
| Term Length | Typical Rate | Payout Condition | Premium Pattern | Use Case |
|------------|--------------|------------------|-----------------|----------|
| **10-Year** | $20-35/mo | Death within 10 years | Level, guaranteed | Young family |
| **20-Year** | $30-60/mo | Death within 20 years | Level, guaranteed | Mortgage payoff |
| **30-Year** | $40-100/mo | Death within 30 years | Level, guaranteed | Career span |
| **Renewable Term** | Rising | Death during term | Increases at renewal | Temporary coverage |
| **Convertible Term** | Lower | Death within term (→ whole life later) | Level initially | Flexibility option |
| **Decreasing Term** | Lowest | Death within term (benefit ↓) | Level premium | Mortgage/loan protection |

## 3. Examples + Counterexamples

**Simple Example:**  
Age 35, $300K, 20-year term, APV ≈ $12,500 (4.2%); Level net premium ≈ $625/year ($52/mo)

**Failure Case:**  
Using perpetuity formula (1/i = 25 × annual premium): $625 × 25 = $15,625 > $12,500; ignores probability of surviving full term

**Edge Case:**  
Age 75, 10-year term: APV ≈ 12% of benefit (vs 4% at age 35) because 1-in-5 chance of dying within decade; premium very high

## 4. Layer Breakdown
```
Term Insurance Structure:
├─ Definition & Valuation:
│   ├─ APV = Aₓ:n̄| = ∑ᵏ₌₁ⁿ v^k · ₖ₋₁pₓ · qₓ₊ₖ₋₁
│   ├─ Probability of dying in year k: ₖ₋₁pₓ · qₓ₊ₖ₋₁
│   ├─ Discount for year k: v^k = 1/(1+i)^k
│   └─ Net Premium P = APV / a̅ₓ:n̄|  (annuity due for ongoing payments)
├─ Key Characteristics:
│   ├─ Term expires: No value after n years (pure risk)
│   ├─ Level death benefit: B constant throughout
│   ├─ Level premium: P constant (if guaranteed level)
│   ├─ No cash value: Cannot surrender/borrow against
│   └─ Renewable/Convertible: Options to extend or convert to permanent
├─ Rider Options:
│   ├─ Waiver of Premium: Premiums waived if insured disabled
│   ├─ Accidental Death Benefit: Additional payout for accidents
│   ├─ Return of Premium: If survive n years, return premiums (increases cost 20-50%)
│   └─ Guaranteed Insurability: Option to buy additional coverage without medical
├─ Pricing Components:
│   ├─ Mortality assumption: Standard, Preferred, Smoker tables
│   ├─ Interest rate: 3-5% typical (conservative)
│   ├─ Expenses: Acquisition (10-15% of year 1 premium), maintenance ($20-50/year)
│   ├─ Profit load: 10-20% margin on net premium
│   └─ Gross Premium = [APV + PV(expenses) + Profit]/PV(annuity of premiums)
├─ Product Variations:
│   ├─ Annually Renewable: ART, premium increases each year
│   ├─ Level-Premium Term: Cost fixed for n years, then expires
│   ├─ Return of Premium (ROP): Surrender value = premiums paid (high cost)
│   ├─ Convertible: Can convert to whole life without medical exam
│   ├─ Decreasing Term: Benefit ↓ linearly (original amount falls with balance)
│   └─ Increasing Term: Benefit ↑ (rare; cost-prohibitive)
├─ Underwriting Considerations:
│   ├─ Medical exam: Age 40+, higher amounts typically required
│   ├─ Tobacco use: 2× premium if smoker
│   ├─ Build/Height: BMI-based ratings (underweight, overweight)
│   ├─ Occupational hazard: Pilot, mining adds 10-50% load
│   ├─ Avocation: Hazardous hobbies (skydiving, mountaineering)
│   └─ Moral hazard: High benefit amount relative to income (deny or cap)
└─ Comparative Advantage:
    ├─ vs. Whole Life: 70% cheaper; no cash value; expires
    ├─ vs. Universal Life: Cheaper, simpler; fixed not flexible
    ├─ vs. Self-Insurance: Pooled risk more efficient than personal savings
    └─ vs. Group Insurance: Individual underwriting; can't lose coverage if change jobs
```

**Key Decision:** Select term length matching liability duration (mortgage 30 years, child support 18 years, etc.)

## 5. Mini-Project
Calculate premiums, build profitability analysis, and model policyholder behavior:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

# 1. MORTALITY TABLE & ASSUMPTIONS
print("=" * 80)
print("TERM INSURANCE: PREMIUM CALCULATION & PROFITABILITY ANALYSIS")
print("=" * 80)

# Illustrative mortality (age, qx)
mortality_data = {
    25: 0.00067,
    30: 0.00084,
    35: 0.00103,
    40: 0.00131,
    45: 0.00172,
    50: 0.00233,
    55: 0.00325,
    60: 0.00459,
    65: 0.00653
}

# Separate by smoking status
smoker_load = 2.0  # Smokers pay 2× mortality
preferred_discount = 0.8  # Preferred pays 80% of standard

# Interest assumptions
annual_rate = 0.04
v = 1 / (1 + annual_rate)

# Expense assumptions
acquisition_expense_pct = 0.15  # 15% of year 1 premium
maintenance_per_year = 25  # $25/year
profit_margin_pct = 0.15  # 15% load on net premium

print(f"\nAssumptions:")
print(f"  Interest Rate: {annual_rate*100:.1f}%")
print(f"  Acquisition Expense: {acquisition_expense_pct*100:.0f}% of Year 1 Premium")
print(f"  Maintenance Expense: ${maintenance_per_year:.0f}/year")
print(f"  Profit Margin: {profit_margin_pct*100:.0f}% of Net Premium")
print()

# 2. NET PREMIUM CALCULATION
print("=" * 80)
print("NET PREMIUM: 20-YEAR LEVEL TERM, STANDARD MORTALITY")
print("=" * 80)

def calculate_term_premiums(start_age, benefit, term_length, mortality_dict, annual_rate_calc, 
                            smoker_mult=1.0, preferred_mult=1.0, expense_pct=0, expense_fixed=0, 
                            profit_pct=0):
    """Calculate net and gross premiums for term insurance"""
    
    # Adjust mortality
    mortality_adjusted = {}
    for age, qx in mortality_dict.items():
        mortality_adjusted[age] = qx * smoker_mult * preferred_mult
    
    # Get ages for term
    ages_term = []
    qx_values = []
    
    for k in range(1, term_length + 1):
        age_in_year = start_age + k - 1
        
        # Linear interpolation for ages between table values
        if age_in_year in mortality_adjusted:
            qx = mortality_adjusted[age_in_year]
        else:
            ages_available = sorted(mortality_adjusted.keys())
            # Find bracketing ages
            age_below = max([a for a in ages_available if a <= age_in_year], default=ages_available[0])
            age_above = min([a for a in ages_available if a >= age_in_year], default=ages_available[-1])
            
            if age_below == age_above:
                qx = mortality_adjusted[age_below]
            else:
                # Linear interpolation
                qx_below = mortality_adjusted[age_below]
                qx_above = mortality_adjusted[age_above]
                qx = qx_below + (qx_above - qx_below) * (age_in_year - age_below) / (age_above - age_below)
        
        ages_term.append(age_in_year)
        qx_values.append(qx)
    
    # Calculate APV of benefits
    apv_benefits = 0
    kpx = 1.0  # k-year survival probability
    
    for k, qx_k in enumerate(qx_values, 1):
        vk = (1 / (1 + annual_rate_calc)) ** k
        pv_benefit_k = kpx * qx_k * vk * benefit
        apv_benefits += pv_benefit_k
        
        px_k = 1 - qx_k
        kpx *= px_k
    
    # Calculate PV of annuity due (premium payments at beginning of each year)
    pv_annuity_due = 0
    kpx_annuity = 1.0
    
    for k in range(0, term_length):
        vk = (1 / (1 + annual_rate_calc)) ** k
        pv_annuity_due += kpx_annuity * vk
        
        if k < term_length - 1:
            qx_k_ann = qx_values[k]
            px_k_ann = 1 - qx_k_ann
            kpx_annuity *= px_k_ann
    
    # Net premium (no expenses, no profit)
    net_premium_annual = apv_benefits / pv_annuity_due
    
    # Gross premium with expenses and profit
    pv_expense_y1 = expense_pct * net_premium_annual + expense_fixed  # Year 1 expense
    pv_expense_ongoing = 0
    kpx_exp = 1.0
    
    for k in range(1, term_length):
        vk = (1 / (1 + annual_rate_calc)) ** k
        pv_expense_ongoing += kpx_exp * vk * expense_fixed
        qx_k_exp = qx_values[k-1]
        px_k_exp = 1 - qx_k_exp
        kpx_exp *= px_k_exp
    
    total_pv_expense = pv_expense_y1 + pv_expense_ongoing
    pv_profit_margin = profit_pct * apv_benefits
    
    gross_premium_annual = (apv_benefits + total_pv_expense + pv_profit_margin) / pv_annuity_due
    
    return {
        'net_annual': net_premium_annual,
        'net_monthly': net_premium_annual / 12,
        'gross_annual': gross_premium_annual,
        'gross_monthly': gross_premium_annual / 12,
        'apv_benefits': apv_benefits,
        'pv_annuity': pv_annuity_due,
        'pv_expense': total_pv_expense,
        'pv_profit': pv_profit_margin
    }

# Calculate for standard mortality, age 35, $300,000 benefit
start_age = 35
benefit = 300000
term_years = 20

results_standard = calculate_term_premiums(
    start_age, benefit, term_years, mortality_data, annual_rate,
    smoker_mult=1.0, preferred_mult=1.0,
    expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
    profit_pct=profit_margin_pct
)

print(f"\nScenario: Age {start_age}, ${benefit:,.0f}, {term_years}-Year Term")
print(f"\n{'Metric':<35} {'Amount':<20}")
print("-" * 55)
print(f"{'APV of Benefits':<35} ${results_standard['apv_benefits']:>18,.2f}")
print(f"{'PV of Annuity Due (premiums)':<35} {results_standard['pv_annuity']:>19,.2f}")
print()
print(f"{'Net Premium (annual)':<35} ${results_standard['net_annual']:>18,.2f}")
print(f"{'Net Premium (monthly)':<35} ${results_standard['net_monthly']:>18,.2f}")
print()
print(f"{'Gross Premium (annual)':<35} ${results_standard['gross_annual']:>18,.2f}")
print(f"{'Gross Premium (monthly)':<35} ${results_standard['gross_monthly']:>18,.2f}")
print()
print(f"PV of Expenses: ${results_standard['pv_expense']:>18,.2f}")
print(f"PV of Profit Margin: ${results_standard['pv_profit']:>18,.2f}")
print()

# 3. COMPARATIVE ANALYSIS: STANDARD VS SMOKER VS PREFERRED
print("=" * 80)
print("PREMIUM COMPARISON: STANDARD vs SMOKER vs PREFERRED")
print("=" * 80)

scenarios = {
    'Standard': {'smoker': 1.0, 'preferred': 1.0},
    'Smoker': {'smoker': 2.0, 'preferred': 1.0},
    'Preferred Non-Smoker': {'smoker': 1.0, 'preferred': 0.8}
}

print(f"\nAge {start_age}, ${benefit:,.0f}, {term_years}-Year Term\n")
print(f"{'Scenario':<25} {'Monthly':<15} {'Annual':<15} {'% vs Standard':<15}")
print("-" * 70)

standard_monthly = results_standard['gross_monthly']

for scenario, adjustments in scenarios.items():
    results = calculate_term_premiums(
        start_age, benefit, term_years, mortality_data, annual_rate,
        smoker_mult=adjustments['smoker'], preferred_mult=adjustments['preferred'],
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=profit_margin_pct
    )
    
    pct_diff = (results['gross_monthly'] / standard_monthly - 1) * 100
    print(f"{scenario:<25} ${results['gross_monthly']:<14,.2f} ${results['gross_annual']:<14,.2f} {pct_diff:>13.1f}%")

print()

# 4. PREMIUMS BY AGE & TERM LENGTH
print("=" * 80)
print("PREMIUM MATRIX: BY AGE AND TERM LENGTH")
print("=" * 80)

ages_matrix = [25, 30, 35, 40, 45, 50]
terms_matrix = [10, 20, 30]

print(f"\nBenefit: ${benefit:,.0f}\nMonthly Premium:\n")
print(f"{'Age':<8}", end='')
for term in terms_matrix:
    print(f"{term}-Yr", end='\t')
print()
print("-" * 50)

for age in ages_matrix:
    print(f"{age:<8}", end='')
    
    for term in terms_matrix:
        # Expand mortality table if needed using Gompertz
        mortality_expanded = mortality_data.copy()
        
        for test_age in range(age, age + term):
            if test_age not in mortality_expanded:
                # Simple projection beyond table
                if test_age < 25:
                    mortality_expanded[test_age] = 0.0005
                elif test_age > 65:
                    # Gompertz extrapolation
                    mu = 0.0001 * (1.075 ** test_age)
                    mortality_expanded[test_age] = 1 - np.exp(-mu)
                else:
                    # Interpolation
                    ages_avail = sorted(mortality_expanded.keys())
                    if test_age in ages_avail:
                        continue
                    age_below = max([a for a in ages_avail if a < test_age])
                    age_above = min([a for a in ages_avail if a > test_age])
                    qx_below = mortality_expanded[age_below]
                    qx_above = mortality_expanded[age_above]
                    mortality_expanded[test_age] = qx_below + (qx_above - qx_below) * \
                                                  (test_age - age_below) / (age_above - age_below)
        
        results_matrix = calculate_term_premiums(
            age, benefit, term, mortality_expanded, annual_rate,
            expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
            profit_pct=profit_margin_pct
        )
        
        print(f"${results_matrix['gross_monthly']:<7,.0f}", end='\t')
    
    print()

print()

# 5. CASH FLOW PROJECTION (10-YEAR EXAMPLE)
print("=" * 80)
print("CASH FLOW PROJECTION: YEAR-BY-YEAR (10-YEAR TERM)")
print("=" * 80)

term_cashflow = 10
results_cf = calculate_term_premiums(
    35, benefit, term_cashflow, mortality_data, annual_rate,
    expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
    profit_pct=profit_margin_pct
)

annual_premium = results_cf['gross_annual']
annual_net_premium = results_cf['net_annual']

print(f"\nAge 35, ${benefit:,.0f}, {term_cashflow}-Year Term")
print(f"Annual Premium: ${annual_premium:,.2f}")
print()

ages_cf = []
qx_cf = []
kpx = 1.0

for k in range(1, term_cashflow + 1):
    age_k = 35 + k - 1
    ages_cf.append(age_k)
    
    if age_k in mortality_data:
        qx_k = mortality_data[age_k]
    else:
        qx_k = 0.001  # Placeholder
    
    qx_cf.append(qx_k)

print(f"{'Year':<8} {'Age':<8} {'qₓ':<12} {'ₚₓ':<12} {'Deaths Exp*':<15} {'Premium':<15} {'Exp Paid':<15} {'Profit/Loss**':<15}")
print("-" * 110)

surplus_cumulative = 0

for k in range(1, term_cashflow + 1):
    age_cf = ages_cf[k-1]
    qx = qx_cf[k-1]
    px = 1 - qx
    
    # Expected number of deaths per 100 policies
    expected_deaths = qx * 100
    
    # Claims paid (expected value)
    claims_expected = expected_deaths / 100 * benefit
    
    # Expenses
    expense_y1 = acquisition_expense_pct * annual_premium if k == 1 else 0
    expense_maintenance = maintenance_per_year
    total_expenses = expense_y1 + expense_maintenance
    
    # Revenue
    premium_revenue = annual_premium * 100  # Per 100 policies
    
    # Profit/loss (not accounting for interest or discounting)
    profit_loss = premium_revenue - claims_expected * 100 - total_expenses * 100
    
    surplus_cumulative += profit_loss
    
    print(f"{k:<8} {age_cf:<8} {qx:<12.6f} {px:<12.6f} {expected_deaths:<15.2f} ${annual_premium:<14,.2f} ${total_expenses:<14,.2f} ${profit_loss/100:<14,.2f}")

print("*Deaths per 100 policies; **Profit/loss per 100 policies before interest")
print()

# 6. BREAK-EVEN & PROFITABILITY ANALYSIS
print("=" * 80)
print("PROFITABILITY ANALYSIS: BREAK-EVEN SCENARIOS")
print("=" * 80)

# Calculate what mortality would need to be for break-even
def calculate_breakeven_mortality(net_prem_given, pv_annuity_given, benefit_amount, 
                                  term_given, annual_rate_given):
    """Find mortality rate that yields given premium"""
    # This is complex; we'll estimate by testing mortality multiples
    
    mortality_test = {}
    for age in range(30, 70):
        if age in mortality_data:
            mortality_test[age] = mortality_data[age]
        else:
            mortality_test[age] = 0.001
    
    # Solve numerically
    def objective(mult):
        test_mort = {age: qx * mult for age, qx in mortality_test.items()}
        
        apv_test = 0
        kpx_test = 1.0
        
        for k in range(1, term_given + 1):
            age_test = 35 + k - 1
            qx_test = test_mort.get(age_test, 0.001)
            vk_test = (1 / (1 + annual_rate_given)) ** k
            apv_test += kpx_test * qx_test * vk_test * benefit_amount
            
            px_test = 1 - qx_test
            kpx_test *= px_test
        
        implied_premium = apv_test / pv_annuity_given
        return (implied_premium - net_prem_given) ** 2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x

print(f"\nGross Premium: ${results_standard['gross_annual']:,.2f}/year")
print(f"Net Premium (actuarial cost): ${results_standard['net_annual']:,.2f}/year")
print(f"Expense & Profit Component: ${results_standard['gross_annual'] - results_standard['net_annual']:,.2f}/year")
print()

# Margin analysis
expense_margin = results_standard['pv_expense']
profit_margin = results_standard['pv_profit']
total_margin = expense_margin + profit_margin

print(f"Expense margin (PV): ${expense_margin:,.2f}")
print(f"Profit margin (PV): ${profit_margin:,.2f}")
print(f"Total margin (PV): ${total_margin:,.2f}")
print(f"Margin as % of benefit: {total_margin / results_standard['apv_benefits'] * 100:.1f}%")
print()

# What if mortality worse than expected?
print("Sensitivity to Adverse Experience:")
print()

for mortality_increase in [0.10, 0.25, 0.50]:  # 10%, 25%, 50% worse
    mortality_stressed = {age: qx * (1 + mortality_increase) for age, qx in mortality_data.items()}
    
    results_stressed = calculate_term_premiums(
        35, benefit, 20, mortality_stressed, annual_rate,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=0  # No profit to absorb
    )
    
    inadequacy = results_stressed['net_annual'] - results_standard['net_annual']
    margin_cover_pct = results_standard['gross_annual'] / results_stressed['net_annual'] if results_stressed['net_annual'] > 0 else 0
    
    print(f"  Mortality {mortality_increase*100:>3.0f}% worse: Net premium required ${results_stressed['net_annual']:>10,.2f}")
    print(f"    Gross premium covers {margin_cover_pct*100:>5.1f}% of cost (margin adequacy)")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Premium by age and term
ax = axes[0, 0]
ages_for_plot = list(range(25, 66, 5))
terms_for_plot = [10, 20, 30]

for term in terms_for_plot:
    premiums_by_age = []
    
    for age in ages_for_plot:
        mort_expanded = mortality_data.copy()
        for test_age in range(age, age + term):
            if test_age not in mort_expanded:
                if test_age > 65:
                    mu = 0.0001 * (1.075 ** test_age)
                    mort_expanded[test_age] = 1 - np.exp(-mu)
                else:
                    mort_expanded[test_age] = 0.001
        
        res = calculate_term_premiums(
            age, benefit, term, mort_expanded, annual_rate,
            expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
            profit_pct=profit_margin_pct
        )
        premiums_by_age.append(res['gross_monthly'])
    
    ax.plot(ages_for_plot, premiums_by_age, linewidth=2.5, marker='o', markersize=6, label=f'{term}-Year')

ax.set_xlabel('Starting Age', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title(f'Term Insurance Premiums (${benefit/1000:.0f}K Benefit)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Smoker vs Standard vs Preferred
ax = axes[0, 1]
categories = ['Smoker', 'Standard', 'Preferred']
multipliers = [2.0, 1.0, 0.8]
colors_cat = ['red', 'blue', 'green']

premiums_cat = []
for mult in multipliers:
    res_cat = calculate_term_premiums(
        35, benefit, 20, mortality_data, annual_rate,
        smoker_mult=1.0, preferred_mult=mult,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=profit_margin_pct
    )
    premiums_cat.append(res_cat['gross_monthly'])

bars = ax.bar(categories, premiums_cat, color=colors_cat, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title('Underwriting Class Impact (Age 35, 20-Year Term)', fontsize=12, fontweight='bold')

for bar, prem in zip(bars, premiums_cat):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${prem:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Cash flow and profitability
ax = axes[1, 0]
years_cf_plot = np.arange(1, 11)
premiums_cf = []
expected_claims_cf = []
expenses_cf = []

for k in years_cf_plot:
    age_cf_plot = 35 + k - 1
    qx_cf_plot = mortality_data.get(age_cf_plot, 0.001)
    
    premiums_cf.append(annual_premium)
    expected_claims_cf.append(qx_cf_plot * benefit)
    
    exp_y1 = acquisition_expense_pct * annual_premium if k == 1 else 0
    expenses_cf.append(exp_y1 + maintenance_per_year)

ax.bar(years_cf_plot - 0.25, premiums_cf, width=0.5, label='Premium', alpha=0.6, color='green')
ax.bar(years_cf_plot + 0.25, expected_claims_cf, width=0.5, label='Expected Claims (per policy)', alpha=0.6, color='red')
ax.plot(years_cf_plot, expenses_cf, linewidth=2.5, marker='o', color='orange', label='Expenses', markersize=6)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Annual Cash Flows (10-Year Term)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Benefit amount sensitivity
ax = axes[1, 1]
benefits_for_sens = np.array([100, 250, 500, 750, 1000]) * 1000
premiums_by_benefit = []

for ben in benefits_for_sens:
    res_ben = calculate_term_premiums(
        35, ben, 20, mortality_data, annual_rate,
        expense_pct=acquisition_expense_pct, expense_fixed=maintenance_per_year,
        profit_pct=profit_margin_pct
    )
    premiums_by_benefit.append(res_ben['gross_monthly'])

ax.plot(benefits_for_sens / 1000, premiums_by_benefit, linewidth=2.5, marker='s', markersize=7, color='purple')
ax.fill_between(benefits_for_sens / 1000, 0, premiums_by_benefit, alpha=0.2, color='purple')

ax.set_xlabel('Benefit Amount ($000s)', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title('Premium vs Benefit Amount (Age 35, 20-Year)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('term_insurance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When term insurance fails to protect:
- **Conversion risk**: If premium prohibitive at end of term, must accept whole life rates at older age; renewal shock leads to lapse
- **Mortality experience worse than expected**: Insufficient reserves if actual deaths > assumption; losses spike immediately
- **Persistent inflation**: $300K benefit today worth $150K in 20 years; need increasing benefit rider (costly)
- **Income continuation risk**: If income falls, cannot afford renewal; coverage lapses when needed most
- **Lapse correlation**: When market crashes, policyholders drop term; claims concentration in downturns
- **Underwriting integrity**: Misrepresentation (smoking, alcohol, occupation) discovered post-claim; claim denied or reduced

## 7. Key References
- [SOA Exam FM Term Insurance Pricing](https://www.soa.org/education/exam-req/edu-exam-fm-detail.aspx) - Real examples, case studies
- [Bowers et al., Actuarial Mathematics (Chapter 6)](https://www.soa.org/) - Detailed formulations
- [LIMRA Fact Book (Term Insurance Market Data)](https://www.limra.com/) - Industry statistics, retention rates

---
**Status:** Core product | **Complements:** Convertible Term, Return of Premium Term, Whole Life Insurance
