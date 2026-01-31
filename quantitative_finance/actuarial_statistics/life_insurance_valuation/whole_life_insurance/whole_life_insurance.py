# Auto-extracted from markdown file
# Source: whole_life_insurance.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

# 1. MORTALITY & ASSUMPTIONS
print("=" * 80)
print("WHOLE LIFE INSURANCE: PREMIUM CALCULATION & RESERVE BUILDUP")
print("=" * 80)

# Mortality data by age (standard table)
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873, 90: 0.04213
}

# Gompertz projection for older ages
def gompertz_mortality(age, A=0.0001, B=1.075):
    """Force of mortality: μₓ = A·B^x"""
    return A * (B ** age)

annual_rate = 0.04
v = 1 / (1 + annual_rate)

# Expense assumptions
acquisition_pct = 0.15  # 15% of year 1 premium
maintenance = 25  # $25/year
profit_pct = 0.20  # 20% of net premium

print(f"\nAssumptions:")
print(f"  Interest Rate: {annual_rate*100:.1f}%")
print(f"  Acquisition Expense: {acquisition_pct*100:.0f}% Year 1")
print(f"  Maintenance: ${maintenance}/year")
print(f"  Profit Margin: {profit_pct*100:.0f}% of Net Premium\n")

# 2. WHOLE LIFE PREMIUM CALCULATION
print("=" * 80)
print("WHOLE LIFE PREMIUM CALCULATION: AGE 35, $500,000")
print("=" * 80)

def calculate_whole_life_apv(start_age, mortality_dict, annual_rate_calc, max_age=120):
    """Calculate APV of whole life insurance (benefit of 1)"""
    
    apv = 0
    kpx = 1.0  # k-year survival probability
    
    for k in range(1, max_age - start_age + 1):
        age_in_year = start_age + k - 1
        
        # Get or estimate mortality
        if age_in_year in mortality_dict:
            qx_k = mortality_dict[age_in_year]
        else:
            # Gompertz extrapolation
            mu = gompertz_mortality(age_in_year)
            qx_k = 1 - np.exp(-mu)  # Convert force to annual probability
        
        vk = v ** k
        contribution = kpx * qx_k * vk
        apv += contribution
        
        if contribution < 1e-10:  # Stop when negligible
            break
        
        px_k = 1 - qx_k
        kpx *= px_k
    
    return apv

def calculate_annuity_due(start_age, mortality_dict, annual_rate_calc, max_age=120):
    """Calculate PV of annuity due (premium payments throughout life)"""
    
    pv_annuity = 0
    kpx = 1.0
    
    for k in range(0, max_age - start_age + 1):
        age_in_year = start_age + k
        
        if age_in_year > start_age:
            # Get mortality for year of decrement
            age_decrement = age_in_year - 1
            if age_decrement in mortality_dict:
                qx_prev = mortality_dict[age_decrement]
            else:
                mu_prev = gompertz_mortality(age_decrement)
                qx_prev = 1 - np.exp(-mu_prev)
            
            px_prev = 1 - qx_prev
            kpx *= px_prev
        
        vk = v ** k
        pv_annuity += kpx * vk
        
        if kpx < 1e-10:  # Stop when negligible survival
            break
    
    return pv_annuity

# Calculate for age 35
start_age = 35
benefit = 500000

apv_benefit = calculate_whole_life_apv(start_age, mortality_data, annual_rate)
pv_annuity_life = calculate_annuity_due(start_age, mortality_data, annual_rate)

net_premium_annual = apv_benefit * benefit / pv_annuity_life
net_premium_monthly = net_premium_annual / 12

# Gross premium with expenses and profit
pv_exp_year1 = acquisition_pct * net_premium_annual
pv_exp_ongoing = maintenance  # Simplified

total_expense_pv = pv_exp_year1 + pv_exp_ongoing
profit_pv = profit_pct * apv_benefit * benefit

gross_premium_annual = (apv_benefit * benefit + total_expense_pv + profit_pv) / pv_annuity_life
gross_premium_monthly = gross_premium_annual / 12

print(f"\nAge {start_age}, ${benefit:,.0f} Whole Life\n")
print(f"{'Metric':<35} {'Amount':<20}")
print("-" * 55)
print(f"{'APV of Benefits (per $1)':<35} {apv_benefit:<20.6f}")
print(f"{'APV of Benefits':<35} ${apv_benefit * benefit:>18,.2f}")
print(f"{'PV of Annuity Due':<35} {pv_annuity_life:>19.4f}")
print()
print(f"{'Net Premium (Annual)':<35} ${net_premium_annual:>18,.2f}")
print(f"{'Net Premium (Monthly)':<35} ${net_premium_monthly:>18,.2f}")
print()
print(f"{'Gross Premium (Annual)':<35} ${gross_premium_annual:>18,.2f}")
print(f"{'Gross Premium (Monthly)':<35} ${gross_premium_monthly:>18,.2f}")
print()

# 3. RESERVE ACCUMULATION OVER 40 YEARS
print("=" * 80)
print("POLICY RESERVE ACCUMULATION & CASH SURRENDER VALUE")
print("=" * 80)

reserve_years = 40
reserves = []
csv_values = []
years_vec = []

print(f"\n{'Year':<8} {'Age':<8} {'Reserve':<15} {'CSV (80% of Res)':<18} {'Net Premium':<15} {'Interest Earned':<15}")
print("-" * 90)

reserve_previous = 0

for year in range(1, reserve_years + 1):
    age_year = start_age + year - 1
    
    # Prospective reserve: V = Aₓ₊ₙ - P·ä∞|ₓ₊ₙ
    apv_future = calculate_whole_life_apv(age_year, mortality_data, annual_rate)
    pv_future_premiums = calculate_annuity_due(age_year, mortality_data, annual_rate)
    
    reserve_prospective = (apv_future * benefit - net_premium_annual * pv_future_premiums)
    
    # Retrospective calculation (accumulated premiums + interest - claims)
    reserve_retrospective = reserve_previous * (1 + annual_rate) + net_premium_annual - 0  # No claims yet
    
    reserve = reserve_prospective  # Use prospective
    interest_earned = reserve - reserve_previous * (1 + annual_rate)
    
    csv = reserve * 0.80  # 80% of reserve (typical surrender charge)
    
    reserves.append(reserve)
    csv_values.append(csv)
    years_vec.append(year)
    
    if year <= 5 or year % 5 == 0 or year == 40:
        print(f"{year:<8} {age_year:<8} ${reserve:<14,.0f} ${csv:<17,.0f} ${net_premium_annual:<14,.2f} ${interest_earned:<14,.2f}")
    
    reserve_previous = reserve

print(f"\nReserve at age {start_age + 40}: ${reserve:,.0f}")
print(f"As % of benefit: {reserve/benefit*100:.1f}%")
print()

# 4. COMPARISON: WHOLE LIFE vs TERM INSURANCE
print("=" * 80)
print("WHOLE LIFE vs TERM INSURANCE: CUMULATIVE COST ANALYSIS")
print("=" * 80)

# Calculate term insurance premium for comparison
def calculate_term_premium(start_age_t, benefit_t, term_length, mortality_dict, 
                          annual_rate_calc, margin_pct=0.20):
    """Simplified term premium calculation"""
    
    apv_t = 0
    kpx_t = 1.0
    
    for k in range(1, term_length + 1):
        age_t = start_age_t + k - 1
        
        if age_t in mortality_dict:
            qx_t = mortality_dict[age_t]
        else:
            mu_t = gompertz_mortality(age_t)
            qx_t = 1 - np.exp(-mu_t)
        
        vk_t = (1 / (1 + annual_rate_calc)) ** k
        apv_t += kpx_t * qx_t * vk_t * benefit_t
        
        px_t = 1 - qx_t
        kpx_t *= px_t
    
    pv_annuity_t = 0
    kpx_ann = 1.0
    for k in range(0, term_length):
        vk_ann = (1 / (1 + annual_rate_calc)) ** k
        pv_annuity_t += kpx_ann * vk_ann
        
        if k < term_length - 1:
            age_ann = start_age_t + k
            if age_ann in mortality_dict:
                qx_ann = mortality_dict[age_ann]
            else:
                mu_ann = gompertz_mortality(age_ann)
                qx_ann = 1 - np.exp(-mu_ann)
            
            px_ann = 1 - qx_ann
            kpx_ann *= px_ann
    
    net_prem_t = apv_t / pv_annuity_t
    gross_prem_t = net_prem_t * (1 + margin_pct)
    
    return gross_prem_t

# Compare different term lengths
term_lengths = [10, 20, 30, 'WL']
comparison_data = []

print(f"\nAge {start_age}, ${benefit:,.0f}\n")
print(f"{'Product':<20} {'Monthly Premium':<20} {'20-Year Cost':<20} {'30-Year Cost':<20}")
print("-" * 80)

# Whole life
cost_20yr = gross_premium_monthly * 12 * 20
cost_30yr = gross_premium_monthly * 12 * 30

print(f"{'Whole Life':<20} ${gross_premium_monthly:<19,.2f} ${cost_20yr:<19,.0f} ${cost_30yr:<19,.0f}")

# Term insurance at different periods
for term in [10, 20, 30]:
    term_prem = calculate_term_premium(start_age, benefit, term, mortality_data, annual_rate)
    
    # Cost for 20 years
    cost_20_term = term_prem * 12 * min(20, term)
    if 20 > term:
        age_after_term = start_age + term
        term_renewal_prem = calculate_term_premium(age_after_term, benefit, min(20 - term, 30), 
                                                   mortality_data, annual_rate)
        cost_20_term += term_renewal_prem * 12 * (20 - term)
    
    # Cost for 30 years
    cost_30_term = term_prem * 12 * min(30, term)
    if 30 > term:
        age_after_term = start_age + term
        term_renewal_prem = calculate_term_premium(age_after_term, benefit, min(30 - term, 30), 
                                                   mortality_data, annual_rate)
        cost_30_term += term_renewal_prem * 12 * (30 - term)
    
    print(f"{f'{term}-Year Term':<20} ${term_prem:<19,.2f} ${cost_20_term:<19,.0f} ${cost_30_term:<19,.0f}")

print()
print(f"Break-even analysis:")
print(f"  If need permanent coverage: Whole life justified (no expiration risk)")
print(f"  If need temporary coverage: 20-30yr term + invest difference (cost over 20 yrs: {cost_20_term/cost_20yr:.1f}×)")
print()

# 5. DIVIDEND & PARTICIPATING WHOLE LIFE
print("=" * 80)
print("PARTICIPATING WHOLE LIFE (WITH DIVIDENDS)")
print("=" * 80)

# Assume dividends reduce net cost by 20-40% (conservative mutual company)
dividend_assumption = 0.30  # 30% of gross premium returned as dividend

gross_premium_wl_par = gross_premium_annual
dividend_annual = gross_premium_wl_par * dividend_assumption
net_cost_par = gross_premium_wl_par - dividend_annual

print(f"\nNon-Participating (Fixed): ${gross_premium_annual:,.2f}/year (${gross_premium_monthly:.2f}/mo)")
print(f"Participating (With Dividends):")
print(f"  Initial premium: ${gross_premium_annual:,.2f}/year")
print(f"  Est. annual dividend ({dividend_assumption*100:.0f}%): ${dividend_annual:,.2f}")
print(f"  Net annual cost: ${net_cost_par:,.2f}")
print(f"  Reduction: {(1 - net_cost_par/gross_premium_annual)*100:.1f}%")
print()

# Dividend usage options
print(f"Dividend options:")
print(f"  1) Take as cash: ${dividend_annual:,.2f}/year")
print(f"  2) Buy paid-up additions: Increase benefit")
print(f"  3) Reduce premium: Annual cost = ${net_cost_par:,.2f}")
print(f"  4) Accumulate at interest: Grow for emergency fund")
print()

# 6. SENSITIVITY TO AGE, RATE, AND BENEFIT
print("=" * 80)
print("SENSITIVITY ANALYSIS: WHOLE LIFE PREMIUMS BY AGE & BENEFIT")
print("=" * 80)

ages_sens = [25, 35, 45, 55]
benefits_sens = [100000, 250000, 500000, 1000000]

print(f"\nMonthly Gross Premium by Starting Age and Benefit Amount:\n")
print(f"{'Age':<10}", end='')
for ben in benefits_sens:
    print(f"${ben/1000:>6.0f}K", end='\t')
print()
print("-" * 70)

for age_s in ages_sens:
    print(f"{age_s:<10}", end='')
    
    for ben_s in benefits_sens:
        apv_s = calculate_whole_life_apv(age_s, mortality_data, annual_rate)
        pv_annuity_s = calculate_annuity_due(age_s, mortality_data, annual_rate)
        
        net_prem_s = apv_s * ben_s / pv_annuity_s
        gross_prem_s = net_prem_s * (1 + profit_pct) * 1.15  # Add expense load
        
        print(f"${gross_prem_s/12:>6.0f}", end='\t')
    
    print()

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reserve accumulation and CSV
ax = axes[0, 0]
ax.plot(years_vec, reserves, linewidth=2.5, marker='o', markersize=4, label='Net Reserve', color='darkblue')
ax.plot(years_vec, csv_values, linewidth=2.5, marker='s', markersize=4, label='Cash Surrender Value (80%)', color='green')
ax.fill_between(years_vec, reserves, csv_values, alpha=0.2, color='red', label='Surrender Charge')
ax.axhline(y=benefit, color='gray', linestyle='--', linewidth=1.5, label=f'Face Amount (${benefit/1000:.0f}K)')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Value ($)', fontsize=11)
ax.set_title('Policy Reserve & Cash Surrender Value Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 2: Whole Life vs Term cumulative cost
ax = axes[0, 1]
years_plot = np.array([5, 10, 15, 20, 25, 30])
wl_cost = years_plot * gross_premium_annual
term10_cost = np.minimum(years_plot, 10) * calculate_term_premium(start_age, benefit, 10, 
                                                                   mortality_data, annual_rate) * 12
term10_cost += np.maximum(0, (years_plot - 10)) * calculate_term_premium(start_age + 10, benefit, 20, 
                                                                          mortality_data, annual_rate) * 12

ax.plot(years_plot, wl_cost, linewidth=2.5, marker='o', markersize=7, label='Whole Life', color='darkred')
ax.plot(years_plot, term10_cost, linewidth=2.5, marker='s', markersize=7, label='Term (10-yr renewable)', color='steelblue')
ax.fill_between(years_plot, term10_cost, wl_cost, alpha=0.2, color='gray')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Cumulative Cost ($)', fontsize=11)
ax.set_title('Cumulative Premium Cost: Whole Life vs Term (Renewable)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 3: Premium by age (whole life)
ax = axes[1, 0]
ages_for_plot = list(range(20, 71, 5))
premiums_wl = []

for age_plot in ages_for_plot:
    apv_p = calculate_whole_life_apv(age_plot, mortality_data, annual_rate)
    pv_ann_p = calculate_annuity_due(age_plot, mortality_data, annual_rate)
    net_prem_p = apv_p * benefit / pv_ann_p
    gross_prem_p = net_prem_p * 1.35
    premiums_wl.append(gross_prem_p / 12)

ax.plot(ages_for_plot, premiums_wl, linewidth=2.5, marker='o', markersize=6, color='darkgreen')
ax.fill_between(ages_for_plot, 0, premiums_wl, alpha=0.2, color='green')

ax.set_xlabel('Age at Issue', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title(f'Whole Life Premium by Age (${benefit/1000:.0f}K Benefit)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Reserve as % of benefit
ax = axes[1, 1]
reserve_pct = np.array(reserves) / benefit * 100

ax.plot(years_vec, reserve_pct, linewidth=2.5, marker='o', markersize=4, color='purple')
ax.fill_between(years_vec, 0, reserve_pct, alpha=0.2, color='purple')
ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Face Amount (100%)')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Reserve as % of Benefit', fontsize=11)
ax.set_title('Reserve Accumulation as % of Face Amount', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('whole_life_insurance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

