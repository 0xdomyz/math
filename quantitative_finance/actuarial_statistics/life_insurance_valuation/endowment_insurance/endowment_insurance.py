# Auto-extracted from markdown file
# Source: endowment_insurance.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. MORTALITY & ASSUMPTIONS
print("=" * 80)
print("ENDOWMENT INSURANCE: PREMIUM CALCULATION & RESERVE ANALYSIS")
print("=" * 80)

# Standard mortality table
mortality_data = {
    25: 0.00067, 30: 0.00084, 35: 0.00103, 40: 0.00131, 45: 0.00172,
    50: 0.00233, 55: 0.00325, 60: 0.00459, 65: 0.00653, 70: 0.00933,
    75: 0.01330, 80: 0.01934, 85: 0.02873, 90: 0.04213
}

def gompertz_mortality(age, A=0.0001, B=1.075):
    """Extrapolate mortality for ages beyond table"""
    return A * (B ** age)

annual_rate = 0.04
v = 1 / (1 + annual_rate)

# Expense assumptions
acquisition_pct = 0.15
maintenance = 25
profit_pct = 0.10  # Lower profit margin than term

print(f"\nAssumptions:")
print(f"  Interest Rate: {annual_rate*100:.1f}%")
print(f"  Acquisition: {acquisition_pct*100:.0f}% Year 1")
print(f"  Maintenance: ${maintenance}/year")
print(f"  Profit Margin: {profit_pct*100:.0f}%\n")

# 2. ENDOWMENT PREMIUM CALCULATION
print("=" * 80)
print("ENDOWMENT PREMIUM: AGE 35, 20-YEAR TERM, $200,000 BENEFIT")
print("=" * 80)

def calculate_endowment_apv(start_age, term_length, mortality_dict, annual_rate_calc):
    """Calculate APV of endowment insurance"""
    
    # Death benefit component (term insurance)
    apv_death = 0
    kpx = 1.0
    
    for k in range(1, term_length + 1):
        age_k = start_age + k - 1
        
        if age_k in mortality_dict:
            qx_k = mortality_dict[age_k]
        else:
            mu = gompertz_mortality(age_k)
            qx_k = 1 - np.exp(-mu)
        
        vk = v ** k
        apv_death += kpx * qx_k * vk
        
        px_k = 1 - qx_k
        kpx *= px_k
    
    # Survival benefit component (pure endowment)
    survival_prob = kpx  # Probability survive n years
    vn = v ** term_length
    apv_survival = vn * survival_prob
    
    # Total APV (benefit of $1)
    apv_total = apv_death + apv_survival
    
    return apv_death, apv_survival, apv_total, survival_prob

def calculate_annuity_due_limited(start_age, term_length, mortality_dict, annual_rate_calc):
    """Calculate PV of annuity due for n years"""
    
    pv_annuity = 0
    kpx = 1.0
    
    for k in range(0, term_length):
        vk = v ** k
        pv_annuity += kpx * vk
        
        if k < term_length - 1:
            age_k = start_age + k
            
            if age_k in mortality_dict:
                qx_k = mortality_dict[age_k]
            else:
                mu = gompertz_mortality(age_k)
                qx_k = 1 - np.exp(-mu)
            
            px_k = 1 - qx_k
            kpx *= px_k
    
    return pv_annuity

# Calculate for age 35, 20-year endowment
start_age = 35
term_length = 20
benefit = 200000

apv_death, apv_survival, apv_total, prob_survive = calculate_endowment_apv(
    start_age, term_length, mortality_data, annual_rate
)

pv_annuity_limited = calculate_annuity_due_limited(
    start_age, term_length, mortality_data, annual_rate
)

# Net premium
net_premium_annual = apv_total * benefit / pv_annuity_limited
net_premium_monthly = net_premium_annual / 12

# Gross premium
expense_pv = acquisition_pct * net_premium_annual + maintenance
profit_pv = profit_pct * apv_total * benefit

gross_premium_annual = (apv_total * benefit + expense_pv + profit_pv) / pv_annuity_limited
gross_premium_monthly = gross_premium_annual / 12

print(f"\nAge {start_age}, {term_length}-Year Endowment, ${benefit:,.0f}\n")
print(f"{'Metric':<35} {'Amount':<20}")
print("-" * 55)
print(f"{'APV of Death Benefit (per $1)':<35} {apv_death:<20.6f}")
print(f"{'APV of Survival Benefit (per $1)':<35} {apv_survival:<20.6f}")
print(f"{'Total APV (per $1)':<35} {apv_total:<20.6f}")
print(f"{'Probability Survive {term_length} years':<35} {prob_survive:<20.4f}")
print()
print(f"{'APV of Death Benefits':<35} ${apv_death * benefit:>18,.2f}")
print(f"{'APV of Maturity Benefit':<35} ${apv_survival * benefit:>18,.2f}")
print(f"{'Total APV':<35} ${apv_total * benefit:>18,.2f}")
print()
print(f"{'Net Premium (Annual)':<35} ${net_premium_annual:>18,.2f}")
print(f"{'Net Premium (Monthly)':<35} ${net_premium_monthly:>18,.2f}")
print()
print(f"{'Gross Premium (Annual)':<35} ${gross_premium_annual:>18,.2f}")
print(f"{'Gross Premium (Monthly)':<35} ${gross_premium_monthly:>18,.2f}")
print()

# 3. RESERVE ACCUMULATION
print("=" * 80)
print("POLICY RESERVE & CASH SURRENDER VALUE ACCUMULATION")
print("=" * 80)

reserves = []
csv_values = []
surrender_charges = []
years_res = []

print(f"\n{'Year':<8} {'Age':<8} {'Reserve':<15} {'CSV (90%)':<15} {'Surr Charge':<15} {'Death Benefit':<15}")
print("-" * 90)

reserve_previous = 0

for year in range(1, term_length + 1):
    age_year = start_age + year - 1
    
    # Prospective reserve: V = Aₓ₊ₖ:ₙ₋ₖ̄| + v^(n-k) · ₙ₋ₖpₓ₊ₖ
    remaining_term = term_length - year
    
    if remaining_term > 0:
        apv_death_rem, apv_surv_rem, apv_tot_rem, prob_surv_rem = calculate_endowment_apv(
            age_year, remaining_term, mortality_data, annual_rate
        )
        
        pv_ann_rem = calculate_annuity_due_limited(
            age_year, remaining_term, mortality_data, annual_rate
        )
        
        # Future premiums PV
        pv_future_premiums = net_premium_annual * pv_ann_rem
        
        # Reserve
        reserve_prospective = (apv_tot_rem * benefit - pv_future_premiums)
    else:
        # At maturity
        reserve_prospective = benefit
    
    # Cash surrender value (typically 90-95% of reserve in later years, less in early)
    surrender_charge_rate = max(0.15 - year * 0.01, 0) if year <= 10 else 0.02  # 15% year 1, declining
    csv_reserve = reserve_prospective * (1 - surrender_charge_rate)
    
    reserves.append(reserve_prospective)
    csv_values.append(csv_reserve)
    surrender_charges.append(reserve_prospective - csv_reserve)
    years_res.append(year)
    
    if year <= 5 or year % 5 == 0 or year == term_length:
        surr_charge_display = reserve_prospective - csv_reserve
        print(f"{year:<8} {age_year:<8} ${reserve_prospective:<14,.0f} ${csv_reserve:<14,.0f} ${surr_charge_display:<14,.0f} ${benefit:<14,.0f}")

print()

# 4. COMPARISON: ENDOWMENT VS TERM + INVESTMENT
print("=" * 80)
print("ENDOWMENT VS TERM INSURANCE + EXTERNAL INVESTMENT")
print("=" * 80)

# Calculate term insurance premium for comparison
def calculate_term_premium(start_age_t, benefit_t, term_length_t, mortality_dict):
    """Simplified term premium calculation"""
    
    apv_t = 0
    kpx_t = 1.0
    
    for k in range(1, term_length_t + 1):
        age_t = start_age_t + k - 1
        
        if age_t in mortality_dict:
            qx_t = mortality_dict[age_t]
        else:
            mu_t = gompertz_mortality(age_t)
            qx_t = 1 - np.exp(-mu_t)
        
        vk_t = v ** k
        apv_t += kpx_t * qx_t * vk_t * benefit_t
        
        px_t = 1 - qx_t
        kpx_t *= px_t
    
    pv_annuity_t = 0
    kpx_ann = 1.0
    
    for k in range(0, term_length_t):
        vk_ann = v ** k
        pv_annuity_t += kpx_ann * vk_ann
        
        if k < term_length_t - 1:
            age_ann = start_age_t + k
            
            if age_ann in mortality_dict:
                qx_ann = mortality_dict[age_ann]
            else:
                mu_ann = gompertz_mortality(age_ann)
                qx_ann = 1 - np.exp(-mu_ann)
            
            px_ann = 1 - qx_ann
            kpx_ann *= px_ann
    
    net_prem_t = apv_t / pv_annuity_t
    gross_prem_t = net_prem_t * 1.25  # Add 25% for expenses/profit
    
    return gross_prem_t

term_premium = calculate_term_premium(start_age, benefit, term_length, mortality_data)
premium_difference = gross_premium_annual - term_premium

print(f"\nAge {start_age}, {term_length}-Year Term, ${benefit:,.0f}\n")
print(f"{'Product':<35} {'Annual Premium':<20} {'Difference':<20}")
print("-" * 75)
print(f"{'Endowment Insurance':<35} ${gross_premium_annual:<19,.2f}")
print(f"{'Term Insurance (same term)':<35} ${term_premium:<19,.2f}")
print(f"{'Annual excess cost':<35} ${premium_difference:<19,.2f}")
print()

# Calculate what term + investment would accumulate
print(f"If invest premium difference at {annual_rate*100:.1f}% for {term_length} years:\n")

savings_invested = 0
print(f"{'Year':<8} {'Endowment CSV':<20} {'Term + Invest (FV)':<20} {'Difference':<20}")
print("-" * 68)

for year in range(1, term_length + 1):
    # Future value of premium difference invested at interest
    # From now (year 0) to year n
    
    fv_invested = 0
    for y in range(1, year + 1):
        # Invest difference at end of each year
        fv_invested += premium_difference * (1 + annual_rate) ** (year - y)
    
    endowment_csv = csv_values[year - 1] if year <= len(csv_values) else benefit
    term_fv = fv_invested
    
    if year <= 5 or year % 5 == 0 or year == term_length:
        print(f"{year:<8} ${endowment_csv:<19,.0f} ${term_fv:<19,.0f} ${endowment_csv - term_fv:<19,.0f}")

print()

# 5. WITH-PROFIT ENDOWMENT
print("=" * 80)
print("ENDOWMENT WITH PROFIT (DIVIDEND SCENARIO)")
print("=" * 80)

# Assume guaranteed + reversionary bonus structure
guaranteed_maturity = benefit
assumed_bonus_rate = 0.03  # 3% annual bonus on premium
terminal_bonus_rate = 0.10  # 10% terminal bonus at maturity

expected_maturity_with_profit = guaranteed_maturity * (1 + terminal_bonus_rate)

# Average annual bonus applied
expected_total_bonus = premium_difference * term_length * assumed_bonus_rate * (1 + terminal_bonus_rate)

print(f"\nWith-Profit Endowment Structure:")
print(f"  Guaranteed maturity benefit: ${guaranteed_maturity:,.0f}")
print(f"  Assumed reversionary bonus: {assumed_bonus_rate*100:.1f}% annual (on premium)")
print(f"  Terminal bonus (at maturity): {terminal_bonus_rate*100:.0f}%")
print()
print(f"Expected maturity value: ${expected_maturity_with_profit:,.0f}")
print(f"  Growth above guaranteed: {(expected_maturity_with_profit/guaranteed_maturity - 1)*100:.1f}%")
print()
print(f"Note: With-profit endowments become less attractive with low interest rates")
print(f"      (bonus rates typically tied to company's investment returns)")
print()

# 6. SENSITIVITY: DIFFERENT AGES & TERMS
print("=" * 80)
print("ENDOWMENT PREMIUMS BY AGE & TERM LENGTH")
print("=" * 80)

ages_sens = [30, 40, 50]
terms_sens = [10, 15, 20]

print(f"\nMonthly Gross Premium (${benefit:,.0f} Benefit)\n")
print(f"{'Age':<10}", end='')
for term_s in terms_sens:
    print(f"{term_s:>2}-Yr", end='\t')
print()
print("-" * 50)

for age_s in ages_sens:
    print(f"{age_s:<10}", end='')
    
    for term_s in terms_sens:
        apv_d_s, apv_surv_s, apv_t_s, prob_s = calculate_endowment_apv(
            age_s, term_s, mortality_data, annual_rate
        )
        pv_ann_s = calculate_annuity_due_limited(age_s, term_s, mortality_data, annual_rate)
        
        net_prem_s = apv_t_s * benefit / pv_ann_s
        gross_prem_s = net_prem_s * 1.25
        
        print(f"${gross_prem_s/12:>6,.0f}", end='\t')
    
    print()

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reserve vs CSV
ax = axes[0, 0]
ax.plot(years_res, reserves, linewidth=2.5, marker='o', markersize=5, label='Net Reserve', color='darkblue')
ax.plot(years_res, csv_values, linewidth=2.5, marker='s', markersize=5, label='Cash Surrender Value', color='green')
ax.fill_between(years_res, csv_values, reserves, alpha=0.2, color='red', label='Surrender Charge')
ax.axhline(y=benefit, color='gray', linestyle='--', linewidth=1.5, label=f'Maturity Benefit (${benefit/1000:.0f}K)')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Value ($)', fontsize=11)
ax.set_title('Endowment Reserve & Cash Surrender Value', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Plot 2: APV components (death vs survival)
ax = axes[0, 1]
components = ['Death Benefit\nComponent', 'Survival Benefit\nComponent', 'Total APV']
apv_values = [apv_death, apv_survival, apv_total]
colors_apv = ['steelblue', 'green', 'orange']

bars = ax.bar(components, apv_values, color=colors_apv, alpha=0.6, edgecolor='black', linewidth=1.5)
ax.set_ylabel('APV per $1 of Benefit', fontsize=11)
ax.set_title('Endowment APV Components', fontsize=12, fontweight='bold')

for bar, val in zip(bars, apv_values):
    height = bar.get_height()
    pct = val / apv_total * 100 if apv_total > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Endowment vs Term + Investment
ax = axes[1, 0]
term_plus_invest = []

for year in years_res:
    fv_inv = 0
    for y in range(1, year + 1):
        fv_inv += premium_difference * (1 + annual_rate) ** (year - y)
    term_plus_invest.append(fv_inv)

ax.plot(years_res, reserves, linewidth=2.5, marker='o', markersize=5, label='Endowment CSV', color='darkblue')
ax.plot(years_res, term_plus_invest, linewidth=2.5, marker='s', markersize=5, label='Term + Invest Difference', color='red')
ax.fill_between(years_res, reserves, term_plus_invest, alpha=0.2, color='gray')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Accumulated Value ($)', fontsize=11)
ax.set_title('Endowment vs Term + External Investment Strategy', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Premium by age (all terms)
ax = axes[1, 1]
ages_for_plot = list(range(25, 61, 5))

for term_plot in [10, 15, 20]:
    premiums_by_age = []
    
    for age_plot in ages_for_plot:
        apv_d_p, apv_s_p, apv_t_p, prob_p = calculate_endowment_apv(
            age_plot, term_plot, mortality_data, annual_rate
        )
        pv_ann_p = calculate_annuity_due_limited(age_plot, term_plot, mortality_data, annual_rate)
        
        net_prem_p = apv_t_p * benefit / pv_ann_p
        gross_prem_p = net_prem_p * 1.25
        
        premiums_by_age.append(gross_prem_p / 12)
    
    ax.plot(ages_for_plot, premiums_by_age, linewidth=2.5, marker='o', markersize=6, label=f'{term_plot}-Year')

ax.set_xlabel('Age at Issue', fontsize=11)
ax.set_ylabel('Monthly Premium ($)', fontsize=11)
ax.set_title(f'Endowment Premium by Age (${benefit/1000:.0f}K Benefit)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('endowment_insurance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

