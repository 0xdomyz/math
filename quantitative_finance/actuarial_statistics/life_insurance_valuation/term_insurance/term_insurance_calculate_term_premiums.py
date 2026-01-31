import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

# Auto-extracted from markdown file
# Source: term_insurance.md

# --- Code Block 1 ---
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