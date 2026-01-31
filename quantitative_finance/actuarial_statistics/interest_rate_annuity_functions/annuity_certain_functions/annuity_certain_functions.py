# Auto-extracted from markdown file
# Source: annuity_certain_functions.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. ANNUITY FACTOR CALCULATIONS
print("=" * 70)
print("ANNUITY-CERTAIN FACTORS")
print("=" * 70)

interest_rate = 0.05
n_terms = np.array([1, 5, 10, 20, 30, 50])

print(f"\nInterest rate: i = {interest_rate*100:.1f}%\n")
print(f"{'n (years)':<12} {'aₙ̄| (Imm)':<15} {'äₙ̄| (Due)':<15} {'sₙ̄| (Accum)':<15} {'s̈ₙ̄| (A-Due)':<15}")
print("-" * 72)

annuity_data = []

for n in n_terms:
    # Annuity-immediate (PV): aₙ̄| = [1 - v^n]/i
    v_n = 1 / (1 + interest_rate) ** n
    a_n_imm = (1 - v_n) / interest_rate
    
    # Annuity-due (PV): äₙ̄| = aₙ̄| × (1+i)
    a_n_due = a_n_imm * (1 + interest_rate)
    
    # Accumulation-immediate (FV): sₙ̄| = [(1+i)^n - 1]/i
    s_n_imm = ((1 + interest_rate) ** n - 1) / interest_rate
    
    # Accumulation-due (FV): s̈ₙ̄| = sₙ̄| × (1+i)
    s_n_due = s_n_imm * (1 + interest_rate)
    
    print(f"{n:<12} {a_n_imm:<15.6f} {a_n_due:<15.6f} {s_n_imm:<15.6f} {s_n_due:<15.6f}")
    
    annuity_data.append({
        'n': n,
        'aₙ̄|': a_n_imm,
        'äₙ̄|': a_n_due,
        'sₙ̄|': s_n_imm,
        's̈ₙ̄|': s_n_due
    })

print()

# 2. RETIREMENT INCOME PRESENT VALUE
print("=" * 70)
print("RETIREMENT INCOME: PRESENT VALUE CALCULATION")
print("=" * 70)

# Scenario: Retirement planning
annual_income_needed = 50000  # $50,000/year
retirement_years = 30
discount_rate = 0.04  # Conservative 4%

# Payment at end of each year (annuity-immediate)
v_ret = 1 / (1 + discount_rate) ** retirement_years
a_ret_imm = (1 - v_ret) / discount_rate
pv_retirement_imm = annual_income_needed * a_ret_imm

# Payment at start of each year (annuity-due)
a_ret_due = a_ret_imm * (1 + discount_rate)
pv_retirement_due = annual_income_needed * a_ret_due

print(f"\nRetirement Planning Scenario:")
print(f"  Desired annual income: ${annual_income_needed:,.0f}/year")
print(f"  Retirement period: {retirement_years} years")
print(f"  Discount rate: {discount_rate*100:.1f}%")
print(f"  Annuity factor (immediate): aₙ̄| = {a_ret_imm:.6f}")
print(f"  Annuity factor (due): äₙ̄| = {a_ret_due:.6f}\n")

print(f"Lump sum needed:")
print(f"  Payments at end of year: ${pv_retirement_imm:>12,.2f}  (aₙ̄| method)")
print(f"  Payments at start of year: ${pv_retirement_due:>12,.2f}  (äₙ̄| method)")
print(f"  Difference: ${abs(pv_retirement_due - pv_retirement_imm):>12,.2f}")
print()

# 3. PENSION LIABILITY VALUATION
print("=" * 70)
print("PENSION LIABILITY: PRESENT VALUE OF BENEFIT")
print("=" * 70)

# Defined benefit pension
benefit_per_year = 30000  # $30,000 annual benefit
life_expectancy = 25  # Expected 25 years of payments
life_exp_ages = np.array([20, 25, 30])
discount_rates_pension = np.array([0.02, 0.03, 0.04, 0.05])

print(f"\nPension Benefit: ${benefit_per_year:,.0f}/year")
print(f"\nPV by life expectancy and discount rate:\n")

# Create matrix
print(f"{'Life Exp (yrs)':<18}", end='')
for rate in discount_rates_pension:
    print(f"i={rate*100:.1f}%", end='\t')
print()
print("-" * 70)

for life_exp in life_exp_ages:
    print(f"{life_exp:<18}", end='')
    
    for rate in discount_rates_pension:
        v_val = 1 / (1 + rate) ** life_exp
        a_val = (1 - v_val) / rate
        liability = benefit_per_year * a_val
        print(f"${liability:>10,.0f}", end='\t')
    
    print()

print()

# 4. DEFERRED ANNUITY
print("=" * 70)
print("DEFERRED ANNUITY: PAYMENTS STARTING IN THE FUTURE")
print("=" * 70)

# ₘ|aₙ̄| = v^m · aₙ̄|
years_to_defer = 5  # Start payments in 5 years
payment_years = 20  # 20 years of payments
deferral_rate = 0.04

# Deferred annuity factor: m|aₙ̄| = v^m · aₙ̄|
v_defer = 1 / (1 + deferral_rate) ** years_to_defer
a_n = (1 - (1 / (1 + deferral_rate) ** payment_years)) / deferral_rate
deferred_factor = v_defer * a_n

# Also immediate factor for comparison
a_n_immediate = (1 - (1 / (1 + deferral_rate) ** (years_to_defer + payment_years))) / deferral_rate

monthly_payout = 2000

pv_deferred = monthly_payout * deferred_factor
pv_immediate = monthly_payout * a_n_immediate

print(f"\nDeferred Annuity:")
print(f"  Monthly payout: ${monthly_payout:,.0f}")
print(f"  Deferral period: {years_to_defer} years")
print(f"  Payment period: {payment_years} years")
print(f"  Rate: {deferral_rate*100:.1f}%\n")

print(f"Calculation:")
print(f"  ₅|aₙ̄| = v^5 × a₂₀̄|")
print(f"         = {v_defer:.6f} × {a_n:.6f}")
print(f"         = {deferred_factor:.6f}\n")

print(f"Present Value:")
print(f"  Deferred (start in 5 years):  ${pv_deferred:>12,.2f}")
print(f"  Compare: immediate 25 years:  ${pv_immediate:>12,.2f}")
print(f"  Difference: ${pv_deferred - pv_immediate:>12,.2f}")
print()

# 5. ANNUITY VALUATIONS FOR DIFFERENT PAYMENT FREQUENCIES
print("=" * 70)
print("PAYMENT FREQUENCY: SEMI-ANNUAL, QUARTERLY, MONTHLY")
print("=" * 70)

# Assume 4% annual rate
annual_rate = 0.04
annual_payout = 12000
n_years = 20

print(f"\nAnnual payout (equivalent): ${annual_payout:,.0f}")
print(f"Term: {n_years} years, Rate: {annual_rate*100:.1f}%\n")

# Annual payments
v_annual = 1 / (1 + annual_rate) ** n_years
a_annual = (1 - v_annual) / annual_rate
pv_annual = annual_payout * a_annual

# Semi-annual (m=2): i^(2) = 2[(1+i)^(1/2) - 1]
i_semi = 2 * ((1 + annual_rate) ** 0.5 - 1)
payment_semi = annual_payout / 2
n_semi = n_years * 2
v_semi = 1 / (1 + i_semi) ** n_semi
a_semi = (1 - v_semi) / i_semi
pv_semi = payment_semi * a_semi

# Quarterly (m=4)
i_quarterly = 4 * ((1 + annual_rate) ** 0.25 - 1)
payment_quarterly = annual_payout / 4
n_quarterly = n_years * 4
v_quarterly = 1 / (1 + i_quarterly) ** n_quarterly
a_quarterly = (1 - v_quarterly) / i_quarterly
pv_quarterly = payment_quarterly * a_quarterly

# Monthly (m=12)
i_monthly = 12 * ((1 + annual_rate) ** (1/12) - 1)
payment_monthly = annual_payout / 12
n_monthly = n_years * 12
v_monthly = 1 / (1 + i_monthly) ** n_monthly
a_monthly = (1 - v_monthly) / i_monthly
pv_monthly = payment_monthly * a_monthly

freq_data = pd.DataFrame({
    'Frequency': ['Annual', 'Semi-Annual', 'Quarterly', 'Monthly'],
    'Per-Period Payment': [f"${annual_payout:,.0f}", f"${payment_semi:,.2f}", 
                          f"${payment_quarterly:,.2f}", f"${payment_monthly:,.2f}"],
    'Per-Period Rate': [f"{annual_rate*100:.4f}%", f"{i_semi*100:.4f}%",
                       f"{i_quarterly*100:.4f}%", f"{i_monthly*100:.4f}%"],
    'Present Value': [f"${pv_annual:,.2f}", f"${pv_semi:,.2f}",
                     f"${pv_quarterly:,.2f}", f"${pv_monthly:,.2f}"]
})

print(freq_data.to_string(index=False))
print()

# 6. INCREASING ANNUITY (Ia)ₙ̄|
print("=" * 70)
print("INCREASING ANNUITY: Payments Grow Over Time")
print("=" * 70)

# (Ia)ₙ̄| = ∑ₖ₌₁ⁿ k·v^k = [aₙ̄| - n·v^n]/i
i_incr = 0.05
n_incr = 10
initial_payment = 1000

# Calculate using formula
v_incr = 1 / (1 + i_incr) ** n_incr
a_incr = (1 - v_incr) / i_incr
ia_factor = (a_incr - n_incr * v_incr) / i_incr

print(f"\nIncreasing Annuity: Payments grow by 1 unit each year")
print(f"  Initial payment: ${initial_payment:,.0f} (year 1)")
print(f"  Term: {n_incr} years")
print(f"  Rate: {i_incr*100:.1f}%\n")

# Year-by-year breakdown
print(f"{'Year':<8} {'Payment':<15} {'PV Factor':<15} {'Present Value':<15}")
print("-" * 53)

total_pv_incr = 0
for year in range(1, n_incr + 1):
    payment = initial_payment * year
    pv_factor = (1 / (1 + i_incr) ** year)
    pv = payment * pv_factor
    total_pv_incr += pv
    print(f"{year:<8} ${payment:<14,.0f} {pv_factor:<15.6f} ${pv:<14,.2f}")

print(f"\nTotal Present Value: ${total_pv_incr:>12,.2f}")
print(f"Using formula: ${initial_payment * ia_factor:>12,.2f}")
print()

# 7. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Annuity factors vs term
ax = axes[0, 0]
n_range = np.arange(1, 51)
rates_plot = [0.02, 0.05, 0.08]

for rate in rates_plot:
    v_range = 1 / (1 + rate) ** n_range
    a_range = (1 - v_range) / rate
    ax.plot(n_range, a_range, linewidth=2.5, label=f'i = {rate*100:.1f}%', marker='o', 
           markersize=3, alpha=0.7)

ax.set_xlabel('Term (years)', fontsize=11)
ax.set_ylabel('Annuity Factor aₙ̄|', fontsize=11)
ax.set_title('Annuity Present Value Factors vs Term', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Immediate vs Due comparison
ax = axes[0, 1]
n_range_small = np.arange(1, 31)
rate_comp = 0.05

a_imm = (1 - (1 / (1 + rate_comp) ** n_range_small)) / rate_comp
a_due = a_imm * (1 + rate_comp)
difference = a_due - a_imm

ax.plot(n_range_small, a_imm, linewidth=2.5, label='Annuity-Immediate (aₙ̄|)', marker='o', markersize=4)
ax.plot(n_range_small, a_due, linewidth=2.5, label='Annuity-Due (äₙ̄|)', marker='s', markersize=4)
ax.fill_between(n_range_small, a_imm, a_due, alpha=0.2, color='gray')
ax.set_xlabel('Term (years)', fontsize=11)
ax.set_ylabel('Annuity Factor', fontsize=11)
ax.set_title('Immediate vs Due: Effect of Payment Timing', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Deferred annuity sensitivity
ax = axes[1, 0]
m_defer_range = np.arange(0, 21)  # Years to deferral
n_payment = 20
i_defer = 0.05

pv_deferred_range = []
for m in m_defer_range:
    v_m = 1 / (1 + i_defer) ** m
    a_n = (1 - (1 / (1 + i_defer) ** n_payment)) / i_defer
    pv_def = v_m * a_n
    pv_deferred_range.append(pv_def)

ax.plot(m_defer_range, pv_deferred_range, linewidth=2.5, color='darkgreen', marker='o', markersize=5)
ax.fill_between(m_defer_range, 0, pv_deferred_range, alpha=0.2, color='green')
ax.set_xlabel('Years of Deferral (m)', fontsize=11)
ax.set_ylabel('Deferred Annuity Factor (ₘ|aₙ̄|)', fontsize=11)
ax.set_title('Deferred Annuity: Impact of Deferral Period', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Increasing annuity pattern
ax = axes[1, 1]
years_incr = np.arange(1, n_incr + 1)
payments_incr = initial_payment * years_incr
pv_payments = payments_incr / (1 + i_incr) ** years_incr

ax.bar(years_incr, payments_incr, alpha=0.5, label='Nominal payments', color='steelblue', edgecolor='black')
ax.plot(years_incr, pv_payments, linewidth=2.5, color='red', marker='o', markersize=6, label='Present value of payments')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Increasing Annuity: Payments & Present Values', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('annuity_certain_functions_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

