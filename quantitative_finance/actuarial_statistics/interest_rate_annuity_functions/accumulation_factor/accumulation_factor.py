# Auto-extracted from markdown file
# Source: accumulation_factor.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. BASIC ACCUMULATION CALCULATIONS
print("=" * 70)
print("ACCUMULATION FACTOR FUNDAMENTALS")
print("=" * 70)

principal = 1000  # $1,000 initial investment
rates = np.array([0.02, 0.05, 0.10])
years = np.array([1, 5, 10, 20, 30, 50])

print(f"\nInitial investment: ${principal:,.0f}\n")
print(f"{'Years':<10}", end='')
for rate in rates:
    print(f"i={rate*100:>4.1f}%", end='\t')
print()
print("-" * 60)

for n in years:
    print(f"{n:<10}", end='')
    for rate in rates:
        accumulation = (1 + rate) ** n
        future_value = principal * accumulation
        print(f"${future_value:>8,.0f}", end='\t')
    print()

print("\nWith compound interest formulas:")
print(f"FV = PV × (1+i)^n")
print()

# 2. VISUALIZATION OF COMPOUND GROWTH
print("=" * 70)
print("COMPOUND GROWTH OVER TIME")
print("=" * 70)

time_horizon = 50
times = np.arange(0, time_horizon + 1)
initial = 1000

print(f"\nCompound growth table (selected years):")
print(f"{'Year':<8} {'2%':<15} {'5%':<15} {'10%':<15} {'Real Return %':<15}")
print("-" * 58)

# Assume 2% inflation
inflation = 0.02

for year in [0, 10, 20, 30, 40, 50]:
    for rate in [0.02, 0.05, 0.10]:
        fv = initial * (1 + rate) ** year
        real_return = ((1 + rate) / (1 + inflation)) ** year - 1
        
        if rate == 0.02:
            print(f"{year:<8} ${fv:<14,.0f} ", end='')
        elif rate == 0.05:
            print(f"${fv:<14,.0f} ", end='')
        elif rate == 0.10:
            print(f"${fv:<14,.0f} {real_return*100:>13.1f}%")

print()

# 3. ANNUITY ACCUMULATION
print("=" * 70)
print("ANNUITY ACCUMULATION (sₙ̄|)")
print("=" * 70)

annual_payment = 1000
accumulation_rate = 0.05

print(f"\nAnnual payment: ${annual_payment:,.0f}")
print(f"Interest rate: {accumulation_rate*100:.1f}%")
print(f"\nAnnuity accumulation factor: sₙ̄| = [(1+i)^n - 1]/i\n")

print(f"{'n (years)':<12} {'Accum Factor':<15} {'Future Value':<15} {'Total Paid':<15} {'Interest Earned':<15}")
print("-" * 72)

n_annuity = np.array([1, 5, 10, 20, 30])

for n in n_annuity:
    s_n = ((1 + accumulation_rate) ** n - 1) / accumulation_rate
    fv_annuity = annual_payment * s_n
    total_paid = annual_payment * n
    interest = fv_annuity - total_paid
    
    print(f"{n:<12} {s_n:<15.6f} ${fv_annuity:<14,.0f} ${total_paid:<14,.0f} ${interest:<14,.0f}")

print()

# 4. RULE OF 72
print("=" * 70)
print("RULE OF 72: DOUBLING TIME")
print("=" * 70)

print("\nRule: Years to double ≈ 72 / rate(%)")
print("Allows quick mental estimation of compound growth\n")

rates_test = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])

print(f"{'Rate (%)':<12} {'Rule of 72':<18} {'Exact (ln(2)/ln(1+r))':<25} {'Error %':<12}")
print("-" * 67)

for rate in rates_test:
    estimate = 72 / (rate * 100)
    exact = np.log(2) / np.log(1 + rate)
    error = abs((estimate - exact) / exact) * 100
    
    print(f"{rate*100:<11.2f} {estimate:<18.2f} {exact:<25.3f} {error:<11.2f}%")

print()

# 5. MORTGAGE/LOAN ACCUMULATION (backward perspective)
print("=" * 70)
print("DEBT ACCUMULATION: How Principal Grows")
print("=" * 70)

loan_principal = 100000
loan_rate = 0.06
loan_years = 30

print(f"\nLoan details:")
print(f"  Principal: ${loan_principal:,.0f}")
print(f"  Rate: {loan_rate*100:.1f}%")
print(f"  Maturity: {loan_years} years (no payments during this analysis)\n")

# Without payments, accumulated debt
debt_accumulated = loan_principal * (1 + loan_rate) ** loan_years

print(f"Accumulated debt (if no payments): ${debt_accumulated:,.0f}")
print(f"Total interest: ${debt_accumulated - loan_principal:,.0f}")
print(f"Interest as % of principal: {((debt_accumulated - loan_principal)/loan_principal)*100:.1f}%\n")

# Compare with monthly payments (debt decreases instead)
monthly_rate = loan_rate / 12
n_payments = loan_years * 12

# Monthly payment formula: P = L[r(1+r)^n]/[(1+r)^n - 1]
numerator = monthly_rate * (1 + monthly_rate) ** n_payments
denominator = (1 + monthly_rate) ** n_payments - 1
monthly_payment = loan_principal * numerator / denominator

total_paid_with_payments = monthly_payment * n_payments
total_interest_with_payments = total_paid_with_payments - loan_principal

print(f"With monthly payments of ${monthly_payment:,.2f}:")
print(f"  Total paid: ${total_paid_with_payments:,.0f}")
print(f"  Total interest: ${total_interest_with_payments:,.0f}")
print(f"  Interest as % of principal: {(total_interest_with_payments/loan_principal)*100:.1f}%")

print()

# 6. INVESTMENT COMPARISON
print("=" * 70)
print("COMPARING INVESTMENT OPTIONS")
print("=" * 70)

print(f"\nInvest $10,000 for 30 years:\n")

options = [
    ("Savings account (2%)", 0.02),
    ("Bonds (4%)", 0.04),
    ("Stock market (8%)", 0.08),
    ("Aggressive growth (10%)", 0.10),
    ("Real estate (6% with inflation 2%)", 0.06)
]

results = []

for name, rate in options:
    fv = 10000 * (1 + rate) ** 30
    growth = fv - 10000
    multiple = fv / 10000
    
    results.append({
        'Investment': name,
        'Annual Rate': f"{rate*100:.1f}%",
        'Future Value': f"${fv:,.0f}",
        'Growth': f"${growth:,.0f}",
        'Multiple': f"{multiple:.1f}x"
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print()

# 7. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Growth curves at different rates
ax = axes[0, 0]
times_detail = np.linspace(0, 50, 100)

for rate in [0.02, 0.05, 0.08, 0.12]:
    fv_curve = 1000 * (1 + rate) ** times_detail
    ax.plot(times_detail, fv_curve, linewidth=2.5, label=f'i = {rate*100:.1f}%')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Future Value ($)', fontsize=11)
ax.set_title('$1,000 Accumulation at Different Rates', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Rule of 72 accuracy
ax = axes[0, 1]
rates_range = np.linspace(0.001, 0.30, 100)
estimates = 72 / (rates_range * 100)
exact_values = np.log(2) / np.log(1 + rates_range)
errors = abs(estimates - exact_values)

ax.plot(rates_range * 100, errors, linewidth=2.5, color='darkblue')
ax.fill_between(rates_range * 100, 0, errors, alpha=0.2)
ax.set_xlabel('Annual Rate (%)', fontsize=11)
ax.set_ylabel('Error in Doubling Time (years)', fontsize=11)
ax.set_title('Rule of 72: Estimation Error', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Annuity accumulation growth
ax = axes[1, 0]
n_range = np.arange(1, 41)
rates_annuity = [0.03, 0.05, 0.08]

for rate in rates_annuity:
    s_factors = ((1 + rate) ** n_range - 1) / rate
    fv_annuity = 1000 * s_factors
    ax.plot(n_range, fv_annuity, linewidth=2.5, marker='o', markersize=3, 
           alpha=0.7, label=f'i = {rate*100:.1f}%')

ax.set_xlabel('Number of Periods (n)', fontsize=11)
ax.set_ylabel('Future Value of Annuity ($)', fontsize=11)
ax.set_title('Annual $1,000 Payments Accumulated', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Log-scale long-term growth
ax = axes[1, 1]
times_log = np.linspace(0, 100, 100)

for rate in [0.02, 0.05, 0.10]:
    fv_log = 1000 * (1 + rate) ** times_log
    ax.semilogy(times_log, fv_log, linewidth=2.5, label=f'i = {rate*100:.1f}%')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Future Value ($, log scale)', fontsize=11)
ax.set_title('Long-Term Growth: 100-Year Horizon', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('accumulation_factor_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

