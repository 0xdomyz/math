# Auto-extracted from markdown file
# Source: nominal_rate.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. CONVERSION: NOMINAL TO EFFECTIVE
print("=" * 70)
print("NOMINAL TO EFFECTIVE RATE CONVERSIONS")
print("=" * 70)

i_nom_scenarios = [
    ("Bank savings", 0.02, 12),      # 2% annual, monthly compounding
    ("Bond coupon", 0.05, 2),        # 5% annual, semi-annual (typical)
    ("Mortgage", 0.06, 12),          # 6% annual, monthly
    ("Quarterly accrual", 0.04, 4),  # 4% annual, quarterly
    ("Daily accrual", 0.03, 365)     # 3% annual, daily
]

conversion_results = []

for desc, i_nom, m in i_nom_scenarios:
    i_effective = (1 + i_nom/m) ** m - 1
    delta = np.log(1 + i_effective)
    
    # Per-period rate
    per_period = i_nom / m
    
    conversion_results.append({
        'Instrument': desc,
        'Nominal i^(m)': f"{i_nom*100:.2f}%",
        'Frequency (m)': m,
        'Per-Period': f"{per_period*100:.4f}%",
        'Effective i': f"{i_effective*100:.4f}%",
        'Force δ': f"{delta*100:.4f}%",
        'Difference (i-i^(m))': f"{(i_effective - i_nom)*100:.2f} bps"
    })

conv_df = pd.DataFrame(conversion_results)
print("\n" + conv_df.to_string(index=False))
print()

# 2. REVERSE: EFFECTIVE TO NOMINAL
print("=" * 70)
print("EFFECTIVE TO NOMINAL RATE CONVERSIONS")
print("=" * 70)

i_effective_val = 0.05  # 5% effective

print(f"\nGiven: Effective annual rate i = {i_effective_val*100:.2f}%\n")

frequencies = [
    ('Annual (m=1)', 1),
    ('Semi-Annual (m=2)', 2),
    ('Quarterly (m=4)', 4),
    ('Monthly (m=12)', 12),
    ('Daily (m=365)', 365),
    ('Continuous', np.inf)
]

reverse_results = []

for freq_name, m in frequencies:
    if m == np.inf:
        i_nom = np.log(1 + i_effective_val)  # Force of interest
        per_period = "N/A (continuous)"
    else:
        i_nom = m * ((1 + i_effective_val) ** (1/m) - 1)
        per_period = f"{(i_nom/m)*100:.4f}%"
    
    reverse_results.append({
        'Frequency': freq_name,
        'Nominal i^(m)': f"{i_nom*100:.4f}%" if m != np.inf else f"δ={i_nom*100:.4f}%",
        'Per-Period Rate': per_period,
        'Verification': f"{((1 + i_nom/m if m != np.inf else i_nom)**m if m != np.inf else np.exp(i_nom) - 1)*100:.4f}%" if m != np.inf else f"{(np.exp(i_nom) - 1)*100:.4f}%"
    })

rev_df = pd.DataFrame(reverse_results)
print(rev_df.to_string(index=False))
print()

# 3. BOND PRICING EXAMPLE
print("=" * 70)
print("BOND PRICING WITH NOMINAL SEMI-ANNUAL RATES")
print("=" * 70)

# Bond specifications
face_value = 1000
coupon_rate_nom = 0.05  # 5% coupon, semi-annual payments
coupon_payment = face_value * coupon_rate_nom / 2  # $25 per half-year
years_to_maturity = 5
semi_periods = years_to_maturity * 2

# Market yield (semi-annual)
market_yield_nom = 0.06  # 6% annual semi-annual
semi_annual_yield = market_yield_nom / 2

# Calculate bond price
cf_dates = np.arange(1, semi_periods + 1)  # Period numbers
cf_amounts = np.full(semi_periods, coupon_payment)
cf_amounts[-1] += face_value  # Add face value to last coupon

pv_factors = (1 + semi_annual_yield) ** (-cf_dates)
pv_cfs = cf_amounts * pv_factors

bond_price = pv_cfs.sum()

print(f"\nBond Specification:")
print(f"  Face value: ${face_value}")
print(f"  Coupon: {coupon_rate_nom*100:.1f}% annual (semi-annual) = ${coupon_payment:.2f} per half-year")
print(f"  Maturity: {years_to_maturity} years ({semi_periods} half-years)")
print(f"  Market yield: {market_yield_nom*100:.1f}% annual semi-annual")
print(f"  Semi-annual discount rate: {semi_annual_yield*100:.2f}%")

print(f"\nBond Price Calculation:")
print(f"{'Period':<8} {'Cash Flow':<12} {'Discount Factor':<18} {'PV':<12}")
print("-" * 50)
for i, (cf, pv_factor, pv) in enumerate(zip(cf_amounts, pv_factors, pv_cfs), 1):
    print(f"{i:<8} ${cf:<11.2f} {pv_factor:<18.6f} ${pv:<11.2f}")
print("-" * 50)
print(f"{'Bond Price':<29} ${bond_price:<11.2f}")

# Price-yield relationship
print(f"\nPrice-Yield Analysis:")
yields_test = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
for yield_test in yields_test:
    semi_yield_test = yield_test / 2
    pv_test = np.sum(cf_amounts * (1 + semi_yield_test) ** (-cf_dates))
    status = "PAR" if abs(pv_test - face_value) < 1 else ("PREMIUM" if pv_test > face_value else "DISCOUNT")
    print(f"  Yield {yield_test*100:.1f}%: Price ${pv_test:.2f} ({status})")

print()

# 4. MORTGAGE CALCULATION
print("=" * 70)
print("MORTGAGE CALCULATION WITH NOMINAL MONTHLY RATE")
print("=" * 70)

# Mortgage specs
principal = 300000
annual_rate_nom = 0.06  # 6% annual, compounded monthly
monthly_rate = annual_rate_nom / 12
years = 30
months = years * 12

# Monthly payment using annuity formula
# P = L · [r(1+r)^n] / [(1+r)^n - 1]
numerator = monthly_rate * (1 + monthly_rate) ** months
denominator = (1 + monthly_rate) ** months - 1
monthly_payment = principal * (numerator / denominator)

print(f"\nMortgage Specification:")
print(f"  Principal: ${principal:,.0f}")
print(f"  Annual rate (nominal): {annual_rate_nom*100:.2f}%")
print(f"  Compounding: Monthly")
print(f"  Monthly rate: {monthly_rate*100:.4f}%")
print(f"  Term: {years} years ({months} months)")

print(f"\nMonthly Payment Calculation:")
print(f"  Formula: Payment = P × [r(1+r)^n] / [(1+r)^n - 1]")
print(f"  Payment = ${monthly_payment:,.2f}")

# Total interest paid
total_paid = monthly_payment * months
total_interest = total_paid - principal

print(f"\nOver {years}-year term:")
print(f"  Total paid: ${total_paid:,.2f}")
print(f"  Total interest: ${total_interest:,.2f}")
print(f"  Interest as % of principal: {(total_interest/principal)*100:.1f}%")

# Amortization schedule (first 6 months + last 6 months)
print(f"\nAmortization Schedule (sample months):")
print(f"{'Month':<8} {'Payment':<12} {'Principal':<12} {'Interest':<12} {'Balance':<12}")
print("-" * 56)

balance = principal

# First 6 months
for month in range(1, 7):
    interest = balance * monthly_rate
    principal_payment = monthly_payment - interest
    balance -= principal_payment
    print(f"{month:<8} ${monthly_payment:<11,.2f} ${principal_payment:<11,.2f} ${interest:<11,.2f} ${balance:<11,.2f}")

print("...")

# Last 6 months (manually calculated)
balance = 0  # Reset for last calculation
for month in range(months - 5, months + 1):
    # Estimate balance near end
    remaining_months = months - month + 1
    balance_estimate = monthly_payment * ((1 + monthly_rate)**remaining_months - 1) / (monthly_rate * (1 + monthly_rate)**remaining_months)
    interest = balance_estimate * monthly_rate
    principal_payment = monthly_payment - interest
    balance_estimate -= principal_payment
    if month >= months - 5:
        print(f"{month:<8} ${monthly_payment:<11,.2f} ${principal_payment:<11,.2f} ${interest:<11,.2f} ${balance_estimate:<11,.2f}")

print()

# 5. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Nominal vs Effective over frequencies
ax = axes[0, 0]
m_range = np.array([1, 2, 4, 12, 52, 365])
i_nominal_base = 0.06  # 6% nominal

effective_from_nominal = [(1 + i_nominal_base/m)**m - 1 for m in m_range]
effective_labels = ['Annual', 'Semi-Anl', 'Quarterly', 'Monthly', 'Weekly', 'Daily']

colors = plt.cm.viridis(np.linspace(0, 1, len(m_range)))
bars = ax.bar(effective_labels, [e*100 for e in effective_from_nominal], 
             color=colors, edgecolor='black', linewidth=1.5)

ax.axhline(i_nominal_base*100, color='red', linestyle='--', linewidth=2, 
          label=f'Nominal {i_nominal_base*100:.1f}%')
ax.set_ylabel('Effective Rate (%)', fontsize=11)
ax.set_title('Effect of Compounding Frequency on Effective Rate', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar, e in zip(bars, effective_from_nominal):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{e*100:.3f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Bond price vs yield
ax = axes[0, 1]
yields_range = np.linspace(0.02, 0.10, 50)
bond_prices = []

for y in yields_range:
    semi_y = y / 2
    cf_pv = np.sum(cf_amounts * (1 + semi_y) ** (-cf_dates))
    bond_prices.append(cf_pv)

ax.plot(yields_range*100, bond_prices, linewidth=2.5, color='darkblue')
ax.axhline(face_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Par value')
ax.scatter([market_yield_nom*100], [bond_price], color='red', s=200, zorder=5, 
          label=f'Current (yield {market_yield_nom*100:.1f}%)')
ax.set_xlabel('Market Yield (%)', fontsize=11)
ax.set_ylabel('Bond Price ($)', fontsize=11)
ax.set_title('Bond Price-Yield Relationship', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Mortgage balance amortization
ax = axes[1, 0]
amortization_months = np.linspace(0, months, 1000)
remaining_balance = principal * (((1 + monthly_rate)**months - (1 + monthly_rate)**amortization_months) / 
                                  ((1 + monthly_rate)**months - 1))

ax.plot(amortization_months/12, remaining_balance/1000, linewidth=2.5, color='darkgreen')
ax.fill_between(amortization_months/12, 0, remaining_balance/1000, alpha=0.2, color='green')
ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Remaining Balance ($1000s)', fontsize=11)
ax.set_title(f'Mortgage Amortization (${principal/1000:.0f}K @ {annual_rate_nom*100:.1f}%, {years} years)', 
            fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Interest vs Principal payments over time
ax = axes[1, 1]
months_detail = np.arange(1, months + 1)
balance_detail = principal
interest_payments = []
principal_payments = []

for m in months_detail:
    interest = balance_detail * monthly_rate
    principal_pmt = monthly_payment - interest
    balance_detail -= principal_pmt
    interest_payments.append(interest)
    principal_payments.append(principal_pmt)

months_annual = np.arange(1, 12*years + 1, 12)  # Sample each year
ax.plot(months_detail[::12]/12, np.array(principal_payments)[::12], 
       linewidth=2.5, color='darkblue', marker='o', markersize=4, label='Principal')
ax.plot(months_detail[::12]/12, np.array(interest_payments)[::12], 
       linewidth=2.5, color='coral', marker='s', markersize=4, label='Interest')
ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Monthly Payment Component ($)', fontsize=11)
ax.set_title('Mortgage Decomposition: Principal vs Interest Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('nominal_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")

