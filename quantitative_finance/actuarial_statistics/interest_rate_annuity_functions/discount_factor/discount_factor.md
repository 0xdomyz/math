# Discount Factor

## 1. Concept Skeleton
**Definition:** Present value multiplier v = 1/(1+i); discounts future $1 to today's equivalent value  
**Purpose:** Convert all future cashflows to present value, price bonds/annuities, calculate liability reserves, compare cashflow timing  
**Prerequisites:** Effective annual rate, present value concept, time value of money, reciprocal relationships

## 2. Comparative Framing
| Component | Discount Factor v | Accumulation (1+i) | Discount Rate d |
|-----------|------------------|-------------------|-----------------|
| **Formula** | v = 1/(1+i) | (1+i) | d = i/(1+i) = 1-v |
| **Range** | (0, 1) for i > 0 | (1, ∞) for i > 0 | (0, 1) for i > 0 |
| **Use** | Discounting future | Growth from today | Adjusting returns |
| **n-year** | v^n = 1/(1+i)^n | (1+i)^n | d^n (rarely used) |
| **Example** | $1 in 1 year ≈ $0.95 at 5% | $1 today → $1.05 | 5%/1.05 ≈ 4.76% |

## 3. Examples + Counterexamples

**Simple Example:**  
i = 5%: v = 1/1.05 ≈ 0.9524; $1,000 received in 1 year ≈ $952.38 in present value

**Failure Case:**  
Using v = 1 (no discounting): Ignores time value; $1,000 one year out treated as $1,000 today; massive reserve underestimation

**Edge Case:**  
Negative rates (δ = -0.02): v = 1/0.98 ≈ 1.0204 > 1; future cashflows worth MORE than present (rare, ECB regime)

## 4. Layer Breakdown
```
Discount Factor Structure:
├─ Fundamental Relationships:
│   ├─ v = 1/(1+i)  [basic definition]
│   ├─ v^n = 1/(1+i)^n  [n-period discounting]
│   ├─ d = 1 - v = i/(1+i)  [discount rate]
│   ├─ i = v^{-1} - 1  [solve for rate]
│   └─ i·v + d = 1  (always true)
├─ Cashflow Valuation:
│   ├─ Single: PV = FV · v^n
│   ├─ Annuity-immediate: PV = PMT · ∑v^k = PMT · [1-v^n]/i
│   ├─ Annuity-due: PV = PMT · ∑v^k · (1+i) = PMT · [1-v^n]/i · (1+i)
│   ├─ Perpetuity-immediate: PV = PMT / i = PMT / (1 - v)
│   └─ Perpetuity-due: PV = PMT / d = PMT / (1 - v) + PMT
├─ Multi-Rate Scenarios:
│   ├─ Forward discount: v_{x,y} = v^{y-x}
│   ├─ Spot curve: v(t) varies by time t (not constant)
│   ├─ Discount bond price: B = Face · v^T  (zero coupon)
│   └─ Yield-to-maturity: Solve for i to match observed price
├─ Term-Structure Applications:
│   ├─ Spot discount curve: v(t) = 1 / [1 + y(t)]^t
│   ├─ Forward rates: f_{t,Δt} derived from ratio of discount factors
│   ├─ Coupon bond: ∑ Coupon·v(t) + Face·v(T)
│   └─ Floating rate: Coupon indexed + spread, v(t) still applied
└─ Actuarial Reserving:
    ├─ Life insurance: Reserve = ∑ Benefit·v^t·ₜp_x
    ├─ Annuity: Reserve = Annuity PV = Payment·aₙ̄|·v^issue_age
    ├─ Liability: DC amount·v^duration (present-value method)
    └─ ALM: Duration · interest rate change = ΔPrice/Price
```

**Interaction:** Forecast cashflows → Apply age-appropriate v → Sum present values → Result is liability/reserve

## 5. Mini-Project
Calculate discount factors, price instruments, and perform duration analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. BASIC DISCOUNT FACTOR CALCULATIONS
print("=" * 70)
print("DISCOUNT FACTOR FUNDAMENTALS")
print("=" * 70)

interest_rates = np.array([0.01, 0.03, 0.05, 0.07, 0.10])
years = np.array([1, 5, 10, 20, 30])

print("\nDiscount Factors v^n = 1/(1+i)^n\n")
print(f"{'i':<8}", end='')
for n in years:
    print(f"v^{n:<2d}", end='\t')
print()
print("-" * 70)

for i in interest_rates:
    print(f"{i*100:<7.1f}%", end='\t')
    for n in years:
        v_n = 1 / (1 + i) ** n
        print(f"{v_n:.4f}", end='\t')
    print()

# Calculate discount rates d = i/(1+i) for each effective rate
print("\n\nDiscount Rates d = i/(1+i):\n")
print(f"{'Effective i':<15} {'Discount Rate d':<15} {'Relationship: 1-v':<20}")
print("-" * 50)

for i in interest_rates:
    v = 1 / (1 + i)
    d = i / (1 + i)
    relationship = 1 - v
    print(f"{i*100:>6.2f}%        {d*100:>6.4f}%        {relationship*100:>6.4f}%")

print()

# 2. FUTURE VALUE TO PRESENT VALUE
print("=" * 70)
print("FUTURE VALUE TO PRESENT VALUE CONVERSIONS")
print("=" * 70)

cashflows = [
    ("Annual savings", 1000, [1, 2, 3, 4, 5]),
    ("Lump sum payment", 10000, [10]),
    ("Inheritance", 50000, [25]),
    ("Pension benefit", 30000, [15, 16, 17, 18, 19, 20])
]

i_discount = 0.05

print(f"\nDiscount rate: {i_discount*100:.1f}% effective annual\n")

total_pv = 0

for cf_name, cf_amount, years_list in cashflows:
    print(f"{cf_name}:")
    cf_pv = 0
    
    for year in years_list:
        v_n = 1 / (1 + i_discount) ** year
        pv = cf_amount * v_n
        cf_pv += pv
        print(f"  Year {year:2d}: ${cf_amount:>10,} × {v_n:.6f} = ${pv:>10,.2f}")
    
    print(f"  Subtotal: ${cf_pv:>10,.2f}\n")
    total_pv += cf_pv

print("-" * 70)
print(f"Total Present Value: ${total_pv:>10,.2f}")
print()

# 3. BOND PRICING WITH DISCOUNT FACTORS
print("=" * 70)
print("BOND PRICING USING DISCOUNT FACTORS")
print("=" * 70)

# Bond details
face_value = 1000
coupon_annual = 0.06  # 6% annual coupon
coupon_payment = face_value * coupon_annual  # Assume annual for simplicity
years_to_maturity = 5
semi_periods = years_to_maturity * 2
coupon_semi = coupon_payment / 2

# Market yields
yields = [0.04, 0.06, 0.08]

print(f"\nBond Specification:")
print(f"  Face Value: ${face_value}")
print(f"  Coupon: {coupon_annual*100:.1f}% annual (${coupon_payment:.0f}/year)")
print(f"  Maturity: {years_to_maturity} years")
print(f"  Payment: Semi-annual (${coupon_semi:.2f} per half-year)\n")

bond_pricing = []

for ytm in yields:
    semi_ytm = ytm / 2
    
    # Calculate bond price using discount factors
    price = 0
    cf_details = []
    
    # Coupon payments
    for period in range(1, semi_periods + 1):
        v_period = 1 / (1 + semi_ytm) ** period
        pv = coupon_semi * v_period
        price += pv
        cf_details.append(('Coupon', period, coupon_semi, v_period, pv))
    
    # Face value at maturity
    v_final = 1 / (1 + semi_ytm) ** semi_periods
    pv_face = face_value * v_final
    price += pv_face
    cf_details.append(('Face Value', semi_periods, face_value, v_final, pv_face))
    
    bond_pricing.append({
        'Yield (%)': f"{ytm*100:.1f}",
        'Semi-Annual Rate': f"{semi_ytm*100:.3f}",
        'Bond Price': price,
        'Premium/Discount': 'Premium' if price > face_value else ('Par' if abs(price - face_value) < 0.01 else 'Discount'),
        'Price/Face': f"{(price/face_value)*100:.2f}%"
    })
    
    print(f"YTM = {ytm*100:.1f}% (semi-annual: {semi_ytm*100:.3f}%)")
    print(f"{'Period':<10} {'Type':<12} {'Cash Flow':<12} {'v^n':<12} {'PV':<12}")
    print("-" * 58)
    
    for cf_type, period, cf, v_n, pv in cf_details:
        print(f"{period:<10} {cf_type:<12} ${cf:<11,.2f} {v_n:<12.6f} ${pv:<11,.2f}")
    
    print(f"{'Bond Price':<10} {'':12} {'':12} {'':12} ${price:>11,.2f}\n")

bond_df = pd.DataFrame(bond_pricing)
print("Bond Pricing Summary:")
print(bond_df.to_string(index=False))
print()

# 4. DURATION AND CONVEXITY (interest rate sensitivity)
print("=" * 70)
print("DURATION: INTEREST RATE SENSITIVITY")
print("=" * 70)

ytm_base = 0.05
semi_ytm_base = ytm_base / 2

# Calculate duration using discount factors
duration = 0
price = 0
macaulay_duration = 0

for period in range(1, semi_periods + 1):
    v_period = 1 / (1 + semi_ytm_base) ** period
    pv = coupon_semi * v_period
    price += pv
    macaulay_duration += period * pv

# Add final face value
v_final = 1 / (1 + semi_ytm_base) ** semi_periods
pv_face = face_value * v_final
price += pv_face
macaulay_duration += semi_periods * pv_face

# Duration in years (convert from semi-periods)
macaulay_duration /= price
macaulay_duration /= 2  # Convert from semi-periods to years

# Modified duration
modified_duration = macaulay_duration / (1 + semi_ytm_base * 2)

print(f"\nBond at YTM = {ytm_base*100:.1f}%:")
print(f"  Price: ${price:,.2f}")
print(f"  Macaulay Duration: {macaulay_duration:.3f} years")
print(f"  Modified Duration: {modified_duration:.3f} years")
print(f"\nDuration Interpretation:")
print(f"  For 1% increase in YTM:")
print(f"    Price change ≈ -{modified_duration*100:.2f}% = ${price * (-modified_duration * 0.01):,.2f}")
print()

# 5. SPOT CURVE AND TERM STRUCTURE
print("=" * 70)
print("SPOT CURVE: VARYING DISCOUNT FACTORS BY TERM")
print("=" * 70)

# Example: Upward-sloping yield curve
maturities = np.array([0.5, 1, 2, 3, 5, 10, 20, 30])
spot_rates = 0.02 + 0.02 * (1 - np.exp(-maturities / 5))  # Nelson-Siegel inspired

print(f"\n{'Maturity':<12} {'Spot Rate':<15} {'Discount Factor':<18} {'$1 PV':<12}")
print("-" * 57)

for mat, rate in zip(maturities, spot_rates):
    v_t = 1 / (1 + rate) ** mat
    pv_unit = 1 * v_t
    print(f"{mat:>6.1f} years  {rate*100:>7.3f}%        {v_t:>15.6f}       ${pv_unit:>10.4f}")

print()

# 6. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Discount factors decay
ax = axes[0, 0]
times = np.linspace(0, 30, 100)
rates_example = [0.02, 0.05, 0.08]

for rate in rates_example:
    v_decay = 1 / (1 + rate) ** times
    ax.plot(times, v_decay, linewidth=2.5, label=f'i = {rate*100:.1f}%')

ax.set_xlabel('Years', fontsize=11)
ax.set_ylabel('Discount Factor v^n', fontsize=11)
ax.set_title('Discount Factor Decay Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Bond price sensitivity to yield
ax = axes[0, 1]
yields_range = np.linspace(0.02, 0.10, 50)
bond_prices = []

for y in yields_range:
    semi_y = y / 2
    bond_price = 0
    
    # Coupons
    for period in range(1, semi_periods + 1):
        bond_price += coupon_semi / (1 + semi_y) ** period
    
    # Face value
    bond_price += face_value / (1 + semi_y) ** semi_periods
    
    bond_prices.append(bond_price)

ax.plot(yields_range * 100, bond_prices, linewidth=2.5, color='darkblue')
ax.axhline(face_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Par value')
ax.scatter([ytm_base * 100], [price], color='green', s=200, zorder=5, label=f'Current YTM {ytm_base*100:.1f}%')
ax.set_xlabel('Yield-to-Maturity (%)', fontsize=11)
ax.set_ylabel('Bond Price ($)', fontsize=11)
ax.set_title('Bond Price-Yield Relationship', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Spot curve
ax = axes[1, 0]
ax.plot(maturities, spot_rates * 100, linewidth=2.5, color='darkgreen', marker='o', markersize=6)
ax.fill_between(maturities, 0, spot_rates * 100, alpha=0.2, color='green')
ax.set_xlabel('Maturity (years)', fontsize=11)
ax.set_ylabel('Spot Rate (%)', fontsize=11)
ax.set_title('Yield Curve: Upward-Sloping Example', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Annuity present values using discount factors
ax = axes[1, 1]
annuity_terms = np.arange(1, 51)
annuity_i = 0.05
annuities_pv = []

for n in annuity_terms:
    # aₙ̄| = [1 - v^n]/i
    v_n = 1 / (1 + annuity_i) ** n
    a_n = (1 - v_n) / annuity_i
    annuities_pv.append(a_n)

perpetuity_value = 1 / annuity_i

ax.plot(annuity_terms, annuities_pv, linewidth=2.5, color='purple', marker='o', markersize=3, alpha=0.7)
ax.axhline(perpetuity_value, color='red', linestyle='--', linewidth=2, label=f'Perpetuity = {perpetuity_value:.1f}')
ax.set_xlabel('Term (years)', fontsize=11)
ax.set_ylabel('Annuity Factor aₙ̄|', fontsize=11)
ax.set_title('Annuity Present Value Factors (i = 5%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('discount_factor_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When discount factors mislead:
- **Negative rates**: v > 1 breaks intuition; money worth more later than now; requires careful communication to stakeholders
- **Curve inversion**: Long-term discount v_30 > short-term v_5 when curve inverts; creates negative forward rates
- **Floating coupons**: v^n applies to expectation of rate-linked cash, not known amount; requires separate analysis
- **Early redemption**: Callable bonds have uncertain maturity; option-adjusted duration needed, not simple duration
- **Credit risk**: Discount only returns to investor after credit losses; v^n assumes 100% recovery; must adjust for default probability

## 7. Key References
- [Discount Factor (Wikipedia)](https://en.wikipedia.org/wiki/Discount_factor) - Mathematical foundation
- [Bond Duration & Convexity](https://en.wikipedia.org/wiki/Bond_duration) - Interest rate sensitivity measures
- [Bowers et al., Actuarial Mathematics (Chapters 2-3)](https://www.soa.org/) - Annuity and bond calculations

---
**Status:** Foundational valuation tool | **Complements:** Effective Annual Rate, Annuity Functions, Bond Pricing
