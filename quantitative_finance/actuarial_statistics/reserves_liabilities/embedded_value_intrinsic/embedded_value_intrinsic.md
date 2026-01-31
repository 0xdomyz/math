# Embedded Value & Intrinsic Value

## 1. Concept Skeleton
**Definition:** Embedded Value (EV) = Net worth + PV(future profits from in-force business); measures total shareholder value in insurance company  
**Purpose:** Shareholder valuation; measure management performance; appraisal value; compare profitability across portfolios  
**Prerequisites:** Statutory reserves, projected cash flows, risk discount rate, cost of capital

## 2. Comparative Framing
| Value Metric | Basis | Formula | Purpose | User |
|--------------|-------|---------|---------|------|
| **Statutory value** | Net worth | Assets - Liabilities | Regulatory solvency | Regulators |
| **Book value** | GAAP accounting | Assets - (Reserves + Liab) | Financial statements | Investors |
| **Intrinsic value** | EV - option cost | PV(profits) | Baseline value | Actuaries |
| **Embedded value** | Full EV | NW + Intrinsic | Total shareholder value | Management |
| **Market value** | Stock price | Share price × Shares out | Market opinion | Traders |

## 3. Examples + Counterexamples

**Simple Example:**  
Company A: Net worth $100M, PV(future profits) $50M → EV = $150M; if book value $100M, intrinsic profit = $50M

**Failure Case:**  
Company B: Book value $80M, but PV(future profits) = -$10M (bad portfolio); EV = $70M < Book; shareholder value destroyed

**Edge Case:**  
Company C: Whole life portfolio generating low current profits but high reserve = huge statutory liability; EV may be depressed if new business hurts

## 4. Layer Breakdown
```
Embedded Value (EV) Structure:
├─ EV Components:
│   ├─ Net Worth (Statutory):
│   │   ├─ Assets: Total values on balance sheet
│   │   ├─ Reserves: Policy liabilities at statutory basis
│   │   ├─ Other liabilities: Debt, payables, contingencies
│   │   ├─ Net worth: Assets - (Reserves + Other liab)
│   │   ├─ Typical range: 5-20% of reserves
│   │   └─ Purpose: Solvency cushion, capital adequacy
│   ├─ Free Surplus:
│   │   ├─ Definition: Net worth above regulatory minimum
│   │   ├─ Calculation: Net worth - Required capital minimum
│   │   ├─ Example: Net worth $100M, Min capital $60M → Free surplus $40M
│   │   ├─ Use: Available for distribution, growth investment
│   │   └─ Constraint: Regulator may freeze if depleted
│   ├─ Present Value of Future Profits (PVFP / Intrinsic Value):
│   │   ├─ Definition: Σ(Future year profits) × Discount factor
│   │   ├─ Profit sources:
│   │   │   ├─ Mortality gain: Actual < Assumed (win for insurer)
│   │   │   ├─ Interest gain: Actual > Assumed (win for insurer)
│   │   │   ├─ Expense gain: Actual < Assumed (win for insurer)
│   │   │   ├─ Lapse gain: Actual > Assumed (win for insurer - fewer policies to fund)
│   │   │   └─ Total: Sum of all experiences vs assumptions
│   │   ├─ Projection period: 20-40 years (contract lifetime)
│   │   ├─ Terminal value: Tail-end profit (small after many years)
│   │   ├─ Discount rate: Risk-adjusted (cost of capital + risk margin)
│   │   └─ Typical range: 10-50% of net worth (varies by product quality)
│   └─ Total EV:
│       ├─ Formula: EV = Net worth + PVFP
│       ├─ Example: $100M (NW) + $40M (PVFP) = $140M (EV)
│       └─ Interpretation: Total value to shareholders if liquidated + continuity
├─ Embedded Value Definition (IEV):
│   ├─ European Embedded Value (EEV):
│   │   ├─ Standard: Best estimate assumptions + explicit risk margin
│   │   ├─ Net worth: Full economic basis (not statutory)
│   │   ├─ PVFP basis: Realistic assumptions + risk margin on PVFP
│   │   ├─ Disclosure: Formal standards (IEV Association guidelines)
│   │   └─ Calculation: Annual to track management performance
│   ├─ Market-Consistent Embedded Value (MCEV):
│   │   ├─ Enhancement: EEV calibrated to market prices
│   │   ├─ Assets: Marked-to-market (not amortized cost)
│   │   ├─ Liabilities: Matched to market yields
│   │   ├─ Option cost: Adjust PVFP for embedded options (rate guarantees, etc.)
│   │   └─ Benefit: Better reflects economic reality in volatile markets
│   └─ Adjusted Embedded Value (AEV):
│       ├─ Adjustment: Remove one-time items (restructuring, tax gains)
│       ├─ Purpose: Steady-state valuation (comparable year-to-year)
│       └─ Use: Bonuses tied to AEV growth
├─ Option Adjustment in MCEV:
│   ├─ Embedded options in insurance contracts:
│   │   ├─ Guarantees: Minimum rate guaranteed (e.g., 2% on whole life)
│   │   ├─ Surrender options: Policyholder can exit at fair value
│   │   ├─ Lapse options: Policyholder can let policy lapse
│   │   └─ Rider options: Add/remove riders at set cost
│   ├─ Impact if rates rise (favorable):
│   │   ├─ Policyholder keeps 2% guarantee; market yielding 5%
│   │   ├─ Insurer forgoes 3% return differential
│   │   ├─ Option cost = PV(foregone returns)
│   │   └─ MCEV reduced by option cost
│   ├─ Impact if rates fall (unfavorable):
│   │   ├─ Policyholder still gets 2% guarantee
│   │   ├─ Market yields 1%; insurer matches guarantee
│   │   ├─ Option cost minimal (insurer not worse off)
│   │   └─ MCEV reduced less (asymmetric)
│   └─ Hedging strategies:
│       ├─ Interest rate derivatives: Swaps, caps, floors
│       ├─ Dynamic hedging: Rebalance as rates change
│       ├─ Cost: 1-5% of PVFP over time
│       └─ Benefit: Reduces capital requirement
├─ Intrinsic Value Components:
│   ├─ Year-by-year profit waterfall:
│   │   ├─ Premium income: Gross premium × Policies in force
│   │   ├─ Less: Claims paid
│   │   ├─ Less: Operating expenses
│   │   ├─ Plus: Investment income
│   │   ├─ Less: Change in reserve
│   │   └─ Equals: Statutory profit (per year)
│   ├─ Adjustments to statutory profit:
│   │   ├─ Add back: Reserve increase (liability created, not cash cost)
│   │   ├─ Add back: Depreciation (non-cash)
│   │   └─ Less: Capital required (new reserves need capital)
│   ├─ Risk-adjusted discount rate:
│   │   ├─ Base: Risk-free rate (government bond) 2-3%
│   │   ├─ Add: Equity risk premium 5-7%
│   │   ├─ Add: Insurance risk premium 2-4%
│   │   ├─ Total: Typically 10-15% (vs 4-5% for bonds)
│   │   └─ Impact: Higher rate → Lower PVFP
│   └─ Terminal value (tail):
│       ├─ After explicit projection (30-40 years), assume steady-state
│       ├─ Remaining PVFP often small (<5% of total)
│       └─ Treated conservatively (zero or minimal)
├─ Intrinsic Value Application:
│   ├─ New business value:
│   │   ├─ PVFP from new policies issued in year
│   │   ├─ Cost: Acquisition expense + capital requirement
│   │   ├─ NBV = New business PVFP - New business cost
│   │   ├─ Metric: Track whether new business adds/destroys value
│   │   └─ Decision: Stop selling if NBV < 0 (destroying value)
│   ├─ Portfolio valuation:
│   │   ├─ Life insurance: 60-70% from in-force book
│   │   ├─ Term insurance: 40-50% from in-force (high lapse, mortality gain)
│   │   ├─ Whole life: 80%+ from in-force (long duration, high profit potential)
│   │   └─ Comparison: Identify which products most profitable
│   ├─ Acquisition target valuation:
│   │   ├─ Price buyer willing to pay ≈ EV + synergy value
│   │   ├─ EV baseline: Stand-alone economics
│   │   ├─ Synergies: Cost savings, tax benefits, scale economies
│   │   └─ Deal price: Often 1.0-1.5× EV
│   └─ Performance metrics (tied to bonuses):
│       ├─ EV growth: % increase year-over-year (target 10-15%)
│       ├─ Intrinsic value growth: % increase in PVFP
│       ├─ NBV as % of new premium: (Higher % = better profitability)
│       └─ Adjusted for changes: Strip out one-time items
└─ Risks & Limitations:
    ├─ Assumption sensitivity:
    │   ├─ Discount rate ±1% → PVFP changes 15-25%
    │   ├─ Mortality ±10% → PVFP changes 5-15%
    │   ├─ Lapse ±2% → PVFP changes 10-20%
    │   └─ Combined stress → PVFP could halve or double
    ├─ Model risk:
    │   ├─ Complex calculations; implementation errors possible
    │   ├─ Calibration to market prices (MCEV) adds complexity
    │   ├─ Embedded option valuation uses stochastic models (Monte Carlo)
    │   └─ Validation difficult; small changes → large value swings
    ├─ Forecasting risk:
    │   ├─ Projection 30-40 years forward; inherent uncertainty
    │   ├─ Market conditions, regulations, demographics unknowable
    │   ├─ Terminal value tail often ignored (but can be material)
    │   └─ One-time events (pandemic) invalidate all assumptions
    └─ Comparability issues:
        ├─ Different methodologies (EEV vs MCEV vs non-standard)
        ├─ Different assumption bases (conservative vs optimistic)
        ├─ Different risk margins applied (5% vs 10%)
        └─ Cross-company comparisons problematic
```

**Key Insight:** EV captures economic value beyond regulatory net worth; gap between EV and book value shows hidden profit potential

## 5. Mini-Project
Calculate embedded value and analyze sensitivity to key drivers:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP
print("=" * 80)
print("EMBEDDED VALUE ANALYSIS: INTRINSIC VALUE & SHAREHOLDER VALUE")
print("=" * 80)

# Assumptions
mortality_data = {
    40: 0.00131, 45: 0.00172, 50: 0.00233, 55: 0.00325,
    60: 0.00459, 65: 0.00653, 70: 0.00933
}

issue_age = 40
contract_term = 20
gross_premium = 450
net_premium = 340
benefit_amount = 250000
policies_inforce = 100000

# Financial assumptions
interest_assumption = 0.04
cost_of_capital = 0.12  # Risk-adjusted discount rate
expense_maintenance = 30
lapse_rate = 0.05

print(f"\nAssumptions:")
print(f"  Policies in force: {policies_inforce:,}")
print(f"  Gross premium: ${gross_premium:,.0f}/year")
print(f"  Interest rate: {interest_assumption*100:.1f}%")
print(f"  Cost of capital: {cost_of_capital*100:.1f}%\n")

v_assumption = 1 / (1 + interest_assumption)
v_discount = 1 / (1 + cost_of_capital)

def calculate_annual_profit(policies, age, remaining_term, mortality_dict, 
                           premium, maintenance, benefit_amt):
    """Calculate statutory profit for year"""
    
    qx = mortality_dict.get(age, 0.001)
    px = 1 - qx
    
    # Policy count
    deaths = int(policies * qx)
    lapses = int(policies * (1 - qx) * lapse_rate)
    policies_end = policies - deaths - lapses
    
    # Revenue
    premium_income = policies * premium
    
    # Expenses
    claims_paid = deaths * benefit_amt
    maintenance_cost = policies * maintenance
    
    # Simplified profit
    profit = premium_income - claims_paid - maintenance_cost
    
    return profit, policies_end

# 2. CALCULATE STATUTORY NET WORTH
print("=" * 80)
print("STATUTORY NET WORTH CALCULATION")
print("=" * 80)

# Simplified balance sheet
total_assets = 750000000  # $750M (for 100k policies)
total_reserves = 650000000  # $650M (at statutory basis)
other_liabilities = 50000000  # $50M (debt, payables)

net_worth = total_assets - total_reserves - other_liabilities

print(f"\n{'Item':<40} {'Amount':<20}")
print("-" * 60)
print(f"{'Assets':<40} ${total_assets:>18,.0f}")
print(f"{'Less: Policy reserves':<40} ${total_reserves:>18,.0f}")
print(f"{'Less: Other liabilities':<40} ${other_liabilities:>18,.0f}")
print(f"{'Net worth (Statutory)':<40} ${net_worth:>18,.0f}")
print()

# Estimate minimum required capital
min_capital = total_reserves * 0.08  # Simplified: 8% of reserves
free_surplus = net_worth - min_capital

print(f"{'Required minimum capital':<40} ${min_capital:>18,.0f}")
print(f"{'Free surplus':<40} ${free_surplus:>18,.0f}\n")

# 3. PRESENT VALUE OF FUTURE PROFITS (PVFP)
print("=" * 80)
print("PRESENT VALUE OF FUTURE PROFITS (INTRINSIC VALUE)")
print("=" * 80)

print(f"\nYear-by-Year Profit Projection:\n")
print(f"{'Yr':<4} {'Policies':<12} {'Profit':<15} {'PV Factor':<12} {'PV Profit':<15}")
print("-" * 65)

pvfp = 0
policies_current = policies_inforce
cumulative_pv_profit = 0

for year in range(1, contract_term + 1):
    age_now = issue_age + year - 1
    
    profit, policies_next = calculate_annual_profit(policies_current, age_now, contract_term - year + 1,
                                                   mortality_data, gross_premium, expense_maintenance,
                                                   benefit_amount)
    
    pv_factor = v_discount ** year
    pv_profit = profit * pv_factor
    
    pvfp += pv_profit
    cumulative_pv_profit += pv_profit
    
    if year in [1, 5, 10, 15, 20]:
        print(f"{year:<4} {policies_current:<12,} ${profit:<14,.0f} {pv_factor:<12.6f} ${pv_profit:<14,.0f}")
    
    policies_current = policies_next

print()
print(f"Total Present Value of Future Profits: ${pvfp:,.0f}\n")

# 4. EMBEDDED VALUE CALCULATION
print("=" * 80)
print("EMBEDDED VALUE (EV) = NET WORTH + PVFP")
print("=" * 80)

embedded_value = net_worth + pvfp

print(f"\n{'Component':<40} {'Amount':<20} {'% of EV':<15}")
print("-" * 75)
print(f"{'Statutory net worth':<40} ${net_worth:>18,.0f} {(net_worth/embedded_value)*100:>13.1f}%")
print(f"{'Add: PVFP (intrinsic value)':<40} ${pvfp:>18,.0f} {(pvfp/embedded_value)*100:>13.1f}%")
print(f"{'Embedded Value':<40} ${embedded_value:>18,.0f} {'100.0%':>14}")
print()

# Compare to book value
book_value = net_worth
hidden_value = embedded_value - book_value
hidden_value_pct = (hidden_value / embedded_value) * 100

print(f"Book value (statutory NW): ${book_value:,.0f}")
print(f"Embedded value: ${embedded_value:,.0f}")
print(f"Hidden value: ${hidden_value:,.0f} ({hidden_value_pct:.1f}% of EV)\n")

# 5. NEW BUSINESS VALUE
print("=" * 80)
print("NEW BUSINESS VALUE (NBV)")
print("=" * 80)

# Assume 5% growth in policies
new_policies_year = int(policies_inforce * 0.05)
new_business_profit = new_policies_year * (gross_premium - net_premium)  # Simplified

acquisition_cost = new_policies_year * (gross_premium * 0.20)  # 20% acquisition load
capital_required = new_policies_year * 50  # Rough capital need per policy

nbv = (new_business_profit - acquisition_cost - capital_required) * v_discount

print(f"\nNew business assumption: {new_policies_year:,} new policies\n")
print(f"{'Item':<40} {'Amount':<20}")
print("-" * 60)
print(f"{'New policies':<40} {new_policies_year:>18,}")
print(f"{'Expected profit per policy':<40} ${(gross_premium - net_premium):>18,.0f}")
print(f"{'Total expected profit':<40} ${new_business_profit:>18,.0f}")
print()
print(f"{'Less: Acquisition cost':<40} ${acquisition_cost:>18,.0f}")
print(f"{'Less: Capital required':<40} ${capital_required:>18,.0f}")
print(f"{'Net profit':<40} ${new_business_profit - acquisition_cost - capital_required:>18,.0f}")
print()
print(f"{'New Business Value (PV)':<40} ${nbv:>18,.0f}\n")

# 6. SENSITIVITY ANALYSIS
print("=" * 80)
print("SENSITIVITY: PVFP TO KEY DRIVERS")
print("=" * 80)

discount_rates = [0.08, 0.10, 0.12, 0.14, 0.16]
mortality_multipliers = [0.80, 0.90, 1.00, 1.10, 1.20]

print(f"\nBase Case PVFP: ${pvfp:,.0f}\n")

# Discount rate sensitivity
print("Discount Rate Impact:\n")
print(f"{'Rate':<15} {'PVFP':<20} {'vs Base':<15} {'EV':<20}")
print("-" * 70)

for rate in discount_rates:
    v_rate = 1 / (1 + rate)
    pvfp_test = 0
    policies_test = policies_inforce
    
    for year in range(1, contract_term + 1):
        age = issue_age + year - 1
        profit_test, policies_test = calculate_annual_profit(policies_test, age, contract_term - year + 1,
                                                             mortality_data, gross_premium, expense_maintenance,
                                                             benefit_amount)
        pvfp_test += profit_test * (v_rate ** year)
    
    ev_test = net_worth + pvfp_test
    pct_vs_base = ((pvfp_test / pvfp) - 1) * 100
    
    print(f"{rate*100:>13.1f}% ${pvfp_test:>18,.0f} {pct_vs_base:>+13.1f}% ${ev_test:>18,.0f}")

print()

# 7. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: EV composition
ax = axes[0, 0]
components = ['Statutory\nNet Worth', 'PVFP\n(Intrinsic)', 'Embedded\nValue']
amounts = [net_worth, pvfp, embedded_value]
colors = ['steelblue', 'green', 'darkblue']

bars = ax.bar(components, amounts, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)

for bar, amt in zip(bars, amounts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${amt/1e6:.0f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Amount ($)', fontsize=11)
ax.set_title('Embedded Value Composition', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 2: Cumulative PV profit
ax = axes[0, 1]
years_plot = np.arange(1, contract_term + 1)
cumulative_pv = []

pvfp_cumul = 0
policies_cumul = policies_inforce

for year in years_plot:
    age = issue_age + year - 1
    profit, policies_cumul = calculate_annual_profit(policies_cumul, age, contract_term - year + 1,
                                                    mortality_data, gross_premium, expense_maintenance,
                                                    benefit_amount)
    pv_factor = v_discount ** year
    pvfp_cumul += profit * pv_factor
    cumulative_pv.append(pvfp_cumul)

ax.plot(years_plot, np.array(cumulative_pv) / 1e6, linewidth=2.5, marker='o', markersize=5, color='green')
ax.fill_between(years_plot, 0, np.array(cumulative_pv) / 1e6, alpha=0.3, color='green')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Cumulative PV Profit ($M)', fontsize=11)
ax.set_title('Profit Development Path', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Discount rate sensitivity
ax = axes[1, 0]
rates_plot = np.array([r * 100 for r in discount_rates])
pvfp_by_rate = []

for rate in discount_rates:
    v_r = 1 / (1 + rate)
    pvfp_r = 0
    pol_r = policies_inforce
    
    for year in range(1, contract_term + 1):
        age = issue_age + year - 1
        prof_r, pol_r = calculate_annual_profit(pol_r, age, contract_term - year + 1,
                                              mortality_data, gross_premium, expense_maintenance,
                                              benefit_amount)
        pvfp_r += prof_r * (v_r ** year)
    
    pvfp_by_rate.append(pvfp_r / 1e6)

ax.plot(rates_plot, pvfp_by_rate, linewidth=2.5, marker='o', markersize=6, color='steelblue')
ax.axvline(x=cost_of_capital * 100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Base (12%)')

ax.set_xlabel('Discount Rate (%)', fontsize=11)
ax.set_ylabel('PVFP ($M)', fontsize=11)
ax.set_title('Sensitivity: Discount Rate Impact', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: EV multiples
ax = axes[1, 1]
multiples = ['Book Value', 'EV / Book', 'EV / Premium\nIncome', 'PVFP /\nNet Worth']
values = [1.0, embedded_value / book_value, 
         embedded_value / (gross_premium * policies_inforce),
         pvfp / net_worth]

bars = ax.bar(range(len(multiples)), values, alpha=0.6, edgecolor='black', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(multiples)))
ax.set_xticklabels(multiples, fontsize=10)
ax.set_ylabel('Multiple / Ratio', fontsize=11)
ax.set_title('Valuation Multiples & Ratios', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('embedded_value.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete. Chart saved.")
```

## 6. Challenge Round
When embedded value assumptions break down:
- **Model recalibration**: Discover mortality assumption wrong; PVFP drops 30%; shareholder value destroyed; management credibility damaged
- **Market-consistency shock**: Interest rates plummet; option cost explodes (due to guarantees); MCEV drops 20% while statutory book stable
- **Assumption drift**: Year 1: Assume 5% lapse; Year 3 reality 8%; PVFP erodes gradually; forward guidance repeatedly cut
- **Tail risk materialization**: Pandemic mortality; PVFP assumes deaths 5 years hence; sudden surge 10 years ahead; entire projection invalid
- **Capital requirement creep**: Regulatory requirement calculation changes; minimum capital needed increases 50%; free surplus vanishes
- **Terminal value surprise**: Projected 30-year steady state assumed 2% profit; actual 1% (due to competition); terminal value halves

## 7. Key References
- [European Embedded Value Standards (IEV Association)](https://www.actuaries.org.uk/) - EV/MCEV guidelines
- [Bowers et al., Actuarial Mathematics (Chapter 6-8)](https://www.soa.org/) - Profit and value calculations
- [SOA Embedded Value Standards](https://www.soa.org/) - US framework

---
**Status:** Shareholder value measurement | **Complements:** Statutory Reserves, Profit Testing, Capital Requirements
