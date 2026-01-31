# Exposure at Default (EAD)

## 1. Concept Skeleton
**Definition:** Total value of credit exposure at time of borrower default; principal + accrued interest + drawn amounts  
**Purpose:** Quantify maximum potential loss amount, size capital reserves, price credit products  
**Prerequisites:** Credit products, amortization schedules, credit line mechanics, commitment accounting

## 2. Comparative Framing
| Product Type | EAD Calculation | Uncertainty | Example |
|--------------|-----------------|------------|---------|
| **Installment Loan** | Remaining principal | Low (fixed schedule) | $95K on $100K mortgage (5 yrs in) |
| **Credit Card** | Drawn balance + interest | Medium (utilization varies) | $8K drawn on $10K limit |
| **Committed Credit Line** | Drawn + undrawn (CCF) | High (future drawdowns) | $5M committed, $2M drawn; CCF=80% → $6.6M EAD |
| **Derivative** | Mark-to-market + potential future | Very high (path-dependent) | FX swap notional + credit exposure |

## 3. Examples + Counterexamples

**Simple Example:**  
$100K loan, 5-year term, 2 years into loan at 3% interest. Principal paid down to $60K. Default occurs: EAD = $60K + accrued interest ≈ $61.5K

**Failure Case:**  
Credit card default: Assuming EAD = drawn balance ignores interest accrual and penalty fees. True EAD = $5K + $500 interest + $200 fees = $5.7K

**Edge Case:**  
Revolving credit during crisis: Borrower draws on available credit as default risk rises. EAD > original commitment if additional funds advanced near default

## 4. Layer Breakdown
```
Exposure at Default Framework:
├─ EAD Components:
│   ├─ Principal outstanding: Core loan amount
│   ├─ Accrued interest: Interest earned but unpaid
│   ├─ Fees & penalties: Late payment charges, over-limit fees
│   ├─ Foreign exchange impact: FX loans, mark-to-market
│   └─ Undrawn commitments: Available but not yet drawn
├─ EAD for Different Products:
│   ├─ Term Loan/Mortgage: Principal + accrued interest
│   ├─ Revolving Credit: Drawn + undrawn*CCF
│   │   └─ Credit Conversion Factor (CCF): Probability undrawn is accessed
│   ├─ Guarantees: Maximum loss if principal obligor fails
│   └─ Derivatives: Replacement cost + potential future exposure
├─ Credit Conversion Factors:
│   ├─ CCF=0: No exposure from undrawn (unlikely)
│   ├─ CCF=0.2-0.5: Partial utilization (uncommitted lines)
│   ├─ CCF=0.75-1.0: High utilization near default (committed)
│   └─ CCF=1.0: Full commitment guaranteed
├─ Time Dimension:
│   ├─ Current exposure: Today's outstanding amount
│   ├─ Potential exposure: Future expected exposure (for derivatives)
│   └─ Total exposure: Current + potential
└─ EAD Dynamics:
    ├─ Amortizing loans: EAD decreases over time
    ├─ Revolving credit: EAD uncertain (utilization varies)
    ├─ Bullet loans: EAD stable until maturity
    └─ Default correlation: EAD often increases at default (draws on backup lines)
```

## 5. Mini-Project
Model EAD across product types:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# Simulation 1: Mortgage amortization and EAD
print("=== Mortgage EAD Over Time ===")
principal = 300000
annual_rate = 0.04
years = 30
monthly_rate = annual_rate / 12
n_months = years * 12

# Calculate monthly payment
monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)

# Generate payment schedule
months = np.arange(0, n_months + 1)
balance = np.zeros(len(months))
balance[0] = principal

for month in range(1, len(months)):
    interest_paid = balance[month-1] * monthly_rate
    principal_paid = monthly_payment - interest_paid
    balance[month] = max(0, balance[month-1] - principal_paid)

# EAD = remaining principal + accrued interest
accrued_interest = np.zeros(len(months))
ead_mortgage = balance + accrued_interest

# Specific points
print(f"Mortgage: ${principal:,.0f}")
print(f"Monthly payment: ${monthly_payment:,.0f}")
print(f"Years | Months | Principal | Interest | EAD")
for year in [0, 5, 10, 15, 20, 25, 30]:
    month_idx = year * 12
    if month_idx < len(months):
        print(f"{year:5d} | {month_idx:6d} | ${balance[month_idx]:9,.0f} | ${accrued_interest[month_idx]:8,.0f} | ${ead_mortgage[month_idx]:10,.0f}")

# Simulation 2: Credit card utilization and EAD
print("\n=== Credit Card EAD ===")
credit_limits = np.array([5000, 10000, 15000, 20000, 25000])
utilization_rates = np.array([0.25, 0.50, 0.75, 0.90, 0.95])
interest_rates = np.array([0.18, 0.20, 0.21, 0.22, 0.23])  # Annual APR

card_data = []
for limit, util, apr in zip(credit_limits, utilization_rates, interest_rates):
    drawn = limit * util
    monthly_interest = drawn * (apr / 12)
    # Assume minimum payment is 2% of balance
    min_payment = drawn * 0.02
    # If customer defaults next month after payment
    ead = drawn + monthly_interest
    card_data.append({
        'Credit Limit': limit,
        'Utilization %': util * 100,
        'Drawn Amount': drawn,
        'Monthly Interest': monthly_interest,
        'EAD (with 1-month interest)': ead
    })

card_df = pd.DataFrame(card_data)
print(card_df.to_string(index=False))

# Simulation 3: Committed credit line with CCF
print("\n=== Credit Line EAD with Credit Conversion Factor ===")
n_accounts = 1000
committed_amount = np.random.lognormal(12, 1.5, n_accounts)  # Committed size
drawn_pct = np.random.beta(2, 5, n_accounts)  # 0-100% drawn
drawn_amount = committed_amount * drawn_pct
accrued_interest_line = drawn_amount * 0.015  # 1.5% monthly interest equivalent

# CCF varies by seniority
ccf_normal = 0.75  # When borrower is healthy
ccf_stressed = 0.95  # When borrower is stressed
ccf_default = 1.00  # At default, full commitment at risk

# Scenario 1: Normal times
ead_normal = drawn_amount + accrued_interest_line + (committed_amount - drawn_amount) * ccf_normal

# Scenario 2: Stressed (near default)
ead_stressed = drawn_amount + accrued_interest_line + (committed_amount - drawn_amount) * ccf_stressed

# Scenario 3: At default
ead_at_default = drawn_amount + accrued_interest_line + (committed_amount - drawn_amount) * ccf_default

results_line = pd.DataFrame({
    'Committed': committed_amount,
    'Drawn %': drawn_pct * 100,
    'EAD (Normal)': ead_normal,
    'EAD (Stressed)': ead_stressed,
    'EAD (At Default)': ead_at_default
})

print(f"Average Committed Amount: ${results_line['Committed'].mean():,.0f}")
print(f"Average % Drawn: {results_line['Drawn %'].mean():.1f}%")
print(f"Average EAD (Normal): ${results_line['EAD (Normal)'].mean():,.0f}")
print(f"Average EAD (Stressed): ${results_line['EAD (Stressed)'].mean():,.0f}")
print(f"Average EAD (At Default): ${results_line['EAD (At Default)'].mean():,.0f}")
print(f"EAD Increase from Normal to Default: {(results_line['EAD (At Default)'].mean() / results_line['EAD (Normal)'].mean() - 1) * 100:.1f}%")

# Simulation 4: Derivative exposure (simplified)
print("\n=== Derivative EAD (Simplified FX Swap) ===")
n_swaps = 100
notional_usd = np.random.lognormal(15, 1, n_swaps)
maturity_years = np.random.uniform(1, 10, n_swaps)
time_to_maturity = maturity_years

# Mark-to-market varies randomly (interest rate changes)
mtm = np.random.normal(0, notional_usd * 0.02, n_swaps)

# Potential future exposure (PFE) simplified: max potential move
volatility = 0.10
pfe = 2 * volatility * notional_usd * np.sqrt(time_to_maturity)

# EAD = max(MTM, 0) + PFE
ead_derivative = np.maximum(mtm, 0) + pfe

deriv_df = pd.DataFrame({
    'Notional': notional_usd,
    'Maturity (years)': maturity_years,
    'MTM': mtm,
    'PFE': pfe,
    'EAD': ead_derivative
})

print(f"Average Notional: ${deriv_df['Notional'].mean():,.0f}")
print(f"Average MTM: ${deriv_df['MTM'].mean():,.0f}")
print(f"Average PFE: ${deriv_df['PFE'].mean():,.0f}")
print(f"Average EAD: ${deriv_df['EAD'].mean():,.0f}")
print(f"Max PFE: ${deriv_df['PFE'].max():,.0f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Mortgage EAD amortization
ax1 = axes[0, 0]
time_years = months / 12
ax1.plot(time_years, ead_mortgage, linewidth=2)
ax1.fill_between(time_years, 0, ead_mortgage, alpha=0.2)
ax1.set_xlabel('Years')
ax1.set_ylabel('EAD ($)')
ax1.set_title('Mortgage EAD Over Time\n(Declining as loan paid down)')
ax1.grid(True, alpha=0.3)

# Plot 2: Credit card EAD
ax2 = axes[0, 1]
x_card = np.arange(len(card_df))
width = 0.35
ax2.bar(x_card - width/2, card_df['Drawn Amount'], width, label='Drawn', alpha=0.7, edgecolor='black')
ax2.bar(x_card + width/2, card_df['Monthly Interest'], width, label='Interest', alpha=0.7, edgecolor='black')
ax2.set_ylabel('Amount ($)')
ax2.set_title('Credit Card EAD Components')
ax2.set_xticks(x_card)
ax2.set_xticklabels([f'${l/1000:.0f}K' for l in card_df['Credit Limit']])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Credit line EAD scenarios
ax3 = axes[0, 2]
ead_scenarios = [results_line['EAD (Normal)'].mean(),
                 results_line['EAD (Stressed)'].mean(),
                 results_line['EAD (At Default)'].mean()]
scenario_names = ['Normal', 'Stressed', 'At Default']
colors_scenario = ['green', 'yellow', 'red']
bars = ax3.bar(scenario_names, ead_scenarios, color=colors_scenario, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Average EAD ($)')
ax3.set_title('Credit Line EAD by Scenario\n(CCF increases near default)')
for bar, val in zip(bars, ead_scenarios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'${val/1e6:.1f}M', ha='center', va='bottom')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Utilization vs EAD
ax4 = axes[1, 0]
scatter = ax4.scatter(results_line['Drawn %'], results_line['EAD (At Default)'], 
                     s=results_line['Committed']/1000, alpha=0.5, edgecolors='black')
ax4.set_xlabel('Utilization %')
ax4.set_ylabel('EAD at Default ($)')
ax4.set_title('EAD vs Utilization\n(Bubble size = Committed amount)')
ax4.grid(True, alpha=0.3)

# Plot 5: Derivative EAD distribution
ax5 = axes[1, 1]
ax5.hist(deriv_df['EAD']/deriv_df['Notional'], bins=30, edgecolor='black', alpha=0.7)
ax5.axvline((deriv_df['EAD']/deriv_df['Notional']).mean(), color='r', 
           linestyle='--', linewidth=2, label=f'Mean={deriv_df["EAD"].mean()/deriv_df["Notional"].mean():.1%}')
ax5.set_xlabel('EAD as % of Notional')
ax5.set_ylabel('Frequency')
ax5.set_title('Derivative EAD Distribution\n(EAD > Notional due to PFE)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: CCF sensitivity
ax6 = axes[1, 2]
ccf_range = np.linspace(0, 1, 50)
committed_avg = results_line['Committed'].mean()
drawn_avg = results_line['Drawn Amount'].mean()
ead_by_ccf = drawn_avg + (committed_avg - drawn_avg) * ccf_range

ax6.plot(ccf_range, ead_by_ccf/committed_avg, linewidth=2)
ax6.axvline(ccf_normal, color='g', linestyle='--', label=f'Normal CCF={ccf_normal}', alpha=0.7)
ax6.axvline(ccf_stressed, color='orange', linestyle='--', label=f'Stressed CCF={ccf_stressed}', alpha=0.7)
ax6.axvline(ccf_default, color='r', linestyle='--', label=f'Default CCF={ccf_default}', alpha=0.7)
ax6.set_xlabel('Credit Conversion Factor (CCF)')
ax6.set_ylabel('EAD as % of Commitment')
ax6.set_title('EAD Sensitivity to CCF\n(Undrawn drawdown risk)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is EAD estimation problematic?
- **Hidden exposure**: Off-balance sheet items, guarantees, contingent liabilities not tracked
- **Correlated drawdowns**: During crisis, borrowers draw on all available credit as default risk rises
- **FX volatility**: Exposure in foreign currency; MTM can swing significantly with exchange rates
- **Netting complexity**: Master agreements allow netting; counterparty dependence on intermediary solvency
- **Term uncertainty**: Commitments with embedded options (cancellation, renewal); maturity variable

## 7. Key References
- [Basel III EAD Standards](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory EAD definitions, CCF guidelines
- [Credit Conversion Factors](https://www.bis.org/publ/bcbs128.pdf) - Historical CCF calibration by product
- [Derivative Exposure](https://www.bis.org/publ/bcbs279.pdf) - CVA charge, replacement cost methodology

---
**Status:** Quantifies exposure size for credit losses | **Complements:** Credit Risk Definition, PD, LGD, Expected Loss
