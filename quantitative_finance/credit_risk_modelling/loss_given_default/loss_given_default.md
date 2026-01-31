# Loss Given Default (LGD)

## 1. Concept Skeleton
**Definition:** Proportion of exposure lost when borrower defaults; 1 - recovery rate; LGD = 1 - (recoveries / exposure)  
**Purpose:** Quantify severity of default impact, determine capital required per unit of exposure, price credit spread  
**Prerequisites:** Collateral valuation, recovery processes, portfolio analysis, risk-adjusted return calculation

## 2. Comparative Framing
| Loan Type | Typical LGD | Collateral | Recovery Rate | Key Driver |
|-----------|------------|-----------|---------------|------------|
| **Secured Mortgage** | 10%-25% | Real estate | 75%-90% | Property value decline |
| **Auto Loan** | 20%-35% | Vehicle | 65%-80% | Repossession/sale delays |
| **Unsecured Credit Card** | 70%-100% | None | 0%-30% | Bankruptcy treatment |
| **Corporate Bonds** | 30%-50% | Firm assets | 50%-70% | Capital structure, bankruptcy code |
| **Trade Finance** | 5%-20% | Inventory/goods | 80%-95% | Collateral liquidity |

## 3. Examples + Counterexamples

**Simple Example:**  
$100K mortgage, property worth $120K at default. After sale costs (5%), net recovery $114K. LGD = (100-114)/100 = -14% (gain!) → Use LGD=0

**Failure Case:**  
Assuming constant LGD across economic cycle. 2008: Real estate LGD doubled as property prices fell 50%. Fixed models missed this

**Edge Case:**  
Unsecured loan during pandemic; borrower has no recovery value but later returns to work. Time-dependent LGD; recovery may take years

## 4. Layer Breakdown
```
Loss Given Default Framework:
├─ LGD Components:
│   ├─ Collateral value at default: Market price at time of loss
│   ├─ Recovery amount: Proceeds from liquidation
│   ├─ Recovery costs: Legal, administrative, sales friction
│   ├─ Recovery timing: When received (present value discount)
│   └─ Seniority: Priority in bankruptcy (affects recovery rank)
├─ Types of Recovery:
│   ├─ Collateral sales: Secured assets liquidated
│   ├─ Debt restructuring: Waive/extend obligations
│   ├─ Guarantees: Third-party payment
│   └─ Bankruptcy proceeds: Distribution from estate
├─ LGD Dynamics:
│   ├─ Pro-cyclical: LGD rises in downturns (collateral values fall)
│   ├─ Correlation with PD: High default rates + low recoveries compound losses
│   └─ Workout duration: Short-term vs long-term recovery timelines
├─ LGD Levels by Collateral:
│   ├─ High recovery (LGD 5-20%): Liquid collateral (cash, securities)
│   ├─ Medium recovery (LGD 20-50%): Real estate, inventory
│   ├─ Low recovery (LGD 50-100%): Unsecured, subordinated debt
│   └─ Recovery hierarchy: Senior secured → unsecured → subordinated
└─ Valuation Methods:
    ├─ Appraisal: Professional assessment
    ├─ Market-based: Comparable sales
    ├─ Income approach: Cash flow valuation
    └─ Liquidation value: Fire-sale price
```

## 5. Mini-Project
Simulate LGD under different collateral scenarios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Mortgage pool simulation
print("=== Mortgage LGD Analysis ===")
n_mortgages = 1000
mortgages = pd.DataFrame({
    'original_balance': np.random.normal(300000, 100000, n_mortgages),
    'ltv_at_origination': np.random.uniform(0.70, 0.95, n_mortgages),
    'property_type': np.random.choice(['Single Family', 'Condo', 'Multi-family'], n_mortgages),
    'months_seasoned': np.random.uniform(0, 360, n_mortgages)
})

# Ensure positive loan amounts
mortgages['original_balance'] = mortgages['original_balance'].abs()

# Property value appreciation/depreciation (economic cycle)
economic_condition = np.random.choice(['Boom', 'Normal', 'Crisis'], n_mortgages, p=[0.2, 0.5, 0.3])
value_change = np.where(economic_condition == 'Boom', 1.15,
                       np.where(economic_condition == 'Normal', 1.02,
                               np.random.uniform(0.7, 0.85, n_mortgages)))

mortgages['current_property_value'] = mortgages['original_balance'] / mortgages['ltv_at_origination'] * value_change

# Loan balance remaining (simplified amortization)
amortization_factor = 1 - (mortgages['months_seasoned'] / 360) * 0.3
mortgages['current_balance'] = mortgages['original_balance'] * amortization_factor

# Calculate LGD components
sale_costs_pct = 0.08  # 8% for realtor, closing costs
legal_costs_pct = 0.02  # 2% for legal fees
recovery_timeline = np.random.uniform(3, 24, n_mortgages)  # months to sell

mortgages['gross_recovery'] = mortgages['current_property_value']
mortgages['net_recovery'] = mortgages['gross_recovery'] * (1 - sale_costs_pct - legal_costs_pct)
mortgages['lgd'] = np.maximum((mortgages['current_balance'] - mortgages['net_recovery']) / mortgages['current_balance'], 0)

# Summary statistics
print(f"Average LGD: {mortgages['lgd'].mean():.1%}")
print(f"Median LGD: {mortgages['lgd'].median():.1%}")
print(f"75th percentile LGD: {mortgages['lgd'].quantile(0.75):.1%}")
print(f"Percentage with LGD=0 (no loss): {(mortgages['lgd']==0).sum()/len(mortgages):.1%}")

# LGD by economic condition
print("\n=== LGD by Economic Condition ===")
for condition in ['Boom', 'Normal', 'Crisis']:
    lgd_subset = mortgages[mortgages['original_balance'].isin(mortgages['original_balance'].index)]['lgd']
    # Reassign based on condition
    mask = economic_condition == condition
    print(f"{condition:8s}: Mean LGD = {mortgages[mask]['lgd'].mean():.1%}, " + 
          f"95th pctile = {mortgages[mask]['lgd'].quantile(0.95):.1%}")

# Corporate bond recovery analysis
print("\n=== Corporate Bond Recovery Analysis ===")
n_defaults = 100
bonds = pd.DataFrame({
    'debt_amount': np.random.lognormal(15, 2, n_defaults),
    'seniority': np.random.choice(['Senior Secured', 'Senior Unsecured', 'Subordinated'], n_defaults, p=[0.3, 0.5, 0.2]),
    'firm_value': np.random.lognormal(17, 1, n_defaults)
})

# Recovery depends on seniority and firm value
recovery_rates = {'Senior Secured': 0.70, 'Senior Unsecured': 0.40, 'Subordinated': 0.15}
bonds['recovery_rate'] = bonds['seniority'].map(recovery_rates)
bonds['recovery_amount'] = bonds['debt_amount'] * bonds['recovery_rate']
bonds['lgd'] = 1 - bonds['recovery_rate']

print("\n| Seniority          | Recovery Rate | Avg LGD |")
for seniority in ['Senior Secured', 'Senior Unsecured', 'Subordinated']:
    subset = bonds[bonds['seniority'] == seniority]
    print(f"| {seniority:18s} | {subset['recovery_rate'].iloc[0]:12.1%} | {subset['lgd'].iloc[0]:6.1%} |")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: LGD distribution
ax1 = axes[0, 0]
ax1.hist(mortgages['lgd'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(mortgages['lgd'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={mortgages["lgd"].mean():.1%}')
ax1.set_xlabel('LGD')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Mortgage LGD')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: LGD by economic condition
ax2 = axes[0, 1]
conditions = ['Boom', 'Normal', 'Crisis']
lgds_by_condition = [mortgages[economic_condition == c]['lgd'].mean() for c in conditions]
colors_econ = ['green', 'yellow', 'red']
ax2.bar(conditions, lgds_by_condition, color=colors_econ, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Average LGD')
ax2.set_title('LGD by Economic Condition\n(Pro-cyclical behavior)')
ax2.set_ylim([0, max(lgds_by_condition) * 1.2])
for i, lgd in enumerate(lgds_by_condition):
    ax2.text(i, lgd + 0.01, f'{lgd:.1%}', ha='center', va='bottom')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: LGD vs Current LTV
ax3 = axes[0, 2]
current_ltv = mortgages['current_balance'] / mortgages['current_property_value']
ax3.scatter(current_ltv, mortgages['lgd'], alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Current LTV')
ax3.set_ylabel('LGD')
ax3.set_title('LGD vs Current Loan-to-Value\n(Higher LTV = Higher LGD)')
ax3.set_xlim([0, 1.2])
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3)

# Plot 4: Recovery waterfall (example default)
ax4 = axes[1, 0]
example_idx = 0
example = mortgages.iloc[example_idx]
waterfall_values = [example['current_balance'], 
                   -example['current_balance'] + example['gross_recovery'],
                   -(example['gross_recovery'] - example['net_recovery']),
                   example['net_recovery']]
categories = ['Loan\nBalance', 'Property\nValue\nChange', 'Sale &\nLegal Costs', 'Net\nRecovery']
colors_waterfall = ['blue', 'green' if waterfall_values[1] > 0 else 'red', 'red', 'orange']

x_pos = np.arange(len(categories))
cumulative = [0, example['current_balance'], example['gross_recovery'], example['gross_recovery']]
for i, (cat, val, cum, col) in enumerate(zip(categories, waterfall_values[:-1], cumulative, colors_waterfall)):
    ax4.bar(i, abs(val), bottom=min(cum, cum + val), color=col, alpha=0.7, edgecolor='black')

ax4.bar(len(categories)-1, example['net_recovery'], color='orange', alpha=0.7, edgecolor='black')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories)
ax4.set_ylabel('Amount ($)')
ax4.set_title(f'Recovery Waterfall Example\n(LGD={example["lgd"]:.1%})')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Bond recovery by seniority
ax5 = axes[1, 1]
bond_seniorities = ['Senior\nSecured', 'Senior\nUnsecured', 'Subordinated']
recovery_pcts = [recovery_rates['Senior Secured']*100, 
                 recovery_rates['Senior Unsecured']*100,
                 recovery_rates['Subordinated']*100]
lgd_pcts = [100*recovery_rates['Senior Secured'], 
           100*recovery_rates['Senior Unsecured'],
           100*recovery_rates['Subordinated']]

x_bond = np.arange(len(bond_seniorities))
width = 0.35
ax5.bar(x_bond - width/2, recovery_pcts, width, label='Recovery Rate', alpha=0.7, edgecolor='black')
ax5.bar(x_bond + width/2, [100-r for r in recovery_pcts], width, label='LGD', alpha=0.7, edgecolor='black')
ax5.set_ylabel('Percentage (%)')
ax5.set_title('Bond Recovery by Seniority')
ax5.set_xticks(x_bond)
ax5.set_xticklabels(bond_seniorities)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Pro-cyclicality of LGD
ax6 = axes[1, 2]
# Simulate economic conditions over time
time_periods = np.arange(20)
gdp_growth = 2 + 2*np.sin(time_periods*np.pi/10) + np.random.normal(0, 0.5, 20)
property_value_index = 100 * (1 + np.cumsum(gdp_growth-2)/100)
lgd_cycle = 100 - property_value_index/100

ax6.plot(time_periods, property_value_index, 'b-', linewidth=2, label='Property Value Index')
ax6.set_xlabel('Time Period')
ax6.set_ylabel('Index (Blue Axis)')
ax6.set_title('Pro-cyclical LGD\n(LGD rises when property values fall)')
ax6_2 = ax6.twinx()
ax6_2.plot(time_periods, lgd_cycle, 'r-', linewidth=2, label='LGD')
ax6_2.set_ylabel('LGD (Red Axis)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When is LGD estimation problematic?
- **Collateral value volatility**: Real estate, equity collateral highly pro-cyclical; price crashes during defaults
- **Recovery correlation**: PD and LGD often positively correlated (defaults cluster with falling collateral values)
- **Workout uncertainty**: Recovery timelines variable; months to years affect present value significantly
- **Seniority complexity**: Multiple creditors, subordination structures; recovery depends on bankruptcy code
- **Fraud/deterioration**: Collateral may be hidden or rapidly deteriorate post-default (reputational damage)

## 7. Key References
- [Basel III LGD Standards](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory LGD definitions, recovery rates
- [Collateral Valuation Methods](https://en.wikipedia.org/wiki/Collateral_(finance)) - Appraisal approaches, market-based methods
- [Bankruptcy Code Recovery](https://en.wikipedia.org/wiki/Priority_of_claims) - Seniority treatment, creditor hierarchy

---
**Status:** Key severity parameter for credit losses | **Complements:** Credit Risk Definition, PD, EAD, Expected Loss
