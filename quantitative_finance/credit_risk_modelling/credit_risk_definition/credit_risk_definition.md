# Credit Risk Definition

## 1. Concept Skeleton
**Definition:** Risk of loss from borrower default or deterioration in creditworthiness; core risk in lending institutions  
**Purpose:** Quantify and manage potential losses from credit exposure, inform capital allocation and pricing  
**Prerequisites:** Probability theory, financial instruments, lending basics, risk management concepts

## 2. Comparative Framing
| Risk Type | Credit Risk | Market Risk | Operational Risk | Liquidity Risk |
|-----------|------------|------------|-----------------|----------------|
| **Source** | Counterparty failure | Price movements | Internal failures | Funding constraints |
| **Horizon** | Medium to long-term | Short-term | Variable | Short-term |
| **Correlation** | Business cycles | Systematic factors | Idiosyncratic | Market-driven |
| **Mitigation** | Diversification, collateral | Hedging | Controls, insurance | Reserve buffers |

## 3. Examples + Counterexamples

**Simple Example:**  
Bank lends $100K mortgage at 5%; borrower loses job and misses payments. Credit loss materializes when default occurs

**Failure Case:**  
Assuming zero default risk on AAA-rated bonds during financial crisis. 2008: Lehman Brothers default surprised many risk models

**Edge Case:**  
Government bonds (sovereign credit risk): Technically can default (Argentina 2001) despite perceived safety; not truly risk-free

## 4. Layer Breakdown
```
Credit Risk Ecosystem:
├─ Risk Components (Loss Drivers):
│   ├─ Probability of Default (PD): Likelihood of non-payment
│   ├─ Loss Given Default (LGD): Proportion lost if default occurs
│   ├─ Exposure at Default (EAD): Amount at risk when default happens
│   └─ Maturity (M): Time to potential loss realization
├─ Loss Quantification:
│   ├─ Expected Loss: EL = PD × LGD × EAD (first moment)
│   ├─ Unexpected Loss: Deviation from expected (second moment)
│   └─ Capital Requirement: Regulatory buffer for unexpected losses
├─ Time Dimensions:
│   ├─ Point-in-time (PIT): Current economic conditions
│   ├─ Through-the-cycle (TTC): Long-term average cycle
│   └─ Economic cycle dependency: Impacts both PD and LGD
├─ Borrower Segments:
│   ├─ Retail: Individual consumers
│   ├─ SME: Small and medium enterprises
│   ├─ Corporate: Large companies
│   └─ Sovereign: Government borrowers
└─ Manifestations:
    ├─ Default: Contractual obligation breach
    ├─ Downgrade: Credit rating deterioration
    ├─ Spread widening: Higher yield demanded
    └─ Recovery: Partial or full compensation post-default
```

## 5. Mini-Project
Analyze credit risk components across loan types:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define typical credit risk profiles by loan type
loan_types = {
    'Government Bond': {'PD': 0.001, 'LGD': 0.40, 'EAD': 1000000},
    'AAA Corporate': {'PD': 0.005, 'LGD': 0.30, 'EAD': 5000000},
    'BBB Corporate': {'PD': 0.020, 'LGD': 0.40, 'EAD': 5000000},
    'High-Yield Bond': {'PD': 0.080, 'LGD': 0.50, 'EAD': 3000000},
    'Prime Auto Loan': {'PD': 0.015, 'LGD': 0.20, 'EAD': 25000},
    'Subprime Auto Loan': {'PD': 0.080, 'LGD': 0.35, 'EAD': 20000},
    'Prime Mortgage': {'PD': 0.005, 'LGD': 0.25, 'EAD': 400000},
    'Subprime Mortgage': {'PD': 0.050, 'LGD': 0.40, 'EAD': 350000}
}

# Calculate expected loss
results = []
for loan_type, params in loan_types.items():
    pd_val = params['PD']
    lgd_val = params['LGD']
    ead_val = params['EAD']
    el = pd_val * lgd_val * ead_val
    
    results.append({
        'Loan Type': loan_type,
        'PD (%)': pd_val * 100,
        'LGD (%)': lgd_val * 100,
        'EAD': ead_val,
        'Expected Loss': el,
        'EL per $1K': el / ead_val * 1000
    })

df = pd.DataFrame(results)

print("=== Credit Risk Across Loan Types ===")
print(df.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Expected Loss by Loan Type
ax1 = axes[0, 0]
colors = ['green' if el < 1000 else 'orange' if el < 5000 else 'red' 
          for el in df['Expected Loss']]
ax1.barh(df['Loan Type'], df['Expected Loss'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Expected Loss ($)')
ax1.set_title('Expected Loss by Loan Type\n(EL = PD × LGD × EAD)')
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Risk Components
ax2 = axes[0, 1]
x = np.arange(len(df))
width = 0.25
ax2.bar(x - width, df['PD (%)'], width, label='PD (%)', alpha=0.7, edgecolor='black')
ax2.bar(x, df['LGD (%)'], width, label='LGD (%)', alpha=0.7, edgecolor='black')
ax2.bar(x + width, df['EL per $1K'] * 10, width, label='EL per $1K × 10', alpha=0.7, edgecolor='black')
ax2.set_ylabel('Percentage / Scaled Value')
ax2.set_title('Credit Risk Components Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Loan Type'], rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: PD vs LGD scatter
ax3 = axes[1, 0]
scatter = ax3.scatter(df['PD (%)'], df['LGD (%)'], s=df['EAD']/10000, 
                     alpha=0.6, edgecolors='black', linewidth=1)
for idx, row in df.iterrows():
    ax3.annotate(row['Loan Type'], (row['PD (%)'], row['LGD (%)']),
                fontsize=8, ha='right')
ax3.set_xlabel('Probability of Default (%)')
ax3.set_ylabel('Loss Given Default (%)')
ax3.set_title('PD vs LGD (Bubble size = EAD)')
ax3.grid(True, alpha=0.3)

# Plot 4: Risk decomposition
ax4 = axes[1, 1]
# Show why subprime loans are riskier
risk_drivers = ['Government\nBond', 'Prime\nMortgage', 'Subprime\nMortgage']
pd_vals = [0.001, 0.005, 0.050]
lgd_vals = [0.40, 0.25, 0.40]
el_vals = [pd * lgd for pd, lgd in zip(pd_vals, lgd_vals)]

x_pos = np.arange(len(risk_drivers))
bars1 = ax4.bar(x_pos, [p*100 for p in pd_vals], label='PD (%)', 
               alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x_pos, [l*100 for l in lgd_vals], bottom=[p*100 for p in pd_vals],
               label='LGD (%)', alpha=0.7, edgecolor='black')

ax4.set_ylabel('Risk Contribution (%)')
ax4.set_title('Risk Decomposition: PD vs LGD Impact')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(risk_drivers)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, el in enumerate(el_vals):
    ax4.text(i, el*100 + 1, f'EL={el*100:.2f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# Time series of default rates (economic cycle)
print("\n=== Default Rate Variation Over Economic Cycle ===")
years = np.arange(2000, 2021)
# Simulated default rate with business cycle
cycle_default = 2 + 1.5 * np.sin(2*np.pi*years/10) + 0.5 * np.random.normal(0, 1, len(years))
cycle_default = np.maximum(cycle_default, 0.5)  # Floor at 0.5%

plt.figure(figsize=(12, 5))
plt.plot(years, cycle_default, 'o-', linewidth=2, markersize=6)
plt.axhspan(3.5, 6, alpha=0.2, color='red', label='Crisis Period')
plt.axhspan(0.5, 2.5, alpha=0.2, color='green', label='Normal Period')
plt.xlabel('Year')
plt.ylabel('Corporate Default Rate (%)')
plt.title('Default Rate Variation: Point-in-Time vs Through-the-Cycle')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("PIT rates spike in downturns; TTC rates average over full cycle")
```

## 6. Challenge Round
When is credit risk definition insufficient?
- **Correlation dynamics**: Linear PD/LGD assumptions break down during crises; defaults cluster
- **Systemic risk**: Portfolio diversification fails during systemic events (2008 financial crisis)
- **Emerging risks**: Climate change, cybersecurity, geopolitical events not captured in historical data
- **Model risk**: Reliance on rating agencies that may be slow to downgrade (Enron, Lehman Brothers)
- **Reverse causality**: Credit losses impact economic conditions, which further deteriorate credit quality

## 7. Key References
- [Basel III Credit Risk Framework](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory definition, PD/LGD/EAD specifications
- [Credit Risk Modelling Theory](https://www.investopedia.com/terms/c/creditrisk.asp) - Comprehensive overview of concepts
- [Altman Z-Score](https://en.wikipedia.org/wiki/Altman_Z-score) - Early distress prediction model, credit risk signal

---
**Status:** Foundational concept for all credit analysis | **Complements:** PD, LGD, EAD, Expected Loss
