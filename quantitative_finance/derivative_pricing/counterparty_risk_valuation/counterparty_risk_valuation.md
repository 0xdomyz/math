# Counterparty Risk Valuation (CVA/DVA/FVA)

## 1. Concept Skeleton
**Definition:** Counterparty risk valuation adjusts derivative value for the risk that one party defaults before contract maturity. Key adjustments include CVA (credit valuation adjustment), DVA (debit valuation adjustment), and FVA (funding valuation adjustment).  
**Purpose:** Incorporate default risk, funding cost, and collateralization into fair value pricing of OTC derivatives.  
**Prerequisites:** Default probability, hazard rates, discounting, exposure profiles, netting/CSA frameworks.

## 2. Comparative Framing
| Adjustment | Perspective | Formula Core | Increases Value? | Notes |
|------------|-------------|--------------|------------------|------|
| **CVA** | Bank loss from counterparty default | (1-R) × EE × PD × DF | Decreases | Counterparty credit risk cost |
| **DVA** | Bank benefit from own default | (1-R) × NE × PD × DF | Increases | Controversial in reporting |
| **FVA** | Funding cost of uncollateralized exposure | Funding spread × EE × DF | Decreases | Depends on funding strategy |
| **Collateralization** | Reduces exposure | EE reduced by collateral | Increases | CSA lowers CVA/FVA |
| **Netting** | Portfolio-level offset | EE of netted portfolio | Increases | Reduces CVA materially |

## 3. Examples + Counterexamples

**Simple Example (CVA):**  
Swap exposure EE=$2M average, counterparty hazard rate 2%, recovery 40%, discount factor 0.95.  
CVA ≈ (1-0.40) × $2M × 2% × 0.95 = $22.8K.

**Counterexample (Ignoring Netting):**  
If exposures are netted across trades, ignoring netting overstates CVA by 30–60%.  
Netting sets reduce EE; CVA should be computed on net portfolio, not trade-by-trade.

**Edge Case (Wrong-Way Risk):**  
Counterparty’s default probability rises when exposure increases (e.g., commodity producer in a falling commodity price).  
Standard CVA underestimates risk unless WWR is modeled.

## 4. Layer Breakdown
```
Counterparty Valuation Adjustments:
├─ Exposure Profile
│   ├─ E(t): Expected Exposure
│   ├─ EE(t): Average positive exposure
│   ├─ EPE(t): High-quantile exposure
│   └─ NE(t): Negative exposure (for DVA)
├─ Default Risk
│   ├─ Hazard rate λ(t)
│   ├─ Survival S(t) = exp(-∫λ dt)
│   └─ Default probability PD(t) = S(t-1) - S(t)
├─ Loss Given Default
│   ├─ Recovery rate R
│   └─ LGD = 1 - R
├─ Discounting
│   ├─ DF(t) = exp(-r t)
│   └─ Risk-free vs funding curves
├─ Portfolio Effects
│   ├─ Netting sets
│   ├─ Collateral thresholds
│   └─ Margin period of risk
└─ Adjustments
    ├─ CVA: Counterparty risk cost
    ├─ DVA: Own credit benefit
    └─ FVA: Funding cost for uncollateralized exposure
```

## 5. Mini-Project
Simulate exposure profiles and compute CVA sensitivity:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Counterparty Risk Valuation (CVA) ===")

# Parameters
notional = 100e6
r = 0.03
recovery = 0.40
hazard = 0.02
n_steps = 20
maturity = 5

# Simulate exposure paths (simplified swap exposure)
paths = 1000
exposure_paths = []

for _ in range(paths):
    exposures = []
    level = 0
    for t in range(n_steps):
        level += np.random.normal(0, 1) * 0.5
        exposure = max(0, level) * 1e6  # scale to $1M
        exposures.append(exposure)
    exposure_paths.append(exposures)

exposure_paths = np.array(exposure_paths)

# Expected Exposure (EE) and Expected Positive Exposure (EPE)
ee = exposure_paths.mean(axis=0)
percentile_95 = np.percentile(exposure_paths, 95, axis=0)

# Default probabilities per step
time_grid = np.linspace(0.25, maturity, n_steps)
survival = np.exp(-hazard * time_grid)
pd_increments = np.append(1, survival[:-1]) - survival

# Discount factors
discount_factors = np.exp(-r * time_grid)

# CVA calculation
cva = np.sum((1 - recovery) * ee * pd_increments * discount_factors)

print(f"CVA (base case): ${cva/1e6:.2f}M")

# Sensitivity to hazard rate
hazard_range = np.linspace(0.005, 0.05, 10)
cva_sens = []

for h in hazard_range:
    survival_h = np.exp(-h * time_grid)
    pd_inc_h = np.append(1, survival_h[:-1]) - survival_h
    cva_h = np.sum((1 - recovery) * ee * pd_inc_h * discount_factors)
    cva_sens.append(cva_h)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Exposure profile
ax1 = axes[0, 0]
ax1.plot(time_grid, ee/1e6, label='EE', linewidth=2.5)
ax1.plot(time_grid, percentile_95/1e6, label='95% PFE', linestyle='--', linewidth=2.0)
ax1.set_title('Exposure Profile')
ax1.set_xlabel('Years')
ax1.set_ylabel('Exposure ($M)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Default probability increments
ax2 = axes[0, 1]
ax2.bar(time_grid, pd_increments*100, color='orange', alpha=0.7, edgecolor='black')
ax2.set_title('Default Probability by Period')
ax2.set_xlabel('Years')
ax2.set_ylabel('Default Probability (%)')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: CVA contribution by time
ax3 = axes[1, 0]
contrib = (1 - recovery) * ee * pd_increments * discount_factors
ax3.bar(time_grid, contrib/1e6, color='green', alpha=0.7, edgecolor='black')
ax3.set_title('CVA Contribution by Time')
ax3.set_xlabel('Years')
ax3.set_ylabel('CVA Contribution ($M)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: CVA sensitivity to hazard
ax4 = axes[1, 1]
ax4.plot(hazard_range*100, np.array(cva_sens)/1e6, linewidth=2.5, color='red')
ax4.set_title('CVA Sensitivity to Hazard Rate')
ax4.set_xlabel('Hazard Rate (%)')
ax4.set_ylabel('CVA ($M)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('counterparty_risk_cva.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Key Insights ===")
print("• CVA depends on both exposure profile and default timing")
print("• Netting/collateral reduces EE and CVA materially")
print("• Higher hazard rates increase CVA almost linearly")
```
