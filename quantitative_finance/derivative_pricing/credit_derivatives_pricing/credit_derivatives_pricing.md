# Credit Derivatives Pricing (CDS Fundamentals)

## 1. Concept Skeleton
**Definition:** Credit derivatives transfer credit risk without transferring ownership. The most common is the CDS, where protection buyer pays a spread to receive payout if default occurs.  
**Purpose:** Hedge credit exposure, speculate on default risk, and isolate credit risk from interest rate risk.  
**Prerequisites:** Default probability, hazard rates, recovery, discounting, swap pricing.

## 2. Comparative Framing
| Instrument | Cash Flow Structure | Key Pricing Driver | Typical Use | Risk |
|------------|---------------------|--------------------|-------------|------|
| **CDS** | Premium leg vs protection leg | Hazard rate, recovery | Hedge credit risk | Counterparty + basis risk |
| **Bond** | Coupons + principal | Yield curve, credit spread | Long credit | Interest rate + credit |
| **Total Return Swap** | Total return vs funding | Spread + funding | Synthetic exposure | Funding risk |
| **CDO Tranche** | Structured loss sharing | Default correlation | Risk transfer | Correlation/structural risk |

## 3. Examples + Counterexamples

**Simple Example (Par CDS Spread):**  
If hazard rate λ=2%, recovery=40%, risk-free rate 3%, a 5y CDS spread is roughly:  
Spread ≈ (1-R) × λ = 0.60 × 2% = 1.2% (120 bps).  

**Counterexample (Ignoring Discounting):**  
Ignoring discount factors overstates the CDS spread, especially for longer maturities.

**Edge Case (Front-Loaded Default Risk):**  
If default risk is concentrated in early years, spread must be higher than constant λ approximation.

## 4. Layer Breakdown
```
CDS Pricing Framework:
├─ Survival Curve
│   ├─ Hazard rate λ(t)
│   ├─ Survival S(t) = exp(-∫λ dt)
│   └─ Default probability = S(t-1) - S(t)
├─ Premium Leg
│   ├─ Spread × Notional × DF(t) × Survival
│   ├─ Paid until default or maturity
│   └─ PV_premium = Σ spread × DF × survival
├─ Protection Leg
│   ├─ LGD × Notional × DF × default_prob
│   └─ PV_protection = Σ LGD × DF × PD
├─ Par Spread
│   ├─ Set PV_premium = PV_protection
│   └─ Spread = PV_protection / PV_premium_annuity
└─ Basis & Adjustments
    ├─ CDS-Bond basis
    ├─ Counterparty risk (CVA)
    └─ Liquidity premiums
```

## 5. Mini-Project
Compute par CDS spreads and sensitivity to hazard rates:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=== Credit Derivatives Pricing (CDS) ===")

# Parameters
notional = 10e6
recovery = 0.40
r = 0.03
maturity = 5

# Hazard rates to test
hazard_rates = np.linspace(0.005, 0.05, 10)

# Time grid (quarterly payments)
steps = maturity * 4
time_grid = np.linspace(0.25, maturity, steps)

# Discount factors
df = np.exp(-r * time_grid)

spreads = []

for h in hazard_rates:
    survival = np.exp(-h * time_grid)
    pd_increments = np.append(1, survival[:-1]) - survival

    # Premium leg annuity
    premium_leg = np.sum(df * survival) * 0.25

    # Protection leg
    protection_leg = np.sum((1 - recovery) * df * pd_increments)

    spread = protection_leg / premium_leg
    spreads.append(spread)

# Example calculation
h_example = 0.02
survival_ex = np.exp(-h_example * time_grid)
pd_ex = np.append(1, survival_ex[:-1]) - survival_ex
premium_leg_ex = np.sum(df * survival_ex) * 0.25
protection_leg_ex = np.sum((1 - recovery) * df * pd_ex)
spread_ex = protection_leg_ex / premium_leg_ex

print(f"Par CDS Spread (hazard 2%): {spread_ex*10000:.0f} bps")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival curve
ax1 = axes[0, 0]
ax1.plot(time_grid, survival_ex, linewidth=2.5, color='blue')
ax1.set_title('Survival Curve (Hazard 2%)')
ax1.set_xlabel('Years')
ax1.set_ylabel('Survival Probability')
ax1.grid(True, alpha=0.3)

# Plot 2: Default probability increments
ax2 = axes[0, 1]
ax2.bar(time_grid, pd_ex*100, color='orange', alpha=0.7, edgecolor='black')
ax2.set_title('Default Probability by Period')
ax2.set_xlabel('Years')
ax2.set_ylabel('Default Probability (%)')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Spread vs hazard rate
ax3 = axes[1, 0]
ax3.plot(hazard_rates*100, np.array(spreads)*10000, linewidth=2.5, color='green')
ax3.set_title('CDS Spread vs Hazard Rate')
ax3.set_xlabel('Hazard Rate (%)')
ax3.set_ylabel('Spread (bps)')
ax3.grid(True, alpha=0.3)

# Plot 4: Premium vs protection leg
ax4 = axes[1, 1]
ax4.bar(['Premium Leg', 'Protection Leg'], [premium_leg_ex, protection_leg_ex],
        color=['steelblue', 'red'], alpha=0.7, edgecolor='black')
ax4.set_title('Leg PV Comparison (Hazard 2%)')
ax4.set_ylabel('PV')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cds_pricing.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Key Insights ===")
print("• Par CDS spread balances expected default losses and premium payments")
print("• Higher hazard rates increase spreads nearly linearly")
print("• Discounting reduces long-dated premium leg contributions")
```
