# Basel II Framework

## 1. Concept Skeleton
**Definition:** International regulatory framework establishing minimum capital standards for banks; introduces three pillars (minimum capital requirements, supervisory review, market discipline) with risk-sensitive approach  
**Purpose:** Strengthen banking system stability; align regulatory capital with actual risk exposure; accommodate advanced risk management techniques; reduce pro-cyclicality of capital requirements  
**Prerequisites:** Credit risk modeling, market risk, operational risk, regulatory capital concepts, bank balance sheets

## 2. Comparative Framing
| Aspect | Basel I | Basel II | Basel III |
|--------|---------|---------|----------|
| **Risk Sensitivity** | Low (flat 8% for most assets) | High (PD/LGD/EAD based) | Very High (CVaR, leverage ratio) |
| **Capital Requirement** | 8% fixed | 8% risk-weighted | 10.5% minimum (with buffers) |
| **Risk Approaches** | Standardized only | Standardized + IRB | Both + simplified |
| **Operational Risk** | Not included | Explicit charge | Enhanced methodology |
| **Market Risk** | Minimal coverage | Integrated | Stressed calibration |
| **Pillar Structure** | Implicit | Explicit three-pillar | Maintained + enhanced |
| **Adoption** | 1988-2006 | 2004-2007 (phased) | 2008-ongoing |
| **Pro-Cyclicality** | Moderate | High (amplified cycles) | Reduced (countercyclical buffer) |

## 3. Examples + Counterexamples

**Simple Capital Calc (Basel II Standardized):**  
Bank holds $100M AAA corporate bond. Basel I: 8% × $100M = $8M capital required. Basel II Standardized: Risk weight = 20% (vs 20% Basel I), so 8% × 20% × $100M = $1.6M capital. More risk-sensitive.

**High-Grade vs Low-Grade:**  
$100M portfolio: 50% AAA (RW=20%) + 50% BBB (RW=100%). Basel I: Flat 8% × $100M = $8M. Basel II Standardized: [0.5 × 20% + 0.5 × 100%] × 8% × $100M = $4.8M. Still simpler than IRB.

**Internal Ratings-Based (IRB):**  
Bank's proprietary model: PD=2%, LGD=40%, EAD=$50M. Basel II IRB: Calculate risk weight via formula using PD/LGD → say RW=80%. Capital = 8% × 80% × $50M = $3.2M. Bank's own risk estimates replace regulatory assumptions.

**Problem: Pro-Cyclicality:**  
Boom time: Credit spreads tight, PD estimates low. IRB capital requirements low → banks lend more → boom amplified. Bust time: Spreads widen, PD spikes. Capital required spikes → forced deleveraging → bust amplified. Basel II's risk-sensitivity created feedback loop.

**Mortgage Portfolio:**  
Prime mortgages: 35% LTV, RW=35%. Capital = 1.6%. Alt-A mortgages: 85% LTV, RW=100%. Capital = 4%. Seemed risk-appropriate pre-2008. But correlated collapse: Alt-A + Prime both defaulted when housing fell. Diversification within portfolio failed.

**Operational Risk (New in Basel II):**  
Before: Capital only for credit/market risk. Basel II: Added charge for operational failures (fraud, system failure, legal risk). Estimated via three methods: Basic Indicator, Standardized, Advanced Measurement Approach (AMA).

## 4. Layer Breakdown
```
Basel II Framework Architecture:

├─ Three Pillars:
│  ├─ Pillar I: Minimum Capital Requirements
│  │   ├─ Credit Risk:
│  │   │   ├─ Standardized Approach: Regulatory risk weights
│  │   │   ├─ Foundation IRB: Bank provides PD, regulator sets LGD/EAD
│  │   │   └─ Advanced IRB: Bank provides all (PD/LGD/EAD)
│  │   ├─ Market Risk:
│  │   │   ├─ Standard method: Fixed percentages by asset class
│  │   │   └─ Internal Models: Bank's VaR calculation
│  │   └─ Operational Risk:
│  │       ├─ Basic Indicator: 15% of gross income
│  │       ├─ Standardized: 12-18% of gross income by business line
│  │       └─ Advanced Measurement (AMA): Bank's loss data model
│  ├─ Pillar II: Supervisory Review
│  │   ├─ Bank's Internal Capital Adequacy Assessment Process (ICAAP)
│  │   ├─ Regulator evaluates adequacy beyond Pillar I minimum
│  │   ├─ Stress testing, concentration risk, interest rate risk
│  │   └─ Pillar II Guidance (P2G) may require higher capital
│  └─ Pillar III: Market Discipline
│      ├─ Public disclosure of capital position
│      ├─ Risk exposures (credit, market, operational)
│      ├─ Risk management framework
│      └─ Transparency enables market monitoring
├─ Credit Risk: Standardized Approach
│  ├─ Risk Weights by Asset Class & Counterparty Rating:
│  │   ├─ Sovereign risk: 0% (AAA-AA), 20% (A-BBB), 50% (BB-B), 100% (Below B)
│  │   ├─ Bank/Corporate: 20% (AAA-AA), 50% (A), 100% (BBB-unrated), 150% (Below BBB)
│  │   ├─ Retail mortgages: 35%
│  │   ├─ Retail other: 75%
│  │   └─ Unrated: 100% default (conservative)
│  ├─ Adjustment Factors:
│  │   ├─ Collateral: Reduce RW if secured
│  │   ├─ Guarantees: Risk weight of guarantor
│  │   ├─ Credit derivatives: Protection provider's RW
│  │   └─ Supervisory haircuts: Volatility-based adjustments
│  └─ Formula:
│      Capital (Pillar I) = 8% × Σ(RW_i × Exposure_i)
├─ Credit Risk: Internal Ratings-Based (IRB)
│  ├─ Foundation IRB:
│  │   ├─ Bank estimates: PD only
│  │   ├─ Regulator provides: LGD, EAD, correlation parameters
│  │   ├─ Risk Weight Formula: Function of PD, LGD, EAD, correlation
│  │   └─ Capital = 8% × Σ(RW_i(PD_i) × EAD_i)
│  ├─ Advanced IRB:
│  │   ├─ Bank estimates: PD, LGD, EAD, correlation all
│  │   ├─ Requires 5+ years of historical data
│  │   ├─ Robust backtesting of estimates
│  │   └─ Higher capital requirements for poor track records
│  ├─ Risk Weight Function:
│  │   RW(PD, LGD, EAD, M, correlation) ≈
│  │   [LGD × N((Φ⁻¹(PD) + √correlation × Z) / √(1 - correlation)) 
│  │    - LGD × PD] × (1 + (M - 2.5) × b) / (1 - 1.5 × b)
│  │   Where:
│  │   - N = cumulative normal distribution
│  │   - M = maturity factor (~1 for retail, 1-5 for corp)
│  │   - b = maturity adjustment (formula depends on PD)
│  │   - correlation ≈ 0.12 (corporate), 0.04 (retail)
│  ├─ PD Estimation Methods:
│  │   ├─ Scorecard models: Logistic regression on historical defaults
│  │   ├─ Expert judgment: Credit analyst override
│  │   ├─ Market-implied: CDS spreads or bond yields
│  │   └─ Hybrid: Combining multiple approaches
│  ├─ LGD Estimation:
│  │   ├─ Collateralized loans: Loss = Max(0, Exposure - Collateral_Value)
│  │   ├─ Unsecured: Historical recovery rates (typically 30-50%)
│  │   ├─ Stressed LGD: At-crisis recovery (may be 10-20% lower)
│  │   └─ Downturn LGD: Worst economic conditions (required for Pillar I)
│  ├─ EAD Estimation:
│  │   ├─ Term loans: EAD = Drawn amount (100% of principal)
│  │   ├─ Revolving credit: EAD = Drawn + (Undrawn × Credit Conversion Factor)
│  │   ├─ CCF typically 20-75% (higher for commitments closer to drawdown)
│  │   └─ Derivatives: EAD = Replacement cost + Potential future exposure
│  └─ Validation Requirements:
│      ├─ Backtesting: Compare predicted defaults vs actual
│      ├─ Benchmarking: PD estimates consistent with external ratings
│      ├─ Stability analysis: Parameters stable over time
│      └─ Stress testing: Performance under adverse scenarios
├─ Operational Risk
│  ├─ Definition:
│  │   Risk of loss from inadequate/failed internal processes,
│  │   people, systems, or external events
│  │   Excludes strategic & reputational risk
│  ├─ Loss Categories:
│  │   ├─ Internal fraud: Employee misconduct, theft
│  │   ├─ External fraud: Robbery, forgery, cyber attack
│  │   ├─ Employment practices: Discrimination, unsafe work environment
│  │   ├─ Clients/Products: Errors, mis-selling, product flaws
│  │   ├─ Damage to assets: Natural disasters, terrorism
│  │   ├─ Business disruption: System failure, supply chain disruption
│  │   └─ Execution/Delivery: Transaction error, counterparty failure
│  ├─ Capital Calculation Methods:
│  │   ├─ Basic Indicator Approach:
│  │   │   OpRisk_Capital = 15% × Average(Gross_Income_3_years)
│  │   │   Simplest, least risk-sensitive
│  │   ├─ Standardized Approach:
│  │   │   OpRisk_Capital = Σ(β_i × Gross_Income_i) for each business line
│  │   │   β coefficients: 12-18% by business line (higher for trading)
│  │   └─ Advanced Measurement Approach (AMA):
│  │       Bank uses loss data + scenario analysis + controls
│  │       Combines: Historical losses + Scenarios + Control indicators
│  │       Requires regulatory approval
│  └─ Data Requirements (AMA):
│      ├─ Internal loss data: 5-10 year history minimum
│      ├─ External data: Industry losses from consortiums
│      ├─ Scenario analysis: Expert estimates of potential losses
│      └─ Controls/Risk drivers: Correlation with losses
├─ Transitional Provisions
│  ├─ Adoption Timeline:
│  │   ├─ 2004: Basel II finalized
│  │   ├─ 2006: Initial implementation (Europe, major banks)
│  │   ├─ 2008: Financial crisis interrupts full rollout
│  │   ├─ 2009+: Transition to Basel III (supersedes Basel II)
│  ├─ Grandfathering:
│  │   ├─ Basel I floor: Capital can't fall below Basel I level
│  │   ├─ Phase-in of new rules (multi-year ramp)
│  │   └─ Transitional arrangements for lesser-developed countries
│  └─ Parallel Run:
│      ├─ Banks calculate both Basel I and Basel II
│      ├─ Report both to regulators
│      ├─ Gradual shift to Basel II requirement
└─ Issues & Criticisms
   ├─ Pro-Cyclicality:
   │   ├─ Good times: Low PD, low capital → excess lending
   │   ├─ Bad times: High PD, high capital → credit crunch
   │   ├─ Amplifies boom-bust cycles
   │   └─ Basel III added countercyclical buffer to mitigate
   ├─ Calibration Risk:
   │   ├─ Risk weights based on historical relationships
   │   ├─ Correlations spike in crisis → RWs underestimate tail risk
   │   ├─ Correlation estimates may be wrong → capital mispriced
   │   └─ 2008 showed many "low-risk" mortgages defaults together
   ├─ Model Risk:
   │   ├─ IRB complexity → more parameter estimates → more error
   │   ├─ Banks incentivized to minimize PD/LGD → lower capital
   │   ├─ Regulatory scrutiny of models insufficient pre-2008
   │   └─ Model validation lag behind model sophistication
   ├─ Complexity:
   │   ├─ Basel II extremely detailed (1000s of pages)
   │   ├─ Implementation complex, expensive for banks
   │   ├─ Regulatory arbitrage opportunities (exploit loopholes)
   │   └─ Simplified approaches available but less risk-accurate
   ├─ Interconnectedness Blind Spot:
   │   ├─ Addresses single-bank capital, not system-wide risk
   │   ├─ Procyclical behavior of all banks together → systemic crisis
   │   ├─ 2008: Lehman failure cascaded through interconnected network
   │   └─ Basel III adds systemic risk overlay (not in Basel II)
   └─ Application Gaps:
       ├─ Shadow banking not covered (now regulatory gap)
       ├─ Derivatives valuation models had faults (CVA risk)
       ├─ Counterparty risk concentration underestimated
       └─ Behavioral aspects (herding, model consensus) not captured
```

**Interaction:** Credit risk classification → Risk weight or IRB PD/LGD/EAD → Capital requirement (8% × RW × Exposure) → Supervisory review (Pillar II) → Market disclosure (Pillar III).

## 5. Mini-Project
Calculate minimum capital requirement under Basel II Standardized Approach:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bank portfolio (simplified)
portfolio = [
    {'description': 'US Treasury Bonds', 'exposure': 50, 'rating': 'AAA', 'rw': 0.00},
    {'description': 'Corporate Bonds (A-rated)', 'exposure': 100, 'rating': 'A', 'rw': 0.50},
    {'description': 'Corporate Bonds (BBB-rated)', 'exposure': 80, 'rating': 'BBB', 'rw': 1.00},
    {'description': 'Mortgages (Prime)', 'exposure': 200, 'rating': 'Secured', 'rw': 0.35},
    {'description': 'Mortgages (Subprime)', 'exposure': 150, 'rating': 'Secured (Low LTV)', 'rw': 0.75},
    {'description': 'Business Loans (Small firms)', 'exposure': 120, 'rating': 'Unrated', 'rw': 1.00},
    {'description': 'Derivatives (Credit exposure)', 'exposure': 30, 'rating': 'BBB', 'rw': 1.00},
    {'description': 'Equity Holdings', 'exposure': 60, 'rating': 'Unrated', 'rw': 1.50},
]

df = pd.DataFrame(portfolio)

# Calculate risk-weighted assets
df['risk_weighted_exposure'] = df['exposure'] * df['rw']

# Minimum capital ratio (Pillar I)
min_capital_ratio = 0.08

# Calculate capital requirements
df['capital_required'] = df['risk_weighted_exposure'] * min_capital_ratio

# Summary calculations
total_exposure = df['exposure'].sum()
total_rwa = df['risk_weighted_exposure'].sum()
total_capital_required = df['capital_required'].sum()

print("="*100)
print("BASEL II STANDARDIZED APPROACH - CAPITAL REQUIREMENT CALCULATION")
print("="*100)
print(f"\nBank Portfolio (Millions USD):\n")
print(df[['description', 'exposure', 'rating', 'rw', 'risk_weighted_exposure', 'capital_required']].to_string(index=False))

print(f"\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"Total Exposure: ${total_exposure:.1f}M")
print(f"Total Risk-Weighted Assets (RWA): ${total_rwa:.1f}M")
print(f"Risk-Weighted Ratio: {total_rwa / total_exposure * 100:.1f}% (overall risk weight)")
print(f"Minimum Capital Requirement (8% of RWA): ${total_capital_required:.1f}M")
print(f"Required Capital Ratio: {total_capital_required / total_exposure * 100:.1f}% of total exposure")

# Scenario: Adding Pillar II buffer
pillar_ii_buffer = 0.025  # 2.5% for supervisory discretion
pillar_ii_capital = (min_capital_ratio + pillar_ii_buffer) * total_rwa

print(f"\nWith Pillar II Guidance Buffer ({pillar_ii_buffer*100:.1f}%): ${pillar_ii_capital:.1f}M")
print(f"Capital Ratio (Pillar I + Pillar II): {(min_capital_ratio + pillar_ii_buffer)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Risk weight distribution
ax = axes[0, 0]
colors_rw = plt.cm.RdYlGn_r(df['rw'] / df['rw'].max())
ax.barh(df['description'], df['rw'], color=colors_rw, alpha=0.8)
ax.set_xlabel('Risk Weight')
ax.set_title('Risk Weight by Asset Type (Basel II Standardized)')
ax.grid(alpha=0.3, axis='x')

# Plot 2: Risk-weighted assets breakdown
ax = axes[0, 1]
colors_assets = plt.cm.Set3(np.linspace(0, 1, len(df)))
ax.pie(df['risk_weighted_exposure'], labels=df['description'], autopct='%1.1f%%',
      colors=colors_assets, startangle=90)
ax.set_title('Risk-Weighted Assets (RWA) Composition')

# Plot 3: Capital requirement by asset
ax = axes[1, 0]
ax.bar(range(len(df)), df['capital_required'], color=colors_assets, alpha=0.8)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['description'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Capital Required ($M)')
ax.set_title('Capital Requirement by Asset Type')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Exposure vs Risk-Weighted Assets
ax = axes[1, 1]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['exposure'], width, label='Total Exposure', alpha=0.8)
ax.bar(x + width/2, df['risk_weighted_exposure'], width, label='Risk-Weighted Assets', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df['description'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Amount ($M)')
ax.set_title('Exposure vs Risk-Weighted Assets')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Sensitivity: What if risk weights change?
print(f"\n" + "="*100)
print("SENSITIVITY ANALYSIS: Impact of Risk Weight Changes")
print("="*100)

scenarios = {
    'Baseline': 1.0,
    'Conservative (+50% RW)': 1.5,
    'Aggressive (-30% RW)': 0.7,
}

for scenario_name, multiplier in scenarios.items():
    rwa_scenario = (df['rw'] * multiplier * df['exposure']).sum()
    capital_scenario = rwa_scenario * min_capital_ratio
    print(f"{scenario_name}: RWA = ${rwa_scenario:.1f}M, Capital = ${capital_scenario:.1f}M")
```

## 6. Challenge Round
- Calculate capital requirement for corporate loan using Basel II IRB (given PD, LGD, EAD)
- Compare capital: Standardized approach vs IRB for retail mortgage portfolio
- Design collateral adjustment: How does collateral reduce RW under Basel II?
- Analyze pro-cyclicality: Model capital requirement as PD rises in recession
- Explain Foundation vs Advanced IRB: Trade-offs in data requirements vs capital savings

## 7. Key References
- [Basel Committee, "Basel II: International Convergence of Capital Measurement and Capital Standards" (2004)](https://www.bis.org/publ/bcbs107.pdf) — Official framework
- [BIS, "Basel II Framework" (https://www.bis.org/basel_framework/)](https://www.bis.org/basel_framework/) — Complete regulatory text
- [Crouhy et al, "The Essentials of Risk Management" (2014)](https://www.mheducation.com/) — Practical guide
- [Jones, "Operational Risk" (2009)](https://www.wiley.com/en-us/Operational+Risk-p-9780470516782) — OpRisk in detail

---
**Status:** International regulatory standard 2004-2008 | **Complements:** Basel I, Basel III, Credit Risk Modeling, IRB Approaches
