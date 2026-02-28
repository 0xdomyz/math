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
Bank holds $100M AAA corporate bond. Basel I: 8% Ã— $100M = $8M capital required. Basel II Standardized: Risk weight = 20% (vs 20% Basel I), so 8% Ã— 20% Ã— $100M = $1.6M capital. More risk-sensitive.

**High-Grade vs Low-Grade:**  
$100M portfolio: 50% AAA (RW=20%) + 50% BBB (RW=100%). Basel I: Flat 8% Ã— $100M = $8M. Basel II Standardized: [0.5 Ã— 20% + 0.5 Ã— 100%] Ã— 8% Ã— $100M = $4.8M. Still simpler than IRB.

**Internal Ratings-Based (IRB):**  
Bank's proprietary model: PD=2%, LGD=40%, EAD=$50M. Basel II IRB: Calculate risk weight via formula using PD/LGD â†’ say RW=80%. Capital = 8% Ã— 80% Ã— $50M = $3.2M. Bank's own risk estimates replace regulatory assumptions.

**Problem: Pro-Cyclicality:**  
Boom time: Credit spreads tight, PD estimates low. IRB capital requirements low â†’ banks lend more â†’ boom amplified. Bust time: Spreads widen, PD spikes. Capital required spikes â†’ forced deleveraging â†’ bust amplified. Basel II's risk-sensitivity created feedback loop.

**Mortgage Portfolio:**  
Prime mortgages: 35% LTV, RW=35%. Capital = 1.6%. Alt-A mortgages: 85% LTV, RW=100%. Capital = 4%. Seemed risk-appropriate pre-2008. But correlated collapse: Alt-A + Prime both defaulted when housing fell. Diversification within portfolio failed.

**Operational Risk (New in Basel II):**  
Before: Capital only for credit/market risk. Basel II: Added charge for operational failures (fraud, system failure, legal risk). Estimated via three methods: Basic Indicator, Standardized, Advanced Measurement Approach (AMA).

## 4. Layer Breakdown
```
Basel II Framework Architecture:

â”œâ”€ Three Pillars:
â”‚  â”œâ”€ Pillar I: Minimum Capital Requirements
â”‚  â”‚   â”œâ”€ Credit Risk:
â”‚  â”‚   â”‚   â”œâ”€ Standardized Approach: Regulatory risk weights
â”‚  â”‚   â”‚   â”œâ”€ Foundation IRB: Bank provides PD, regulator sets LGD/EAD
â”‚  â”‚   â”‚   â””â”€ Advanced IRB: Bank provides all (PD/LGD/EAD)
â”‚  â”‚   â”œâ”€ Market Risk:
â”‚  â”‚   â”‚   â”œâ”€ Standard method: Fixed percentages by asset class
â”‚  â”‚   â”‚   â””â”€ Internal Models: Bank's VaR calculation
â”‚  â”‚   â””â”€ Operational Risk:
â”‚  â”‚       â”œâ”€ Basic Indicator: 15% of gross income
â”‚  â”‚       â”œâ”€ Standardized: 12-18% of gross income by business line
â”‚  â”‚       â””â”€ Advanced Measurement (AMA): Bank's loss data model
â”‚  â”œâ”€ Pillar II: Supervisory Review
â”‚  â”‚   â”œâ”€ Bank's Internal Capital Adequacy Assessment Process (ICAAP)
â”‚  â”‚   â”œâ”€ Regulator evaluates adequacy beyond Pillar I minimum
â”‚  â”‚   â”œâ”€ Stress testing, concentration risk, interest rate risk
â”‚  â”‚   â””â”€ Pillar II Guidance (P2G) may require higher capital
â”‚  â””â”€ Pillar III: Market Discipline
â”‚      â”œâ”€ Public disclosure of capital position
â”‚      â”œâ”€ Risk exposures (credit, market, operational)
â”‚      â”œâ”€ Risk management framework
â”‚      â””â”€ Transparency enables market monitoring
â”œâ”€ Credit Risk: Standardized Approach
â”‚  â”œâ”€ Risk Weights by Asset Class & Counterparty Rating:
â”‚  â”‚   â”œâ”€ Sovereign risk: 0% (AAA-AA), 20% (A-BBB), 50% (BB-B), 100% (Below B)
â”‚  â”‚   â”œâ”€ Bank/Corporate: 20% (AAA-AA), 50% (A), 100% (BBB-unrated), 150% (Below BBB)
â”‚  â”‚   â”œâ”€ Retail mortgages: 35%
â”‚  â”‚   â”œâ”€ Retail other: 75%
â”‚  â”‚   â””â”€ Unrated: 100% default (conservative)
â”‚  â”œâ”€ Adjustment Factors:
â”‚  â”‚   â”œâ”€ Collateral: Reduce RW if secured
â”‚  â”‚   â”œâ”€ Guarantees: Risk weight of guarantor
â”‚  â”‚   â”œâ”€ Credit derivatives: Protection provider's RW
â”‚  â”‚   â””â”€ Supervisory haircuts: Volatility-based adjustments
â”‚  â””â”€ Formula:
â”‚      Capital (Pillar I) = 8% Ã— Î£(RW_i Ã— Exposure_i)
â”œâ”€ Credit Risk: Internal Ratings-Based (IRB)
â”‚  â”œâ”€ Foundation IRB:
â”‚  â”‚   â”œâ”€ Bank estimates: PD only
â”‚  â”‚   â”œâ”€ Regulator provides: LGD, EAD, correlation parameters
â”‚  â”‚   â”œâ”€ Risk Weight Formula: Function of PD, LGD, EAD, correlation
â”‚  â”‚   â””â”€ Capital = 8% Ã— Î£(RW_i(PD_i) Ã— EAD_i)
â”‚  â”œâ”€ Advanced IRB:
â”‚  â”‚   â”œâ”€ Bank estimates: PD, LGD, EAD, correlation all
â”‚  â”‚   â”œâ”€ Requires 5+ years of historical data
â”‚  â”‚   â”œâ”€ Robust backtesting of estimates
â”‚  â”‚   â””â”€ Higher capital requirements for poor track records
â”‚  â”œâ”€ Risk Weight Function:
â”‚  â”‚   RW(PD, LGD, EAD, M, correlation) â‰ˆ
â”‚  â”‚   [LGD Ã— N((Î¦â»Â¹(PD) + âˆšcorrelation Ã— Z) / âˆš(1 - correlation)) 
â”‚  â”‚    - LGD Ã— PD] Ã— (1 + (M - 2.5) Ã— b) / (1 - 1.5 Ã— b)
â”‚  â”‚   Where:
â”‚  â”‚   - N = cumulative normal distribution
â”‚  â”‚   - M = maturity factor (~1 for retail, 1-5 for corp)
â”‚  â”‚   - b = maturity adjustment (formula depends on PD)
â”‚  â”‚   - correlation â‰ˆ 0.12 (corporate), 0.04 (retail)
â”‚  â”œâ”€ PD Estimation Methods:
â”‚  â”‚   â”œâ”€ Scorecard models: Logistic regression on historical defaults
â”‚  â”‚   â”œâ”€ Expert judgment: Credit analyst override
â”‚  â”‚   â”œâ”€ Market-implied: CDS spreads or bond yields
â”‚  â”‚   â””â”€ Hybrid: Combining multiple approaches
â”‚  â”œâ”€ LGD Estimation:
â”‚  â”‚   â”œâ”€ Collateralized loans: Loss = Max(0, Exposure - Collateral_Value)
â”‚  â”‚   â”œâ”€ Unsecured: Historical recovery rates (typically 30-50%)
â”‚  â”‚   â”œâ”€ Stressed LGD: At-crisis recovery (may be 10-20% lower)
â”‚  â”‚   â””â”€ Downturn LGD: Worst economic conditions (required for Pillar I)
â”‚  â”œâ”€ EAD Estimation:
â”‚  â”‚   â”œâ”€ Term loans: EAD = Drawn amount (100% of principal)
â”‚  â”‚   â”œâ”€ Revolving credit: EAD = Drawn + (Undrawn Ã— Credit Conversion Factor)
â”‚  â”‚   â”œâ”€ CCF typically 20-75% (higher for commitments closer to drawdown)
â”‚  â”‚   â””â”€ Derivatives: EAD = Replacement cost + Potential future exposure
â”‚  â””â”€ Validation Requirements:
â”‚      â”œâ”€ Backtesting: Compare predicted defaults vs actual
â”‚      â”œâ”€ Benchmarking: PD estimates consistent with external ratings
â”‚      â”œâ”€ Stability analysis: Parameters stable over time
â”‚      â””â”€ Stress testing: Performance under adverse scenarios
â”œâ”€ Operational Risk
â”‚  â”œâ”€ Definition:
â”‚  â”‚   Risk of loss from inadequate/failed internal processes,
â”‚  â”‚   people, systems, or external events
â”‚  â”‚   Excludes strategic & reputational risk
â”‚  â”œâ”€ Loss Categories:
â”‚  â”‚   â”œâ”€ Internal fraud: Employee misconduct, theft
â”‚  â”‚   â”œâ”€ External fraud: Robbery, forgery, cyber attack
â”‚  â”‚   â”œâ”€ Employment practices: Discrimination, unsafe work environment
â”‚  â”‚   â”œâ”€ Clients/Products: Errors, mis-selling, product flaws
â”‚  â”‚   â”œâ”€ Damage to assets: Natural disasters, terrorism
â”‚  â”‚   â”œâ”€ Business disruption: System failure, supply chain disruption
â”‚  â”‚   â””â”€ Execution/Delivery: Transaction error, counterparty failure
â”‚  â”œâ”€ Capital Calculation Methods:
â”‚  â”‚   â”œâ”€ Basic Indicator Approach:
â”‚  â”‚   â”‚   OpRisk_Capital = 15% Ã— Average(Gross_Income_3_years)
â”‚  â”‚   â”‚   Simplest, least risk-sensitive
â”‚  â”‚   â”œâ”€ Standardized Approach:
â”‚  â”‚   â”‚   OpRisk_Capital = Î£(Î²_i Ã— Gross_Income_i) for each business line
â”‚  â”‚   â”‚   Î² coefficients: 12-18% by business line (higher for trading)
â”‚  â”‚   â””â”€ Advanced Measurement Approach (AMA):
â”‚  â”‚       Bank uses loss data + scenario analysis + controls
â”‚  â”‚       Combines: Historical losses + Scenarios + Control indicators
â”‚  â”‚       Requires regulatory approval
â”‚  â””â”€ Data Requirements (AMA):
â”‚      â”œâ”€ Internal loss data: 5-10 year history minimum
â”‚      â”œâ”€ External data: Industry losses from consortiums
â”‚      â”œâ”€ Scenario analysis: Expert estimates of potential losses
â”‚      â””â”€ Controls/Risk drivers: Correlation with losses
â”œâ”€ Transitional Provisions
â”‚  â”œâ”€ Adoption Timeline:
â”‚  â”‚   â”œâ”€ 2004: Basel II finalized
â”‚  â”‚   â”œâ”€ 2006: Initial implementation (Europe, major banks)
â”‚  â”‚   â”œâ”€ 2008: Financial crisis interrupts full rollout
â”‚  â”‚   â”œâ”€ 2009+: Transition to Basel III (supersedes Basel II)
â”‚  â”œâ”€ Grandfathering:
â”‚  â”‚   â”œâ”€ Basel I floor: Capital can't fall below Basel I level
â”‚  â”‚   â”œâ”€ Phase-in of new rules (multi-year ramp)
â”‚  â”‚   â””â”€ Transitional arrangements for lesser-developed countries
â”‚  â””â”€ Parallel Run:
â”‚      â”œâ”€ Banks calculate both Basel I and Basel II
â”‚      â”œâ”€ Report both to regulators
â”‚      â”œâ”€ Gradual shift to Basel II requirement
â””â”€ Issues & Criticisms
   â”œâ”€ Pro-Cyclicality:
   â”‚   â”œâ”€ Good times: Low PD, low capital â†’ excess lending
   â”‚   â”œâ”€ Bad times: High PD, high capital â†’ credit crunch
   â”‚   â”œâ”€ Amplifies boom-bust cycles
   â”‚   â””â”€ Basel III added countercyclical buffer to mitigate
   â”œâ”€ Calibration Risk:
   â”‚   â”œâ”€ Risk weights based on historical relationships
   â”‚   â”œâ”€ Correlations spike in crisis â†’ RWs underestimate tail risk
   â”‚   â”œâ”€ Correlation estimates may be wrong â†’ capital mispriced
   â”‚   â””â”€ 2008 showed many "low-risk" mortgages defaults together
   â”œâ”€ Model Risk:
   â”‚   â”œâ”€ IRB complexity â†’ more parameter estimates â†’ more error
   â”‚   â”œâ”€ Banks incentivized to minimize PD/LGD â†’ lower capital
   â”‚   â”œâ”€ Regulatory scrutiny of models insufficient pre-2008
   â”‚   â””â”€ Model validation lag behind model sophistication
   â”œâ”€ Complexity:
   â”‚   â”œâ”€ Basel II extremely detailed (1000s of pages)
   â”‚   â”œâ”€ Implementation complex, expensive for banks
   â”‚   â”œâ”€ Regulatory arbitrage opportunities (exploit loopholes)
   â”‚   â””â”€ Simplified approaches available but less risk-accurate
   â”œâ”€ Interconnectedness Blind Spot:
   â”‚   â”œâ”€ Addresses single-bank capital, not system-wide risk
   â”‚   â”œâ”€ Procyclical behavior of all banks together â†’ systemic crisis
   â”‚   â”œâ”€ 2008: Lehman failure cascaded through interconnected network
   â”‚   â””â”€ Basel III adds systemic risk overlay (not in Basel II)
   â””â”€ Application Gaps:
       â”œâ”€ Shadow banking not covered (now regulatory gap)
       â”œâ”€ Derivatives valuation models had faults (CVA risk)
       â”œâ”€ Counterparty risk concentration underestimated
       â””â”€ Behavioral aspects (herding, model consensus) not captured
```

**Interaction:** Credit risk classification â†’ Risk weight or IRB PD/LGD/EAD â†’ Capital requirement (8% Ã— RW Ã— Exposure) â†’ Supervisory review (Pillar II) â†’ Market disclosure (Pillar III).

## 5. Challenge Round
- Calculate capital requirement for corporate loan using Basel II IRB (given PD, LGD, EAD)
- Compare capital: Standardized approach vs IRB for retail mortgage portfolio
- Design collateral adjustment: How does collateral reduce RW under Basel II?
- Analyze pro-cyclicality: Model capital requirement as PD rises in recession
- Explain Foundation vs Advanced IRB: Trade-offs in data requirements vs capital savings

## 6. Key References
- [Basel Committee, "Basel II: International Convergence of Capital Measurement and Capital Standards" (2004)](https://www.bis.org/publ/bcbs107.pdf) â€” Official framework
- [BIS, "Basel II Framework" (https://www.bis.org/basel_framework/)](https://www.bis.org/basel_framework/) â€” Complete regulatory text
- [Crouhy et al, "The Essentials of Risk Management" (2014)](https://www.mheducation.com/) â€” Practical guide
- [Jones, "Operational Risk" (2009)](https://www.wiley.com/en-us/Operational+Risk-p-9780470516782) â€” OpRisk in detail

---
**Status:** International regulatory standard 2004-2008 | **Complements:** Basel I, Basel III, Credit Risk Modeling, IRB Approaches
