# Basel III Framework

## 1. Concept Skeleton
**Definition:** Enhanced international regulatory framework post-2008 crisis; strengthens minimum capital requirements, introduces liquidity standards, adds macroprudential buffers, requires leverage ratio and stress testing  
**Purpose:** Prevent banking system-wide collapses through higher capital/liquidity; reduce pro-cyclicality via countercyclical buffers; address systemically important institutions; capture tail risk and correlations; enforce living wills  
**Prerequisites:** Basel II, credit/market/operational risk, liquidity risk, systemic risk, regulatory capital, stress testing frameworks

## 2. Comparative Framing
| Aspect | Basel II | Basel III | Post-Crisis (2024) |
|--------|---------|----------|-------------------|
| **Minimum Capital** | 8% total, 4% Tier 1 | 10.5% total, 7% Tier 1 | 10.5% total + buffers |
| **Liquidity Standards** | None | LCR (30-day) + NSFR (1-year) | Enhanced, stress-dependent |
| **Leverage Ratio** | None (VaR-based) | 3% non-risk-weighted | 3% + 50% surcharge for systemic banks |
| **Countercyclical Buffer** | None | 0-2.5% in booms | 0-5% (country/time dependent) |
| **G-SIB Surcharge** | None | 1-3.5% for systemic banks | 1.5-3.5% (bank-specific) |
| **CVaR Charge** | Not captured | Introduced (Stressed | Enhanced (Expected Shortfall) |
| **Correlation Assumptions** | Static | Updated annually | Dynamic stress-derived |
| **Stress Testing** | Recommended | Annual Fed stress tests (US) | Regular (quarterly updates) |
| **Resolution Planning** | Not required | Required for large banks | Enhanced (living wills) |

## 3. Examples + Counterexamples

**Capital Ratio Increase:**  
2006 major bank: 8% Tier 1, 12% total (Basel II minimum). 2023 same bank: Must hold 7% Tier 1 + 2.5% capital buffer + 1.5% G-SIB surcharge = 11% Tier 1 equivalent. 4x leverage reduction → safer but higher cost.

**Pro-Cyclicality Addressed:**  
Boom: Bank decides countercyclical buffer = 1% (regulator sets). Capital requirement: 7% base + 2.5% capital buffer + 1% CyCB = 10.5%. High but discipline against excess lending. Bust: Regulator releases CyCB → 7% + 2.5% = 9.5%. Smaller but some relief (vs forced tightening).

**Liquidity Coverage Ratio (LCR):**  
Bank's 30-day liquidity stress test. Scenario: 30% deposit outflow + derivatives collateral call. Required: High-Quality Liquid Assets (HQLA) ≥ Net Cash Outflow. If shortfall, bank must hold more Treasury/CB deposits (less profitable but safe).

**Systemically Important Bank (G-SIB):**  
JPMorgan: Size $3.7T, interconnectedness, substitutability high → G-SIB bucket 5 (4-4.5% capital surcharge). Community bank: $5B assets → no surcharge. Different risk profiles = different capital rules.

**Stress Test Failure:**  
Fed stress test: Severe recession scenario (unemployment +5%, house prices -35%, equity -50%). Bank A projected: $20B loss, capital ratio falls to 2% (below 7% minimum). Result: "Capital plan rejected" → must raise equity or cut dividends immediately.

**CVaR (Expected Shortfall) vs VaR:**  
Basel II: Market risk capital = VaR(99%,10-day). Basel III: Adds CVaR charge = average of tail losses. Result: Capital nearly doubles for some trading desks (tail risk penalty). Tail hedging becomes mandatory.

**Leverage Ratio (Non-Risk-Weighted Floor):**  
Bank: $100B assets, 80% "low-risk" (RW=20%), 20% "high-risk" (RW=100%). RWA = 36B. Risk-weighted capital = 8% × 36B = 2.88B. Leverage ratio floor: 3% × $100B = 3B. Leverage ratio is binding (more restrictive than RW). Forces de-leveraging regardless of mix.

## 4. Layer Breakdown
```
Basel III Framework Architecture:

├─ Pillar I: Minimum Capital Requirements (Enhanced)
│  ├─ Capital Definitions (Tier-1 vs Tier-2 refined):
│  │   ├─ Common Equity Tier 1 (CET1):
│  │   │   ├─ Common shares, retained earnings (core capital)
│  │   │   ├─ Must represent at least 4.5% of RWA (hard floor)
│  │   │   └─ Fully loss-absorbing, most subordinated
│  │   ├─ Tier 1 Capital:
│  │   │   ├─ CET1 + Additional Tier 1 (AT1) instruments
│  │   │   ├─ AT1: Perpetual subordinated bonds (contingent write-down)
│  │   │   ├─ Must represent at least 6% of RWA
│  │   │   └─ Write-down at CET1 ratio ≤ 5.125%
│  │   └─ Tier 2 Capital:
│  │       ├─ Subordinated debt (5-10 year maturity)
│  │       ├─ Loan loss reserves (limited to 1.25% RWA)
│  │       ├─ Total capital = Tier 1 + Tier 2 ≥ 10.5% RWA
│  │       └─ Tier 2 limited to 50% of Tier 1
│  ├─ Capital Buffers (Pillars I+II):
│  │   ├─ Capital Conservation Buffer (CCB):
│  │   │   ├─ 2.5% of RWA (additional requirement)
│  │   │   ├─ All banks must hold (no discretion)
│  │   │   ├─ Prevents dividend/compensation when buffer breached
│  │   │   └─ Phased in: 0.625% (2011) → 2.5% (2019)
│  │   ├─ Countercyclical Buffer (CyCB):
│  │   │   ├─ 0-2.5% of RWA (set by national regulators)
│  │   │   ├─ Activated in credit booms (prevent leverage buildup)
│  │   │   ├─ Released in busts (allow lending room)
│  │   │   ├─ Example: COVID 2020 → CyCB released 0→0 (already 0)
│  │   │   └─ Pre-2008 would have been set to 1.5-2% in boom
│  │   ├─ G-SIB Capital Buffer:
│  │   │   ├─ 1-3.5% of RWA (based on systemic importance score)
│  │   │   ├─ Annual ranking by Fed/national regulator
│  │   │   ├─ Score factors: Size, interconnectedness, complexity, substitutability
│  │   │   └─ Only for systemically important banks
│  │   └─ Other Systemically Important Institution (O-SIB) Buffer:
│  │       ├─ Similar concept, applies to non-bank systemic entities
│  │       ├─ Insurance companies, market infrastructure
│  │       └─ Country-specific identification
│  ├─ Capital Requirement Floors:
│  │   ├─ Total Capital: 10.5% RWA minimum
│  │   │   └─ 4.5% CET1 + 1.5% AT1 + 2.5% Tier2 + 2% CCB + 0-2.5% CyCB + 0-3.5% G-SIB
│  │   ├─ Tier 1 Capital: 8.5% RWA minimum
│  │   ├─ CET1 Capital: 7% RWA minimum
│  │   └─ Typical large bank: 12-15% CET1 (well above minimum)
│  ├─ Risk-Weighted Assets (RWA) Calculation (Refined):
│  │   ├─ Credit Risk:
│  │   │   ├─ Standardized Approach (unchanged from Basel II)
│  │   │   ├─ Foundation/Advanced IRB (enhanced calibration)
│  │   │   └─ IRB floor: 72.5% of Standardized RW (prevents gaming)
│  │   ├─ Market Risk (Fundamental Review):
│  │   │   ├─ Replaces Basel II VaR with Expected Shortfall (CVaR)
│  │   │   ├─ 10-day horizon, 99% confidence
│  │   │   ├─ Adds stressed ES (using crisis calibration)
│  │   │   ├─ Incremental Risk Charge (IRC): Jump-to-default risk
│  │   │   ├─ Comprehensive Risk Measure (CRM): Non-linear derivatives
│  │   │   └─ Typically doubles capital for trading desks vs Basel II
│  │   ├─ Operational Risk:
│  │   │   ├─ Standardized Approach (replaces AMA for most):
│  │   │   │   OpRisk = 12% × [Indicator] × [Loss Component Factor]
│  │   │   │   Indicator = average revenue over 3 years
│  │   │   ├─ Advanced Approach (limited banks only):
│  │   │   │   OpRisk = 9.5% × [Expected Loss] + [CVaR-weighted Tail]
│  │   │   │   ILM adjustment (Internal Loss Multiplier) for severe events
│  │   │   └─ Credit valuation adjustment (CVA) risk:
│  │   │       Risk that counterparty becomes less creditworthy
│  │   │       Separate capital charge on derivatives positions
│  │   └─ Floor: RWA ≥ 72.5% of Standardized RWA (binding for complex banks)
│  └─ Output Floor (As of 2023):
│      ├─ Finalized output floor: 72.5%
│      ├─ Means: RWA(Internal models) ≥ 72.5% × RWA(Standardized)
│      ├─ Prevents excessive RWA reduction from IRB optimization
│      └─ Effective 2028 (phase-in period extended)
├─ Pillar II: Supervisory Review (Enhanced)
│  ├─ Internal Capital Adequacy Assessment (ICAAP):
│  │   ├─ Banks must model own capital needs over 3-year horizon
│  │   ├─ Includes: Credit, market, operational, concentration, interest rate risks
│  │   ├─ Stress scenarios designed by bank (regulator can override)
│  │   ├─ Board-level review and approval required
│  │   └─ Submitted to regulator for evaluation
│  ├─ Supervisory Stress Testing (CCAR/DFAST in US):
│  │   ├─ Fed-designed scenarios: Baseline, adverse, severely adverse
│  │   ├─ Banks run models on own portfolios → report results
│  │   ├─ Fed compares results for reasonableness
│  │   ├─ "Fail" results → reject capital plans (dividends/buybacks frozen)
│  │   ├─ Annual exercise, published results
│  │   └─ Increasingly stringent assumptions (tail risk focus)
│  ├─ Pillar II Guidance (P2G):
│  │   ├─ Regulator-set additional capital if ICAAP/stress test inadequate
│  │   ├─ Discretionary add-on (not formulaic like CCB/CyCB)
│  │   ├─ Addresses: Concentration, interconnectedness, business model risk
│  │   ├─ Example: Bank heavily concentrated in commercial real estate → +2% P2G
│  │   └─ Can be released if risk profile improves
│  ├─ Concentration Risk (New Focus):
│  │   ├─ Single counterparty exposure limit: 10-25% depending on bank size
│  │   ├─ Large exposure framework: Capital charge if exposure >10% Tier1
│  │   ├─ Sector concentration: Monitored separately
│  │   └─ Interconnectedness: Factor into required capital
│  └─ Other Risks:
│      ├─ Interest Rate Risk (in the Banking Book): Non-trading portfolio
│      ├─ Business Model Risk: Profitability sustainability
│      ├─ Reputational Risk: Fines, loss of market confidence
│      └─ Macroeconomic Risk: System-wide stresses
├─ Pillar III: Market Discipline (Enhanced Disclosure)
│  ├─ Quantitative Disclosure (Quarterly):
│  │   ├─ Capital composition (CET1, Tier 1, Total)
│  │   ├─ Capital ratios vs requirements
│  │   ├─ RWA breakdown (credit, market, operational)
│  │   ├─ Leverage ratio
│  │   ├─ LCR, NSFR (liquidity metrics)
│  │   └─ Remuneration, risk-weighted positions
│  ├─ Qualitative Disclosure:
│  │   ├─ Risk governance framework
│  │   ├─ Risk management policies by risk type
│  │   ├─ Stress testing methodology
│  │   ├─ Concentration risk disclosures
│  │   └─ Regulatory framework compliance
│  ├─ Standardized Templates (COREP/FINREP):
│  │   ├─ Regulatory technical standards (EBA, OCC, others)
│  │   ├─ Machines-readable formats (XML)
│  │   ├─ Enables direct comparison across banks
│  │   └─ Public vs confidential component (sensitive data protected)
│  └─ Transparency Goals:
│      ├─ Investors assess capital adequacy
│      ├─ Depositors/creditors price risk accurately
│      ├─ Peer comparison drives discipline
│      ├─ Regulatory arbitrage reduced
│      └─ Data-driven market monitoring
├─ Liquidity Standards (New - Major Innovation)
│  ├─ Liquidity Coverage Ratio (LCR):
│  │   ├─ Ensure bank survives 30-day stress scenario
│  │   ├─ Formula: LCR = HQLA / Net Cash Outflow ≥ 100%
│  │   ├─ HQLA: High-Quality Liquid Assets (Level 1/2)
│  │   │   ├─ Level 1: Cash, CB reserves, government bonds (no haircut)
│  │   │   ├─ Level 2a: AAA/AA bank/corporate bonds (15% haircut)
│  │   │   └─ Level 2b: BBB+ corporate, equities (25-50% haircut)
│  │   ├─ Net Cash Outflow: Liability runoff + collateral needs
│  │   ├─ Example: Bank $100B assets, $80B deposits (100% runoff), $20B commitments
│  │   │   Required HQLA = $100B, bank has $30B cash + $80B Treasuries = $110B ✓
│  │   ├─ Phase-in: 60% (2015) → 100% (2019)
│  │   └─ More restrictive for investment banks
│  ├─ Net Stable Funding Ratio (NSFR):
│  │   ├─ Ensure structural funding stability (1-year horizon)
│  │   ├─ Formula: NSFR = Available Stable Funding / Required Stable Funding ≥ 100%
│  │   ├─ Stable Funding (ASF): Long-term liabilities, core deposits
│  │   │   ├─ Deposits from retail customers: 90% ASF
│  │   ├─ Stable Funding (RSF): Difficulty to convert to cash
│  │   │   ├─ Illiquid assets (commercial loans): 85% RSF
│  │   │   ├─ Unencumbered corporates: 50% RSF
│  │   │   └─ Encumbered assets: 100% RSF
│  │   ├─ Example: Bank funding = $50B retail deposits (90% stable) + $30B wholesale (30% stable)
│  │   │   ASF = 0.9×$50B + 0.3×$30B = $54B
│  │   │   RSF = 0.85×$50B + 0.5×$40B + 1.0×$10B = 75B
│  │   │   NSFR = $54B / $75B = 72% ✗ (below 100%, needs stable funding)
│  │   ├─ Phase-in: 80% (2018) → 100% (2020)
│  │   └─ Discourages reliance on wholesale funding
│  └─ Intraday Liquidity Monitoring:
│      ├─ Track daily cash flows
│      ├─ Ensure capability to meet obligations throughout day
│      ├─ Critical for systemically important payment systems
│      └─ Real-time management, not regulatory floor
├─ Leverage Ratio (Non-Risk-Weighted Floor)
│  ├─ Purpose:
│  │   ├─ Backstop to risk-weighted capital (limits RWA gaming)
│  │   ├─ Ensures banks don't take excessive unweighted risk
│  │   ├─ Binding for some institutions (high leverage users)
│  ├─ Definition:
│  │   ├─ Leverage Ratio = Tier 1 Capital / Exposure Measure
│  │   ├─ Exposure measure ≈ Total assets + derivatives + commitments (minimal haircuts)
│  │   ├─ Minimum = 3% (proposed enhancement to 3.6% for systemic banks)
│  ├─ Example (Illustrative):
│  │   Bank: Tier 1 = $20B, Total Assets = $500B
│  │   LR = $20B / $500B = 4% ✓ (above 3% floor)
│  ├─ Comparison to Risk-Weighted:
│  │   If RWA = 250B (50% of assets): RW-based Tier 1 req = 8% × $250B = $20B ✓
│  │   LR floor achieved same capital, different basis
│  └─ Effectiveness:
│      ├─ Prevents high-leverage strategies (carry trades, dark pooling)
│      ├─ Less risk-sensitive but more transparent
│      ├─ Often non-binding (RW capital more restrictive for normal portfolios)
├─ Macroprudential Tools & Resolution
│  ├─ Countercyclical Capital Buffer (CyCB):
│  │   ├─ National regulator sets 0-2.5% (up to 5% with approval)
│  │   ├─ Activation: Credit growth signals boom (e.g., >15% annual)
│  │   ├─ Deactivation: Credit stress signals bust (e.g., defaults spike)
│  │   ├─ Example: 2012 EU CyCB most set to 0%, 2013+ gradually increased to 0.5-1%
│  │   └─ Goal: Brake on lending in good times, relief in bad times
│  ├─ Systemic Risk Buffer:
│  │   ├─ National regulator discretion
│  │   ├─ Applied to systemically important institutions
│  │   ├─ Can vary by sector (e.g., real estate concentration)
│  │   └─ Prevents local financial instability
│  ├─ Resolution & Recovery Planning (RRP):
│  │   ├─ Living Wills: Banks must show how to unwind without systemic harm
│  │   ├─ Resolvability: Regulators test annual ("Is this bank dissolvable?")
│  │   ├─ Recovery Plans: How bank raises capital/liquidity in stress
│  │   ├─ Barriers to Resolution: Legal, structural (spin-offs required if barrier too high)
│  │   └─ Goal: Avoid Lehman-type contagion
│  ├─ Total Loss-Absorbing Capacity (TLAC):
│  │   ├─ G-SIBs must have 16-18% of RWA in capital/debt that can absorb losses
│  │   ├─ Ensures sufficient loss buffer in resolution
│  │   ├─ Bail-in provisions: Creditors share losses before public/taxpayer
│  │   └─ Reduces moral hazard ("too big to fail" expectations)
│  └─ Single Counterparty Exposure:
│      ├─ Limit large exposure to one counterparty
│      ├─ Large exposure = >10% Tier1 capital
│      ├─ Capital charge on excess: Up to 50%
│      ├─ Prevents concentration risk contagion
│      └─ Example: Bank with $30B Tier1, Large client = $3B exposure OK, $4B faces charge
├─ Implementation Timeline
│  ├─ 2010-2019: Phased in globally
│  │   ├─ 2010: Framework agreed (G20 in Seoul)
│  │   ├─ 2011: US, EU, Japan, etc. begin rule-making
│  │   ├─ 2012: Capital buffers partial (0.625%)
│  │   ├─ 2014: LCR phased to 60%
│  │   ├─ 2019: NSFR = 100%, CCB = 2.5%
│  │   └─ 2019: Endgame issued (output floor finalized)
│  ├─ 2020-2027: Post-COVID & Endgame Implementation
│  │   ├─ 2020: COVID capital relief (CyCB released)
│  │   ├─ 2021-2027: Output floor phase-in
│  │   ├─ 2024-2028: Stress testing enhancement
│  │   └─ Ongoing: Regulatory refinement based on feedback
│  └─ Transition Provisions:
│      ├─ Grandfather clauses for pre-existing instruments
│      ├─ Phase-out of hybrid capital (partially replaced by AT1)
│      ├─ Multi-year transition periods (avoid cliff effects)
│      └─ Flexibility for less developed countries
├─ Criticisms & Challenges
│  ├─ Complexity:
│  │   ├─ Multiple buffers, LCR, NSFR, Leverage → hard to understand
│  │   ├─ Regulatory arbitrage opportunities (find loopholes)
│  │   ├─ Compliance costs (technology, staffing, data systems)
│  │   └─ Smaller banks struggle relative to large banks
│  ├─ Capital Flight to Shadow Banking:
│  │   ├─ Tighter regulations → shift to less-regulated entities
│  │   ├─ Private equity, hedge funds grow while banks contract
│  │   ├─ Systemic risk moves rather than reduced
│  │   └─ Regulatory arbitrage between jurisdictions
│  ├─ Pro-Cyclicality Residual:
│  │   ├─ CyCB helps but discretion leads to late activation
│  │   ├─ G-SIB surcharge may increase during stress (more complex → higher risk)
│  │   ├─ Procyclical components remain despite improvements
│  │   └─ 2023: Rising rates → mark-to-market losses spike CET1 requirements
│  ├─ LCR Constraints:
│  │   ├─ HQLA scarcity post-2008 (not enough government bonds)
│  │   ├─ LCR forces banks into low-return safe assets
│  │   ├─ Liquidity trade-off: Safe but expensive funding
│  │   └─ NSFR discourages secured financing (repo), impacts liquidity provision
│  ├─ Model Risk (IRB Approaches Remain):
│  │   ├─ Output floor reduces (but doesn't eliminate) IRB gaming
│  │   ├─ Banks incentivized to minimize RWA within 72.5% floor
│  │   ├─ Parameter manipulation (PD, LGD, correlation)
│  │   └─ Regulators may tighten further (regulatory capital arbitrage wars)
│  └─ Unintended Consequences:
│      ├─ Liquidity coverage focus on stable funding → less financial system efficiency
│      ├─ Leverage ratio floor may be binding (non-risk-sensitive) → mispricingof risk
│      ├─ Interconnectedness definition drives capital → could reduce market liquidity
│      └─ Regulatory complexity → smaller competitive disadvantage
```

**Interaction:** Enhanced capital → Liquidity standards → Supervisory review (stress test) → Market discipline (disclosure) → Resolution framework → Macroprudential tools (countercyclical buffers).

## 5. Mini-Project
Run stress test on bank capital ratios under Basel III:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bank starting position (Basel III)
tier1_capital = 50  # $50B
tier2_capital = 20  # $20B
rwa = 400  # $400B risk-weighted assets
total_assets = 1000  # $1000B total assets

# Baseline ratios
baseline = {
    'CET1': tier1_capital / rwa,
    'Tier1': tier1_capital / rwa,
    'Total': (tier1_capital + tier2_capital) / rwa,
    'Leverage': tier1_capital / total_assets,
}

print("="*100)
print("BASEL III CAPITAL RATIO STRESS TEST")
print("="*100)
print(f"\nBank Starting Position:")
print(f"  Tier 1 Capital: ${tier1_capital}B")
print(f"  Tier 2 Capital: ${tier2_capital}B")
print(f"  Risk-Weighted Assets: ${rwa}B")
print(f"  Total Assets: ${total_assets}B")

print(f"\nBaseline Ratios:")
for metric, value in baseline.items():
    print(f"  {metric}: {value*100:.2f}%")

# Basel III Minimums
minimums = {
    'CET1': 0.045,
    'Tier1': 0.065,
    'Total': 0.105,
    'Leverage': 0.03,
}

# Stress scenarios
scenarios = {
    'Mild Downturn': {
        'tier1_loss': -5,  # -$5B
        'rwa_increase': 50,  # +$50B (asset quality deteriorates)
        'tier2_loss': -2,  # -$2B
    },
    'Moderate Recession': {
        'tier1_loss': -15,
        'rwa_increase': 150,
        'tier2_loss': -5,
    },
    'Severe Crisis': {
        'tier1_loss': -25,
        'rwa_increase': 250,
        'tier2_loss': -15,
    },
}

results = {}

for scenario_name, shocks in scenarios.items():
    t1 = tier1_capital + shocks['tier1_loss']
    t2 = tier2_capital + shocks['tier2_loss']
    rwa_stressed = rwa + shocks['rwa_increase']
    
    ratios = {
        'CET1': t1 / rwa_stressed,
        'Tier1': t1 / rwa_stressed,
        'Total': (t1 + t2) / rwa_stressed,
        'Leverage': t1 / total_assets,
    }
    
    # Check vs minimums
    passes = {k: ratios[k] >= minimums[k] for k in minimums}
    
    results[scenario_name] = {
        'ratios': ratios,
        'passes': passes,
        'capital': t1,
        'rwa': rwa_stressed,
    }

# Print results
print(f"\n" + "="*100)
print("STRESS TEST RESULTS")
print("="*100)

for scenario_name, scenario_results in results.items():
    print(f"\n{scenario_name}:")
    print(f"  Tier 1 Capital: ${scenario_results['capital']:.1f}B")
    print(f"  Risk-Weighted Assets: ${scenario_results['rwa']:.1f}B")
    for metric in minimums.keys():
        stressed_ratio = scenario_results['ratios'][metric]
        minimum_ratio = minimums[metric]
        status = "✓ PASS" if scenario_results['passes'][metric] else "✗ FAIL"
        shortfall = (minimum_ratio - stressed_ratio) * scenario_results['rwa']
        print(f"    {metric}: {stressed_ratio*100:.2f}% (min: {minimum_ratio*100:.2f}%) {status}")
        if not scenario_results['passes'][metric]:
            print(f"      Capital shortfall: ${shortfall:.1f}B")

# Add buffers (Capital Conservation Buffer, CyCB)
print(f"\n" + "="*100)
print("WITH REGULATORY BUFFERS (CCB + CyCB)")
print("="*100)

buffers = {
    'Capital Conservation Buffer (CCB)': 0.025,
    'Countercyclical Buffer (CyCB)': 0.010,
    'G-SIB Buffer': 0.015,
}

total_buffer = sum(buffers.values())
print(f"\nTotal Buffer: {total_buffer*100:.2f}%")

# Check vs total buffer requirement
print(f"\nSevere Crisis Scenario with Buffers:")
stressed_tier1 = results['Severe Crisis']['capital']
stressed_rwa = results['Severe Crisis']['rwa']
total_min_with_buffers = minimums['Total'] + total_buffer
stressed_total_ratio = (stressed_tier1 + tier2_capital + shocks['tier2_loss']) / stressed_rwa

print(f"  Total Capital Ratio (stressed): {stressed_total_ratio*100:.2f}%")
print(f"  Total Requirement (with buffers): {total_min_with_buffers*100:.2f}%")
print(f"  Status: {'✓ PASS' if stressed_total_ratio >= total_min_with_buffers else '✗ FAIL'}")

if stressed_total_ratio < total_min_with_buffers:
    shortfall_amount = (total_min_with_buffers - stressed_total_ratio) * stressed_rwa
    print(f"  Capital raise needed: ${shortfall_amount:.1f}B")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Capital ratios across scenarios
ax = axes[0, 0]
scenarios_list = list(results.keys())
ratios_tier1 = [results[s]['ratios']['Tier1']*100 for s in scenarios_list]
ratios_total = [results[s]['ratios']['Total']*100 for s in scenarios_list]

x = np.arange(len(scenarios_list))
width = 0.35
ax.bar(x - width/2, ratios_tier1, width, label='Tier 1', alpha=0.8)
ax.bar(x + width/2, ratios_total, width, label='Total', alpha=0.8)
ax.axhline(y=minimums['Tier1']*100, color='blue', linestyle='--', linewidth=1, label='Min Tier1')
ax.axhline(y=minimums['Total']*100, color='orange', linestyle='--', linewidth=1, label='Min Total')
ax.set_ylabel('Capital Ratio (%)')
ax.set_title('Capital Ratios Under Stress Scenarios')
ax.set_xticks(x)
ax.set_xticklabels(scenarios_list, rotation=15)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Pass/Fail matrix
ax = axes[0, 1]
pass_fail_matrix = []
for scenario_name in scenarios_list:
    scenario_results = results[scenario_name]
    pass_fail_matrix.append([1 if scenario_results['passes'][m] else 0 for m in minimums.keys()])

im = ax.imshow(pass_fail_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(np.arange(len(minimums)))
ax.set_yticks(np.arange(len(scenarios_list)))
ax.set_xticklabels(list(minimums.keys()))
ax.set_yticklabels(scenarios_list)
ax.set_title('Pass/Fail Regulatory Requirements')
for i in range(len(scenarios_list)):
    for j in range(len(minimums)):
        text_color = 'white' if pass_fail_matrix[i][j] == 1 else 'black'
        text = '✓' if pass_fail_matrix[i][j] == 1 else '✗'
        ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')

# Plot 3: Leverage ratio
ax = axes[1, 0]
leverage_ratios = [results[s]['ratios']['Leverage']*100 for s in scenarios_list]
ax.bar(scenarios_list, leverage_ratios, color=['green' if r >= minimums['Leverage']*100 else 'red' for r in leverage_ratios], alpha=0.7)
ax.axhline(y=minimums['Leverage']*100, color='black', linestyle='--', linewidth=2, label=f"Min ({minimums['Leverage']*100:.1f}%)")
ax.set_ylabel('Leverage Ratio (%)')
ax.set_title('Leverage Ratio Under Stress (Non-Risk-Weighted Floor)')
ax.tick_params(axis='x', rotation=15)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Capital shortfall/surplus
ax = axes[1, 1]
shortfalls = []
for scenario_name in scenarios_list:
    stressed_total = results[scenario_name]['ratios']['Total']
    min_total = minimums['Total']
    if stressed_total < min_total:
        shortfall = (min_total - stressed_total) * results[scenario_name]['rwa']
        shortfalls.append(-shortfall)
    else:
        shortfalls.append(0)

colors_shortfall = ['red' if s < 0 else 'green' for s in shortfalls]
ax.bar(scenarios_list, shortfalls, color=colors_shortfall, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Capital Surplus/(Shortfall) ($B)')
ax.set_title('Capital Shortfall/(Surplus) vs Minimum Requirement')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
- Calculate LCR for bank balance sheet; assess 30-day liquidity stress
- Design countercyclical buffer schedule for 7-year economic cycle
- Compare capital requirement: Basel II IRB vs Basel III (same portfolio)
- Run reverse stress test: What losses cause CET1 ratio to fall below minimum?
- Explain TLAC and bail-in mechanics for G-SIB resolution

## 7. Key References
- [Basel Committee, "Basel III: A global regulatory framework" (2010-2023)](https://www.bis.org/basel_framework/) — Authoritative source
- [Federal Reserve, "Comprehensive Capital Analysis and Review (CCAR)" (Annual)](https://www.federalreserve.gov/) — US stress testing framework
- [BIS, "Basel III: Post Crisis Reforms" (2017 update)](https://www.bis.org/bcbs/publ/d424.pdf) — Endgame framework
- [Blundell-Wignall & Atkinson, "The Financial Crisis & Policy Implications" (2010)](https://www.oecd.org/) — Basel III motivation

---
**Status:** Current regulatory standard (phased 2008-2028) | **Complements:** Basel II, Credit Risk Modeling, Liquidity Risk, Stress Testing
