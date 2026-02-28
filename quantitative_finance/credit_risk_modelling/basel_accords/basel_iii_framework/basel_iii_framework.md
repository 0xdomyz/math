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
2006 major bank: 8% Tier 1, 12% total (Basel II minimum). 2023 same bank: Must hold 7% Tier 1 + 2.5% capital buffer + 1.5% G-SIB surcharge = 11% Tier 1 equivalent. 4x leverage reduction â†’ safer but higher cost.

**Pro-Cyclicality Addressed:**  
Boom: Bank decides countercyclical buffer = 1% (regulator sets). Capital requirement: 7% base + 2.5% capital buffer + 1% CyCB = 10.5%. High but discipline against excess lending. Bust: Regulator releases CyCB â†’ 7% + 2.5% = 9.5%. Smaller but some relief (vs forced tightening).

**Liquidity Coverage Ratio (LCR):**  
Bank's 30-day liquidity stress test. Scenario: 30% deposit outflow + derivatives collateral call. Required: High-Quality Liquid Assets (HQLA) â‰¥ Net Cash Outflow. If shortfall, bank must hold more Treasury/CB deposits (less profitable but safe).

**Systemically Important Bank (G-SIB):**  
JPMorgan: Size $3.7T, interconnectedness, substitutability high â†’ G-SIB bucket 5 (4-4.5% capital surcharge). Community bank: $5B assets â†’ no surcharge. Different risk profiles = different capital rules.

**Stress Test Failure:**  
Fed stress test: Severe recession scenario (unemployment +5%, house prices -35%, equity -50%). Bank A projected: $20B loss, capital ratio falls to 2% (below 7% minimum). Result: "Capital plan rejected" â†’ must raise equity or cut dividends immediately.

**CVaR (Expected Shortfall) vs VaR:**  
Basel II: Market risk capital = VaR(99%,10-day). Basel III: Adds CVaR charge = average of tail losses. Result: Capital nearly doubles for some trading desks (tail risk penalty). Tail hedging becomes mandatory.

**Leverage Ratio (Non-Risk-Weighted Floor):**  
Bank: $100B assets, 80% "low-risk" (RW=20%), 20% "high-risk" (RW=100%). RWA = 36B. Risk-weighted capital = 8% Ã— 36B = 2.88B. Leverage ratio floor: 3% Ã— $100B = 3B. Leverage ratio is binding (more restrictive than RW). Forces de-leveraging regardless of mix.

## 4. Layer Breakdown
```
Basel III Framework Architecture:

â”œâ”€ Pillar I: Minimum Capital Requirements (Enhanced)
â”‚  â”œâ”€ Capital Definitions (Tier-1 vs Tier-2 refined):
â”‚  â”‚   â”œâ”€ Common Equity Tier 1 (CET1):
â”‚  â”‚   â”‚   â”œâ”€ Common shares, retained earnings (core capital)
â”‚  â”‚   â”‚   â”œâ”€ Must represent at least 4.5% of RWA (hard floor)
â”‚  â”‚   â”‚   â””â”€ Fully loss-absorbing, most subordinated
â”‚  â”‚   â”œâ”€ Tier 1 Capital:
â”‚  â”‚   â”‚   â”œâ”€ CET1 + Additional Tier 1 (AT1) instruments
â”‚  â”‚   â”‚   â”œâ”€ AT1: Perpetual subordinated bonds (contingent write-down)
â”‚  â”‚   â”‚   â”œâ”€ Must represent at least 6% of RWA
â”‚  â”‚   â”‚   â””â”€ Write-down at CET1 ratio â‰¤ 5.125%
â”‚  â”‚   â””â”€ Tier 2 Capital:
â”‚  â”‚       â”œâ”€ Subordinated debt (5-10 year maturity)
â”‚  â”‚       â”œâ”€ Loan loss reserves (limited to 1.25% RWA)
â”‚  â”‚       â”œâ”€ Total capital = Tier 1 + Tier 2 â‰¥ 10.5% RWA
â”‚  â”‚       â””â”€ Tier 2 limited to 50% of Tier 1
â”‚  â”œâ”€ Capital Buffers (Pillars I+II):
â”‚  â”‚   â”œâ”€ Capital Conservation Buffer (CCB):
â”‚  â”‚   â”‚   â”œâ”€ 2.5% of RWA (additional requirement)
â”‚  â”‚   â”‚   â”œâ”€ All banks must hold (no discretion)
â”‚  â”‚   â”‚   â”œâ”€ Prevents dividend/compensation when buffer breached
â”‚  â”‚   â”‚   â””â”€ Phased in: 0.625% (2011) â†’ 2.5% (2019)
â”‚  â”‚   â”œâ”€ Countercyclical Buffer (CyCB):
â”‚  â”‚   â”‚   â”œâ”€ 0-2.5% of RWA (set by national regulators)
â”‚  â”‚   â”‚   â”œâ”€ Activated in credit booms (prevent leverage buildup)
â”‚  â”‚   â”‚   â”œâ”€ Released in busts (allow lending room)
â”‚  â”‚   â”‚   â”œâ”€ Example: COVID 2020 â†’ CyCB released 0â†’0 (already 0)
â”‚  â”‚   â”‚   â””â”€ Pre-2008 would have been set to 1.5-2% in boom
â”‚  â”‚   â”œâ”€ G-SIB Capital Buffer:
â”‚  â”‚   â”‚   â”œâ”€ 1-3.5% of RWA (based on systemic importance score)
â”‚  â”‚   â”‚   â”œâ”€ Annual ranking by Fed/national regulator
â”‚  â”‚   â”‚   â”œâ”€ Score factors: Size, interconnectedness, complexity, substitutability
â”‚  â”‚   â”‚   â””â”€ Only for systemically important banks
â”‚  â”‚   â””â”€ Other Systemically Important Institution (O-SIB) Buffer:
â”‚  â”‚       â”œâ”€ Similar concept, applies to non-bank systemic entities
â”‚  â”‚       â”œâ”€ Insurance companies, market infrastructure
â”‚  â”‚       â””â”€ Country-specific identification
â”‚  â”œâ”€ Capital Requirement Floors:
â”‚  â”‚   â”œâ”€ Total Capital: 10.5% RWA minimum
â”‚  â”‚   â”‚   â””â”€ 4.5% CET1 + 1.5% AT1 + 2.5% Tier2 + 2% CCB + 0-2.5% CyCB + 0-3.5% G-SIB
â”‚  â”‚   â”œâ”€ Tier 1 Capital: 8.5% RWA minimum
â”‚  â”‚   â”œâ”€ CET1 Capital: 7% RWA minimum
â”‚  â”‚   â””â”€ Typical large bank: 12-15% CET1 (well above minimum)
â”‚  â”œâ”€ Risk-Weighted Assets (RWA) Calculation (Refined):
â”‚  â”‚   â”œâ”€ Credit Risk:
â”‚  â”‚   â”‚   â”œâ”€ Standardized Approach (unchanged from Basel II)
â”‚  â”‚   â”‚   â”œâ”€ Foundation/Advanced IRB (enhanced calibration)
â”‚  â”‚   â”‚   â””â”€ IRB floor: 72.5% of Standardized RW (prevents gaming)
â”‚  â”‚   â”œâ”€ Market Risk (Fundamental Review):
â”‚  â”‚   â”‚   â”œâ”€ Replaces Basel II VaR with Expected Shortfall (CVaR)
â”‚  â”‚   â”‚   â”œâ”€ 10-day horizon, 99% confidence
â”‚  â”‚   â”‚   â”œâ”€ Adds stressed ES (using crisis calibration)
â”‚  â”‚   â”‚   â”œâ”€ Incremental Risk Charge (IRC): Jump-to-default risk
â”‚  â”‚   â”‚   â”œâ”€ Comprehensive Risk Measure (CRM): Non-linear derivatives
â”‚  â”‚   â”‚   â””â”€ Typically doubles capital for trading desks vs Basel II
â”‚  â”‚   â”œâ”€ Operational Risk:
â”‚  â”‚   â”‚   â”œâ”€ Standardized Approach (replaces AMA for most):
â”‚  â”‚   â”‚   â”‚   OpRisk = 12% Ã— [Indicator] Ã— [Loss Component Factor]
â”‚  â”‚   â”‚   â”‚   Indicator = average revenue over 3 years
â”‚  â”‚   â”‚   â”œâ”€ Advanced Approach (limited banks only):
â”‚  â”‚   â”‚   â”‚   OpRisk = 9.5% Ã— [Expected Loss] + [CVaR-weighted Tail]
â”‚  â”‚   â”‚   â”‚   ILM adjustment (Internal Loss Multiplier) for severe events
â”‚  â”‚   â”‚   â””â”€ Credit valuation adjustment (CVA) risk:
â”‚  â”‚   â”‚       Risk that counterparty becomes less creditworthy
â”‚  â”‚   â”‚       Separate capital charge on derivatives positions
â”‚  â”‚   â””â”€ Floor: RWA â‰¥ 72.5% of Standardized RWA (binding for complex banks)
â”‚  â””â”€ Output Floor (As of 2023):
â”‚      â”œâ”€ Finalized output floor: 72.5%
â”‚      â”œâ”€ Means: RWA(Internal models) â‰¥ 72.5% Ã— RWA(Standardized)
â”‚      â”œâ”€ Prevents excessive RWA reduction from IRB optimization
â”‚      â””â”€ Effective 2028 (phase-in period extended)
â”œâ”€ Pillar II: Supervisory Review (Enhanced)
â”‚  â”œâ”€ Internal Capital Adequacy Assessment (ICAAP):
â”‚  â”‚   â”œâ”€ Banks must model own capital needs over 3-year horizon
â”‚  â”‚   â”œâ”€ Includes: Credit, market, operational, concentration, interest rate risks
â”‚  â”‚   â”œâ”€ Stress scenarios designed by bank (regulator can override)
â”‚  â”‚   â”œâ”€ Board-level review and approval required
â”‚  â”‚   â””â”€ Submitted to regulator for evaluation
â”‚  â”œâ”€ Supervisory Stress Testing (CCAR/DFAST in US):
â”‚  â”‚   â”œâ”€ Fed-designed scenarios: Baseline, adverse, severely adverse
â”‚  â”‚   â”œâ”€ Banks run models on own portfolios â†’ report results
â”‚  â”‚   â”œâ”€ Fed compares results for reasonableness
â”‚  â”‚   â”œâ”€ "Fail" results â†’ reject capital plans (dividends/buybacks frozen)
â”‚  â”‚   â”œâ”€ Annual exercise, published results
â”‚  â”‚   â””â”€ Increasingly stringent assumptions (tail risk focus)
â”‚  â”œâ”€ Pillar II Guidance (P2G):
â”‚  â”‚   â”œâ”€ Regulator-set additional capital if ICAAP/stress test inadequate
â”‚  â”‚   â”œâ”€ Discretionary add-on (not formulaic like CCB/CyCB)
â”‚  â”‚   â”œâ”€ Addresses: Concentration, interconnectedness, business model risk
â”‚  â”‚   â”œâ”€ Example: Bank heavily concentrated in commercial real estate â†’ +2% P2G
â”‚  â”‚   â””â”€ Can be released if risk profile improves
â”‚  â”œâ”€ Concentration Risk (New Focus):
â”‚  â”‚   â”œâ”€ Single counterparty exposure limit: 10-25% depending on bank size
â”‚  â”‚   â”œâ”€ Large exposure framework: Capital charge if exposure >10% Tier1
â”‚  â”‚   â”œâ”€ Sector concentration: Monitored separately
â”‚  â”‚   â””â”€ Interconnectedness: Factor into required capital
â”‚  â””â”€ Other Risks:
â”‚      â”œâ”€ Interest Rate Risk (in the Banking Book): Non-trading portfolio
â”‚      â”œâ”€ Business Model Risk: Profitability sustainability
â”‚      â”œâ”€ Reputational Risk: Fines, loss of market confidence
â”‚      â””â”€ Macroeconomic Risk: System-wide stresses
â”œâ”€ Pillar III: Market Discipline (Enhanced Disclosure)
â”‚  â”œâ”€ Quantitative Disclosure (Quarterly):
â”‚  â”‚   â”œâ”€ Capital composition (CET1, Tier 1, Total)
â”‚  â”‚   â”œâ”€ Capital ratios vs requirements
â”‚  â”‚   â”œâ”€ RWA breakdown (credit, market, operational)
â”‚  â”‚   â”œâ”€ Leverage ratio
â”‚  â”‚   â”œâ”€ LCR, NSFR (liquidity metrics)
â”‚  â”‚   â””â”€ Remuneration, risk-weighted positions
â”‚  â”œâ”€ Qualitative Disclosure:
â”‚  â”‚   â”œâ”€ Risk governance framework
â”‚  â”‚   â”œâ”€ Risk management policies by risk type
â”‚  â”‚   â”œâ”€ Stress testing methodology
â”‚  â”‚   â”œâ”€ Concentration risk disclosures
â”‚  â”‚   â””â”€ Regulatory framework compliance
â”‚  â”œâ”€ Standardized Templates (COREP/FINREP):
â”‚  â”‚   â”œâ”€ Regulatory technical standards (EBA, OCC, others)
â”‚  â”‚   â”œâ”€ Machines-readable formats (XML)
â”‚  â”‚   â”œâ”€ Enables direct comparison across banks
â”‚  â”‚   â””â”€ Public vs confidential component (sensitive data protected)
â”‚  â””â”€ Transparency Goals:
â”‚      â”œâ”€ Investors assess capital adequacy
â”‚      â”œâ”€ Depositors/creditors price risk accurately
â”‚      â”œâ”€ Peer comparison drives discipline
â”‚      â”œâ”€ Regulatory arbitrage reduced
â”‚      â””â”€ Data-driven market monitoring
â”œâ”€ Liquidity Standards (New - Major Innovation)
â”‚  â”œâ”€ Liquidity Coverage Ratio (LCR):
â”‚  â”‚   â”œâ”€ Ensure bank survives 30-day stress scenario
â”‚  â”‚   â”œâ”€ Formula: LCR = HQLA / Net Cash Outflow â‰¥ 100%
â”‚  â”‚   â”œâ”€ HQLA: High-Quality Liquid Assets (Level 1/2)
â”‚  â”‚   â”‚   â”œâ”€ Level 1: Cash, CB reserves, government bonds (no haircut)
â”‚  â”‚   â”‚   â”œâ”€ Level 2a: AAA/AA bank/corporate bonds (15% haircut)
â”‚  â”‚   â”‚   â””â”€ Level 2b: BBB+ corporate, equities (25-50% haircut)
â”‚  â”‚   â”œâ”€ Net Cash Outflow: Liability runoff + collateral needs
â”‚  â”‚   â”œâ”€ Example: Bank $100B assets, $80B deposits (100% runoff), $20B commitments
â”‚  â”‚   â”‚   Required HQLA = $100B, bank has $30B cash + $80B Treasuries = $110B âœ“
â”‚  â”‚   â”œâ”€ Phase-in: 60% (2015) â†’ 100% (2019)
â”‚  â”‚   â””â”€ More restrictive for investment banks
â”‚  â”œâ”€ Net Stable Funding Ratio (NSFR):
â”‚  â”‚   â”œâ”€ Ensure structural funding stability (1-year horizon)
â”‚  â”‚   â”œâ”€ Formula: NSFR = Available Stable Funding / Required Stable Funding â‰¥ 100%
â”‚  â”‚   â”œâ”€ Stable Funding (ASF): Long-term liabilities, core deposits
â”‚  â”‚   â”‚   â”œâ”€ Deposits from retail customers: 90% ASF
â”‚  â”‚   â”œâ”€ Stable Funding (RSF): Difficulty to convert to cash
â”‚  â”‚   â”‚   â”œâ”€ Illiquid assets (commercial loans): 85% RSF
â”‚  â”‚   â”‚   â”œâ”€ Unencumbered corporates: 50% RSF
â”‚  â”‚   â”‚   â””â”€ Encumbered assets: 100% RSF
â”‚  â”‚   â”œâ”€ Example: Bank funding = $50B retail deposits (90% stable) + $30B wholesale (30% stable)
â”‚  â”‚   â”‚   ASF = 0.9Ã—$50B + 0.3Ã—$30B = $54B
â”‚  â”‚   â”‚   RSF = 0.85Ã—$50B + 0.5Ã—$40B + 1.0Ã—$10B = 75B
â”‚  â”‚   â”‚   NSFR = $54B / $75B = 72% âœ— (below 100%, needs stable funding)
â”‚  â”‚   â”œâ”€ Phase-in: 80% (2018) â†’ 100% (2020)
â”‚  â”‚   â””â”€ Discourages reliance on wholesale funding
â”‚  â””â”€ Intraday Liquidity Monitoring:
â”‚      â”œâ”€ Track daily cash flows
â”‚      â”œâ”€ Ensure capability to meet obligations throughout day
â”‚      â”œâ”€ Critical for systemically important payment systems
â”‚      â””â”€ Real-time management, not regulatory floor
â”œâ”€ Leverage Ratio (Non-Risk-Weighted Floor)
â”‚  â”œâ”€ Purpose:
â”‚  â”‚   â”œâ”€ Backstop to risk-weighted capital (limits RWA gaming)
â”‚  â”‚   â”œâ”€ Ensures banks don't take excessive unweighted risk
â”‚  â”‚   â”œâ”€ Binding for some institutions (high leverage users)
â”‚  â”œâ”€ Definition:
â”‚  â”‚   â”œâ”€ Leverage Ratio = Tier 1 Capital / Exposure Measure
â”‚  â”‚   â”œâ”€ Exposure measure â‰ˆ Total assets + derivatives + commitments (minimal haircuts)
â”‚  â”‚   â”œâ”€ Minimum = 3% (proposed enhancement to 3.6% for systemic banks)
â”‚  â”œâ”€ Example (Illustrative):
â”‚  â”‚   Bank: Tier 1 = $20B, Total Assets = $500B
â”‚  â”‚   LR = $20B / $500B = 4% âœ“ (above 3% floor)
â”‚  â”œâ”€ Comparison to Risk-Weighted:
â”‚  â”‚   If RWA = 250B (50% of assets): RW-based Tier 1 req = 8% Ã— $250B = $20B âœ“
â”‚  â”‚   LR floor achieved same capital, different basis
â”‚  â””â”€ Effectiveness:
â”‚      â”œâ”€ Prevents high-leverage strategies (carry trades, dark pooling)
â”‚      â”œâ”€ Less risk-sensitive but more transparent
â”‚      â”œâ”€ Often non-binding (RW capital more restrictive for normal portfolios)
â”œâ”€ Macroprudential Tools & Resolution
â”‚  â”œâ”€ Countercyclical Capital Buffer (CyCB):
â”‚  â”‚   â”œâ”€ National regulator sets 0-2.5% (up to 5% with approval)
â”‚  â”‚   â”œâ”€ Activation: Credit growth signals boom (e.g., >15% annual)
â”‚  â”‚   â”œâ”€ Deactivation: Credit stress signals bust (e.g., defaults spike)
â”‚  â”‚   â”œâ”€ Example: 2012 EU CyCB most set to 0%, 2013+ gradually increased to 0.5-1%
â”‚  â”‚   â””â”€ Goal: Brake on lending in good times, relief in bad times
â”‚  â”œâ”€ Systemic Risk Buffer:
â”‚  â”‚   â”œâ”€ National regulator discretion
â”‚  â”‚   â”œâ”€ Applied to systemically important institutions
â”‚  â”‚   â”œâ”€ Can vary by sector (e.g., real estate concentration)
â”‚  â”‚   â””â”€ Prevents local financial instability
â”‚  â”œâ”€ Resolution & Recovery Planning (RRP):
â”‚  â”‚   â”œâ”€ Living Wills: Banks must show how to unwind without systemic harm
â”‚  â”‚   â”œâ”€ Resolvability: Regulators test annual ("Is this bank dissolvable?")
â”‚  â”‚   â”œâ”€ Recovery Plans: How bank raises capital/liquidity in stress
â”‚  â”‚   â”œâ”€ Barriers to Resolution: Legal, structural (spin-offs required if barrier too high)
â”‚  â”‚   â””â”€ Goal: Avoid Lehman-type contagion
â”‚  â”œâ”€ Total Loss-Absorbing Capacity (TLAC):
â”‚  â”‚   â”œâ”€ G-SIBs must have 16-18% of RWA in capital/debt that can absorb losses
â”‚  â”‚   â”œâ”€ Ensures sufficient loss buffer in resolution
â”‚  â”‚   â”œâ”€ Bail-in provisions: Creditors share losses before public/taxpayer
â”‚  â”‚   â””â”€ Reduces moral hazard ("too big to fail" expectations)
â”‚  â””â”€ Single Counterparty Exposure:
â”‚      â”œâ”€ Limit large exposure to one counterparty
â”‚      â”œâ”€ Large exposure = >10% Tier1 capital
â”‚      â”œâ”€ Capital charge on excess: Up to 50%
â”‚      â”œâ”€ Prevents concentration risk contagion
â”‚      â””â”€ Example: Bank with $30B Tier1, Large client = $3B exposure OK, $4B faces charge
â”œâ”€ Implementation Timeline
â”‚  â”œâ”€ 2010-2019: Phased in globally
â”‚  â”‚   â”œâ”€ 2010: Framework agreed (G20 in Seoul)
â”‚  â”‚   â”œâ”€ 2011: US, EU, Japan, etc. begin rule-making
â”‚  â”‚   â”œâ”€ 2012: Capital buffers partial (0.625%)
â”‚  â”‚   â”œâ”€ 2014: LCR phased to 60%
â”‚  â”‚   â”œâ”€ 2019: NSFR = 100%, CCB = 2.5%
â”‚  â”‚   â””â”€ 2019: Endgame issued (output floor finalized)
â”‚  â”œâ”€ 2020-2027: Post-COVID & Endgame Implementation
â”‚  â”‚   â”œâ”€ 2020: COVID capital relief (CyCB released)
â”‚  â”‚   â”œâ”€ 2021-2027: Output floor phase-in
â”‚  â”‚   â”œâ”€ 2024-2028: Stress testing enhancement
â”‚  â”‚   â””â”€ Ongoing: Regulatory refinement based on feedback
â”‚  â””â”€ Transition Provisions:
â”‚      â”œâ”€ Grandfather clauses for pre-existing instruments
â”‚      â”œâ”€ Phase-out of hybrid capital (partially replaced by AT1)
â”‚      â”œâ”€ Multi-year transition periods (avoid cliff effects)
â”‚      â””â”€ Flexibility for less developed countries
â”œâ”€ Criticisms & Challenges
â”‚  â”œâ”€ Complexity:
â”‚  â”‚   â”œâ”€ Multiple buffers, LCR, NSFR, Leverage â†’ hard to understand
â”‚  â”‚   â”œâ”€ Regulatory arbitrage opportunities (find loopholes)
â”‚  â”‚   â”œâ”€ Compliance costs (technology, staffing, data systems)
â”‚  â”‚   â””â”€ Smaller banks struggle relative to large banks
â”‚  â”œâ”€ Capital Flight to Shadow Banking:
â”‚  â”‚   â”œâ”€ Tighter regulations â†’ shift to less-regulated entities
â”‚  â”‚   â”œâ”€ Private equity, hedge funds grow while banks contract
â”‚  â”‚   â”œâ”€ Systemic risk moves rather than reduced
â”‚  â”‚   â””â”€ Regulatory arbitrage between jurisdictions
â”‚  â”œâ”€ Pro-Cyclicality Residual:
â”‚  â”‚   â”œâ”€ CyCB helps but discretion leads to late activation
â”‚  â”‚   â”œâ”€ G-SIB surcharge may increase during stress (more complex â†’ higher risk)
â”‚  â”‚   â”œâ”€ Procyclical components remain despite improvements
â”‚  â”‚   â””â”€ 2023: Rising rates â†’ mark-to-market losses spike CET1 requirements
â”‚  â”œâ”€ LCR Constraints:
â”‚  â”‚   â”œâ”€ HQLA scarcity post-2008 (not enough government bonds)
â”‚  â”‚   â”œâ”€ LCR forces banks into low-return safe assets
â”‚  â”‚   â”œâ”€ Liquidity trade-off: Safe but expensive funding
â”‚  â”‚   â””â”€ NSFR discourages secured financing (repo), impacts liquidity provision
â”‚  â”œâ”€ Model Risk (IRB Approaches Remain):
â”‚  â”‚   â”œâ”€ Output floor reduces (but doesn't eliminate) IRB gaming
â”‚  â”‚   â”œâ”€ Banks incentivized to minimize RWA within 72.5% floor
â”‚  â”‚   â”œâ”€ Parameter manipulation (PD, LGD, correlation)
â”‚  â”‚   â””â”€ Regulators may tighten further (regulatory capital arbitrage wars)
â”‚  â””â”€ Unintended Consequences:
â”‚      â”œâ”€ Liquidity coverage focus on stable funding â†’ less financial system efficiency
â”‚      â”œâ”€ Leverage ratio floor may be binding (non-risk-sensitive) â†’ mispricingof risk
â”‚      â”œâ”€ Interconnectedness definition drives capital â†’ could reduce market liquidity
â”‚      â””â”€ Regulatory complexity â†’ smaller competitive disadvantage
```

**Interaction:** Enhanced capital â†’ Liquidity standards â†’ Supervisory review (stress test) â†’ Market discipline (disclosure) â†’ Resolution framework â†’ Macroprudential tools (countercyclical buffers).

## 5. Challenge Round
- Calculate LCR for bank balance sheet; assess 30-day liquidity stress
- Design countercyclical buffer schedule for 7-year economic cycle
- Compare capital requirement: Basel II IRB vs Basel III (same portfolio)
- Run reverse stress test: What losses cause CET1 ratio to fall below minimum?
- Explain TLAC and bail-in mechanics for G-SIB resolution

## 6. Key References
- [Basel Committee, "Basel III: A global regulatory framework" (2010-2023)](https://www.bis.org/basel_framework/) â€” Authoritative source
- [Federal Reserve, "Comprehensive Capital Analysis and Review (CCAR)" (Annual)](https://www.federalreserve.gov/) â€” US stress testing framework
- [BIS, "Basel III: Post Crisis Reforms" (2017 update)](https://www.bis.org/bcbs/publ/d424.pdf) â€” Endgame framework
- [Blundell-Wignall & Atkinson, "The Financial Crisis & Policy Implications" (2010)](https://www.oecd.org/) â€” Basel III motivation

---
**Status:** Current regulatory standard (phased 2008-2028) | **Complements:** Basel II, Credit Risk Modeling, Liquidity Risk, Stress Testing
