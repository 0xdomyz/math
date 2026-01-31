# Capital Requirements & Risk-Based Capital

## 1. Concept Skeleton
**Definition:** Regulatory minimum capital; funds unexpected losses; measured by risk components (C1-C4: underwriting, market, credit, operational)  
**Purpose:** Solvency protection; policyholder security; standardized risk assessment; discourage excessive leverage  
**Prerequisites:** Asset allocation, mortality/lapse assumptions, interest rates, regulatory regime (Solvency II, US RBC, etc.)

## 2. Comparative Framing
| Regime | Calculation | Basis | Capital Type | Burden |
|--------|-------------|-------|--------------|--------|
| **Solvency II** | Standard formula (C1-C4) | Economic value; SCR + MCR | Risk margin | High (68% of premiums) |
| **US RBC** | Risk categories (life, health, property) | Statutory reserves; fixed %ages | Own Funds | High (250-400%) |
| **NAIC (Old)** | Fixed minimum (% of reserves) | Reserves only; simple | Surplus | Very low |
| **IAIS** | Risk-based; principle-based | Local regulatory standard | Paid-in capital | Variable |
| **Pillar approach** | Multi-level: minimum + buffer + target | Regulatory discretion | Paid-in + retained | Tiered |

## 3. Examples + Counterexamples

**Simple Example:**  
Life insurer: $100M reserves; Solvency II SCR = 25% × reserves + mortality shock = $25M + $5M = $30M minimum capital required

**Failure Case:**  
Insurer ignores market risk; invests $500M in corporate bonds; credit crunch → bond values fall 20% → $100M loss; only $20M capital → insolvent

**Edge Case:**  
Catastrophe insurer: $50M capital; 1-in-1,000 year event probability = 0.1%; Poisson: expected loss $500K; but actual loss $200M → instant insolvency (extreme tail)

## 4. Layer Breakdown
```
Capital Requirements & Risk-Based Capital Structure:
├─ Solvency Capital Requirement (SCR):
│   ├─ Purpose: Amount to absorb 1-in-200 year loss (99.5% confidence)
│   ├─ Formula (Standard):
│   │   ├─ SCR = √[Σᵢ Σⱼ Corrᵢⱼ × SCRᵢ × SCRⱼ]
│   │   ├─ Where: SCRᵢ = capital for risk module i
│   │   ├─ Corrᵢⱼ = correlation matrix between risks
│   │   └─ Interpretation: Diversification reduces total capital
│   ├─ Risk modules (life insurance):
│   │   ├─ Mortality risk:
│   │   │   ├─ Shock: Mortality rates +15% for 1 year
│   │   │   ├─ Impact on reserve: Higher claims → higher liability
│   │   │   ├─ Calculation: V_shock - V_base
│   │   │   ├─ Example: $100M reserve, 15% mortality shock
│   │   │   │   ├─ Year 1 claims: +$3M (15% × $20M annual claims)
│   │   │   │   ├─ Future years: Death claims shift forward
│   │   │   │   ├─ PV impact: ~$8-12M reserve increase
│   │   │   │   └─ SCR_mortality = $10M
│   │   │   └─ Hedge: Reinsurance reduces capital requirement
│   │   ├─ Longevity risk:
│   │   │   ├─ Shock: Mortality rates -20% (people live longer)
│   │   │   ├─ Impact on reserve: Lower claims → reserve increases
│   │   │   ├─ Example: Annuity portfolio
│   │   │   │   ├─ Reserve: $100M for expected 20-year payouts
│   │   │   │   ├─ If mortality -20%: Some payouts shift to year 25-30
│   │   │   │   ├─ PV increases: Fewer discounted payments far out
│   │   │   │   └─ SCR_longevity = $12M
│   │   │   └─ Hedge: Longevity swaps, reinsurance
│   │   ├─ Disability & morbidity risk:
│   │   │   ├─ Shock: Claim incidence +35%; average duration +25%
│   │   │   ├─ Impact: Increased claims + longer payouts
│   │   │   ├─ Example: Disability insurance
│   │   │   │   ├─ Expected claims: $10M/year
│   │   │   │   ├─ Shock claims: $13.5M/year × 1.25 years = $16.9M
│   │   │   │   ├─ PV impact: ~$8-10M reserve increase
│   │   │   │   └─ SCR_disability = $9M
│   │   │   └─ Sensitivity: High volatility; smaller portfolios riskier
│   │   ├─ Lapse risk:
│   │   │   ├─ Shock patterns:
│   │   │   │   ├─ Upward mass lapse: -50% on profitable contracts
│   │   │   │   │   ├─ Loss: Profit opportunity disappears
│   │   │   │   │   ├─ Plus: Fixed costs continue (overhead loss)
│   │   │   │   │   └─ Impact: -$5-10M on portfolio (depends on surrender charges)
│   │   │   │   ├─ Downward mass lapse: -20% on loss-making contracts
│   │   │   │   │   ├─ Benefit: Bad business stays
│   │   │   │   │   ├─ Problem: Fixed costs now spread thinner
│   │   │   │   │   └─ Impact: -$2-5M (costs not saved)
│   │   │   ├─ Examples: Market shock → everyone surrenders good policies
│   │   │   └─ Hedge: Surrender charge structure; sticky book management
│   │   └─ Catastrophe & revision risk:
│   │       ├─ Revision: Assumption changes mid-year (mortality tables updated)
│   │       ├─ Impact: Reserve recalculation, possible capital charge
│   │       ├─ Example: New mortality table shows -5% mortality; reserves drop $8M; no charge (good)
│   │       └─ Catastrophe: Usually property insurance; occasionally health (pandemic)
│   ├─ Market risk:
│   │   ├─ Equity risk:
│   │   │   ├─ Shock: Equity prices fall 39% (historical: 2008 crisis)
│   │   │   ├─ Impact: Asset values down; surplus reduced
│   │   │   ├─ Example: $500M equity holdings
│   │   │   │   ├─ Post-shock value: $305M (−39%)
│   │   │   │   ├─ Asset loss: $195M
│   │   │   │   ├─ Liability unchanged: $400M
│   │   │   │   ├─ Surplus change: -$195M → capital requirement ~$195M
│   │   │   │   └─ But SCR typically 22% of equity holding
│   │   │   └─ SCR_equity ≈ $110M (22% × $500M)
│   │   ├─ Interest rate risk:
│   │   │   ├─ Scenario 1: Rates rise 2%
│   │   │   │   ├─ Bond asset: $1,000M, duration 5 years
│   │   │   │   ├─ Price decline: -10% (5 years × 2%) = -$100M
│   │   │   │   ├─ Liability: $400M, duration 8 years
│   │   │   │   │   ├─ Reserve increases (PV future payments higher):
│   │   │   │   │   │   ├─ Wait, higher rates → lower PV → lower reserve
│   │   │   │   │   │   └─ Confusion: When discount rate rises, PV falls
│   │   │   │   │   └─ Actually: Reserve decline 12% = -$48M (good for insurer)
│   │   │   │   ├─ Net asset-liability impact:
│   │   │   │   │   ├─ Assets: -$100M
│   │   │   │   │   ├─ Liabilities: -$48M (benefit)
│   │   │   │   │   └─ Surplus: -$52M net loss
│   │   │   │   └─ SCR_interest_up ≈ $52M
│   │   │   ├─ Scenario 2: Rates fall 2%
│   │   │   │   ├─ Bond asset: Price rise +10% = +$100M
│   │   │   │   ├─ Liability: Reserve increases 12% = +$48M (bad)
│   │   │   │   ├─ Net surplus: +$52M
│   │   │   │   └─ SCR_interest_down ≈ $52M
│   │   │   └─ Worst case: Rates fall (liability blows up); typically SCR = $52M
│   │   ├─ Currency risk:
│   │   │   ├─ Shock: FX rates move ±20%
│   │   │   ├─ Impact: Non-matching currency in assets vs liabilities
│   │   │   ├─ Example: $200M liabilities in EUR; only $150M EUR assets
│   │   │   │   ├─ EUR/USD rate: 1.20 → 0.96 (20% decline)
│   │   │   │   ├─ EUR liability in USD: $240M → $192M (good!)
│   │   │   │   ├─ But if other direction: EUR → 1.44 → liability $288M (+20%)
│   │   │   │   └─ SCR_currency ≈ $20M
│   │   │   └─ Hedge: Forwards; offsetting currency positions
│   │   ├─ Property risk:
│   │   │   ├─ Real estate holdings: Market value shock ±15%
│   │   │   ├─ Example: $300M real estate → ±$45M potential loss
│   │   │   └─ SCR_property ≈ $45M
│   │   └─ Spread risk (credit):
│   │       ├─ Credit spreads widen; bond values fall
│   │       ├─ Shock: ±500 bps spread change
│   │       ├─ Impact: Bond portfolio loses value; counterparty risk
│   │       └─ Covered separately in credit risk module
│   ├─ Credit (counterparty) risk:
│   │   ├─ Default probability: PD per rating class
│   │   ├─ Exposure: Gross exposure to single counterparty
│   │   ├─ Loss given default: LGD = 1 - Recovery %
│   │   ├─ Calculation: EL = Exposure × PD × LGD
│   │   ├─ Example: $50M bond exposure
│   │   │   ├─ Bond issuer: BBB rated
│   │   │   ├─ PD (1-year): 0.5% (historical average BBB)
│   │   │   ├─ LGD: 60% (recovery on bonds ~40%)
│   │   │   ├─ Expected loss: $50M × 0.5% × 60% = $150K
│   │   │   ├─ Confidence interval (99.5%): ~$2-3M
│   │   │   └─ SCR_credit ≈ $3M
│   │   ├─ Concentration risk:
│   │   │   ├─ Large single exposure: $500M to Bank of America
│   │   │   ├─ Shock: Bank rating downgrade (AAA → A); spreads widen
│   │   │   ├─ Impact: Bond value -10% = -$50M
│   │   │   ├─ Plus: Default risk increases
│   │   │   └─ SCR_concentration ≈ $55-60M
│   │   └─ Hedging effect:
│   │       ├─ CDS purchased: -$2M/year for 3-year protection
│   │       ├─ Reduces EL and concentration capital
│   │       └─ Trade-off: Cost of hedge vs capital savings
│   ├─ Operational risk:
│   │   ├─ Definition: Losses from failed/inadequate processes, fraud, IT failure
│   │   ├─ Formula (Solvency II): Op Risk = max(0.30% × Income, 0.45% × Expenses)
│   │   ├─ Example: Large insurer
│   │   │   ├─ Gross income: $5,000M
│   │   │   ├─ Operating expenses: $1,500M
│   │   │   ├─ Calculation:
│   │   │   │   ├─ 0.30% × $5,000M = $15M
│   │   │   │   ├─ 0.45% × $1,500M = $6.75M
│   │   │   │   └─ Op Risk capital = max($15M, $6.75M) = $15M
│   │   ├─ Adverse scenarios:
│   │   │   ├─ System outage: Claims processing stopped 1 day
│   │   │   │   ├─ Lost fees: $1M
│   │   │   │   ├─ Reputational: Customer loss estimate $5M
│   │   │   │   └─ Operational loss: $6M
│   │   │   ├─ Fraud: Employee embezzlement
│   │   │   │   ├─ Direct loss: $2-10M
│   │   │   │   ├─ Recovery: 30-50%
│   │   │   │   └─ Net loss: $5M typical
│   │   │   └─ Regulatory event: Non-compliance fine
│   │   │       ├─ Penalty: 1-5% of annual profit
│   │   │       ├─ Example: $50M profit, fine 3% = $1.5M
│   │   │       └─ Reputational damage: Unknown
│   │   └─ Capital need: Reflects organization size & control quality
│   ├─ Aggregation & correlations:
│   │   ├─ Mortality & lapse: Negatively correlated (low rates → low lapse → higher mortality impact)
│   │   ├─ Interest rates & spreads: Positively correlated (rate rises → default risk up)
│   │   ├─ Equity & credit: Positively correlated (recession → equities fall, defaults rise)
│   │   ├─ Currency & equity: Partially correlated (strong currency → export decline → equity fall)
│   │   ├─ Correlation matrix (simplified):
│   │   │   ├─ Mortality ↔ Longevity: -1.0 (opposite shocks)
│   │   │   ├─ Mortality ↔ Equity: +0.3 (recession → low equity + high mortality)
│   │   │   ├─ Interest ↔ Equity: -0.5 (strong inverse relationship)
│   │   │   ├─ Equity ↔ Credit: +0.7 (very correlated in crisis)
│   │   │   └─ Diversification benefit: √(sum) vs simple addition
│   │   └─ Example calculation:
│   │       ├─ Mortality capital: $10M
│   │       ├─ Interest capital: $8M
│   │       ├─ Equity capital: $12M
│   │       ├─ Without correlation: $10 + $8 + $12 = $30M
│   │       ├─ With correlation: √(10² + 8² + 12² + 2×0.3×10×8 + 2×(-0.5)×8×12 + 2×0.7×12×10)
│   │       │   = √(100 + 64 + 144 + 48 - 96 + 168) = √428 = $20.7M
│   │       └─ Diversification benefit: $30M - $20.7M = 31% reduction
│   └─ Minimum Capital Requirement (MCR):
│       ├─ Purpose: Absolute floor; intervene if MCR breached
│       ├─ Calculation (Solvency II): Linear combination of risks
│       │   ├─ MCR = 0.33 × SCR (minimum $2.5M)
│       │   ├─ Or: Principal component method (more granular)
│       │   └─ Typical: MCR = $4-8M for small insurer, $50-100M for large
│       ├─ Regulatory response to MCR breach:
│       │   ├─ Immediate supervisor notification required
│       │   ├─ Urgent recovery plan submission (7 days)
│       │   ├─ Possible restrictions (no new business, dividend freeze)
│       │   ├─ Potential insolvency procedures initiated
│       │   └─ Customer protection mechanism activation
│       └─ Buffer above MCR:
│           ├─ Target: 1.5× SCR (above minimum)
│           ├─ Policy buffer zone: 1.0-1.5× SCR → Management discretion
│           ├─ Stress buffer: 1.5-2.0× SCR → Regulatory approval for growth
│           └─ Strategy: Maintain 2.0-2.5× SCR for safety
├─ US Risk-Based Capital (RBC):
│   ├─ Structure: 5 components (C1-C5)
│   ├─ C1 Asset risk: Fixed income, equity, real estate quality
│   ├─ C2 Insurance risk: Underwriting risk (mortality, lapse, expense)
│   ├─ C3 Interest rate risk: Liability duration mismatch
│   ├─ C4 Business risk: Expense volatility, market competition
│   ├─ C5 Credit/Franchise risk: Counterparty, regulatory, brand
│   ├─ NAIC RBC formula: Covariance method
│   │   ├─ Total RBC = √(ΣΣ Corrᵢⱼ × RBCᵢ × RBCⱼ)
│   │   ├─ Comparable to Solvency II but different components
│   │   └─ Target: 250-400% RBC (vs 100% minimum)
│   ├─ Action levels:
│   │   ├─ >250% RBC: No action
│   │   ├─ 200-250% RBC: Regulatory attention
│   │   ├─ 150-200% RBC: Enhanced examination
│   │   ├─ 100-150% RBC: Corrective action plan required
│   │   └─ <100% RBC: Potential insolvency proceedings
│   └─ Comparison to Solvency II:
│       ├─ Solvency II: 1-in-200 year (99.5% confidence)
│       ├─ US RBC: 1-in-300 to 1-in-500 year (estimated; not explicit)
│       ├─ Solvency II: Higher capital usually; more conservative
│       └─ US RBC: Simpler formula; less granular
├─ Own Funds Composition:
│   ├─ Tier 1 capital (highest quality):
│   │   ├─ Common equity: Shareholder paid capital
│   │   ├─ Share premium & retained earnings
│   │   ├─ Unrestricted AUM (available for loss absorption)
│   │   ├─ Subordinated perpetual debt (Solvency II only certain types)
│   │   └─ Amount typically: 60-80% of total own funds
│   ├─ Tier 2 capital (moderate quality):
│   │   ├─ Subordinated debt (10+ year maturity)
│   │   ├─ Hybrid instruments (debt-like but equity characteristics)
│   │   ├─ Excess reserves (above minimum required)
│   │   ├─ Transition provisions (old assets recognized temporarily)
│   │   └─ Amount typically: 20-40% of total own funds
│   ├─ Tier 3 capital (lower quality):
│   │   ├─ Recent subordinated debt (<5 years remaining)
│   │   ├─ Deferred tax assets (recognized contingently)
│   │   ├─ Restricted only: SCR coverage (not MCR)
│   │   └─ Amount typically: 0-10% of total own funds
│   ├─ Haircuts applied:
│   │   ├─ Interest rate risk: -10% if rates fall (capital value loss)
│   │   ├─ Credit risk: -25% for subordinated debt (credit spread risk)
│   │   ├─ Liquidity: -10% for longer maturity instruments
│   │   └─ Net effect: Actual capital value may be 70-90% of book
│   └─ Capital structure optimization:
│       ├─ Minimize cost: Maximize Tier 2 debt (cheaper) vs Tier 1 equity (expensive)
│       ├─ Regulatory constraint: Min 50% of requirement in Tier 1
│       ├─ Trade-off: Debt is tax-deductible but increases financial risk
│       └─ Target: 70% Tier 1 + 30% Tier 2 (balance cost & flexibility)
├─ Capital Adequacy Ratio:
│   ├─ Definition: (Eligible Own Funds) / (Capital Requirement)
│   ├─ Interpretation:
│   │   ├─ >1.5: Comfortable; growth room; excess capital
│   │   ├─ 1.0-1.5: Adequate; constrained; need growth to add capital
│   │   ├─ <1.0: Deficient; intervention; corrective action required
│   │   └─ <0.5: Urgent; likely insolvency process initiated
│   ├─ Regulatory targets vary by jurisdiction:
│   │   ├─ Solvency II: 1.0× SCR minimum; 1.5× SCR target (buffer)
│   │   ├─ US: 2.5× RBC minimum action; 4.0× for comfort
│   │   ├─ Switzerland (FINMA): 1.15× BES minimum; 1.45× target
│   │   └─ UK (PRA): 1.0× SCR minimum; 1.75× target (stress scenario based)
│   └─ Forecasting capital position:
│       ├─ Profit retention: Typical 50-70% of earnings retained (dividends to shareholders)
│       ├─ Capital growth: Profit + retained earnings - policyholder dividends
│       ├─ Example: $100M profit, 60% retention = $60M capital build
│       ├─ If capital requirement grows 5%/year: Adequate if profit > 5% of requirement
│       └─ Stress test: If loss scenario -$50M, still maintain 1.0× SCR?
├─ Risk Mitigation & Capital Reduction:
│   ├─ Reinsurance:
│   │   ├─ Ceded risk reduces risk charge
│   │   ├─ Quota share 50%: Capital requirement halved (mortality risk)
│   │   ├─ Cost: 5-15% of premiums
│   │   ├─ Trade-off: Save $10M capital cost for $2M reinsurance premium
│   │   └─ ROE: $2M cost / $10M capital freed = 20% better ROE
│   ├─ Derivatives hedging:
│   │   ├─ Interest rate swaps: Reduce interest rate risk capital
│   │   ├─ Mortality swaps: Transfer longevity/mortality risk
│   │   ├─ Cost: 0.5-2% of hedged risk
│   │   └─ Capital benefit: Can reduce requirement 20-40%
│   ├─ Securitization:
│   │   ├─ Transfer catastrophe risk to capital markets
│   │   ├─ CAT bonds: 3-5 year issuance, 300-500 bps yield
│   │   ├─ Cost: 15-25% of risk value
│   │   └─ Capital freed: Immediate; can grow business without new capital
│   └─ Portfolio adjustments:
│       ├─ Shift to lower-risk products: Reduce mortality/lapse risk
│       ├─ Selective underwriting: Decline riskier business
│       ├─ Surrender high-risk policies: Buy back bad book
│       └─ Capital freed: Modest; 5-10% reduction over time
└─ Forward-Looking Capital Projection:
    ├─ Multi-year forecast:
    │   ├─ Year 1: Profit $100M + Capital raised $50M = Eligible own funds +$150M
    │   ├─ Year 2: Profit $120M + Capital raised $60M = +$180M
    │   ├─ Year 3: Profit $140M + Dividends -$70M = +$70M
    │   └─ Year 3 total: $150M + $180M + $70M = $400M eligible capital
    ├─ Requirement growth:
    │   ├─ Baseline: Requirement $100M → grows 5%/year
    │   ├─ Year 3 requirement: $115.76M
    │   ├─ Ratio: $400M / $115.76M = 3.45× (comfortable)
    │   └─ Headroom for growth: Yes, can add business
    ├─ Stress scenario:
    │   ├─ Assumption: Market downturn, equity -20%, bonds -5%
    │   ├─ Impact: Losses -$50M year 1 (impact capital)
    │   ├─ Recovery: Profit +$80M year 2-3 (rebuild)
    │   ├─ Stress ratio: Dips to 2.2× SCR (year 1); recovers to 2.8× (year 3)
    │   └─ Stress test passes: Maintains >1.5× SCR throughout
    └─ Strategic decisions:
        ├─ If forecasted ratio > 2.0×: Consider dividends, share buyback, or new business
        ├─ If forecasted ratio < 1.2×: Limit growth, raise capital, seek reinsurance
        ├─ If ratio volatile: Increase hedging; reduce risk concentration
        └─ If ratio declining: Cost-cutting or premium increases necessary
```

**Key Insight:** Capital structure is complex optimization balancing safety (high capital) against profitability (low capital); regulatory minimums are floors, not targets

## 5. Mini-Project
[Would include calculation of C1-C4 components, scenario testing, capital projection, and visualization of capital adequacy ratios across scenarios]

## 6. Challenge Round
When capital requirements fail:
- **Correlation shock**: All risks spike together (equity -30%, credit spreads +300 bps, mortality +25%); correlation coefficient structure breaks; SCR doubles
- **Model error**: Assumption of C1-C4 independence wrong; during crisis, all correlate +0.9; actual loss 2× model prediction
- **Regulatory change**: Regulator increases confidence level from 99.5% to 99.9%; capital requirement +40% overnight; compliance deadline 6 months
- **Longevity trend**: Mortality improvement accelerates; all annuity reserves inadequate; longevity risk capital triples; company now undercapitalized
- **Credit concentration**: Bank A holds $200M assets, now defaulting; loss $40M; but was only 10% RBC charge; actual loss way worse; capital evaporates
- **Hedging failure**: Swap counterparty downgraded; hedge value falls while risk it covers increases; forced to unwind; crystallizes loss; capital hit twice

## 7. Key References
- [Solvency II Technical Guidance (EIOPA, 2019)](https://www.eiopa.europa.eu/) - EU capital calculation specifics
- [NAIC Risk-Based Capital Manual (US)](https://www.naic.org/) - US RBC formula and action levels
- [Insolvency Risk: Regulatory Framework Comparison (IAIS, 2015)](https://www.iaisweb.org/) - International perspective

---
**Status:** Risk measurement | **Complements:** Reserve Adequacy, Capital Structure, Market Risk Management
