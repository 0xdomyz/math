# Stress Testing

## 1. Concept Skeleton
**Definition:** Evaluation of portfolio losses under severe but plausible scenarios; unlike VaR (historical/statistical), stress tests assume explicit extreme conditions  
**Purpose:** Quantify tail risk outside model distribution, prepare for crisis scenarios, stress management readiness, regulatory compliance  
**Prerequisites:** Portfolio analytics, scenario building, loss distribution modeling, governance frameworks

## 2. Comparative Framing
| Framework | Basis | Scenarios | Effort | Regulatory |
|-----------|-------|-----------|--------|-----------|
| **Historical Stress** | Past crises (2008, 1998) | Discrete, well-defined | Low | Basel III |
| **Hypothetical Stress** | Invented scenarios | Designer's choice | Medium | CCAR/DFAST |
| **Macro Scenarios** | Recession, rates, unemployment | Econometric models | High | ECB, Fed |
| **Reverse Stress** | Capital target â†’ required scenarios | Backward looking | High | UK PRA |

## 3. Examples + Counterexamples

**Simple Example:**  
Stress test: "If unemployment increases 5pp and house prices fall 30%, portfolio loss = $250M". Compare to VaR estimate

**Failure Case:**  
2008: Banks ran stress tests assuming S&P 500 decline 20%; actual decline 50%+. Stress test scenario too conservative

### 3B. Technical Counterexample: Scenario Design Bias and Model-Specific Risk

**Common Misconception:** "I ran stress tests assuming unemployment rises 5 percentage points and house prices fall 30%. I generated multiple scenarios covering recession, financial crisis, and geopolitical shock. My portfolio survives all scenarios, so capital is adequate."

**Why This Fails:** Stress scenarios are internally designed (not market-generated); they tend to reflect historical crises, not novel future shocks. The scenarios missed (pandemic, cyber-attack, unexpected policy changes) often inflict greater losses than tested scenarios. Correlation assumptions also break in untested extremes.

**Quantitative Example:**

**Designed Scenarios (2019, Pre-COVID):**
- Recession: Unemployment +5pp (to 7%), GDP -2%, house prices -20%, spreads +150 bps
- Financial Crisis: Unemployment +8pp, GDP -4%, house prices -35%, spreads +350 bps
- Portfolio 99% VaR across scenarios: $180M (vs baseline $50M)

**2020 COVID Reality (Untested):**
- Unemployment spiked 10pp in 1 month (fastest on record)
- House prices actually rose initially (negative correlation flip)
- Corporate defaults exceeded stressed scenario assumptions
- Portfolio actually lost $320M (vs stress-tested $180M max)

**Why Design Failed:**
1. **Group-Think Scenarios:** Regulatory scenarios tend to converge on similar assumptions (unemployment, GDP, rates) across banks; miss idiosyncratic tail risks
2. **Overly Mechanical:** Unemployment +5pp → Defaults +2.5pp; ignores non-linear relationships that accelerate in severity
3. **Behavioral Dynamics:** Doesn't account for fire-sale feedback loops, liquidity hoarding, credit availability collapse
4. **Model Dependency:** Stress tests assume model relationships hold; but in novel crisis, all model assumptions break simultaneously

**Better Approach:**
- Reverse stress test: "What scenarios would wipe out capital?" Answer: simultaneous triggers of multiple tail events (unemployment 12% + rates +300bps + spreads +400bps + corr → 0.95)
- Out-of-sample validation: Test 2008-2009 scenarios on 2010-2019 data (what would have happened if repeated?)
- Scenario diversity: Include non-historical (pandemic, cyber, policy shock) scenarios

## 4. Layer Breakdown
```
Stress Testing Framework:
â”œâ”€ Scenario Construction:
â”‚   â”œâ”€ Macro variables: Unemployment â†‘, GDP â†“, rates â†‘, spreads â†‘
â”‚   â”œâ”€ Market prices: Equity â†“, FX volatility â†‘, asset prices â†“
â”‚   â”œâ”€ Credit: Default rates â†‘, recovery rates â†“, rating migrations
â”‚   â”œâ”€ Contagion: Sector spillover, counterparty defaults
â”‚   â””â”€ Liquidity: Haircuts â†‘, bid-ask spreads â†‘, funding stress
â”œâ”€ Scenario Development Methods:
â”‚   â”œâ”€ Historical: Reproduce 2008, 1998 LTCM, Asian crisis
â”‚   â”œâ”€ Hypothetical: Design "what-if" scenarios
â”‚   â”œâ”€ Econometric: Build macro-to-credit model
â”‚   â”œâ”€ Expert judgment: Consensus on unlikely outcomes
â”‚   â””â”€ Reverse: Start with capital target, back out scenario
â”œâ”€ Loss Calculation:
â”‚   â”œâ”€ Mark-to-market: Price changes on assets
â”‚   â”œâ”€ Default losses: PD Ã— LGD Ã— EAD under scenario PDs
â”‚   â”œâ”€ Spread widening: Duration Ã— dSpread on fixed income
â”‚   â”œâ”€ FX impact: Exposure Ã— dFX on foreign portfolios
â”‚   â””â”€ Correlation adjustment: Higher correlation in stress
â”œâ”€ Typical Stress Scenarios:
â”‚   â”œâ”€ Recession: 5pp unemployment rise, 2-3% GDP contraction
â”‚   â”œâ”€ Financial Crisis: Spreads 300bps, equity -40%, volatility spike
â”‚   â”œâ”€ Rates shock: 200bps parallel shift, curve twist
â”‚   â”œâ”€ Sector stress: Real estate -50% (real estate portfolio)
â”‚   â”œâ”€ Counterparty: Top 3 counterparties default
â”‚   â””â”€ Liquidity: All illiquid assets hit with haircuts
â”œâ”€ Governance:
â”‚   â”œâ”€ Scenario approval: Board/executive review
â”‚   â”œâ”€ Model validation: Independent review
â”‚   â”œâ”€ Documentation: Explicit assumptions
â”‚   â”œâ”€ Frequency: Quarterly minimum
â”‚   â””â”€ Escalation: Capital depleted â†’ management action
â”œâ”€ Regulatory Stress Tests:
â”‚   â”œâ”€ Fed CCAR (US): Proprietary scenarios
â”‚   â”œâ”€ ECB TLTRO (EU): Common scenarios
â”‚   â”œâ”€ PRA (UK): Reverse stress test requirement
â”‚   â”œâ”€ BIS: Macro scenarios for banks
â”‚   â””â”€ Frequency: Annual, integrated with capital planning
â””â”€ Use Cases:
    â”œâ”€ Capital planning: Ensure capital survives stress
    â”œâ”€ Pricing: Add margin for stress losses to interest rate
    â”œâ”€ Risk appetite: Limits on exposures to stress scenarios
    â”œâ”€ Model validation: Check if VaR/models underestimate
    â””â”€ Disclosure: Communication to investors/regulators
```

## 5. Challenge Round
When is stress testing problematic?
- **Scenario design bias**: Tests may not capture true tail risks; "Group-think" on scenarios
- **Model risk**: Assumed relationships (PD/unemployment) may break in novel scenarios
- **Correlation assumption**: Assumed correlations may be conservative or aggressive
- **Liquidity**: Assumes ability to liquidate; actual market may be frozen
- **Second-order effects**: Doesn't capture feedback loops (liquidity dries up â†’ prices fall â†’ more defaults)

## 6. Key References
- [Basel III Stress Testing Framework](https://www.bis.org/basel_framework/chapter/MRA/40.htm) - Regulatory requirements for stress testing; macroeconomic scenarios; governance and documentation standards; integration with capital planning and ICAAP.

- [Federal Reserve CCAR/DFAST](https://www.federalreserve.gov/supervisionreg/ccar_about.htm) - Comprehensive Capital Analysis and Review; detailed US regulatory stress test scenarios (baseline, adverse, severely adverse); historical scenario design (2008 crisis template).

- [ECB Banking Supervision Stress Test](https://www.bankingsupervision.europa.eu/ecb/pub/stress_test/html/index.en.html) - EU-wide stress test methodology; common scenarios across banking system; transparency in scenario assumptions; publication of bank-by-bank results.

- Borio, C., Furfine, C., & Lowe, P. (2001). "Procyclicality of the Financial System and Financial Stability: Issues and Policy Options." BIS Papers, 1, 1-57. Research on feedback effects; how stress tests must capture nonlinear responses; policy implications for capital buffers in cycles.

- Danielsson, J., & Shin, H. S. (2003). "Endogenous Risk." FMG Discussion Paper DP424. Theory of procyclical margin requirements and fire-sale dynamics; why static stress tests miss feedback effects when market participants respond simultaneously.

- Nelson, B. D. (2019). "Stress Tests and the Distribution of Risk." Journal of Risk, 21(3), 67-94. Empirical study of 2008 crisis stress test performance; banks that passed stress tests suffered large losses; documents model risk in scenario design; proposes robust stress testing under uncertainty.

---
**Status:** Essential risk management and regulatory tool | **Complements:** VaR, Capital planning, Risk governance
