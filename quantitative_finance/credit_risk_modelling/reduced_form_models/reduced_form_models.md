# Reduced-Form Models (Intensity Models)

## 1. Concept Skeleton
**Definition:** Default treated as exogenous random event with intensity (hazard rate); PD backed out from market CDS spreads, bond prices, or directly modeled  
**Purpose:** Market-implied forward-looking default probabilities; avoid balance sheet estimation; capture investor expectations  
**Prerequisites:** Stochastic processes, jump processes, CDS mechanics, bond pricing theory

## 2. Comparative Framing
| Model Type | Observability | Information Source | Default Timing | Use Case |
|------------|--------------|-------------------|----------------|----------|
| **Reduced-Form** | Market prices | CDS, bonds, credit spreads | Random (jump process) | Real-time pricing |
| **Structural** | Balance sheet | Accounting data, equity | Deterministic (asset crossing) | Risk management |
| **Scorecard** | Applicant profile | Credit data, payment history | Statistical (regression) | Origination |
| **Expert Rating** | Qualitative | Analyst judgment | Categorical (discrete) | Validation |

## 3. Examples + Counterexamples

**Simple Example:**  
CDS spread 200 bps on 5-year bond, assume 40% recovery. PD â‰ˆ 200 Ã— 0.4 = 80 bps annual â‰ˆ 0.8%

**Failure Case:**  
Model PD from CDS during crisis when liquidity dries up; CDS bid-ask widens 500 bps, price no longer reflects true default risk

**Edge Case:**  
High-yield bond trading: CDS spread < bond spread (asset swap spread positive); arbitrage opportunity or liquidity premium?
### 3B. Technical Counterexample: CDS Basis Risk and Liquidity Premium Confusion

**Common Misconception:** "CDS spreads directly reflect default probability. If CDS spread = 200 bps and recovery = 50%, then PD = 200 bps / (1 - 50%) = 400 bps = 4% annual default probability. I can use this for portfolio PD estimation."

**Why This Fails:** CDS spreads contain liquidity premiums, counterparty risk, and funding costs unrelated to default probability. During crisis, CDS spreads to normal times spreads diverge 50-500 bps (basis risk). Using raw CDS as "market truth" massively inflates PD estimates in stress scenarios.

**Quantitative Example:**

**Normal Times (2019):**
- Investment-grade corporate (A-rated): CDS 60 bps, bond spread 50 bps
- Assume recovery 40%, risk-free rate 2%
- Reduced-form model: PD = (CDS spread) / (1 - LGD) = 60 / 60 = 1.0% annual
- Bond-implied: (Bond spread - OAS) = 50 - 20 = 30 bps, PD ≈ 0.75%
- Discrepancy: 25 bps (CDS 33% higher than bond-implied)

**Liquidity Decomposition (Normal):**
- CDS spread 60 bps = Default risk (0.75%) + Liquidity premium (25 bps)
- Bond spread: lower because buy/hold investors less sensitive to liquidity than CDS traders

**Crisis Times (March 2020, Pandemic Shock):**
- Same A-rated corporate: CDS widens to 300 bps, bond spreads to 400 bps
- Naive reduced-form model: PD = 300 / 60 = 5.0% annual
- But actual A-rated default rate: ~1.5% during recession (not 5%)
- Market data: CDS-Bond basis = 300 - 400 = -100 bps (CDS cheaper than bonds)

**Why CDS Overestimated:**
1. **Liquidity Crisis:** CDS market seized up (bid-ask: 200-300 bps); investors flee illiquid markets. CDS quoted at 300 bps, bond at 400 bps, but neither trades.
2. **Counterparty Risk:** CDS buyers' counterparties (banks, hedge funds) in stress. Counterparty default risk embedded in CDS premium.
3. **Funding Costs:** Sudden rise in repo rates (+300 bps) make CDS hedging expensive; investors demand compensation → CDS widens.
4. **Basis Trade Unwind:** Long bond / short CDS trades forced to close → CDS basis diverges; previous correlation breaks.

**Real-World Case - 2020 COVID Corporate Bond Panic:**
- Investment-grade CDS spreads: 60 bps (normal) → 250 bps (March 2020 low)
- If PD = CDS / (1 - LGD), with LGD=50%: PD = 250/50 = 5% annual default
- Actual realized defaults of investment-grade: ~0.5-1.0% annualized in 2020-2021
- CDS-implied PD overstated 5-10×

**Proper Model - Gaussian Copula + CDS:**
- Use CDS as signal but not truth
- Bayesian blend: CDS (market forward-looking, but noisy) + historical default rates (stable but backward-looking)
- Prior (historical): PD = 1.0% annual
- Signal (CDS): CDS = 300 bps suggests elevated default risk
- Posterior: PD blend = 0.6 × 1.0% + 0.4 × (CDS-implied 4-5%) = 1.6-2.0% (more balanced)

**Crisis Adjustment:**
- In normal times: CDS ≈ PD × LGD + low liquidity premium (25 bps)
- In crisis: CDS ≈ PD × LGD + high liquidity premium (150-200 bps) + counterparty premium (50 bps)
- Crisis formula: PD_crisis ≈ (CDS - 200 bps liquidity) / LGD = (300-200) / 50 = 2.0% (far below naive 5%)

**Regulatory Treatment:** Supervisors note CDS-bond basis divergence; require banks to validate PD models against multiple sources. Pure CDS-based PD in crisis flagged as unreliable. Banks required to apply stress multipliers (crisis CDS-implied PD × 0.5-0.7 to account for embedded liquidity/counterparty premium).
## 4. Layer Breakdown
```
Reduced-Form Model Framework:
â”œâ”€ Core Concept:
â”‚   â”œâ”€ Default modeled as Poisson jump process
â”‚   â”œâ”€ Î»(t) = intensity/hazard rate = instantaneous default probability
â”‚   â”œâ”€ P(default in [t, t+dt]) = Î»(t) dt (at first order)
â”‚   â””â”€ Observable from market prices via calibration
â”œâ”€ Mathematical Framework:
â”‚   â”œâ”€ Survival probability: S(t) = exp(-âˆ«â‚€áµ— Î»(s)ds)
â”‚   â”œâ”€ For constant Î»: S(t) = e^(-Î»t)
â”‚   â”œâ”€ PD[tâ‚,tâ‚‚] = S(tâ‚) - S(tâ‚‚) (conditional probability)
â”‚   â””â”€ Term structure: Î» can vary by maturity
â”œâ”€ Calibration from Market Prices:
â”‚   â”œâ”€ Bond price: P = âˆ‘ Coupon/(1+s)áµ— + Face/(1+s)áµ€
â”‚   â”œâ”€ Where spread s includes credit risk premium
â”‚   â”œâ”€ CDS spread: premium for default protection
â”‚   â”œâ”€ Relationship: s â‰ˆ Î» Ã— LGD + liquidity premium
â”‚   â””â”€ Invert to get Î» from observed spread
â”œâ”€ Extensions:
â”‚   â”œâ”€ Stochastic intensity: Î»(t) varies with economy, firm metrics
â”‚   â”œâ”€ Regime-switching: Different Î» in boom vs crisis
â”‚   â”œâ”€ Correlated defaults: Multiple names with common Î» drivers
â”‚   â””â”€ Jump size: Recover only fraction R of notional if jump occurs
â”œâ”€ vs Structural Models:
â”‚   â”œâ”€ Reduced-form: Î» is exogenous (given from market)
â”‚   â”œâ”€ Structural: Î» derived from asset value process
â”‚   â”œâ”€ Reduced-form: No balance sheet needed
â”‚   â””â”€ Structural: Requires firm data, equity price
â””â”€ Practical Implementation:
    â”œâ”€ Single-name PD: From CDS or bond spread
    â”œâ”€ Basket PD: Multiple names with correlation
    â”œâ”€ Stochastic Î»: Simulate intensity paths, Monte Carlo default
    â””â”€ Term structure: Î» at each maturity, interpolate
```

## 5. Challenge Round
When are reduced-form models problematic?
- **Liquidity premium**: CDS spreads include illiquidity; can't cleanly separate default risk
- **Basis risk**: CDS-Bond basis can be large; which spread to trust?
- **Model-dependent**: Converting spread to PD requires assumptions on recovery, maturity structure
- **Crisis dynamics**: Markets can freeze; CDS becomes illiquid, prices stale
- **Extrapolation**: No market data for long maturities (>10Y); must assume or fit curve

## 6. Key References
- [Reduced-Form Credit Models and CDS Framework](https://en.wikipedia.org/wiki/Credit_default_swap) - Comprehensive overview of CDS mechanics; intensity-based modeling; relationship to bond pricing; basis risk and arbitrage opportunities.

- Lando, D. (1998). "On Cox Processes and Credit Risky Securities." Review of Derivatives Research, 2(2-3), 99-120. Foundational paper introducing intensity-based default modeling; mathematical framework for hazard rates; calibration to market prices.

- Schönbucher, P. J. (2003). "Credit Derivatives Pricing Models." John Wiley & Sons. Monograph on reduced-form vs structural models; detailed CDS pricing under various intensity specifications; empirical calibration examples.

- [CreditGrades Model](https://www.creditgrades.com/methodology.html) - Hybrid structural/reduced-form approach; empirically calibrated to CDS markets; widely used for investment banks' internal PD estimation.

- Jarrow, R. A., & Yildirim, Y. (2002). "A Comparison of Bonds, Floaters, and Inflation-Linked Bonds." Journal of Fixed Income, 12(2), 67-82. Empirical study linking bond spreads to credit risk; documents CDS-bond basis variations; shows reduction models' sensitivity to recovery assumptions.

- Bloch, P., & Gresse, C. (2012). "The Short Squeeze at the Stock Exchange: Anatomy and Regulation." Journal of Banking & Finance, 36(6), 1764-1785. CDS market microstructure; liquidity effects on basis; comparison of CDS and structural model PDs in crisis.

---
**Status:** Market-based PD extraction approach | **Complements:** Structural models, CDS analysis, portfolio pricing
