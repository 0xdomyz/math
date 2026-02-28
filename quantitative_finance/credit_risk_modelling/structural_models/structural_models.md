# Structural Models (Merton Model)

## 1. Concept Skeleton
**Definition:** Default occurs when firm asset value falls below debt level; PD derived from asset value dynamics and capital structure  
**Purpose:** Theory-driven approach linking default probability to firm economics; market-implied PD from equity volatility  
**Prerequisites:** Option pricing theory, stochastic processes, balance sheet structure, equity volatility

## 2. Comparative Framing
| Model | Input Data | PD Driver | Calibration | Interpretation |
|-------|-----------|-----------|------------|-----------------|
| **Merton/Structural** | Equity vol, leverage | Asset > Debt threshold | Equity price â†’ asset vol | Economic intuition |
| **Scorecard** | Credit attributes | Default odds from variables | Historical default rates | Statistical pattern |
| **CDS-Implied** | Market spreads | Market consensus | Option-adjusted spread | Forward-looking |
| **Transition Matrix** | Credit ratings | Historical migration | Historical rating changes | Empirical regularity |

## 3. Examples + Counterexamples

**Simple Example:**  
Firm: Assets $100M, Debt $60M, Asset volatility 20%, risk-free rate 3%. Black-Scholes: 1-year PD â‰ˆ 2%

**Failure Case:**  
Merton underestimates distress PD; model assumes continuous asset value (no jumps). 2008 Lehman collapse: gap risk missed

### 3B. Technical Counterexample: Gap Risk and Merton Model Failure

**Common Misconception:** "Merton model with GBM asset dynamics (continuous paths) is theoretically sound. Given assets $1B, debt $600M, asset volatility 20%, Merton gives PD = 1.5%. This is my best estimate."

Why This Fails:** Merton assumes continuous asset value paths; real-world assets suffer discontinuous jumps (sudden news, market dislocations). Single jump below debt threshold causes immediate default; distance-to-default becomes meaningless. Merton systematically underestimates tail risk and rare defaults.

**Quantitative Example:**

**Merton Calculation:**
- Asset V = $1B, Debt D = $600M, σ = 20%, T = 1 year, r = 3%
- Distance to Default: DD = [ln(1000/600) + (0.03 - 0.2²/2)×1] / 0.20 = [0.511 + 0.01] / 0.20 = 2.61 standard deviations
- PD = N(-DD) = N(-2.61) = 0.45% (very low default probability)

**Market Reality (Jump Risk):**
- Single shock event (regulatory action, scandal, commodities collapse): assets drop $250M in one day
- New V = $750M (still above $600M debt), but now close: DD = 2.12 sd → PD = 1.7%
- Second shock: V drops another $200M → V = $550M < D = $600M → **IMMEDIATE DEFAULT**
- Realized default but Merton model showed PD only 0.45% before first jump

**Evidence - Lehman Brothers (2008):**
- Lehman: Assets $680B, Debt $613B, leverage 30:1
- Merton model (pre-collapse) estimated PD ~1-2% annually
- August 2008: Asset value fell from $680B to $30B in weeks (massive jumps due to credit losses discovery)
- September 15, 2008: Default
- Merton failed to capture tail risk of repeated large jumps

**Extensions to Address Gap Risk:**
1. **Jump-Diffusion Model:** dV/V = μdt + σdW + J×dN (where dN = Poisson jump process)
   - Additional jump premium: increases PD estimate by 2-5×
   - Merton with jumps: PD increases from 0.45% to 2-3% (more realistic)

2. **Structural Model with Barrier:**
   - Instead of maturity date, default if assets ever fall below debt
   - First-passage-time probability captures earlier default from jumps
   - PD increases 5-10× vs Merton

3. **Equity Volatility as Proxy:**
   - Market equity vol σ_E ≈ 30% (observable, forward-looking)
   - Implied asset vol σ_A (via Ito's lemma) ≈ σ_E × (V/E) ≈ 15-25% depending on leverage
   - Use market-implied vol instead of historical → captures jump risk premia

**Regulatory Treatment:** Advanced IRB banks using structural models must:
- Validate against historical defaults (Merton typically underestimates)
- Apply multiplier (1.5-2.0×) to Merton PD for tail risk
- Include scenario analysis capturing potential asset value jumps
- Use market-implied volatility, not historical

## 4. Layer Breakdown
```
Merton Structural Model Framework:
â”œâ”€ Core Concept:
â”‚   â”œâ”€ Firm modeled as call option on assets
â”‚   â”œâ”€ Equity = max(Assets - Debt, 0)
â”‚   â”œâ”€ Default when Assets < Debt at maturity
â”‚   â””â”€ Firm value follows geometric Brownian motion
â”œâ”€ Mathematical Framework:
â”‚   â”œâ”€ dV/V = Î¼dt + ÏƒdW  (asset dynamics)
â”‚   â”œâ”€ D = default barrier (debt level)
â”‚   â”œâ”€ T = time to maturity
â”‚   â”œâ”€ Distance to Default (DD) = (ln(V/D) + (Î¼ - ÏƒÂ²/2)T) / (ÏƒâˆšT)
â”‚   â””â”€ PD = N(-DD)  (N = standard normal CDF)
â”œâ”€ Implementation Steps:
â”‚   â”œâ”€ 1. Estimate current asset value Vâ‚€
â”‚   â”œâ”€ 2. Estimate asset volatility Ïƒ
â”‚   â”œâ”€ 3. Define default barrier D (debt)
â”‚   â”œâ”€ 4. Calculate distance to default
â”‚   â”œâ”€ 5. Convert DD to PD using normal distribution
â”‚   â””â”€ 6. Validate against market CDS spread
â”œâ”€ Key Parameters:
â”‚   â”œâ”€ Asset value (V): From balance sheet or equity â†’ assets (inverse problem)
â”‚   â”œâ”€ Asset volatility (Ïƒ): From equity vol via ItÃ´ lemma
â”‚   â”œâ”€ Debt (D): Book value or market value
â”‚   â”œâ”€ Time horizon (T): Usually 1 year for comparison
â”‚   â””â”€ Risk-free rate (r): Government bond yield
â”œâ”€ Two-Way Linkages:
â”‚   â”œâ”€ Equity = call option: Equity_value = Assets Ã— N(dâ‚) - Debt Ã— e^(-rT) Ã— N(dâ‚‚)
â”‚   â”œâ”€ Leverage effect: Higher debt â†’ Higher PD
â”‚   â””â”€ Volatility feedback: Higher asset vol â†’ Higher PD
â””â”€ Extensions:
    â”œâ”€ Multi-period: Simulate paths, calculate default probability
    â”œâ”€ Stochastic rates: Interest rates vary over time
    â”œâ”€ Jump risk: Asset value can jump down (gap risk)
    â””â”€ Barrier models: Early warning when DD crosses threshold
```

## 5. Challenge Round
When is Merton problematic?
- **Gap risk**: Asset value can jump below debt suddenly; continuous assumption fails (2008 Lehman collapse)
- **Equity vol instability**: Equity volatility changes rapidly; implied asset vol unstable
- **Capital structure complexity**: Multiple debt classes, covenants, priority; simple debt level insufficient
- **Endogenous default**: Firm can strategically default (Chapter 11 option); not purely economic threshold
- **Balance sheet quality**: Book debt may not reflect true obligations (pensions, contingencies, leases)

## 6. Key References
- [Merton Model - Wikipedia](https://en.wikipedia.org/wiki/Merton_model) - Original 1974 paper framework; mathematical derivation; equity as call option on assets; empirical applications to corporate PD.

- [Black-Scholes Option Pricing - Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) - Foundational option pricing theory underlying Merton; geometric Brownian motion assumptions; sensitivity analysis.

- [Credit Spread Determinants - BIS Working Paper](https://www.bis.org/publ/work291.pdf) - BIS research on structural model empirical performance; compares Merton to reduced-form models; documents underestimation of tails in crisis.

- Vassalou, M., & Xing, Y. (2004). "Default Risk in Equity Returns." The Journal of Finance, 59(2), 831-868. Empirical study linking Merton distance-to-default to equity returns; shows default risk premium observable in equity markets; validates that equity vol contains PD information.

- Eom, Y. H., Helwege, J., & Huang, J. Z. (2004). "Structural Models of Corporate Bond Pricing: An Empirical Analysis." Review of Financial Studies, 17(2), 499-544. Large sample study of Merton model vs actual default rates; documents systematic underestimation; shows Merton PDs 0.5-1% vs realized 2-4%; proposes calibration adjustments.

- Tudela, V., & Young, G. (2003). "A Merton-model Approach to Assessing the Default Risk of the UK Public Finance Initiative." Bank of England Working Paper. Application of structural models to sovereign/project finance; discusses practical implementation; leverage effects and maturity adjustments.

---
**Status:** Theory-driven PD approach with economic intuition | **Complements:** Scorecard models, market-implied PD, validation
