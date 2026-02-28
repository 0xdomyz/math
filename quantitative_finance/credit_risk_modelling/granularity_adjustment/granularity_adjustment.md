# Granularity Adjustment (Pg)

## 1. Concept Skeleton
**Definition:** Capital add-on (pg) accounting for finite portfolio effects and concentration risk; adjusts from infinite granular portfolio assumption to realistic finite portfolio with large exposures; derived from Basel III IRB framework to ensure capital reflects true portfolio tail risk.

**Purpose:**
- Account for concentration amplification of tail risk in finite portfolios (theoretical infinite assumes independent defaults)
- Align regulatory capital with actual portfolio structure and default clustering effects
- Bridge gap between analytical formulas (infinite granularity) and empirical losses (finite concentration)
- Capture correlation amplification when portfolio composed of just a few large exposures
- Prevent regulatory arbitrage through portfolio concentration optimization
- Apply stress-calibrated granularity adjustment under elevated correlation scenarios

**Prerequisites:**
- Portfolio risk models (Vasicek IRB framework, single-factor default models)
- Concentration metrics: Herfindahl-Hirschman Index (HHI), Numbers Equivalent (N_eq = 1/HHI)
- Basel III IRB framework and capital charge formula knowledge  
- Credit risk parameters: PD, LGD, EAD, maturity adjustment factors
- Understanding finite vs infinite portfolio mathematics and tail risk assumptions

## 2. Comparative Framing
| Approach | Assumption | Capital Impact | Accuracy | Computational |
|----------|-----------|-----------------|-----------|---------------|
| **Infinite Granularity** | N â†’ âˆž, all sizes equal | Low (underestimates) | Poor for concentrated | Simple |
| **Granularity Adjustment** | Finite N, correction factor | Medium adjustment | Good for most portfolios | Moderate |
| **Full Monte Carlo** | Exact simulation | Exact loss distribution | Excellent | Complex |
| **Single-Name Limits** | Cap per exposure | Regulatory minimum | Variable | Simple |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio: $1B capital, 100 exposures (HHI=0.01). Capital under infinite granularity: $80M. With granularity adjustment (pg=0.5%): $85M. Adjustment captures finite portfolio effect

**Failure Case:**  
Assuming granularity adjustment for portfolio of 5 huge exposures; pg formula breaks down. Need full Monte Carlo instead

**Edge Case:**  
Highly granular portfolio (N_eq=1000+): Granularity adjustment â‰ˆ 0; model approaches infinite assumption. Minimal impact

## 4. Layer Breakdown
```
Granularity Adjustment Framework:
â”œâ”€ Basel III Formula:
â”‚   â”œâ”€ pg = (1 - exp(-2Ã—HHI)) / (2Ã—HHI)
â”‚   â”œâ”€ Simplified: pg â‰ˆ HHI for small HHI (Taylor expansion)
â”‚   â”œâ”€ Range: pg âˆˆ [0, 0.5] (max at HHI=0.5, single large exposure)
â”‚   â””â”€ Applied as capital add-on: K_adj = K_granular + pg
â”œâ”€ Intuition:
â”‚   â”œâ”€ Finite portfolio has more tail risk than infinite model
â”‚   â”œâ”€ Large borrower defaults have bigger portfolio impact
â”‚   â”œâ”€ Multiple large defaults more likely to occur
â”‚   â””â”€ pg captures this concentration premium
â”œâ”€ Theoretical Basis:
â”‚   â”œâ”€ Infinite granularity: K_âˆž = âˆšN Ã— Ïƒ(PD, LGD)
â”‚   â”œâ”€ Finite portfolio: K_finite > K_âˆž due to concentration
â”‚   â”œâ”€ Adjustment: pg â‰ˆ E[max loss] - E[mean loss]
â”‚   â””â”€ Probability: Tail events more likely with concentration
â”œâ”€ Alternative Formulas:
â”‚   â”œâ”€ Merton's formula: More complex, inputs HHI + correlation
â”‚   â”œâ”€ Simplified linear: pg = c Ã— HHI (c â‰ˆ 0.5-1.0)
â”‚   â”œâ”€ Regime-dependent: pg increases in crisis
â”‚   â””â”€ Maturity-adjusted: pg rises for longer horizons
â”œâ”€ Granular Portfolio Definition:
â”‚   â”œâ”€ N_eq â‰¥ 100: Typically considered sufficiently granular
â”‚   â”œâ”€ HHI â‰¤ 0.01: Granular (pg â‰ˆ 0.0001)
â”‚   â”œâ”€ HHI âˆˆ [0.01, 0.05]: Moderately granular (pg â‰ˆ 0.5-2%)
â”‚   â”œâ”€ HHI > 0.05: Concentrated (pg > 2%)
â”‚   â””â”€ Regulatory cap: Some jurisdictions cap pg at 2.5%
â”œâ”€ Portfolio Characteristics Impact:
â”‚   â”œâ”€ Size distribution: More skewed â†’ higher pg
â”‚   â”œâ”€ Correlation: Higher correlation â†’ higher pg
â”‚   â”œâ”€ Default probability: Higher PD â†’ higher pg
â”‚   â”œâ”€ Loss given default: Higher LGD â†’ higher pg
â”‚   â””â”€ Maturity: Longer maturity â†’ higher pg
â””â”€ Capital Application:
    â”œâ”€ Under IRB: K_total = K_granular + pg Ã— exposure
    â”œâ”€ Regulatory: Typically capped at 2.5% of RWA
    â”œâ”€ Portfolio-specific: pg varies by segment
    â””â”€ Multiple risk factors: Separate pg for each segment
```

## 5. Challenge Round
When is granularity adjustment problematic?
- **Model risk**: pg formula may not capture all concentration effects in crisis
- **Regime changes**: pg estimated in normal times; breaks in crisis when correlation spikes
- **Regulatory arbitrage**: Banks may structure portfolios to minimize pg without reducing true risk
- **Multi-dimensional**: Handles concentration but not other portfolio effects (collateral correlation, etc.)
- **Tail underestimation**: Still may miss extreme tail events (Expected Shortfall better for risk management)

## 6. Key References
- [Basel III IRB pg Formula](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Official regulatory definition of granularity adjustment; mathematical formula pg = (1-e^(-2×HHI))/(2×HHI); application in capital charge; limits on pg surcharge.

- [Granularity Effect Theory - BIS Working Paper 155](https://www.bis.org/publ/work155.pdf) - In-depth research on finite portfolio effects; derivation of granularity adjustment; empirical calibration to default loss distributions; impact on capital requirements across concentration levels.

- [Merton Portfolio Model](https://en.wikipedia.org/wiki/Merton_model) - Theoretical foundation for credit models; single-factor framework underlying granularity analysis; connection between firm-level PD and portfolio concentration.

- Gordy, M. B. (2003). "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules." Journal of Financial Intermediation, 12(3), 199-232. Rigorous derivation of IRB capital formula including granularity adjustment; shows how pg emerges from finite portfolio convexity; comparison to infinite granularity limit.

- Martin, R., & Wilde, T. (2002). "Unsystematic Surprise and Systemic Surprise." Risk Magazine, 15(11), 89-94. Explanation of granularity adjustment intuition; relationship between portfolio size, concentration, and tail risk; numerical examples with real portfolios.

- Lütkebohmert, E. (2009). "Concentration Risk in Credit Portfolios." Springer-Verlag. Monograph dedicated to concentration and granularity risk; detailed treatment of pg formula; practical implementation guidance; case studies from major banks.

---
**Status:** Regulatory capital add-on accounting for portfolio concentration | **Complements:** HHI, Credit VaR, Basel III
