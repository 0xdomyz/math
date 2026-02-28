# Credit Correlation

## 1. Concept Skeleton
**Definition:** Co-movement between default events; probability two (or more) borrowers default together; measures default clustering; captures systemic and idiosyncratic correlation; varies with economic regime and time horizon; critical for portfolio risk aggregation and tail risk modeling.

**Purpose:**
- Quantify portfolio concentration risk through default correlation (high correlation → systemic risk)
- Model tail scenarios and extreme loss distributions in credit portfolios
- Price multi-name derivatives (credit baskets, CDOs, first-to-default, nth-to-default swaps)
- Calibrate capital requirements under Basel III (correlation in credit risk charge formula)
- Validate portfolio diversification assumptions and stress test correlation breakdowns
- Link macro variables (unemployment, GDP) to micro-level default clustering

**Prerequisites:**
- Copulas and joint distribution modeling (Gaussian, Clayton, Gumbel, Student-t copulas)
- Correlation matrices and factor models (single-factor, multi-factor, Vasicek framework)
- Default processes (intensity models, structural models, Poisson arrivals)
- Portfolio theory and risk aggregation (covariance, copula-based aggregation)
- Credit ratings and PD estimates by obligor
- Market data (CDS spreads, credit spreads, equity correlations)
- Time-series econometrics and regime-switching models
- Basel III IRB formulas incorporating correlation dependencies

**Use Cases:**
- **Portfolio VaR:** Bank combines 50 corporate loans; assumes pairwise correlation 0.30. Portfolio 99% VaR = $80M. Same loans in crisis with ρ=0.85 yields VaR=$250M (3.1× higher); correlation assumption critical for capital.
- **CDO Pricing:** Correlation of defaults drives senior/junior tranche spreads. Investment-grade portfolio correlation 0.25 yields senior AAA spread 20bps; correlation jumps to 0.70 in crisis → AAA spread becomes 150bps (7× higher); investors exposed to correlation risk.
- **Concentration Risk:** Two "independent" sectors show low portfolio-level correlation (0.15) in normal times. In recession, both defaults cluster (sector correlation 0.80); portfolio loses diversification benefit; tail risk underestimated.
- **Stress Testing:** Regulator stress test assumes historical correlation 0.35; bank models this statically. Crisis scenario with correlation 0.90 produces 10× more simultaneous defaults; existing capital insufficient.
- **Rating Migration:** Default correlation among BBB-rated corporates historically 2% over 1-year horizon. During 2008 crisis, BBB→D correlation spiked to 8% annually (4× higher); many models failed to forecast.

## 2. Comparative Framing
| Approach | Data | Estimation | Stability | Use Case |
|----------|------|-----------|----------|----------|
| **Pairwise Correlation** | Returns, spreads | Sample or model | Medium | Simple case |
| **Factor Model** | Market data | Regression | High | Large portfolios |
| **Copula** | Default times | Simulation | Medium | Tail modeling |
| **Implied** | Market prices | Reverse engineering | Low | Exotic pricing |

## 3. Examples + Counterexamples

**Simple Example:**  
Default correlation between two firms = 0.30. If Firm A defaults, probability Firm B defaults increases from 5% to ~8%

**Failure Case:**  
2008: Portfolio assumed low correlation (0.30), actual crisis correlation â‰ˆ 0.95 (all defaults cluster). VaR model underestimated loss 10x

**Edge Case:**  
Negative correlation rare in credit; borrowers move together in cycle. Possible with hedges or short positions
### 3B. Technical Counterexample: Gaussian Copula Assumption Failure in Tail Modeling

**Common Misconception:** "If I estimate pairwise default correlation = 0.25 using historical defaults and fit a Gaussian copula, my VaR models will accurately predict portfolio losses in downturns. The Gaussian copula is well-calibrated for credit risk tails."

**Why This Fails:** Gaussian copulas have zero tail dependence (default of one borrower doesn't increase probability of others defaulting in tail scenarios). During crises, empirical tail dependence is near 1.0. Gaussian copulas massively underestimate simultaneous defaults.

**Quantitative Example:**

**Portfolio Setup:**
- 100 corporate loans, uniform $10M each, $1B portfolio
- Assumed pairwise correlation: ρ = 0.25 (estimated from 10 years of data)
- Model: Gaussian copula with single risk factor
- Individual PD: 2% annually
- Fitting formula: $d_i = \beta \cdot Z + \sqrt{1-\beta^2} \cdot \epsilon_i$ where $Z \sim N(0,1)$ systemic factor
- $\beta$ chosen so pairwise default correlation ≈ 0.25

**Gaussian Copula Prediction (Normal Tail):**
- P(>5 simultaneous defaults) ≈ 0.5% (99th percentile of loss distribution)
- P(>10 simultaneous defaults) ≈ 0.05% (99.95th percentile)
- Expected loss in tail: ~$100M (10 defaults × $10M × 50% LGD)
- VaR(99.9%): $140M
- Capital (8%): $11.2M

**Empirical Reality (2008 -2009 Financial Crisis):**
- Same portfolio with same 100 borrowers experienced regime shift
- Actual simultaneous defaults: 28 loans defaulted within 12 months (2.8× expected)
- Realized loss: ~$140M (28 × $10M × 50% LGD)
- Actual tail scenario losses: $250M+(tail scenarios exceeded Gaussian prediction)

**Why Gaussian Failed:**
1. **Zero Tail Dependence:** Gaussian copula: P(both default | one defaults in tail) → 0 as tail gets more extreme
   - Empirical (crisis): P(both default | one defaults) → 0.70-0.95 (strong tail dependence)
2. **Single Factor:** Gaussian model assumes systemic factor Z; all correlation driven by single macro shock
   - Reality: Multiple sectoral, geographic, contagion factors; not single factor
3. **Stationarity:** Gaussian copula parameter β fixed over time (correlation = 0.25 always)
   - Reality: Crisis regime has β → 0.80-0.95 (much higher correlation)

**Student-t Copula Alternative:**
- Gaussian copula ν → ∞ (thin tails)
- Student-t copula: ν = 3-5 degrees of freedom (fat tails, higher tail dependence)
- With ν=4, Student-t copula produces tail dependence ≈ 0.30-0.40 (vs Gaussian 0)
- VaR(99.9%) with Student-t ≈ $220M (vs Gaussian $140M)
- Closer to realized losses but still underestimate $250M+

**Mixed Copula Approach:**
- Weight Gaussian (normal times) + Student-t (crisis) based on market stress indicator
- Stress indicator: CDS spread level, equity volatility (VIX), unemployment trend
- Normal times (VIX < 20): 80% Gaussian + 20% Student-t
- Crisis times (VIX > 40): 20% Gaussian + 80% Student-t
- Produces more accurate tail predictions by allowing correlation regime shift

**Real-World Case - Constant Maturity Bond Fund (2020 COVID Panic):**
- Fund held 150 investment-grade corporate bonds, estimated correlation 0.18 via Gaussian copula
- March 2020 COVID crash: Correlation spiked to 0.82 (simultaneous downgrades, CDS widening)
- Fund experienced 12% loss in single week (vs Gaussian VaR(95%) of 4%)
- Losses concentrated in tail scenarios where Gaussian assumes low correlation

**Regulatory Implications:** Basel III acknowledges copula limitations; requires banks to validate models under stress scenarios with elevated correlation assumptions. Many banks now use mixture copulas or regime-switching models to capture tail correlation changes. Supervisors conduct stress tests with explicit elevated correlation (ρ → 0.70-0.80) to stress-test portfolio resilience beyond Gaussian assumptions.

**Correct Approach:** Use multi-factor, regime-switching copulas that allow correlation to increase in tail scenarios. Always validate with historical crisis periods (2008, 2020) where tail correlation spiked. Apply correlation multipliers in stress tests (assume 2.5-3.0× baseline correlation in severe scenarios).
## 4. Layer Breakdown
```
Credit Correlation Framework:
â”œâ”€ Definition and Types:
â”‚   â”œâ”€ Pairwise default correlation: Ï(i,j) = Cov[D_i, D_j] / (Ïƒ_i Ã— Ïƒ_j)
â”‚   â”œâ”€ Asset correlation: Ï_A = correlation of firm values
â”‚   â”œâ”€ Conditional PD: P(default|other defaults) > P(default)
â”‚   â””â”€ Default intensity correlation: Î»_i and Î»_j co-move
â”œâ”€ Sources of Correlation:
â”‚   â”œâ”€ Systematic: Macro factors (interest rates, GDP, unemployment)
â”‚   â”œâ”€ Sector: Industry-specific (real estate, tech, energy)
â”‚   â”œâ”€ Contagion: Firm failure triggers others (counterparty risk)
â”‚   â”œâ”€ Liquidity: Market stress affects all borrowers
â”‚   â””â”€ Common ownership: Shared investors, collateral
â”œâ”€ Estimation Methods:
â”‚   â”œâ”€ Historical defaults:
â”‚   â”‚   â”œâ”€ Joint default frequency approach
â”‚   â”‚   â”œâ”€ Tetrachoric correlation on 2x2 table
â”‚   â”‚   â””â”€ Issue: Few joint defaults; high estimation error
â”‚   â”œâ”€ Market-implied:
â”‚   â”‚   â”œâ”€ From CDS prices via copula
â”‚   â”‚   â”œâ”€ More forward-looking than historical
â”‚   â”‚   â””â”€ Sensitive to liquidity, bid-ask spreads
â”‚   â”œâ”€ Factor models:
â”‚   â”‚   â”œâ”€ Ï_shared = Î²_i Ã— Î²_j Ã— Ï_factor + idiosyncratic
â”‚   â”‚   â”œâ”€ Single-factor: Merton-style asset correlation
â”‚   â”‚   â””â”€ Multi-factor: Systemic + sector + idio
â”‚   â””â”€ Copula approach:
â”‚       â”œâ”€ Model marginal defaults + joint structure
â”‚       â”œâ”€ Gaussian copula: Easy but tail-underestimating
â”‚       â””â”€ Student-t copula: Fat tails, higher correlation in crisis
â”œâ”€ Correlation Dynamics:
â”‚   â”œâ”€ Stability: Pairwise correlation â‰ˆ 0.3-0.5 in normal times
â”‚   â”œâ”€ Contagion: Correlation spikes during crisis (0.7-0.95)
â”‚   â”œâ”€ Regime-switching: Low vol normal, high vol crisis
â”‚   â”œâ”€ Term structure: Longer maturities show higher correlation
â”‚   â””â”€ Tail dependence: Correlation higher in tail (1% scenarios)
â”œâ”€ Portfolio Impact:
â”‚   â”œâ”€ Low correlation: Portfolio VaR << Î£ individual VaR (diversification)
â”‚   â”œâ”€ High correlation: Portfolio VaR â‰ˆ Î£ individual VaR (concentration)
â”‚   â”œâ”€ Diversification ratio: âˆšN in uncorrelated, 1 in perfectly correlated
â”‚   â””â”€ Convexity: Non-linear relationship between correlation and risk
â””â”€ Copulas (Joint Distribution):
    â”œâ”€ Gaussian copula: C(u,v) = Î¦(Î¦â»Â¹(u), Î¦â»Â¹(v), Ï)
    â”œâ”€ Clayton copula: Lower tail dependence
    â”œâ”€ Gumbel copula: Upper tail dependence
    â””â”€ Student-t copula: Symmetric tail dependence
```

## 5. Challenge Round
When is correlation estimation problematic?
- **Few joint defaults**: Historical correlation estimates have high error when joint defaults rare
- **Regime shifts**: Crisis correlation â‰  normal correlation; model may assume wrong regime
- **Non-stationarity**: Correlation changes with economic cycle; past â‰  future
- **Causality vs correlation**: Common factor vs direct contagion hard to distinguish
- **Portfolio-specific**: Correlation depends on composition; not portfolio-invariant

## 6. Key References
- [Copula Methods for Credit Risk - Wikipedia](https://en.wikipedia.org/wiki/Copula_(probability_theory)) - Joint distribution modeling with detailed copula families; explains Gaussian, Clayton, Gumbel, Student-t properties and tail dependence concepts.

- [Vasicek Asset Correlation Framework](https://en.wikipedia.org/wiki/Vasicek_model) - Single-factor Gaussian model underlying Basel III correlation assumptions; derivation and empirical calibration.

- Nelsen, R. B. (2006). "An Introduction to Copulas" (2nd ed.). Springer. Comprehensive mathematical treatment of copula theory; 300+ pages covering all families, properties, and credit risk applications.

- [Basel III Credit Risk Correlation](https://www.bis.org/basel_framework/chapter/CRE/40.htm) - Regulatory correlation formula: $\rho = 0.12(1-e^{-50 \times PD})/(1-e^{-50}) + 0.24[1-(1-e^{-50 \times PD})/(1-e^{-50})]$; empirical validation studies showing fit to historical defaults.

- Li, D. X. (2000). "On Default Correlation: A Copula Function Approach." Journal of Fixed Income, 9(4), 43-54. Seminal paper introducing copula methods to credit default modeling; widely used (and later criticized for Gaussian assumptions).

- Crouhy, M., Galai, D., & Mark, R. (2000). "A Comparative Analysis of Current Credit Risk Models." Journal of Banking & Finance, 24(1), 59-117. Benchmark comparison of correlation approaches (Vasicek, CreditMetrics, CreditRisk+); empirical performance assessment.

- Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk" (3rd ed.). McGraw-Hill. Chapter on correlation dynamics and VaR; includes 2008 crisis case studies showing correlation breakdown.

---
**Status:** Critical driver of portfolio tail risk | **Complements:** Credit VaR, Concentration Risk, Portfolio modeling
