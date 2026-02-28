# Concentration Risk

## 1. Concept Skeleton
**Definition:** Risk arising from uneven exposure distribution; large exposures to few borrowers/sectors amplify losses when they default; portfolio tail risk increases non-linearly with exposure inequality; measured via concentration indices (HHI, Gini, Numbers Equivalent)

**Purpose:** 
- Quantify portfolio vulnerability to idiosyncratic shocks and tail events
- Measure diversification effectiveness and identify concentration hotspots  
- Set exposure limits aligned with risk appetite and regulatory frameworks
- Link concentration to capital requirements under Basel III (granularity adjustment)
- Price concentration premiums into loan spreads and term structures
- Monitor portfolio drift and concentration evolution over time

**Prerequisites:** 
- Portfolio statistics (mean, variance, covariance of exposures)
- Concentration metrics (Herfindahl-Hirschman Index, Gini coefficient, CR₅ - concentration ratio of top 5)
- Exposure limits and single-name caps per borrower/sector/geography
- Granularity analysis (effective number of independent exposures)
- Correlation and dependency structures (Pearson, Spearman, copulas)
- Basel III capital framework and IRB approach
- Credit ratings and probability of default by exposure
- Stress testing scenarios for concentration breakdowns

**Use Cases:**
- **Portfolio Risk Management:** Bank with $10B loan portfolio must limit single borrower exposure to 2% ($200M max) to ensure single default doesn't exceed 4% capital loss
- **Stress Testing:** Regulator stress tests portfolio assuming top 10 borrowers default simultaneously; concentration metric predicts 15% portfolio loss (vs 2% average default)
- **M&A Integration:** Acquiring bank absorbs target's loan book; combined concentration may violate limits (HHI jumps from 0.015 to 0.025); must divest or syndicate exposures
- **Capital Optimization:** Bank allocates $2B capital to optimize return; concentrated portfolio of 5 large loans requires 12% capital; diversified 500-loan portfolio requires 8% capital
- **Regulatory Compliance:** EU Large Exposures Directive limits single-name to 25% of capital; concentration monitoring ensures compliance; violations trigger automatic deleveraging

## 2. Comparative Framing
| Metric | Calculation | Interpretation | Use |
|--------|------------|-----------------|-----|
| **Single-Name Limit** | Max exposure per borrower | Direct size constraint | Risk management |
| **Herfindahl Index** | âˆ‘(w_iÂ²) | Overall concentration (0-1) | Portfolio comparison |
| **Gini Coefficient** | Lorenz curve area | Inequality in exposures | Regulation |
| **Numbers Equivalent** | 1/HHI | Effective # independent exposures | Diversification measure |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio A: $1B spread over 1000 loans; Portfolio B: $1B with 10 loans. Portfolio B has 100x higher concentration risk

**Failure Case:**  
Lehman Brothers: Major bank concentration risk went unnoticed; default wiped out counterparties' capital

**Edge Case:**  
Perfectly diversified portfolio with correlated defaults (crisis); concentration looks low but tail risk extremely high

### 3B. Technical Counterexample: Correlation Breakdown and Concentration Metrics Failure

**Common Misconception:** "A portfolio with Herfindahl Index (HHI) of 0.02 is well-diversified and requires minimal capital buffer for concentration risk. As long as no single borrower exceeds 10% of capital, the portfolio is safe."

**Why This Fails:** This assumes statistically independent defaults and ignores systemic correlation dynamics. During crises, correlation jumps from 0.3 (normal times) to 0.8-0.95 (stress). The effective diversification benefit collapses.

**Quantitative Example:**

**Pre-Crisis Scenario (Normal Times):**
- Portfolio: 100 corporate loans, each $10M = $1B total
- Exposure weights: uniform (w_i = 0.01 for all i)
- Assumed pairwise default correlation: ρ = 0.25
- Expected default rate (EL): 2% annually
- HHI = Σ(w_i²) = 100 × (0.01)² = 0.01 (perfectly granular)
- Numbers Equivalent (N_eq) = 1/HHI = 100 (appears highly diversified)
- Mean default loss (MLE): $ 2% × $1B = $20M annually
- VaR (99% confidence, 1-year): Portfolio with correlation 0.25 yields VaR ≈ $60M  (per Vasicek formula)
- Capital requirement (8% ratio): $4.8M

**Crisis Scenario (Regime Shift):**
- Same 100 loans, but correlation spikes to ρ = 0.80 (systemic shock - e.g., pandemic)
- Default rates rise: PD jumps to 5% (previously 2%), LGD rises to 50% (from 40%)
- HHI remains 0.01 (structural unchanged)
- But effective correlation transforms portfolio risk:
  - Realized defaults: ~5 loans fail (vs expected 2)
  - Correlated defaults: Multiple simultaneous failures → cascade effect
  - Actual portfolio loss: $ 5% × $1B × 50% = $25M loss realized (vs $20M expected)
  - True VaR (99%, with ρ=0.80): VaR ≈ $150M+ (2.5× higher than modeled)
- Portfolio actually loses $125M in tail scenario, but capital reserved only $4.8M

**Why Models Failed:**
1. **Correlation Assumption:** HHI and metrics assume constant or low correlation; crisis correlation breaks assumption entirely
2. **Tail vs Mean:** HHI captures average case (mean loss ~$20M) but misses tail risk ($150M+)
3. **Contagion Not Modeled:** When one borrower in sector defaults, others in same sector more likely to fail; HHI treats all defaults independently
4. **Maturity Effects:** All loans mature same cycle; simultaneous defaults more clustered than historical data suggests

**Real-World Case - 2008 Financial Crisis:**
- Banks had portfolios with HHI ≈ 0.015 (seemed well-diversified)
- But exposures were concentrated in mortgages/real estate sector (sector HHI >> 0.10)
- When housing market crashed, correlation of all real estate defaults jumped to 0.95
- Regional bank portfolio with $500M real estate exposure faced 20% loss in one year (not forecasted)
- Capital reserved: $25M (5%); Actual loss: $100M

**Regulatory Response:** Basel III now requires stress testing concentration under high-correlation scenarios. Granularity adjustment (pg) formulas explicitly account for finite portfolio effects. Portfolio managers must report both "normal times HHI" and "stress correlation HHI."

**Correct Approach:** Conduct dual-metric risk assessment:
1. **Diversification metric:** HHI = 0.01 (low concentration)
2. **Stress correlation test:** Assume ρ = 0.70-0.80 in VaR/stress model → reveals true tail risk ($150M+)
3. **Granularity adjustment:** Apply pg = 0.5% × HHI = 0.005 capital surcharge (additional 40bps)
4. **Compare:** $4.8M base capital + ($1B × 40bps) = $8.8M total capital required (not $4.8M)

**Key Insight:** Concentration risk metrics measure CURRENT distributional inequality but fail to predict FUTURE correlation regimes. Always model concentration under stressed-correlation scenarios, not just historical/normal-time correlation.

## 4. Layer Breakdown
```
Concentration Risk Framework:
â”œâ”€ Concentration Metrics:
â”‚   â”œâ”€ Single-name exposure: Largest exposure as % of capital
â”‚   â”œâ”€ Sector exposure: Total exposure to one industry
â”‚   â”œâ”€ Geographic exposure: Regional or country concentration
â”‚   â”œâ”€ Counterparty concentration: Derivatives counterparty risk
â”‚   â””â”€ Collateral concentration: Similar security types
â”œâ”€ Herfindahl-Hirschman Index (HHI):
â”‚   â”œâ”€ Definition: HHI = âˆ‘áµ¢ (Wáµ¢)Â² where Wáµ¢ = weight of exposure i
â”‚   â”œâ”€ Range: 1/n (perfect diversification) to 1 (single exposure)
â”‚   â”œâ”€ Interpretation: HHI > 0.25 typically considered concentrated
â”‚   â””â”€ Regulatory: Used in merger analysis, capital rules
â”œâ”€ Gini Coefficient:
â”‚   â”œâ”€ Definition: Based on Lorenz curve (cumulative % exposures)
â”‚   â”œâ”€ Range: 0 (equal distribution) to 1 (single exposure)
â”‚   â”œâ”€ Formula: G = 1 - 2 âˆ‘áµ¢ (1/n) Ã— Lorenz(i)
â”‚   â””â”€ Interpretation: Higher = more concentrated
â”œâ”€ Numbers Equivalent (N_eq):
â”‚   â”œâ”€ N_eq = 1 / âˆ‘áµ¢ (wáµ¢)Â² = 1 / HHI
â”‚   â”œâ”€ Interpretation: Portfolio equivalent to N_eq equally-sized exposures
â”‚   â”œâ”€ Example: N_eq = 50 means portfolio = 50 diversified exposures
â”‚   â””â”€ Regulatory use: Basel III granularity adjustment
â”œâ”€ Large Exposure Limits:
â”‚   â”œâ”€ Single-name: Typically 10-15% of capital
â”‚   â”œâ”€ Sector cap: 25-50% of capital
â”‚   â”œâ”€ Related parties: Aggregate all connections
â”‚   â”œâ”€ Collateral haircuts: Value adjustments for concentration
â”‚   â””â”€ Stress scenarios: Apply losses to concentrated positions
â”œâ”€ Concentration Risk Sources:
â”‚   â”œâ”€ Business model: Bank lending inherently concentrated
â”‚   â”œâ”€ Portfolio drift: Active management changes concentrations
â”‚   â”œâ”€ Correlation clustering: Related defaults cluster
â”‚   â”œâ”€ Maturity mismatch: All exposures mature together
â”‚   â””â”€ Crisis amplification: Concentration effects magnify in downturns
â”œâ”€ Diversification Benefits:
â”‚   â”œâ”€ Naive: Portfolio risk â†“ as exposures increase
â”‚   â”œâ”€ Granularity: Diminishing returns after N_eq â‰ˆ 50-100
â”‚   â”œâ”€ Correlation benefit: Decreases in crisis
â”‚   â””â”€ Ultimate limit: Systematic risk cannot be diversified away
â””â”€ Regulatory Framework:
    â”œâ”€ Standardized approach: Single-name limits (25% of capital)
    â”œâ”€ IRB approach: Granularity adjustment (pg)
    â”œâ”€ Large exposures: EU limit 25% on single name
    â””â”€ Systemic: Systemically important institution limits
```

## 5. Challenge Round
When is concentration analysis problematic?
- **Correlation masking**: Two "independent" exposures may default together (sector, geographic)
- **Wrong aggregation**: Single-name limit prevents individual concentrations but allows sector concentration
- **Collateral correlation**: Diversified borrowers but same collateral type (real estate); concentrated collateral risk
- **Liquidity**: Diversified portfolio hard to liquidate if all exposures illiquid; fire-sale losses
- **Dynamic effects**: Concentration changes over time as positions mature, new loans added, prepayments occur

## 6. Key References
- Bouchaud, J-P., Potters, M., & Aguilar, J-P. (1997). "Missing Information and Asset Allocation." arXiv preprint cond-mat/9707042. Foundational work on information asymmetry in portfolio concentration; shows how incomplete information inflates concentration risk estimates.

- [Herfindahl Index - Wikipedia](https://en.wikipedia.org/wiki/Herfindahl_index) - Comprehensive overview of HHI origins (industrial organization), mathematical properties, and applications in finance.

- [Basel III Granularity Adjustment](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Official BIS regulatory framework defining granularity adjustment formula pg and its application to IRB capital.

- [Large Exposures Directive (2015/35)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32015L0035) - EU regulatory treatment of large exposures; limits single-name to 25% of capital; details aggregation rules for related parties.

- Gordy, M. B. (2003). "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules." Journal of Financial Intermediation, 12(3), 199-232. Theoretical derivation of concentration adjustments in Basel models; shows how HHI relates to portfolio tail risk.

- Pykhtin, M., & Zhu, S. (2006). "A Guide to Modeling Counterparty Credit Risk." GARP Risk Review, 28, 16-22. Advanced treatment of concentration in multi-counterparty portfolios with empirical backtesting.

- Barth, J. R., Jahera Jr, J. S., & Sauerhaft, D. (1980). "Concentration Risk in Consumer Lending." Journal of Financial Research, 3(2), 83-95. Empirical study linking exposure concentration to default losses during recession; documents pro-cyclicality.

---
**Status:** Key portfolio risk driver, fundamental to diversification | **Complements:** Credit VaR, Correlation, Portfolio limits
