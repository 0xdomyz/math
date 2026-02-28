# Credit Value-at-Risk (Credit VaR)

## 1. Concept Skeleton
**Definition:** Maximum portfolio loss from credit events at specified confidence level over holding period; e.g., 99% Credit VaR = worst-case 1-year loss with 1% probability; tail loss percentile in credit loss distribution; value statistic used in capital planning and risk limits.

**Purpose:**
- Quantify tail credit risk for regulatory capital requirements (Basel III mandates 99.9% confidence)
- Set economic capital buffers aligned with risk tolerance and confidence intervals
- Compare credit portfolios on risk-adjusted basis (VaR density per unit exposure)
- Validate pricing margins sufficient to cover unexpected losses in tail scenarios 
- Monitor VaR evolution and concentration effects dynamically over time
- Stress-test under combined scenarios: elevated PD/LGD with correlated defaults

**Prerequisites:**
- Value-at-Risk framework (parametric, historical simulation, Monte Carlo methods)
- Credit metrics: PD (probability of default), LGD (loss given default), EAD (exposure at default)
- Portfolio statistics: exposure sizes, correlations, maturity distributions
- Loss distributions: empirical or analytical (binomial, beta, lognormal)
- Confidence level selection (95% internal, 99% regulatory standard, 99.9% systemic)

## 2. Comparative Framing
| Risk Measure | Statistic | Time Horizon | Use Case | Advantage |
|--------------|-----------|-------------|----------|-----------|
| **Credit VaR** | Percentile loss | 1-year typical | Capital requirement | Intuitive threshold |
| **Expected Loss** | Mean loss | Annual | Provisioning/pricing | Conservative baseline |
| **Expected Shortfall** | Mean of tail | Beyond VaR | Regulatory (Basel III) | Captures tail severity |
| **Stress Testing** | Scenario loss | Variable | Extreme event | Model-independent |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio 99% Credit VaR = $50M over 1 year. In worst 1% of outcomes, lose â‰¥ $50M; expected loss only $5M

**Failure Case:**  
VaR ignores correlation breakdown during crisis. 2008: Portfolio VaR 99% = $100M assumed, actual loss $500M when correlations â†’ 1

**Edge Case:**  
VaR of $0 when no defaults likely in 99% of paths. True for AAA portfolios; uninformative but technically correct

## 4. Layer Breakdown
```
Credit VaR Framework:
â”œâ”€ Definition:
â”‚   â”œâ”€ VaR(Î±) = Loss amount L such that P(Loss > L) = 1 - Î±
â”‚   â”œâ”€ Example: VaR(99%) = 99th percentile of loss distribution
â”‚   â”œâ”€ Holding period: Typically 1 year for credit risk
â”‚   â””â”€ Confidence level: 99% (regulatory), 95% (internal)
â”œâ”€ Calculation Methods:
â”‚   â”œâ”€ Parametric (delta-normal):
â”‚   â”‚   â”œâ”€ Assume normal loss distribution
â”‚   â”‚   â”œâ”€ VaR = Î¼ + Ïƒ Ã— Z_Î±
â”‚   â”‚   â””â”€ Fast but may underestimate tails
â”‚   â”œâ”€ Historical simulation:
â”‚   â”‚   â”œâ”€ Use empirical loss distribution
â”‚   â”‚   â”œâ”€ Sort historical losses, pick percentile
â”‚   â”‚   â””â”€ No distribution assumption but limited history
â”‚   â”œâ”€ Monte Carlo:
â”‚   â”‚   â”œâ”€ Simulate asset values, default scenarios
â”‚   â”‚   â”œâ”€ Calculate loss in each path
â”‚   â”‚   â””â”€ Flexible but computationally intensive
â”‚   â””â”€ Intensity-based (reduced-form):
â”‚       â”œâ”€ Model default as Poisson jump
â”‚       â”œâ”€ Calibrate to historical/market data
â”‚       â””â”€ Hierarchical computation
â”œâ”€ Loss Distribution Components:
â”‚   â”œâ”€ Expected loss (EL): First moment, mean
â”‚   â”œâ”€ Unexpected loss (UL): Volatility around mean
â”‚   â”œâ”€ VaR: Combines EL + multiple of UL
â”‚   â”œâ”€ Tail risk: Losses beyond VaR (Expected Shortfall)
â”‚   â””â”€ Skewness: Asymmetry (credit losses skewed left)
â”œâ”€ Portfolio VaR:
â”‚   â”œâ”€ Single-name VaR: Individual exposure
â”‚   â”œâ”€ Diversification benefit: VaR_portfolio < Î£ VaR_i
â”‚   â”œâ”€ Correlation impact: Higher correlation â†’ Higher portfolio VaR
â”‚   â””â”€ Concentration: Large exposures dominate VaR
â”œâ”€ VaR Dynamics:
â”‚   â”œâ”€ Time-varying: Increases in crisis (correlation spike)
â”‚   â”œâ”€ Term structure: Multi-year VaR > 1-year
â”‚   â”œâ”€ Regime-dependent: High vs low volatility states
â”‚   â””â”€ Liquidity impact: Illiquid portfolios have higher VaR
â””â”€ Limitations:
    â”œâ”€ Tail risk: VaR ignores magnitude of losses > VaR
    â”œâ”€ Non-subadditivity: Diversifying may increase VaR
    â”œâ”€ Model risk: Sensitive to distributional assumptions
    â””â”€ Fat tails: Actual losses exceed model VaR in crisis
```

## 5. Challenge Round
When is Credit VaR problematic?
- **Model risk**: Normal distribution assumption fails; actual losses have fat tails
- **Correlation changes**: VaR assumes stable correlations; crises show jumps to 1.0
- **Tail events**: VaR ignores losses beyond the threshold (Expected Shortfall better)
- **Liquidity**: Illiquid portfolios harder to value; mark-to-market prices stale
- **Non-additivity**: Diversification can increase VaR (tail concentration risk)

## 6. Key References
- [Basel III Credit Risk VaR Framework](https://www.bis.org/basel_framework/chapter/CRE/40.htm) - Official regulatory credit VaR standards; formula for Standard Approach; IRB correlation assumptions; stress testing requirements for parameter validation.

- Dowd, K. (2007). "Measuring Market Risk" (2nd ed.). John Wiley & Sons. Comprehensive treatment of VaR estimation methods (parametric, historical, Monte Carlo); covers credit VaR applications and limitations including non-subadditivity.

- Cornish, E. A., & Fisher, R. A. (1938). "Moments and Cumulants in the Specification of Distributions." Revue de l'Institut International de Statistique, 5(4), 307-320. Foundation for Cornish-Fisher VaR using higher moments; addresses normal distribution assumption failures and skewness/kurtosis corrections.

- [Vasicek Portfolio Model](https://www.vasicek.com/londonfinancial.pdf) - Foundational one-factor portfolio loss distribution model; derives portfolio VaR formula used in Basel III; shows how correlation parameter drives tail risk aggregation.

- Gordy, M. B. (2000). "A Comparative Anatomy of Credit Risk Models." Journal of Banking & Finance, 24(1), 119-149. Benchmarking of CreditMetrics, CreditRisk+, and portfolio models; empirical VaR estimates under different loss distribution assumptions.

- Lütkebohmert, E. (2009). "Concentration Risk in Credit Portfolios." Springer-Verlag. Monograph on VaR behavior in concentrated portfolios; shows how HHI impacts VaR non-linearly; includes case studies from 2008 crisis highlighting VaR underestimation.

---
**Status:** Core portfolio risk metric for capital planning | **Complements:** Expected Loss, Correlation, Concentration Risk
