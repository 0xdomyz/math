# Modern Portfolio Theory Topics Guide

**Complete reference of portfolio optimization, risk management, and asset allocation concepts with categories, brief descriptions, and sources.**

---

## I. Foundations & Assumptions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Modern Portfolio Theory (MPT)** | N/A | Markowitz framework; optimize expected return vs risk (variance) | [Investopedia - MPT](https://www.investopedia.com/terms/m/modernportfoliotheory.asp) |
| **MPT Assumptions** | N/A | Rational investors, normally distributed returns, no transaction costs, mean-variance utility | [Wiki - MPT](https://en.wikipedia.org/wiki/Modern_portfolio_theory) |
| **Expected Return** | N/A | Weighted average of asset returns; E[Rp] = Σ wi E[Ri] | [Investopedia - Expected Return](https://www.investopedia.com/terms/e/expectedreturn.asp) |
| **Portfolio Variance** | N/A | σp² = Σ wi² σi² + Σ Σ wi wj σi σj ρij; includes covariances | [Wiki - Portfolio Variance](https://en.wikipedia.org/wiki/Modern_portfolio_theory) |
| **Risk-Return Tradeoff** | N/A | Higher return requires accepting higher volatility; fundamental finance principle | [Investopedia - Risk Return](https://www.investopedia.com/terms/r/riskreturntradeoff.asp) |

---

## II. Portfolio Construction & Optimization

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Mean-Variance Optimization** | N/A | Minimize variance for target return or maximize return for target risk | [Wiki - Mean-Variance](https://en.wikipedia.org/wiki/Markowitz_model) |
| **Efficient Frontier** | N/A | Set of optimal portfolios offering highest return per risk level | [Investopedia - Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp) |
| **Minimum Variance Portfolio** | N/A | Portfolio with lowest possible risk; leftmost point on efficient frontier | [Investopedia - Min Variance](https://www.investopedia.com/terms/m/minimum-variance-portfolio.asp) |
| **Tangency Portfolio** | N/A | Portfolio with highest Sharpe ratio; optimal risky portfolio | [Wiki - Tangency Portfolio](https://en.wikipedia.org/wiki/Mutual_fund_separation_theorem) |
| **Portfolio Weights** | N/A | Asset allocation fractions wi; Σ wi = 1; can allow short selling (wi < 0) | [Investopedia - Portfolio Weight](https://www.investopedia.com/terms/p/portfolio-weight.asp) |
| **Constraints in Optimization** | N/A | No short selling (wi ≥ 0), concentration limits, sector constraints | [Wiki - Portfolio Optimization](https://en.wikipedia.org/wiki/Portfolio_optimization) |

---

## III. Diversification & Correlation

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Diversification** | N/A | Risk reduction through uncorrelated assets; "free lunch" of finance | [Investopedia - Diversification](https://www.investopedia.com/terms/d/diversification.asp) |
| **Correlation Coefficient** | N/A | ρij ∈ [-1,1]; measures linear co-movement between assets | [Investopedia - Correlation](https://www.investopedia.com/terms/c/correlationcoefficient.asp) |
| **Covariance** | N/A | Cov(Ri, Rj) = ρij σi σj; joint variability measure | [Investopedia - Covariance](https://www.investopedia.com/terms/c/covariance.asp) |
| **Systematic vs Idiosyncratic Risk** | N/A | Market risk (beta) vs firm-specific risk; diversification eliminates idiosyncratic | [Investopedia - Systematic Risk](https://www.investopedia.com/terms/s/systematicrisk.asp) |
| **Correlation Matrix** | N/A | n×n symmetric matrix of pairwise correlations; diagonal = 1 | [Wiki - Correlation Matrix](https://en.wikipedia.org/wiki/Covariance_matrix) |
| **Limits of Diversification** | N/A | Cannot eliminate systematic risk; diminishing marginal benefit beyond 20-30 stocks | [Investopedia - Over Diversification](https://www.investopedia.com/terms/o/overdiversification.asp) |

---

## IV. Risk-Free Asset & Capital Market Theory

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Risk-Free Rate** | N/A | T-bill rate; σf = 0; no default risk; time value of money baseline | [Investopedia - Risk Free Rate](https://www.investopedia.com/terms/r/risk-freerate.asp) |
| **Capital Allocation Line (CAL)** | N/A | Trade-off between risk-free asset and risky portfolio; linear in mean-std space | [Investopedia - CAL](https://www.investopedia.com/terms/c/cal.asp) |
| **Capital Market Line (CML)** | N/A | CAL using market portfolio; E[Rp] = rf + σp (E[Rm] - rf) / σm | [Investopedia - CML](https://www.investopedia.com/terms/c/cml.asp) |
| **Two-Fund Separation Theorem** | N/A | All investors hold same risky portfolio (market) + risk-free; differs only in proportions | [Wiki - Mutual Fund Separation](https://en.wikipedia.org/wiki/Mutual_fund_separation_theorem) |
| **Leverage & Margin** | N/A | Borrow at rf to invest beyond 100% in risky assets; extends CAL | [Investopedia - Leverage](https://www.investopedia.com/terms/l/leverage.asp) |

---

## V. Capital Asset Pricing Model (CAPM)

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **CAPM** | N/A | E[Ri] = rf + βi (E[Rm] - rf); equilibrium model; Nobel Prize (Sharpe 1990) | [Wiki - CAPM](https://en.wikipedia.org/wiki/Capital_asset_pricing_model) |
| **Beta (β)** | N/A | βi = Cov(Ri, Rm) / Var(Rm); systematic risk measure; market sensitivity | [Investopedia - Beta](https://www.investopedia.com/terms/b/beta.asp) |
| **Alpha (α)** | N/A | Excess return above CAPM prediction; active management skill indicator | [Investopedia - Alpha](https://www.investopedia.com/terms/a/alpha.asp) |
| **Security Market Line (SML)** | N/A | CAPM graphical representation; expected return vs beta | [Investopedia - SML](https://www.investopedia.com/terms/s/sml.asp) |
| **Market Portfolio** | N/A | All investable assets weighted by market cap; theoretically efficient | [Wiki - Market Portfolio](https://en.wikipedia.org/wiki/Market_portfolio) |
| **CAPM Assumptions** | N/A | Homogeneous expectations, mean-variance utility, no taxes/transaction costs | [Wiki - CAPM Assumptions](https://en.wikipedia.org/wiki/Capital_asset_pricing_model) |
| **CAPM Empirical Failures** | N/A | Size effect, value premium, momentum; anomalies violate CAPM predictions | [Wiki - CAPM](https://en.wikipedia.org/wiki/Capital_asset_pricing_model) |

---

## VI. Performance Measurement & Risk-Adjusted Returns

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Sharpe Ratio** | N/A | (E[Rp] - rf) / σp; return per unit total risk; higher is better | [Investopedia - Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) |
| **Treynor Ratio** | N/A | (E[Rp] - rf) / βp; return per unit systematic risk | [Investopedia - Treynor Ratio](https://www.investopedia.com/terms/t/treynorratio.asp) |
| **Information Ratio** | N/A | α / tracking error; active return per unit active risk | [Investopedia - Information Ratio](https://www.investopedia.com/terms/i/informationratio.asp) |
| **Jensen's Alpha** | N/A | Rp - [rf + βp (Rm - rf)]; CAPM-adjusted excess return | [Investopedia - Jensens Alpha](https://www.investopedia.com/terms/j/jensensmeasure.asp) |
| **Sortino Ratio** | N/A | Uses downside deviation instead of total volatility; penalizes only bad volatility | [Investopedia - Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp) |
| **Maximum Drawdown** | N/A | Peak-to-trough decline; measures worst loss experience | [Investopedia - Max Drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp) |
| **Tracking Error** | N/A | Standard deviation of (Rp - Rbenchmark); active risk measure | [Investopedia - Tracking Error](https://www.investopedia.com/terms/t/trackingerror.asp) |

---

## VII. Factor Models & Extensions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Arbitrage Pricing Theory (APT)** | N/A | Multi-factor model; Ri = E[Ri] + Σ βik Fk + εi; less restrictive than CAPM | [Wiki - APT](https://en.wikipedia.org/wiki/Arbitrage_pricing_theory) |
| **Fama-French Three-Factor Model** | N/A | Market + Size (SMB) + Value (HML); explains more variation than CAPM | [Wiki - Fama French](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model) |
| **Carhart Four-Factor Model** | N/A | Fama-French + Momentum (WML); captures trend-following premium | [Wiki - Carhart](https://en.wikipedia.org/wiki/Carhart_four-factor_model) |
| **Fama-French Five-Factor Model** | N/A | + Profitability (RMW) + Investment (CMA); current research standard | [Fama French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |
| **Factor Loadings** | N/A | Sensitivity βik to factor k; analogous to CAPM beta | [Investopedia - Factor Loading](https://www.investopedia.com/terms/f/factor-loading.asp) |
| **Style Analysis** | N/A | Decompose portfolio into factor exposures; Sharpe (1992) returns-based | [Wiki - Style Analysis](https://en.wikipedia.org/wiki/Style_analysis) |

---

## VIII. Utility Theory & Investor Preferences

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Mean-Variance Utility** | N/A | U = E[Rp] - (λ/2) σp²; risk aversion parameter λ | [Wiki - Mean Variance Analysis](https://en.wikipedia.org/wiki/Mean-variance_analysis) |
| **Risk Aversion** | N/A | Concave utility function; prefer certain outcome over gamble with same mean | [Investopedia - Risk Averse](https://www.investopedia.com/terms/r/riskaverse.asp) |
| **Indifference Curves** | N/A | Constant utility combinations of return and risk; upward sloping | [Wiki - Indifference Curve](https://en.wikipedia.org/wiki/Indifference_curve) |
| **Optimal Portfolio Selection** | N/A | Tangency of indifference curve with CAL; depends on risk tolerance | [Investopedia - Optimal Portfolio](https://www.investopedia.com/terms/o/optimal-portfolio.asp) |
| **Certainty Equivalent Return** | N/A | Risk-free rate yielding same utility as risky portfolio | [Investopedia - Certainty Equivalent](https://www.investopedia.com/terms/c/certaintyequivalent.asp) |
| **Expected Utility Theory** | N/A | von Neumann-Morgenstern; axioms of rational choice under uncertainty | [Wiki - Expected Utility](https://en.wikipedia.org/wiki/Expected_utility_hypothesis) |

---

## IX. Behavioral Finance Critiques

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Behavioral Portfolio Theory** | N/A | Mental accounting, loss aversion; deviations from MPT rationality | [Wiki - Behavioral Portfolio](https://en.wikipedia.org/wiki/Behavioral_portfolio_theory) |
| **Prospect Theory** | N/A | Kahneman-Tversky; value function kinked at reference point; Nobel 2002 | [Wiki - Prospect Theory](https://en.wikipedia.org/wiki/Prospect_theory) |
| **Loss Aversion** | N/A | Losses hurt ~2x more than equivalent gains; asymmetric utility | [Investopedia - Loss Aversion](https://www.investopedia.com/terms/l/loss-psychology.asp) |
| **Home Bias** | N/A | Overweight domestic assets; violates global diversification | [Investopedia - Home Bias](https://www.investopedia.com/terms/h/home-bias.asp) |
| **Overconfidence** | N/A | Excessive trading, underestimation of risk; degrades performance | [Investopedia - Overconfidence](https://www.investopedia.com/terms/o/overconfidence.asp) |
| **Herding & Momentum** | N/A | Following crowd; trend-chasing behavior; explains momentum anomaly | [Wiki - Herd Behavior](https://en.wikipedia.org/wiki/Herd_behavior) |

---

## X. Practical Implementation & Limitations

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Estimation Error** | N/A | Input sensitivity; small changes in μ, Σ → large weight changes | [Investopedia - Estimation Risk](https://www.investopedia.com/terms/e/estimation-risk.asp) |
| **Black-Litterman Model** | N/A | Bayesian approach; combines market equilibrium with investor views | [Wiki - Black Litterman](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) |
| **Resampling & Robust Optimization** | N/A | Michaud resampled frontier; address instability in mean-variance | [Investopedia - Resampled Efficiency](https://www.investopedia.com/terms/r/resampled-efficiency-frontier.asp) |
| **Transaction Costs** | N/A | Bid-ask spread, commissions, market impact; reduce turnover | [Investopedia - Transaction Costs](https://www.investopedia.com/terms/t/transactioncosts.asp) |
| **Taxes** | N/A | Capital gains, dividends; tax-loss harvesting; after-tax optimization | [Investopedia - Tax Loss Harvesting](https://www.investopedia.com/terms/t/taxgainlossharvesting.asp) |
| **Non-Normal Returns** | N/A | Fat tails, skewness; variance insufficient; use VaR, CVaR, higher moments | [Investopedia - Fat Tail](https://www.investopedia.com/terms/f/fat-tail.asp) |
| **Time-Varying Parameters** | N/A | μ, σ, ρ change over time; regime-switching models; dynamic allocation | [Wiki - Regime Switching](https://en.wikipedia.org/wiki/Regime_switching) |

---

## XI. Alternative Risk Measures

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Value at Risk (VaR)** | N/A | Maximum loss at confidence level α (e.g., 95%); quantile-based | [Investopedia - VaR](https://www.investopedia.com/terms/v/var.asp) |
| **Conditional VaR (CVaR)** | N/A | Expected shortfall; average loss beyond VaR; coherent risk measure | [Wiki - CVaR](https://en.wikipedia.org/wiki/Expected_shortfall) |
| **Semi-Variance** | N/A | Downside risk only; focuses on returns below mean or target | [Investopedia - Semivariance](https://www.investopedia.com/terms/s/semivariance.asp) |
| **Lower Partial Moments (LPM)** | N/A | Generalized downside risk; LPM(n,τ) = E[max(τ - R, 0)^n] | [Wiki - Lower Partial Moment](https://en.wikipedia.org/wiki/Downside_risk) |
| **Risk Parity** | N/A | Equal risk contribution from each asset; diversifies risk not capital | [Investopedia - Risk Parity](https://www.investopedia.com/terms/r/risk-parity.asp) |
| **Maximum Loss** | N/A | Worst historical or scenario outcome; stress testing | [Investopedia - Stress Testing](https://www.investopedia.com/terms/s/stresstesting.asp) |

---

## XII. Multi-Period & Dynamic Models

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Intertemporal CAPM** | N/A | Merton (1973); hedging demands for changing investment opportunities | [Wiki - ICAPM](https://en.wikipedia.org/wiki/Intertemporal_CAPM) |
| **Consumption CAPM** | N/A | Asset prices linked to consumption growth; equity premium puzzle | [Wiki - CCAPM](https://en.wikipedia.org/wiki/Consumption-based_capital_asset_pricing_model) |
| **Dynamic Programming** | N/A | Bellman equation; optimal control for multi-period decisions | [Wiki - Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) |
| **Strategic Asset Allocation** | N/A | Long-term policy weights; reflects investor goals and constraints | [Investopedia - Strategic Allocation](https://www.investopedia.com/terms/s/strategicassetallocation.asp) |
| **Tactical Asset Allocation** | N/A | Short-term deviations from strategic; market timing attempts | [Investopedia - Tactical Allocation](https://www.investopedia.com/terms/t/tacticalassetallocation.asp) |
| **Rebalancing** | N/A | Periodic adjustment to target weights; contrarian trade; sell high buy low | [Investopedia - Rebalancing](https://www.investopedia.com/terms/r/rebalancing.asp) |

---

## XIII. International & Global Portfolios

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Currency Risk** | N/A | Exchange rate volatility; hedged vs unhedged foreign assets | [Investopedia - Currency Risk](https://www.investopedia.com/terms/c/currencyrisk.asp) |
| **Purchasing Power Parity (PPP)** | N/A | Exchange rates adjust for inflation differentials; long-run equilibrium | [Wiki - PPP](https://en.wikipedia.org/wiki/Purchasing_power_parity) |
| **Interest Rate Parity** | N/A | Forward rates reflect interest differentials; no arbitrage condition | [Wiki - Interest Rate Parity](https://en.wikipedia.org/wiki/Interest_rate_parity) |
| **International Diversification** | N/A | Cross-country correlation < 1; reduces portfolio risk | [Investopedia - International Diversification](https://www.investopedia.com/terms/i/international-portfolio-diversification.asp) |
| **Emerging Markets** | N/A | Higher returns, higher risk; segmentation, political risk, liquidity | [Investopedia - Emerging Markets](https://www.investopedia.com/terms/e/emergingmarketeconomy.asp) |

---

## XIV. Alternative Assets & Extensions

| Topic | File | Description | Source |
|-------|------|-------------|--------|
| **Real Estate** | N/A | Illiquid, low correlation with stocks; inflation hedge | [Investopedia - Real Estate Investment](https://www.investopedia.com/terms/r/realestate.asp) |
| **Commodities** | N/A | Gold, oil, agriculture; inflation protection, backwardation/contango | [Investopedia - Commodities](https://www.investopedia.com/terms/c/commodity.asp) |
| **Private Equity** | N/A | Illiquidity premium, J-curve, selection bias in returns | [Investopedia - Private Equity](https://www.investopedia.com/terms/p/privateequity.asp) |
| **Hedge Funds** | N/A | Alternative strategies; absolute return, leverage, short selling | [Investopedia - Hedge Fund](https://www.investopedia.com/terms/h/hedgefund.asp) |
| **Cryptocurrencies** | N/A | High volatility, speculative; correlation regime shifts | [Investopedia - Cryptocurrency](https://www.investopedia.com/terms/c/cryptocurrency.asp) |

---

## Reference Sources

| Source | URL | Coverage |
|--------|-----|----------|
| **CFA Institute** | https://www.cfainstitute.org | Professional standards; portfolio management curriculum |
| **Investopedia Portfolio Management** | https://www.investopedia.com/portfolio-management-4689745 | Practical guides; MPT, CAPM, performance measurement |
| **Fama-French Data Library** | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | Factor returns data; academic research standard |
| **Wikipedia Modern Portfolio Theory** | https://en.wikipedia.org/wiki/Modern_portfolio_theory | Mathematical foundations; Markowitz optimization |
| **Journal of Portfolio Management** | https://jpm.pm-research.com | Academic research; practitioner applications |

---

## Quick Stats

- **Total Topics Documented**: 95+
- **Workspace Files Created**: 0
- **Categories**: 14
- **Key Contributors**: Markowitz (1952), Sharpe (1964), Fama-French (1992)
- **Coverage**: Theory → Optimization → Risk Management → Implementation
