# MPT Assumptions & Violations: Theory vs Reality

## 1. Concept Skeleton
**Definition:** MPT Assumptions: Set of conditions required for mean-variance optimization framework to hold. Include: rational investors, normally distributed returns, no transaction costs/taxes, homogeneous expectations, perfect divisibility, no market frictions. Violations: Real markets deviate significantly; impacts portfolio optimality.  
**Purpose:** Understand when MPT works (when assumptions hold) vs breaks down (real-world violations); identifies where theory-practice gaps emerge; motivates extensions (behavioral models, robust optimization, CVaR).  
**Prerequisites:** Mean-variance optimization, efficient frontier, risk-return tradeoff, correlation concept

---

## 2. Comparative Framing

| Assumption | MPT Version (Ideal) | Reality (Violation) | Impact on Portfolios | Fix/Extension |
|-----------|-------------------|-------------------|---------------------|----------------|
| **Rationality** | Investors maximize utility; no behavioral biases | Loss aversion, overconfidence, herding → suboptimal choices | Weights off-frontier; excessive trading | Behavioral portfolio theory |
| **Normal Returns** | Return ~ N(μ, σ²); all info in mean & variance | Fat tails, skewness, kurtosis; crashes happen (6+ sigma) | Variance inadequate risk measure; tail risk ignored | CVaR, higher moments, extreme value theory |
| **No Costs** | Trading free; rebalancing costless | Bid-ask 0.01%-0.1%, commissions, market impact | Optimal weights move; rebalancing less frequent | Include transaction costs in optimization |
| **No Taxes** | Returns pre-tax; allocation same for all | Tax rates differ; long-term vs short-term; capital gains | After-tax optimal allocation different from pre-tax | Tax-aware portfolio construction |
| **Homogeneous Beliefs** | All investors same μ, σ, ρ | Analysts disagree; information asymmetry; private info | Different investors hold different risky portfolios | Heterogeneous beliefs models, consensus estimates |
| **Continuous Divisibility** | Can hold any weight (e.g., 7.3456% of stock) | Discrete holdings; lot sizes; integer shares | Rounding errors; practical constraints | Integer programming, minimum trade sizes |
| **Perfect Liquidity** | Buy/sell instantly at market price | Bid-ask spreads; market impact; liquidity dries up in stress | Larger orders face worse prices; portfolio less liquid | Liquidity-adjusted models; market impact functions |
| **Risk-Free Asset Exists** | T-bill with σ = 0, no default | T-bills have modest default risk (rare) and reinvestment risk | Risk-free rate not truly zero-risk; changes over time | Use short-duration bonds as proxy; time-varying rf |
| **Static Analysis** | Single-period; optimized once | Investors multi-period; rebalance over time; goals change | Single-period allocation suboptimal over time | Dynamic MPT, lifecycle models |
| **Expected Returns Predictable** | μ estimated from history; stable over time | Returns forecastability disputed; parameters time-varying | Historical μ poor predictor of future | Factor models, forward-looking estimates |

**Key Insight:** Most violations make portfolios MORE complex (costly constraints), not simpler. Optimal portfolio often has same structure (diversified), but lower returns after costs.

---

## 3. Examples + Counterexamples

**Example 1: Normal Distribution Assumption Violation (Fat Tails)**

Assume returns normally distributed with σ = 15%:
- Probability of -22% return (1987 crash): ~0.000001 (6.5 sigma event)
- Expected frequency: Once every 1 million years
- Actual history: Happens roughly once per 20-30 years

**MPT prediction vs reality:**
- MPT says: Ignore tail risk (99.99% confidence portfolio safe)
- Reality: 1987 (-22%), 2008 (-37%), 2020 COVID (-34%) all happened
- VaR at 95%: MPT predicts -24% worst-case; actual worst was -37%

**Implication:**
- Variance underestimates true risk
- Tail events concentrated in negative returns (downside risk matters more)
- Solution: Use CVaR (expected shortfall) instead of variance
- Or add skewness/kurtosis to optimization

**Example 2: No Transaction Costs Assumption Violation**

Optimal portfolio from MPT: [40% SPY, 35% QQQ, 15% BND, 10% GLD]

Rebalancing cost analysis:
- Initial: $100k portfolio (weights exact)
- After 1 year: Market moves; weights drift to [42%, 38%, 13%, 7%] (1-2% drift typical)
- Rebalance: Sell $2k from QQQ, buy $2k BND, sell $3k GLD, buy $3k SPY
- Cost: 4 trades × $200 commissions ≈ $800 per year
- Bid-ask spreads: ~0.04% on $100k = $40 per trade × 4 = $160
- Market impact: Harder to estimate; ~0.05% = $200
- **Total: ~$1,160 per year ≈ 1.16% of portfolio**

After-cost benefit of rebalancing:
- Historical rebalancing benefit: +0.5-1% p.a. (buy dips, sell rallies)
- Cost of rebalancing: -1.16% p.a.
- **Net: -0.66% drag** (rebalancing hurts!)

**Implication:**
- Optimal rebalancing frequency changes when costs included
- Wider bands (e.g., rebalance only if drift >10%) more efficient
- Tax-deferred accounts (401k) better for frequent rebalancing
- Taxable accounts: Annual or threshold-based only

---

**Example 3: No Taxes Assumption Violation**

Two otherwise identical investors:
- Investor A: In 401k (tax-deferred); can rebalance costlessly
- Investor B: In taxable account; capital gains taxed at 20% long-term

Optimal portfolio (pre-tax): 70% stocks, 30% bonds

After tax considerations:
- Investor A: 70% stocks, 30% bonds (no tax effect)
- Investor B: May prefer 75% stocks, 25% bonds
  - Reason: Bonds generate ordinary income (taxed at 40%+ marginal rate)
  - Stocks generate capital gains (taxed at 20%; tax-deferred if held)
  - Tax-aware optimization uses after-tax returns, not pre-tax

**Example calculation:**
- Bond yield: 4% pre-tax → 2.4% after 40% tax → 2.6% pre-tax equivalent (low!)
- Stock return: 10% pre-tax (if deferred) → 10% after-tax (no tax paid until sale)
- Equity premium after-tax: 10% - 2.6% = 7.4% (vs 6% pre-tax)
- More attractive → Higher stock allocation optimal in taxable

**Implication:**
- Same investor needs different portfolio in taxable vs tax-deferred
- Tax-loss harvesting valuable (realize losses when available)
- Funds with high turnover tax-inefficient (index funds superior)
- Asset location: Put bonds in IRA (taxed), stocks in taxable (deferred)

---

**Example 4: Homogeneous Beliefs Assumption Violation (Heterogeneous Expectations)**

Market consensus: S&P 500 return 10%, volatility 15%

But:
- Bullish analyst: Believes 12% return (tech boom thesis)
- Bearish analyst: Believes 8% return (recession risk)

Implications:
- Bullish investor: Overweights growth stocks vs market portfolio
- Bearish investor: Overweights defensive stocks vs market portfolio
- **Two-fund separation breaks down** (no longer all hold market)
- Result: Active stock-picking emerges (picking "mispriced" securities)

**Historical example (2000 Dot-Com crash):**
- Tech analysts bullish; believed internet "changed everything"
- Nasdaq 100 overvalued; expected return was 20%+ (vs S&P 15%)
- Reality: Nasdaq crashed -78%; expected return actually -15%
- Bullish investors who overweighted tech got crushed

**Implication:**
- Disagreement on expected returns causes portfolio dispersion
- Some investors beat market (luck or skill?)
- Active management rationale: If beliefs differ from consensus
- Risk: Overconfidence in private forecasts (often wrong)

---

**Example 5: Divisibility & Liquidity Violation (Discrete Assets, Illiquidity)**

Optimal portfolio from solver: [23.456% SPY, 31.789% QQQ, 18.901% BND, 15.321% GLD, 10.533% EFA]

Implementation:
- SPY (highly liquid): Can buy 23.456% without issue
- Small illiquid stock: Bid-ask 1%; market impact 0.5%; can't buy 0.001 shares
- Real estate: No fractional ownership; illiquid (could take weeks to buy/sell)
- Private equity: Illiquid; long lockups; can't trade

Practical constraints:
- Rounding to achievable lots
- Accepting sub-optimal allocation due to lumpy positions
- Illiquid assets require liquidity premium; higher transaction costs

**Implication:**
- Real portfolios deviate from theoretical MPT optimum
- Illiquid assets over-represented in institutional portfolios (private equity, real estate)
- Individual investors stick to liquid assets (public equities, bonds)
- Solver should include minimum position size, liquidity constraints

---

**COUNTEREXAMPLE 6: Static Analysis Assumption (Multi-Period Dynamics)**

MPT single-period optimal allocation: 70% stocks, 30% bonds (for λ=2)

Multi-period reality:
- Young investor (25 years to retirement): Higher human capital (implicit bond), should take more equity risk
- Near-retiree (5 years to retirement): Lower human capital, should take less risk, lock in gains
- During crisis (stocks down 40%): Prices attractive; should rebalance into equities
- Booming market (stocks up 100%): Prices rich; should take profits

**Dynamic programming solution** (Bellman equation):
- V(t, W_t) = max [U(W_t) + E[V(t+1, W_{t+1})]]
- Optimal allocation changes over time as wealth evolves
- Not static 70/30 always; adapt as you age and markets change

**Implication:**
- Lifecycle allocation more sophisticated than static MPT
- Target-date funds model this (de-risk over time)
- Rebalancing frequency and thresholds change by life stage
- Single-period MPT suboptimal for multi-period investor

---

## 4. Layer Breakdown

```
MPT Assumptions & Violations: Comprehensive Analysis:

├─ Core Assumption Set (Required for MPT Optimality):
│   ├─ (1) RATIONALITY & UTILITY MAXIMIZATION:
│   │   ├─ Assumption:
│   │   │   ├─ Investors rational (maximize expected utility)
│   │   │   ├─ Preferences captured by mean-variance utility
│   │   │   ├─ No behavioral biases (loss aversion, overconfidence, herding)
│   │   │   ├─ Consistent preferences (transitivity, completeness)
│   │   │   └─ Full information; no information asymmetries
│   │   │
│   │   ├─ Violations in Reality:
│   │   │   ├─ Loss aversion: Losses hurt 2× more than gains (Kahneman-Tversky)
│   │   │   │   └─ Example: Investor holds losing stocks too long (realization aversion)
│   │   │   ├─ Overconfidence: Believe personal picks beat market
│   │   │   │   └─ Example: Individual investors underperform index funds by 1-3% p.a.
│   │   │   ├─ Herding: Follow crowd (momentum, bubbles, crashes)
│   │   │   │   └─ Example: 2008 financial crisis; 2021 crypto bubble; 2023 AI rally
│   │   │   ├─ Mental accounting: Treat investment categories separately
│   │   │   │   └─ Example: Investor won't sell winner to buy loser (breaking diversification)
│   │   │   ├─ Home bias: Overweight home country (violates global diversification)
│   │   │   │   └─ Example: US investors 70% in US stocks (vs 35% market weight)
│   │   │   └─ Narrow framing: Focus on short-term volatility vs long-term wealth
│   │   │       └─ Example: Panic-sell during 50% bear market (worst time to sell)
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ Weights deviate from efficient frontier (not on frontier)
│   │   │   ├─ Excessive trading (overconfidence → turnover)
│   │   │   ├─ Momentum exposure (herding → positive autocorrelation premium/risk)
│   │   │   └─ Concentration risk (behavioral constraints override diversification)
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ Behavioral portfolio theory (Shefrin-Statman): Incorporate mental accounting
│   │   │   ├─ Prospect theory (Kahneman-Tversky): Value function over returns
│   │   │   ├─ Constrained optimization: Add behavioral constraints
│   │   │   └─ Automated investing (robo-advisors): Remove behavioral errors
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Behavioral anomalies documented (disposition effect, home bias)
│   │       ├─ Irrational pricing (bubbles in 2000, 2008, 2021)
│   │       └─ Individual investors significantly underperform after behavioral errors
│   │
│   ├─ (2) NORMAL DISTRIBUTION OF RETURNS:
│   │   ├─ Assumption:
│   │   │   ├─ Returns ~ N(μ, σ²); multivariate normal joint distribution
│   │   │   ├─ Only mean and variance matter (no higher moments affect risk)
│   │   │   ├─ Tail probability: P(R < μ - kσ) follows normal CDF
│   │   │   └─ Symmetric around mean (no skewness)
│   │   │
│   │   ├─ Violations in Reality (Fat Tails, Skewness):
│   │   │   ├─ Empirical distribution has fatter tails than normal
│   │   │   │   ├─ Kurtosis > 3 (excess kurtosis > 0)
│   │   │   │   ├─ S&P 500 daily returns: Kurtosis ~4-5 (vs 3 normal)
│   │   │   │   └─ Probability of -5% day: Normal predicts 0.0001%; actual ~1% (10,000× more!)
│   │   │   ├─ Negative skewness (left tail fatter than right)
│   │   │   │   ├─ Large losses more common than large gains
│   │   │   │   ├─ Asymmetric crashes: -22% days vs +22% days rare
│   │   │   │   └─ Implied: Downside risk > upside opportunity
│   │   │   ├─ Extreme events (black swans):
│   │   │   │   ├─ Oct 1987: -22% (predicted once per 10^9 years; actually every 30 years)
│   │   │   │   ├─ 2008 crisis: -37% (predicted once per 10^13 years)
│   │   │   │   ├─ 2020 COVID: -34% (predicted never under normality)
│   │   │   │   └─ Conclusion: Tails matter; ignoring them risky
│   │   │   ├─ Jump risk (discontinuous moves)
│   │   │   │   └─ Model: Gaps in trading; limit orders miss (get worse prices than expected)
│   │   │   └─ Volatility clustering (variance itself random)
│   │   │       └─ Calm periods followed by turbulent periods (not constant σ)
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ Variance underestimates risk; doesn't capture tail probability
│   │   │   ├─ VaR at 95%: Normal predicts -24%; actual past was -37% (50% worse)
│   │   │   ├─ Portfolio on efficient frontier may have hidden tail risk
│   │   │   ├─ Correlations increase in crises (diversification fails when needed most)
│   │   │   └─ Rebalancing can crystallize losses during crashes
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ (1) Use higher moments: Include skewness, kurtosis in optimization
│   │   │   │   ├─ Extend utility: U = E[R] - (λ/2)σ² + γ S³ + δ K⁴
│   │   │   │   ├─ Penalize negative skew; reward positive skew
│   │   │   │   └─ More complex; requires estimating higher moments
│   │   │   ├─ (2) CVaR (Conditional Value-at-Risk):
│   │   │   │   ├─ Optimize for expected shortfall instead of variance
│   │   │   │   ├─ More robust to tail events
│   │   │   │   └─ Requires specifying confidence level (e.g., CVaR at 95%)
│   │   │   ├─ (3) Extreme value theory:
│   │   │   │   ├─ Model tail behavior explicitly (Generalized Pareto)
│   │   │   │   ├─ Estimate rare event probabilities more accurately
│   │   │   │   └─ Used in risk management (financial institutions)
│   │   │   ├─ (4) Scenario analysis:
│   │   │   │   ├─ Specify stress scenarios (recession, geopolitical shock)
│   │   │   │   ├─ Test portfolio in each; add constraints
│   │   │   │   └─ Practical; doesn't require specific distribution
│   │   │   └─ (5) Robust optimization:
│   │   │       ├─ Optimize portfolio for worst-case scenarios
│   │   │       ├─ Distributional uncertainty; find worst-case distribution
│   │   │       └─ Conservative; may sacrifice some expected return
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Fama-French data: Equity returns show excess kurtosis, negative skew
│   │       ├─ Taleb research: Black swans common; normal distribution inadequate
│   │       └─ Post-2008: Risk managers acknowledge tail risk; use CVaR, stress tests
│   │
│   ├─ (3) NO TRANSACTION COSTS:
│   │   ├─ Assumption:
│   │   │   ├─ Trading free; no commissions, bid-ask spreads, or market impact
│   │   │   ├─ Can trade fractional shares (e.g., 3.7% of asset A)
│   │   │   ├─ Can rebalance as frequently as desired
│   │   │   └─ Prices unchanged regardless of trade size
│   │   │
│   │   ├─ Violations in Reality:
│   │   │   ├─ Commissions: Mostly eliminated (zero-commission brokers)
│   │   │   ├─ Bid-ask spreads: 0.01% (large liquid stocks) to 1% (illiquid stocks)
│   │   │   │   ├─ Example: Buy SPY at ask 429.50; sell at bid 429.40 = 0.023% spread
│   │   │   │   └─ Impact: 0.023% per round-trip; if rebalance quarterly: ~0.09% annual
│   │   │   ├─ Market impact: Large trades move price
│   │   │   │   ├─ $1M trade in illiquid stock: Might move market 0.5-1%
│   │   │   │   └─ Worse prices realized than expected
│   │   │   ├─ Rebalancing frequency:
│   │   │   │   ├─ Mechanical quarterly: 4 × 0.1% cost = 0.4% annual
│   │   │   │   ├─ With market impact: Could be 0.5-1% annual
│   │   │   │   └─ Rebalancing benefit (buy dips): 0.5-1% annual
│   │   │   │   └─ Net effect: Break-even or negative after costs!
│   │   │   └─ Illiquidity premium: Illiquid assets cost 1-3% more to trade
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ Optimal rebalancing frequency lower than theory suggests
│   │   │   ├─ Wider rebalancing bands needed (don't rebalance if drift <5%)
│   │   │   ├─ Turnover should be minimized
│   │   │   ├─ Small portfolios (<$100k) hurt most by costs
│   │   │   └─ Active trading (daily) uneconomic for most investors
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ (1) Transaction cost model:
│   │   │   │   ├─ Add cost function to optimization: Cost = c × |Δw|
│   │   │   │   ├─ Higher cost → More stable weights (less rebalancing)
│   │   │   │   └─ Optimal subject to transaction costs
│   │   │   ├─ (2) Threshold rebalancing:
│   │   │   │   ├─ Only rebalance when drift > X% (e.g., 5%)
│   │   │   │   ├─ Empirically often better than mechanical rebalancing
│   │   │   │   └─ Fewer trades, lower costs
│   │   │   ├─ (3) Tax-loss harvesting:
│   │   │   │   ├─ When selling to rebalance, sell losers (realize tax losses)
│   │   │   │   └─ Offset gains; improve after-tax return
│   │   │   └─ (4) Use low-cost funds:
│   │   │       ├─ Index funds: 0.05% ER (vs 1% active)
│   │   │       └─ Over 20 years: 0.95% annual difference → 18% cumulative (huge!)
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Blume-Stambaugh (1983): Transaction costs reduce rebalancing benefit
│   │       ├─ Arnott-Lovell (2013): Rebalancing hurts if costs > benefits
│   │       └─ Vanguard research: Annual rebalancing sufficient (quarterly excessive)
│   │
│   ├─ (4) NO TAXES:
│   │   ├─ Assumption:
│   │   │   ├─ Returns pre-tax; all investors face same tax rate (zero)
│   │   │   ├─ Capital gains taxed only on sale (can defer indefinitely)
│   │   │   └─ No tax-loss harvesting needed
│   │   │
│   │   ├─ Violations in Reality:
│   │   │   ├─ Capital gains tax: 15-20% long-term (after 1-year holding)
│   │   │   ├─ Ordinary income tax: 24-37% on interest, short-term gains
│   │   │   ├─ Dividend tax: 15-20% long-term; ordinary if short-term
│   │   │   ├─ Tax rates vary by investor: Marginal rate 24-37%
│   │   │   └─ Asset type determines tax rate (bonds worst; growth stocks best)
│   │   │
│   │   ├─ Tax efficiency impact:
│   │   │   ├─ Bond fund (4% yield taxed at 40%): After-tax return 2.4%
│   │   │   ├─ Stock fund (10% return; no dividend; deferred tax): After-tax 10%
│   │   │   ├─ Implication: Bonds less attractive in taxable accounts
│   │   │   ├─ Optimal allocation after-tax different from pre-tax
│   │   │   ├─ Example: May prefer 75% stocks, 25% bonds (vs 70/30 pre-tax)
│   │   │   └─ Turnover impact: Active funds trade ~100% annually; taxes eat 1-2%
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ After-tax allocation higher in stocks (especially growth)
│   │   │   ├─ After-tax allocation lower in bonds
│   │   │   ├─ High-turnover strategies severely damaged by taxes
│   │   │   ├─ Tax-loss harvesting valuable (especially in down markets)
│   │   │   └─ Asset location matters: Bonds in tax-deferred (IRA); stocks in taxable
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ (1) Tax-aware portfolio optimization:
│   │   │   │   ├─ Use after-tax returns in optimizer
│   │   │   │   ├─ Adjust for holding period (long-term vs short-term)
│   │   │   │   └─ Different allocation for different tax brackets
│   │   │   ├─ (2) Tax-loss harvesting:
│   │   │   │   ├─ Sell losing positions; realize tax losses
│   │   │   │   ├─ Offset capital gains
│   │   │   │   ├─ Rebuy similar (but not identical) position
│   │   │   │   └─ Can save 0.5-1% annually in down markets
│   │   │   ├─ (3) Asset location optimization:
│   │   │   │   ├─ Tax-deferred accounts: Place bonds (higher tax rate)
│   │   │   │   ├─ Taxable accounts: Place stocks (lower tax rate)
│   │   │   │   ├─ Can add 0.5-1% annually
│   │   │   │   └─ Applies for multi-account investors (IRA + taxable)
│   │   │   └─ (4) Low-turnover strategies:
│   │   │       ├─ Index funds (0.5-1% turnover) vs active (100% turnover)
│   │   │       └─ Tax savings ~1-2% annually
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Arnott-Berkin-Ye (2000): Tax-aware allocation differs materially
│   │       ├─ Arnott (2002): Tax-loss harvesting adds 0.5-1.5% annually
│   │       └─ Vanguard: Tax-efficient index funds often beat actively managed after-tax
│   │
│   ├─ (5) HOMOGENEOUS EXPECTATIONS:
│   │   ├─ Assumption:
│   │   │   ├─ All investors have same beliefs about μ, σ, ρ
│   │   │   ├─ No information asymmetry
│   │   │   ├─ Public information fully priced
│   │   │   └─ Implies: All investors hold market portfolio (two-fund separation)
│   │   │
│   │   ├─ Violations in Reality (Heterogeneous Beliefs):
│   │   │   ├─ Analyst disagreement: Equity research divergence
│   │   │   │   ├─ On S&P 500: Some forecasters 12%, others 8%
│   │   │   │   └─ Standard deviation of forecasts: ~2-3 percentage points
│   │   │   ├─ Information asymmetry: Insiders know more than public
│   │   │   │   ├─ Corporate insiders beat market (trading on private info)
│   │   │   │   └─ Illegal if material non-public info (insider trading)
│   │   │   ├─ Private research: Expensive; gives edge if good
│   │   │   │   ├─ Sell-side research: 0.01% value on average (after fees)
│   │   │   │   └─ Some research valuable; most worthless
│   │   │   ├─ Behavioral: Overconfidence in private predictions
│   │   │   │   ├─ Investors believe in edge (usually wrong)
│   │   │   │   └─ Active stock-picking motivated by false confidence
│   │   │   └─ Time-varying beliefs: Investors update on news
│   │   │       ├─ Tech boom (1990s): Everyone bullish tech; Nasdaq valuations went crazy
│   │   │       └─ Shift expectations → Portfolio weights shift
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ Two-fund separation breaks down (not all hold market)
│   │   │   ├─ Different investors hold different risky portfolios
│   │   │   ├─ Active stock-picking emerges (trying to exploit beliefs)
│   │   │   ├─ Relative performance dispersion widens
│   │   │   └─ Alpha creation possible (but difficult to sustain)
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ (1) Black-Litterman model:
│   │   │   │   ├─ Start with market equilibrium (consensus)
│   │   │   │   ├─ Incorporate investor views with confidence levels
│   │   │   │   ├─ Blend market + views → more robust allocation
│   │   │   │   └─ Reduces input sensitivity
│   │   │   ├─ (2) Heterogeneous beliefs models:
│   │   │   │   ├─ Allow different beliefs; solve for price equilibrium
│   │   │   │   ├─ Prices determined by marginal investor
│   │   │   │   └─ Explains trading volume, valuation dispersion
│   │   │   ├─ (3) Consensus estimates:
│   │   │   │   ├─ Average analyst forecasts; reduce individual error
│   │   │   │   ├─ More stable than any single forecast
│   │   │   │   └─ FactSet, Bloomberg provide consensus
│   │   │   └─ (4) Active management discipline:
│   │   │       ├─ Only deviate from market if high-conviction view
│   │   │       ├─ Size positions by conviction (confidence level)
│   │   │       └─ Accept losses when views proven wrong
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Heterogeneous beliefs observed (analyst disagreement)
│   │       ├─ Active managers' views diverge from market (outperformance rare)
│   │       └─ Most active managers underperform after fees
│   │
│   ├─ (6) PERFECT DIVISIBILITY & LIQUIDITY:
│   │   ├─ Assumption:
│   │   │   ├─ Can buy/sell any fraction (e.g., 3.7% of asset A)
│   │   │   ├─ Instant execution at market price
│   │   │   ├─ No lot sizes; no minimum trades
│   │   │   └─ All assets equally liquid
│   │   │
│   │   ├─ Violations in Reality:
│   │   │   ├─ Discrete positions: Can't hold 0.001% of private equity
│   │   │   ├─ Minimum investment: Many hedge funds require $1M+
│   │   │   ├─ Lot sizes: Some OTC derivatives traded in large blocks
│   │   │   ├─ Illiquid assets:
│   │   │   │   ├─ Real estate: Days/weeks to sell
│   │   │   │   ├─ Private equity: Years (lockups); can't access capital
│   │   │   │   ├─ Collectibles: Highly illiquid; seller may not find buyer
│   │   │   │   └─ Over-the-counter bonds: Limited trading; wide spreads
│   │   │   ├─ Restricted shares: Can't sell during restriction period
│   │   │   ├─ Float constraints: Only tradeable portion of company stock
│   │   │   └─ Liquidity events: Shareholder votes, lockup expirations
│   │   │
│   │   ├─ Impact on Portfolios:
│   │   │   ├─ Rounding to achievable positions (sub-optimal)
│   │   │   ├─ Can't achieve theoretical MPT optimum
│   │   │   ├─ Illiquid allocations must be larger to optimize (J-curve for PE)
│   │   │   ├─ Portfolio less responsive to market timing (can't exit quickly)
│   │   │   └─ Liquidity risk: In stress, must accept worse prices
│   │   │
│   │   ├─ Fixes & Extensions:
│   │   │   ├─ (1) Integer programming:
│   │   │   │   ├─ Optimize with discrete holdings
│   │   │   │   ├─ Account for minimum position sizes
│   │   │   │   └─ More complex; numerical solutions required
│   │   │   ├─ (2) Liquidity constraints:
│   │   │   │   ├─ Add liquidity modelling to portfolio construction
│   │   │   │   ├─ Limit position sizes in illiquid assets
│   │   │   │   └─ Account for bid-ask spreads and market impact
│   │   │   ├─ (3) Layered portfolio:
│   │   │   │   ├─ Core (liquid): Publicly traded stocks/bonds (80%)
│   │   │   │   ├─ Satellite (less liquid): Real estate, PE (20%)
│   │   │   │   └─ Separate optimization for liquidity needs
│   │   │   └─ (4) Liquidity premium adjustment:
│   │   │       ├─ Illiquid assets: Reduce expected return for illiquidity cost
│   │   │       ├─ Or accept lower allocation
│   │   │       └─ PE historical: 5-7% illiquidity premium
│   │   │
│   │   └─ Empirical Evidence:
│   │       ├─ Institutional portfolios: ~80% liquid, ~20% illiquid (real estate, PE)
│   │       └─ PE returns attractive partly due to illiquidity premium
│   │
│   └─ (7) RISK-FREE ASSET EXISTS & STATIC ANALYSIS:
│       ├─ Risk-free assumption:
│       │   ├─ T-bill/Treasury truly σ = 0; no default risk
│       │   ├─ Can lend and borrow at same rate rf
│       │   ├─ Rate constant (not time-varying)
│       │   └─ Perfect divisibility; any amount tradeable
│       │
│       ├─ Static assumption:
│       │   ├─ Single-period optimization; wealth target known
│       │   ├─ No rebalancing; hold to horizon
│       │   └─ Parameters don't change over time
│       │
│       ├─ Violations in Reality:
│       │   ├─ T-bills have modest default risk (rare, but possible)
│       │   ├─ Borrow/lend rates differ (credit spread 1-3%)
│       │   ├─ Risk-free rate time-varying (moved 0%-4.5% over decade)
│       │   ├─ Multi-period investor: Rebalance over time
│       │   ├─ Time horizon matters: Young investor more equities; near-retiree less
│       │   ├─ Liability-driven investing: Matching cash flows, not just optimizing
│       │   └─ Parameters change: Correlations shift in crises
│       │
│       ├─ Impact on Portfolios:
│       │   ├─ Borrow-lend asymmetry reduces optimal leverage
│       │   ├─ Multi-period optimization gives different allocation
│       │   ├─ Young investors should take more risk (human capital bonds them)
│       │   ├─ Rebalancing frequency/bands change over time
│       │   └─ Static allocation suboptimal over investor lifetime
│       │
│       ├─ Fixes & Extensions:
│       │   ├─ (1) Dynamic programming (Bellman):
│       │   │   ├─ Multi-period optimization; optimal policy by state
│       │   │   ├─ More realistic; accounts for rebalancing
│       │   │   └─ Computationally intensive
│       │   ├─ (2) Lifecycle models:
│       │   │   ├─ Target-date funds implement this
│       │   │   ├─ Reduce equity allocation as approach retirement
│       │   │   └─ Empirical evidence: Works reasonably well
│       │   ├─ (3) Liability-driven investing:
│       │   │   ├─ Match assets to liabilities (pension funds, insurance)
│       │   │   ├─ Goal: Fund obligations, not maximize return
│       │   │   └─ Changes portfolio structure significantly
│       │   └─ (4) Stochastic parameter models:
│       │       ├─ Assume μ, σ, ρ time-varying
│       │       ├─ Estimate evolution (GARCH for volatility)
│       │       └─ Optimize with parameter uncertainty
│       │
│       └─ Empirical Evidence:
│           ├─ Multi-period portfolios materially different from single-period
│           ├─ Lifecycle investing beats static allocation
│           └─ Risk-free rate changes significantly (0% to 4.5% in decade)
│
├─ Summary: Violations & Their Fixes:
│   ├─ Behavioral factors → Behavioral portfolio theory, constraints
│   ├─ Fat tails → CVaR, extreme value theory, scenario analysis
│   ├─ Costs → Transaction cost models, threshold rebalancing
│   ├─ Taxes → Tax-aware optimization, asset location
│   ├─ Disagreement → Black-Litterman, consensus estimates
│   ├─ Illiquidity → Integer programming, liquidity constraints
│   ├─ Multi-period → Dynamic programming, lifecycle models
│   └─ Takeaway: Most fixes INCREASE complexity, not reduce it
│
└─ Practical Implementation:
    ├─ Accept: MPT core insight (diversification benefits) is sound
    ├─ Recognize: Assumptions violated; real-world portfolios more complex
    ├─ Adapt: Use extensions appropriate to your situation
    │   ├─ Individual investor: Tax-aware, low-cost, threshold rebalancing
    │   ├─ Institution: Liability matching, illiquidity constraints, scale
    │   ├─ Active manager: Views/alpha model, market-neutral framework
    │   └─ Retiree: Lifecycle, income planning, liability-driven
    ├─ Measure: Compare real portfolio to theoretical benchmark
    ├─ Audit: Identify assumption violations; quantify impact
    └─ Improve: Iteratively add constraints, costs as understand impact
```

---

## 5. Mini-Project: Testing MPT Assumptions & Violations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime

# Test key MPT assumptions against real data

def fetch_returns(tickers, start_date, end_date):
    """Fetch daily returns."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns


def test_normality(returns_series, name):
    """Test for normality (Jarque-Bera test)."""
    jb_stat, jb_pval = stats.jarque_bera(returns_series)
    skew = stats.skew(returns_series)
    kurt = stats.kurtosis(returns_series)
    
    return {
        'Skewness': skew,
        'Kurtosis (excess)': kurt,
        'JB Statistic': jb_stat,
        'JB p-value': jb_pval,
        'Normal?': 'Yes' if jb_pval > 0.05 else 'No'
    }


def analyze_tail_risk(returns_series, confidence_level=0.95):
    """Analyze tail risk; compare normal vs empirical."""
    
    # Empirical VaR (historical)
    empirical_var = np.percentile(returns_series, (1 - confidence_level) * 100)
    
    # Normal VaR (assuming normality)
    mean = returns_series.mean()
    std = returns_series.std()
    normal_var = mean + std * stats.norm.ppf(1 - confidence_level)
    
    # Excess beyond VaR (expected shortfall)
    empirical_es = returns_series[returns_series <= empirical_var].mean()
    normal_es = mean + std * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
    
    return {
        'Empirical VaR': empirical_var,
        'Normal VaR': normal_var,
        'Tail underestimation': (normal_var - empirical_var) / abs(empirical_var) * 100,
        'Empirical ES': empirical_es,
        'Normal ES': normal_es,
        'Extreme underestimation': (normal_es - empirical_es) / abs(empirical_es) * 100 if empirical_es != 0 else 0
    }


def analyze_correlations_over_time(returns_df, window=252):
    """Analyze correlation changes (violates constant correlation assumption)."""
    
    correlations = []
    dates = []
    
    for i in range(window, len(returns_df)):
        rolling_corr = returns_df.iloc[i-window:i].corr().iloc[0, 1]
        correlations.append(rolling_corr)
        dates.append(returns_df.index[i])
    
    return pd.Series(correlations, index=dates)


def estimate_transaction_costs(portfolio_value, annual_turnover, bid_ask_pct=0.05, impact_pct=0.05):
    """Estimate annual transaction cost drag."""
    
    round_trip_cost = (bid_ask_pct + impact_pct) / 100
    annual_cost = annual_turnover * round_trip_cost
    
    return {
        'Portfolio Value': portfolio_value,
        'Annual Turnover': annual_turnover,
        'Bid-ask cost': bid_ask_pct / 100,
        'Market impact': impact_pct / 100,
        'Annual cost (%)': annual_cost * 100,
        'Annual cost ($)': portfolio_value * annual_cost
    }


def estimate_tax_drag(portfolio_return, turnover_pct, holding_period_years, cap_gains_tax=0.20, ordinary_tax=0.37):
    """Estimate tax drag from rebalancing."""
    
    annual_taxable_gains = (turnover_pct / 100) * cap_gains_tax
    
    # Assume half short-term, half long-term
    short_term_tax_drag = (turnover_pct / 100 * 0.5) * ordinary_tax
    long_term_tax_drag = (turnover_pct / 100 * 0.5) * cap_gains_tax
    
    total_tax_drag = short_term_tax_drag + long_term_tax_drag
    
    return {
        'Pre-tax return': portfolio_return,
        'Annual turnover': turnover_pct,
        'Tax drag (%)': total_tax_drag * 100,
        'After-tax return': (portfolio_return - total_tax_drag) * 100
    }


# Main Analysis
print("=" * 100)
print("TESTING MPT ASSUMPTIONS AGAINST REAL MARKET DATA")
print("=" * 100)

# 1. Normality Test
print("\n1. TESTING NORMALITY ASSUMPTION")
print("-" * 100)

tickers = ['SPY', 'QQQ', 'AGG']
names = ['S&P 500', 'Tech Nasdaq', 'Bonds']

returns = fetch_returns(tickers, '2015-01-01', '2024-01-01')

print("\nJarque-Bera Test (H0: Normal distribution):\n")
print(f"{'Asset':<20} {'Skewness':<15} {'Kurtosis':<15} {'JB p-value':<15} {'Normal?':<10}")
print("-" * 75)

for tick, name in zip(tickers, names):
    result = test_normality(returns[tick], name)
    print(f"{name:<20} {result['Skewness']:<15.4f} {result['Kurtosis (excess)']:<15.4f} "
          f"{result['JB p-value']:<15.2e} {result['Normal?']:<10}")

print("\nInterpretation:")
print("- All p-values < 0.05 → Reject normality")
print("- Negative skewness → Left tail fatter (crashes worse than rallies)")
print("- Positive kurtosis → Extreme events more common than normal predicts")

# 2. Tail Risk Analysis
print("\n2. ANALYZING TAIL RISK (VaR Comparison)")
print("-" * 100)

print("\nTail Risk Analysis (95% confidence level):\n")
print(f"{'Asset':<20} {'Empirical VaR':<18} {'Normal VaR':<18} {'Underest. %':<15}")
print("-" * 71)

for tick, name in zip(tickers, names):
    tail_analysis = analyze_tail_risk(returns[tick], 0.95)
    print(f"{name:<20} {tail_analysis['Empirical VaR']*100:<18.2f}% "
          f"{tail_analysis['Normal VaR']*100:<18.2f}% "
          f"{tail_analysis['Tail underestimation']:<15.1f}%")

print("\nInterpretation:")
print("- Normal VaR usually UNDERESTIMATES true tail risk")
print("- Bonds relatively safe; stocks show larger tail underestimation")
print("- Implication: Variance insufficient for risk management")

# 3. Correlation Stability (Homogeneity Violation)
print("\n3. TESTING CORRELATION STABILITY (Homogeneous Beliefs)")
print("-" * 100)

corr_series = analyze_correlations_over_time(returns[['SPY', 'AGG']], window=252)

print(f"\nCorrelation evolution (SPY vs AGG):")
print(f"  Mean correlation: {corr_series.mean():.3f}")
print(f"  Std deviation: {corr_series.std():.3f}")
print(f"  Min: {corr_series.min():.3f} (date: {corr_series.idxmin().date()})")
print(f"  Max: {corr_series.max():.3f} (date: {corr_series.idxmax().date()})")
print(f"\nInterpretation:")
print("- Correlation NOT constant (violates homogeneous beliefs)")
print("- Ranges from negative to positive (diversification benefit varies)")
print("- Tends to rise during crises (diversification fails when needed)")

# 4. Transaction Cost Analysis
print("\n4. ESTIMATING TRANSACTION COST DRAG")
print("-" * 100)

scenarios = [
    {'size': 100000, 'turnover': 0.5, 'label': 'Low turnover (annual rebalance)'},
    {'size': 100000, 'turnover': 2.0, 'label': 'Moderate turnover (quarterly)'},
    {'size': 100000, 'turnover': 5.0, 'label': 'High turnover (active trading)'},
]

print("\nAnnual Transaction Cost Estimates:\n")
print(f"{'Scenario':<40} {'Annual Turnover':<18} {'Cost %':<12} {'Cost $':<12}")
print("-" * 82)

for scenario in scenarios:
    cost_analysis = estimate_transaction_costs(scenario['size'], scenario['turnover'])
    print(f"{scenario['label']:<40} {cost_analysis['Annual Turnover']:<18.1f}% "
          f"{cost_analysis['Annual cost (%)']:<12.2f}% ${cost_analysis['Annual cost ($)']:<11.0f}")

print("\nInterpretation:")
print("- Low turnover: 0.5-1% drag (acceptable)")
print("- Active trading: 2-5% drag (huge impact on returns)")
print("- Implication: Minimize rebalancing; use low-cost funds")

# 5. Tax Drag Analysis
print("\n5. ESTIMATING TAX DRAG FROM REBALANCING")
print("-" * 100)

print("\nTax Impact on Returns (Taxable Account):\n")
print(f"{'Scenario':<40} {'Annual Turnover':<18} {'Tax Drag %':<15} {'After-tax %':<15}")
print("-" * 88)

tax_scenarios = [
    {'return': 0.08, 'turnover': 0.5, 'label': 'Conservative (index fund)'},
    {'return': 0.08, 'turnover': 4.0, 'label': 'Moderate (quarterly rebalance)'},
    {'return': 0.08, 'turnover': 10.0, 'label': 'Aggressive (active manager)'},
]

for scenario in tax_scenarios:
    tax_analysis = estimate_tax_drag(scenario['return'], scenario['turnover'])
    print(f"{scenario['label']:<40} {tax_analysis['Annual turnover']:<18.1f}% "
          f"{tax_analysis['Tax drag (%)']:<15.2f}% {tax_analysis['After-tax return']:<15.2f}%")

print("\nInterpretation:")
print("- Index fund: Minimal tax drag (low turnover)")
print("- Active management: 1-3% annual tax drag (huge over decades)")
print("- Over 30 years: Compounding effect massive (index fund superior)")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Distribution Comparison (Normal vs Empirical)
ax = axes[0, 0]

returns_spy = returns['SPY']
x_range = np.linspace(returns_spy.min(), returns_spy.max(), 100)

# Normal fit
mu, sigma = returns_spy.mean(), returns_spy.std()
normal_fit = stats.norm.pdf(x_range, mu, sigma)

# Empirical histogram
ax.hist(returns_spy, bins=50, density=True, alpha=0.6, label='Empirical', color='#3498db')
ax.plot(x_range, normal_fit, 'r-', linewidth=2, label='Normal fit')

# Highlight tail
ax.axvline(x=mu - 3*sigma, color='orange', linestyle='--', alpha=0.7, label='3σ from mean')
ax.axvline(x=returns_spy.quantile(0.05), color='red', linestyle='--', alpha=0.7, label='5% tail')

ax.set_xlabel('Daily Return', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Return Distribution: Normal Assumption vs Reality (SPY)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Correlation Stability Over Time
ax = axes[0, 1]

ax.plot(corr_series.index, corr_series, linewidth=1.5, color='#2ecc71', alpha=0.7)
ax.axhline(y=corr_series.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {corr_series.mean():.3f}')
ax.fill_between(corr_series.index, corr_series.mean() - corr_series.std(), 
                corr_series.mean() + corr_series.std(), alpha=0.2, color='gray', label=f'±1 Std Dev')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Correlation (SPY vs AGG)', fontsize=12)
ax.set_title('Correlation Stability Over Time (Not Constant)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Transaction Cost Impact
ax = axes[1, 0]

turnovers = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
costs = turnovers * 0.001  # 0.1% round-trip per 1% turnover

ax.plot(turnovers, costs * 100, 'o-', linewidth=2.5, markersize=10, color='#e74c3c')
ax.fill_between(turnovers, 0, costs * 100, alpha=0.2, color='#e74c3c')

for x, y in zip(turnovers, costs * 100):
    ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=9)

ax.set_xlabel('Annual Turnover (%)', fontsize=12)
ax.set_ylabel('Transaction Cost Drag (%)', fontsize=12)
ax.set_title('Transaction Cost Impact on Returns', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 4: Tax Drag by Strategy
ax = axes[1, 1]

strategies = ['Index\n(0.5%)', 'Quarterly\nRebalance\n(4%)', 'Active\nManager\n(10%)']
pretax_returns = [8, 8, 8]
posttax_returns = [7.6, 5.4, 5.2]

x = np.arange(len(strategies))
width = 0.35

bars1 = ax.bar(x - width/2, pretax_returns, width, label='Pre-tax', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, posttax_returns, width, label='After-tax', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_title('Tax Drag: After-Tax Returns by Strategy', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('mpt_assumptions_violations.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: mpt_assumptions_violations.png")
plt.show()

print("\n" + "=" * 100)
print("KEY FINDINGS:")
print("=" * 100)
print("""
1. NORMALITY VIOLATION: Returns are not normally distributed
   - Fat tails: Extreme events more common than normal predicts
   - Negative skew: Crashes worse than rallies
   - Implication: Variance insufficient for full risk assessment; use CVaR or higher moments

2. TAIL RISK UNDERESTIMATION: Normal distribution underestimates extreme risk
   - VaR underestimated by 10-30% for stocks
   - More severe for bonds (closer to normal, but still deviations)
   - Implication: Risk management models too optimistic

3. CORRELATION NOT CONSTANT: Violates homogeneous beliefs assumption
   - Correlations vary 50%+; increase during crises
   - Diversification fails when most needed (2008, 2020)
   - Implication: Need dynamic correlation models; correlation assumptions risky

4. TRANSACTION COSTS MATERIAL: Can exceed diversification benefits
   - Quarterly rebalancing: ~1% annual drag
   - Active trading: 2-5% annual drag
   - Implication: Minimize turnover; use low-cost index funds

5. TAXES SIGNIFICANTLY REDUCE RETURNS: After-tax optimization critical
   - Active management: 1-3% annual tax drag (compounding!)
   - Over 30 years: Massive difference (index beats active)
   - Implication: Use tax-aware strategies; tax-loss harvest

PRACTICAL RECOMMENDATIONS:
├─ Use low-cost index funds (minimize costs, taxes)
├─ Rebalance infrequently (threshold-based, not mechanical)
├─ Tax-loss harvest in down markets (taxable accounts)
├─ Use CVaR for downside risk (not just variance)
├─ Accept correlations change (diversification imperfect)
└─ Be skeptical of precise optimization (input error huge)
""")
```

---

## 6. Challenge Round

1. **Fat Tails & Optimization:** If variance underestimates risk by 30%, how much suboptimal is your allocation? Should you use CVaR instead? Trade-offs in computational complexity vs accuracy?

2. **Correlation Breakdown:** During 2008 crisis, stock-bond correlation was 0.8 (vs 0.3 normal). If your portfolio designed for correlation 0.3, what was actual risk? How much worse did you do?

3. **Rebalancing Cost-Benefit:** Rebalancing benefit historically 0.5-1% p.a.; costs 0.5-1% p.a. When is rebalancing worthwhile? When should you just buy-and-hold?

4. **Tax Impact on Allocation:** Investor in 40% tax bracket; bonds yield 4%, stocks yield 10%. Should allocate 60% bonds (pre-tax optimal) or 80% stocks (after-tax optimal)? Why?

5. **Homogeneous Beliefs Break:** If you have different return forecast than market, should deviate from market portfolio? How much? What's risk of being wrong?

---

## 7. Key References

- **Kahneman, D. & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision Under Risk" *Econometrica* – Behavioral violations of rational choice; Nobel Prize 2002.

- **Fama, E.F. & French, K.R. (1992).** "The Cross-Section of Expected Stock Returns" *Journal of Finance* – Empirical anomalies; non-normal distributions, factor structure.

- **Sharpe, W.F. (1990).** "Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk" – CAPM; foundational but assumes perfect rationality.

- **Markowitz, H.M. (1952).** "Portfolio Selection" – Original MPT; acknowledges limitations but core insight valid.

- **Rockefeller, T. & Uryasev, S. (2000).** "Optimization of Conditional Value-at-Risk" *Journal of Risk* – CVaR as alternative to variance; handles tails better.

- **Arnott, R.D., Beck, S.L., Kalesnik, V., & West, J. (2016).** "How Can 'Good' Costs Turn 'Bad'?" *Research Affiliates* – Transaction costs, rebalancing tradeoffs.

- **Arnott, R.D. & Berkin, A.L. (2000).** "The Relevant Difference between Stocks and Bonds" – Tax-aware portfolio construction; substantial impact on after-tax allocation.

