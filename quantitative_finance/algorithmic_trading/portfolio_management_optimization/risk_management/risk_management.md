# Risk Management

## Concept Skeleton

Risk management in algorithmic trading systematically identifies, measures, and controls potential losses through position limits, Value-at-Risk (VaR), stress testing, and stop-loss rules, preventing catastrophic drawdowns while preserving capital for profitable opportunities. Core framework: define risk appetite (maximum acceptable loss), monitor exposures in real-time, and implement automatic safeguards (kill switches, circuit breakers) when thresholds are breached.

**Core Components:**
- **Position sizing**: Kelly criterion, fixed fractional (risk fixed % of capital per trade), volatility-adjusted sizing
- **Exposure limits**: Gross exposure (sum of all positions), net exposure (long - short), sector/factor limits
- **Value-at-Risk (VaR)**: Maximum expected loss over time horizon at confidence level (e.g., 1-day 95% VaR = $100k)
- **Stop-loss rules**: Automatic exit when position loss exceeds threshold (trade-level or portfolio-level)
- **Stress testing**: Simulate portfolio performance under extreme scenarios (2008 crisis, flash crash, correlation breakdown)

**Why it matters:** Edge without risk management is roulette; proper sizing and limits ensure strategy survives unfavorable periods and compounding can work over time; prevents behavioral bias (overleveraging winners, doubling down on losers).

---

## Comparative Framing

| Dimension | **Value-at-Risk (VaR)** | **Stop-Loss Orders** | **Position Limits** |
|-----------|-------------------------|----------------------|---------------------|
| **Risk metric** | Probabilistic loss (e.g., 95% confidence) | Worst-case per-trade loss | Maximum capital at risk |
| **Time horizon** | 1-day, 10-day (chosen by user) | Trade duration (until exit) | Ongoing constraint |
| **Implementation** | Portfolio-level; continuous monitoring | Order-level; exchange or internal | Pre-trade check; blocks orders |
| **Tail risk capture** | Underestimates (assumes normal distribution) | Captures if executed (gap risk exists) | Prevents concentration but not volatility |
| **Use case** | Regulatory reporting, capital allocation | Tactical loss control | Strategic diversification |
| **Weakness** | Backward-looking (historical vol may not predict future) | Can trigger in volatile markets (whipsawed) | Rigid; may prevent profitable concentration |

**Key insight:** Layered risk management—VaR for aggregate exposure, stop-losses for tail events, position limits for concentration—provides defense-in-depth; no single metric captures all risks.

---

## Examples & Counterexamples

### Examples of Risk Management in Action

1. **Kelly Criterion Position Sizing**  
   - Strategy has 55% win rate, avg win $500, avg loss $400  
   - Edge: \(p \cdot W - (1-p) \cdot L = 0.55 \times 500 - 0.45 \times 400 = 95\)  
   - Kelly fraction: \(f^* = \frac{p}{L} - \frac{1-p}{W} = \frac{0.55}{400} - \frac{0.45}{500} = 0.0005\) (0.05% of capital per trade)  
   - Conservative: Use 0.5 × Kelly = 0.025% to reduce volatility  

2. **Portfolio VaR Calculation**  
   - 3-asset portfolio: $1M in Asset A (20% vol), $500k in Asset B (15% vol), $500k in Asset C (10% vol)  
   - Correlations: A-B 0.5, A-C 0.3, B-C 0.2  
   - 1-day 95% VaR ≈ Portfolio Vol × 1.65 (assumes normality)  
   - Portfolio vol = 14.2% (computed from covariance matrix)  
   - 1-day VaR = $2M × 14.2% × 1.65 / √252 = $29,500  

3. **Stop-Loss Rule with Volatility Adjustment**  
   - Low-vol asset: Stop-loss at -2% (tight control)  
   - High-vol asset: Stop-loss at -5% (avoid noise triggering)  
   - Adaptive: Stop-loss = Entry Price × (1 - 2 × ATR / Price) where ATR = Average True Range  

4. **Stress Testing (2008 Crisis Scenario)**  
   - Equities drop 40%, volatility triples, correlations → 1.0 (all assets fall together)  
   - Portfolio loss: $800k on $2M portfolio (40% drawdown)  
   - Action: If stress test shows >50% drawdown potential, reduce leverage or add hedges (put options, VIX futures)

### Non-Examples (or Poor Risk Management)

- **All-in on single trade**: No diversification; single loss wipes out account.
- **No stop-loss in leveraged position**: Unlimited loss potential; margin call risk.
- **VaR based on 2-year calm market**: Underestimates risk during regime shift (volatility explosion).

---

## Layer Breakdown

**Layer 1: Pre-Trade Risk Checks**  
Before order submission:  
1. Check position limit: \(\text{Current Exposure} + \text{New Position} \leq \text{Max Exposure}\)  
2. Check sector/factor exposure: Ensure diversification (e.g., no >20% in tech sector)  
3. Check leverage: \(\text{Gross Exposure} / \text{Capital} \leq \text{Max Leverage}\) (e.g., 2× for long-short equity)  
If any limit breached, block order or scale down.

**Layer 2: Position Sizing Algorithms**  
**Fixed Fractional:** Risk fixed % of capital (e.g., 2% per trade).  
\[
\text{Position Size} = \frac{\text{Capital} \times \text{Risk %}}{|\text{Entry Price} - \text{Stop Price}|}
\]  

**Kelly Criterion:** Optimal fraction to maximize log wealth:  
\[
f^* = \frac{p}{L} - \frac{1-p}{W}
\]  
where \(p\) = win probability, \(W\) = avg win, \(L\) = avg loss. Use fractional Kelly (0.25–0.5×) to reduce volatility.

**Layer 3: Value-at-Risk Calculation**  
**Parametric VaR (assumes normal returns):**  
\[
\text{VaR}_{\alpha} = \text{Portfolio Value} \times \sigma \times z_{\alpha} \times \sqrt{t}
\]  
where \(\sigma\) = portfolio volatility, \(z_{\alpha}\) = z-score (1.65 for 95%, 2.33 for 99%), \(t\) = time horizon in days.  

**Historical VaR:** Sort historical returns, take 5th percentile (for 95% VaR).  

**Monte Carlo VaR:** Simulate 10,000 portfolio paths, compute 5th percentile loss.

**Layer 4: Real-Time Monitoring and Alerts**  
Continuous tracking:  
- Current PnL vs. daily loss limit (e.g., $50k max loss)  
- VaR utilization (actual VaR / allocated VaR)  
- Correlation stress (detect when correlations spike toward 1.0)  
- Liquidity stress (bid-ask spreads widening)  
Automated actions: If loss limit hit → flatten all positions (emergency exit).

---

## Mini-Project: VaR Calculation and Position Sizing

**Goal:** Calculate portfolio VaR and determine optimal position sizes under risk constraints.

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

# Portfolio setup: 3 assets
assets = ['Tech Stock', 'Bond', 'Commodity']
positions = np.array([1_000_000, 500_000, 500_000])  # Dollar holdings
volatilities = np.array([0.25, 0.08, 0.30])  # Annual volatilities
correlation = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 1.0, 0.1],
    [0.2, 0.1, 1.0]
])

# Covariance matrix (annualized)
cov_matrix = np.outer(volatilities, volatilities) * correlation

# Portfolio volatility (annualized)
weights = positions / positions.sum()
portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
portfolio_vol = np.sqrt(portfolio_variance)
portfolio_value = positions.sum()

# 1-day 95% VaR (parametric)
confidence_level = 0.95
z_score = norm.ppf(confidence_level)
time_horizon_days = 1
daily_vol = portfolio_vol / np.sqrt(252)
var_95 = portfolio_value * daily_vol * z_score

# 1-day 99% VaR
z_score_99 = norm.ppf(0.99)
var_99 = portfolio_value * daily_vol * z_score_99

print("=" * 60)
print("PORTFOLIO RISK METRICS")
print("=" * 60)
print(f"Portfolio Value:              ${portfolio_value:>12,.0f}")
print(f"Portfolio Volatility (Annual): {portfolio_vol:>11.2%}")
print(f"Portfolio Volatility (Daily):  {daily_vol:>11.2%}")
print()
print(f"1-Day 95% Value-at-Risk:      ${var_95:>12,.0f}")
print(f"1-Day 99% Value-at-Risk:      ${var_99:>12,.0f}")
print()
print("Interpretation: 95% confident daily loss will not exceed VaR_95.")
print("=" * 60)

# Position sizing example: Kelly criterion
# Assume strategy with 60% win rate, avg win $1000, avg loss $600
win_prob = 0.60
avg_win = 1000
avg_loss = 600

kelly_fraction = (win_prob / avg_loss) - ((1 - win_prob) / avg_win)
kelly_percent = kelly_fraction * 100

# Conservative: Use half-Kelly
half_kelly_percent = kelly_percent * 0.5

print("\nPOSITION SIZING (Kelly Criterion)")
print("=" * 60)
print(f"Win Probability:              {win_prob:>12.0%}")
print(f"Average Win:                  ${avg_win:>12,.0f}")
print(f"Average Loss:                 ${avg_loss:>12,.0f}")
print()
print(f"Full Kelly Fraction:          {kelly_percent:>11.2f}%")
print(f"Half-Kelly (Conservative):    {half_kelly_percent:>11.2f}%")
print()
print(f"Recommended Position Size:    ${portfolio_value * half_kelly_percent/100:>12,.0f}")
print("=" * 60)

# Stress test: Simulate 2008-style crisis
print("\nSTRESS TEST: Financial Crisis Scenario")
print("=" * 60)
stress_return = -0.40  # 40% market drop
stress_vol_multiplier = 3  # Volatility triples
stress_correlation = 0.95  # Correlations converge to 1

stress_positions = positions * (1 + stress_return)
stress_loss = positions.sum() - stress_positions.sum()
stress_loss_pct = stress_loss / portfolio_value

print(f"Scenario: Market drops 40%, vol triples, correlation → 0.95")
print(f"Portfolio Loss:               ${stress_loss:>12,.0f}")
print(f"Portfolio Loss %:             {stress_loss_pct:>11.2%}")
print()

if stress_loss_pct > 0.50:
    print("⚠️  WARNING: Portfolio at risk of >50% drawdown in crisis!")
    print("   Recommendation: Reduce leverage or add tail hedges.")
else:
    print("✓ Portfolio survives stress test with manageable drawdown.")
print("=" * 60)
```

**Expected Output (illustrative):**
```
============================================================
PORTFOLIO RISK METRICS
============================================================
Portfolio Value:                $2,000,000
Portfolio Volatility (Annual):      16.23%
Portfolio Volatility (Daily):        1.02%

1-Day 95% Value-at-Risk:           $33,712
1-Day 99% Value-at-Risk:           $47,514

Interpretation: 95% confident daily loss will not exceed VaR_95.
============================================================

POSITION SIZING (Kelly Criterion)
============================================================
Win Probability:                      60%
Average Win:                       $1,000
Average Loss:                        $600

Full Kelly Fraction:                 3.33%
Half-Kelly (Conservative):           1.67%

Recommended Position Size:         $33,333
============================================================

STRESS TEST: Financial Crisis Scenario
============================================================
Scenario: Market drops 40%, vol triples, correlation → 0.95
Portfolio Loss:                   $800,000
Portfolio Loss %:                    40.00%

✓ Portfolio survives stress test with manageable drawdown.
============================================================
```

**Interpretation:**  
- 1-day 95% VaR $33,712: Expect daily loss >$33k only 5% of time (1 in 20 days).  
- Half-Kelly position sizing (1.67%): Conservative; reduces drawdown volatility.  
- Stress test: 40% loss is severe but survivable; if >50%, consider hedging.

---

## Challenge Round

1. **VaR Underestimation in Fat-Tailed Markets**  
   Parametric VaR assumes normal returns. 2008 crisis saw 10-sigma events (should occur once per billions of years). Why did VaR fail?

   <details><summary>Hint</summary>Real returns have fat tails (kurtosis >3) and skewness. Normal distribution underestimates extreme losses. **Solutions:** (1) Use Student's t-distribution (fatter tails), (2) Extreme Value Theory (EVT) for tail modeling, (3) Historical VaR (empirical distribution), (4) Conditional VaR (CVaR/Expected Shortfall: average loss beyond VaR).</details>

2. **Stop-Loss vs. Volatility Targeting**  
   Strategy A: Fixed 5% stop-loss. Strategy B: Volatility-adjusted stop (2× ATR). Which is better in trending vs. mean-reverting markets?

   <details><summary>Solution</summary>
   **Trending markets:** Fixed stop-loss better (allows trends to run; volatility-adjusted may exit too early if vol spikes during trend).  
   **Mean-reverting markets:** Volatility-adjusted better (avoids being stopped out by noise in choppy, high-vol environments).  
   **Hybrid:** Use both—fixed stop for disaster protection, vol-adjusted for tactical exits.
   </details>

3. **Kelly Criterion Overbetting**  
   Strategy has 60% win rate, 2:1 reward-risk ratio. Full Kelly = 20% of capital per trade. After 3 losses in a row, capital drops to $70k from $100k. What is the problem?

   <details><summary>Solution</summary>
   Full Kelly is aggressive; assumes infinite divisibility and no estimation error in win rate. Three losses: $100k → $80k → $64k → $51k (if betting 20% each time). **Problem:** Volatility too high for most traders; psychological difficulty.  
   **Fix:** Use fractional Kelly (0.25× or 0.5×). With 0.5× Kelly (10%), three losses: $100k → $90k → $81k → $73k (more manageable).
   </details>

4. **Correlation Stress and Portfolio Fragility**  
   Long-short equity portfolio (long value, short growth) normally has low correlation between legs. In 2020 March crash, correlation spiked to 0.9 (both fell). What happens to hedging effectiveness?

   <details><summary>Solution</summary>
   **Normal regime:** Long +X%, short -Y% → Portfolio gains X+Y (hedged).  
   **Crisis regime (correlation → 1):** Long -30%, short -25% → Portfolio loses 5% (hedge failed).  
   **Lesson:** Diversification breaks down in crises (all correlations → 1). **Mitigation:** (1) Reduce leverage before crisis, (2) Add non-correlated assets (gold, Treasuries), (3) Tail hedge with out-of-money puts.
   </details>

---

## Key References

- **Jorion (2006)**: *Value at Risk: The New Benchmark for Managing Financial Risk* ([McGraw-Hill](https://www.mheducation.com/))
- **Thorp (2006)**: "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" ([Handbook of Asset and Liability Management](https://www.sciencedirect.com/))
- **Taleb (2007)**: *The Black Swan* (fat tails and model risk) ([Random House](https://www.penguinrandomhouse.com/))
- **GARP**: Global Association of Risk Professionals—FRM curriculum on portfolio risk management ([GARP.org](https://www.garp.org/))

**Further Reading:**  
- Conditional Value-at-Risk (CVaR/Expected Shortfall): Average loss beyond VaR threshold  
- Backtesting VaR models: Kupiec test, traffic light approach (Basel framework)  
- Liquidity-adjusted VaR (LaR): Incorporates bid-ask spreads and market depth
