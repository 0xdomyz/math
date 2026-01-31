# Monte Carlo Simulation

## Concept Skeleton

Monte Carlo simulation in trading strategy validation generates thousands of synthetic equity curves by randomizing historical returns, trade sequences, or parameter values to assess strategy robustness, estimate confidence intervals for performance metrics (Sharpe ratio, drawdown), and stress-test under alternative market scenarios. Unlike historical backtesting (one realized path), Monte Carlo explores distribution of possible outcomes, revealing tail risks and parameter sensitivity.

**Core Components:**
- **Randomization methods**: Bootstrapping returns, shuffling trade order, resampling with replacement, adding noise to parameters
- **Synthetic equity curves**: Generate N paths (e.g., 10,000 simulations) from randomized inputs
- **Performance distribution**: Histogram of Sharpe ratios, max drawdowns across simulations
- **Confidence intervals**: 5th–95th percentile of metrics (e.g., "95% confident Sharpe > 0.8")
- **Stress testing**: Simulate extreme scenarios (volatility spikes, correlation breakdowns)

**Why it matters:** Single historical backtest is one realization; Monte Carlo reveals strategy's statistical distribution of outcomes, identifies worst-case scenarios, and quantifies luck vs. skill.

---

## Comparative Framing

| Dimension | **Monte Carlo Simulation** | **Historical Backtest** | **Walk-Forward Analysis** |
|-----------|----------------------------|-------------------------|---------------------------|
| **Data source** | Synthetic (randomized from historical) | Actual historical prices | Actual historical prices |
| **Number of paths** | Thousands of equity curves | Single realized path | Multiple OOS periods (5–20) |
| **Captures regime changes** | Limited (depends on randomization) | Yes (actual market evolution) | Yes (tests across regimes) |
| **Robustness assessment** | Distribution of outcomes, tail risk | Point estimate (single path) | Parameter stability over time |
| **Use case** | Confidence intervals, stress testing | Baseline performance | Overfitting detection |
| **Computational cost** | High (10,000+ runs) | Low (single run) | Moderate (N optimizations) |

**Key insight:** Monte Carlo complements historical backtesting: backtest shows *what happened*, Monte Carlo shows *what could have happened* under randomized conditions.

---

## Examples & Counterexamples

### Examples of Monte Carlo Simulation in Trading

1. **Bootstrap Resampling of Returns**  
   - Strategy has 500 daily returns from backtest  
   - Generate 10,000 synthetic equity curves by sampling 500 returns *with replacement*  
   - Calculate Sharpe ratio for each curve  
   - Result: 95% confidence interval for Sharpe = [0.85, 1.45], median 1.12  

2. **Shuffling Trade Sequence**  
   - Strategy has 200 trades (wins and losses)  
   - Shuffle order of trades 10,000 times (preserves win/loss distribution but randomizes timing)  
   - Measure max drawdown in each shuffle  
   - Finding: Historical max DD 15%, but 95th percentile across shuffles is 22% (unlucky trade timing could worsen DD)  

3. **Parameter Perturbation**  
   - Optimal moving average length = 50 days (from optimization)  
   - Generate 1,000 simulations with MA length randomly varied: 50 ± N(0, 5)  
   - Calculate OOS Sharpe for each perturbed parameter  
   - Result: Mean OOS Sharpe 1.0, std dev 0.3 → strategy robust to parameter estimation error  

4. **Stress Testing with Increased Volatility**  
   - Historical returns: μ=0.1%, σ=1.5% daily  
   - Simulate 1,000 paths with σ=2.5% (crisis scenario)  
   - Observe: Max drawdown increases from 12% to 28%; strategy struggles in high-vol regime

### Non-Examples (or Misuses)

- **Single Monte Carlo run**: Not simulation; need thousands of paths to estimate distribution.
- **Using Monte Carlo *instead of* historical backtest**: Monte Carlo should validate/stress-test backtest, not replace it (historical path contains actual regime changes).
- **Randomizing prices without preserving correlations**: May destroy autocorrelation structure, creating unrealistic scenarios.

---

## Layer Breakdown

**Layer 1: Randomization Strategies**  
**Bootstrap (Resampling with Replacement):**  
From historical returns \(\{r_1, r_2, \ldots, r_T\}\), draw T samples randomly with replacement. Preserves return distribution but destroys temporal order (assumes i.i.d. returns—oversimplification but useful baseline).

**Block Bootstrap:**  
Sample contiguous blocks of returns (e.g., 20-day blocks) to preserve short-term autocorrelation. Better for time-series data with momentum or mean reversion.

**Parametric Simulation:**  
Fit historical returns to distribution (e.g., normal, Student's t, GARCH model), then generate synthetic returns from fitted parameters. Allows stress-testing by adjusting distribution parameters (e.g., increase tail thickness).

**Layer 2: Generating Synthetic Equity Curves**  
For each simulation \(i = 1, \ldots, N\):  
1. Randomize inputs (returns, trade order, or parameters)  
2. Apply strategy logic to generate PnL series  
3. Compute equity curve: \(\text{Equity}_i(t) = \text{Capital}_0 \prod_{s=1}^{t} (1 + r_{i,s})\)  
4. Calculate metrics: Sharpe\(_i\), MaxDD\(_i\), Calmar\(_i\)

**Layer 3: Performance Distribution Analysis**  
Aggregate metrics across simulations:  
- **Mean and std dev**: Average Sharpe, volatility of Sharpe across sims  
- **Percentiles**: 5th, 50th (median), 95th percentiles for Sharpe, max DD  
- **Probability of loss**: Fraction of sims with negative cumulative return  
- **Worst-case scenario**: Minimum Sharpe or maximum drawdown across all sims

**Layer 4: Confidence Intervals & Statistical Significance**  
Construct 95% confidence interval for metric \(M\):  
\[
\text{CI}_{95} = [M_{5\text{th percentile}}, M_{95\text{th percentile}}]
\]  
**Example:** If CI for Sharpe is [0.6, 1.4] and includes 0, strategy's edge is statistically weak. If CI is [1.0, 1.8], edge is robust.

---

## Mini-Project: Monte Carlo Bootstrap for Sharpe Ratio Distribution

**Goal:** Estimate Sharpe ratio confidence interval via bootstrap resampling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate strategy returns (500 daily returns)
np.random.seed(321)
n_days = 500
true_mean = 0.0008  # 0.08% daily
true_vol = 0.015    # 1.5% daily
strategy_returns = np.random.normal(true_mean, true_vol, n_days)

# Calculate historical Sharpe
historical_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Historical Sharpe Ratio: {historical_sharpe:.2f}")

# Monte Carlo Bootstrap (10,000 simulations)
n_simulations = 10000
bootstrap_sharpes = []

for sim in range(n_simulations):
    # Resample returns with replacement
    resampled_returns = np.random.choice(strategy_returns, size=n_days, replace=True)
    sharpe = resampled_returns.mean() / resampled_returns.std() * np.sqrt(252)
    bootstrap_sharpes.append(sharpe)

bootstrap_sharpes = np.array(bootstrap_sharpes)

# Calculate confidence intervals
ci_5th = np.percentile(bootstrap_sharpes, 5)
ci_50th = np.percentile(bootstrap_sharpes, 50)
ci_95th = np.percentile(bootstrap_sharpes, 95)
mean_sharpe = bootstrap_sharpes.mean()
std_sharpe = bootstrap_sharpes.std()

print(f"\nMonte Carlo Results (n={n_simulations:,}):")
print(f"Mean Sharpe:          {mean_sharpe:.2f}")
print(f"Median Sharpe:        {ci_50th:.2f}")
print(f"Std Dev of Sharpe:    {std_sharpe:.2f}")
print(f"95% Confidence Interval: [{ci_5th:.2f}, {ci_95th:.2f}]")

# Probability of negative Sharpe
prob_negative = (bootstrap_sharpes < 0).mean()
print(f"Probability of Sharpe < 0: {prob_negative:.1%}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_sharpes, bins=50, alpha=0.7, edgecolor='black', density=True)
plt.axvline(historical_sharpe, color='red', linestyle='--', linewidth=2, label=f'Historical Sharpe ({historical_sharpe:.2f})')
plt.axvline(ci_5th, color='orange', linestyle=':', label=f'5th Percentile ({ci_5th:.2f})')
plt.axvline(ci_95th, color='orange', linestyle=':', label=f'95th Percentile ({ci_95th:.2f})')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Density')
plt.title('Monte Carlo Bootstrap: Sharpe Ratio Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Expected Output (illustrative):**
```
Historical Sharpe Ratio: 1.12

Monte Carlo Results (n=10,000):
Mean Sharpe:          1.11
Median Sharpe:        1.12
Std Dev of Sharpe:    0.18
95% Confidence Interval: [0.82, 1.43]
Probability of Sharpe < 0: 0.0%
```

**Chart Interpretation:**  
- Distribution approximately normal (Central Limit Theorem).  
- 95% CI [0.82, 1.43] excludes zero → statistically significant edge.  
- Historical Sharpe 1.12 near median → not lucky outlier.  
- If CI included 0 (e.g., [-0.2, 1.0]), edge would be questionable.

---

## Challenge Round

1. **Bootstrap vs. Parametric Simulation**  
   When would parametric simulation (fitting returns to normal distribution) fail?

   <details><summary>Hint</summary>Real returns exhibit fat tails (kurtosis), skewness, and time-varying volatility (GARCH effects). Normal distribution underestimates tail risk. Bootstrap preserves empirical distribution but assumes i.i.d. returns (destroys autocorrelation). Solution: Use block bootstrap or fit GARCH/Student's t for better tail modeling.</details>

2. **Trade Shuffling Drawdown Analysis**  
   Strategy has 100 trades: 60 wins ($500 avg), 40 losses ($600 avg). Historical max DD = 10%. After shuffling trade order 1,000 times, 95th percentile DD = 18%. What does this tell you?

   <details><summary>Solution</summary>Historical max DD (10%) benefited from favorable trade timing (wins clustered early). 95th percentile (18%) shows worst-case timing if losses cluster. Strategy is vulnerable to sequencing risk. **Implication:** Size positions conservatively; expect DD up to 18% in unlucky scenarios.</details>

3. **Parameter Perturbation Stability**  
   Optimal MA length = 50. Monte Carlo with MA ∈ N(50, 10) yields: Mean OOS Sharpe = 1.0, 5th percentile = 0.3, 95th percentile = 1.5. Is strategy robust?

   <details><summary>Solution</summary>
   Wide range [0.3, 1.5] indicates high sensitivity to parameter choice. If MA drifts to 30 or 70 (within ±2σ), Sharpe could drop to 0.3 (marginal edge). **Not robust**: Small estimation errors in optimal MA significantly impact performance. Prefer strategy with tighter CI, e.g., [0.8, 1.2].
   </details>

4. **Stress Testing with Crisis Scenarios**  
   Historical returns: μ=0.05%, σ=1%. Monte Carlo with σ=3% (crisis) shows max DD increases from 15% to 45%. How should you adjust strategy?

   <details><summary>Solution</summary>Strategy poorly handles volatility spikes. Options: (1) Add volatility filter: scale position size inversely with realized volatility. (2) Implement stop-loss or trailing stop. (3) Diversify across uncorrelated strategies. (4) Reserve capital for max DD of 45% (not 15%) to avoid forced liquidation in crisis.
   </details>

---

## Key References

- **Efron & Tibshirani (1993)**: *An Introduction to the Bootstrap* (foundational bootstrap theory) ([CRC Press](https://www.routledge.com/))
- **Politis & Romano (1994)**: "The Stationary Bootstrap" (block bootstrap for time-series) ([JSTOR](https://www.jstor.org/))
- **López de Prado (2018)**: *Advances in Financial Machine Learning* (Chapter on Monte Carlo for backtests) ([Wiley](https://www.wiley.com/))
- **Pardo (2008)**: *The Evaluation and Optimization of Trading Strategies* (Monte Carlo in strategy validation) ([Wiley](https://www.wiley.com/))

**Further Reading:**  
- Monte Carlo for option pricing (variance reduction techniques: antithetic variates, control variates)  
- Sequential Monte Carlo (particle filters) for dynamic parameter estimation  
- Copula-based simulation for correlated asset returns
