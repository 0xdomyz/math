# Curve Fitting Risk

## Concept Skeleton

Curve fitting risk (overfitting) occurs when strategy parameters are excessively tuned to historical data idiosyncrasies, producing excellent backtest results that fail to generalize to live trading, arising from optimization over too many parameters, insufficient out-of-sample testing, or data-snooping (testing multiple hypotheses until one appears significant). Symptoms: in-sample Sharpe ratio >> out-of-sample Sharpe, parameter sensitivity (small changes cause large performance swings), excessive complexity relative to data quantity.

**Core Components:**
- **Degrees of freedom**: Number of free parameters (moving average lengths, thresholds, look-backs) vs. data points
- **In-sample vs. out-of-sample divergence**: Large performance gap indicates overfitting
- **P-hacking**: Testing multiple strategies/parameters until one shows statistical significance by chance
- **Model complexity penalty**: Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC) penalize parameter count
- **Cross-validation**: k-fold or walk-forward testing to validate parameter stability

**Why it matters:** Overfitted strategies lose money in live trading despite stellar backtests; source of "backtest optimism"; primary reason algorithmic strategies fail post-deployment.

---

## Comparative Framing

| Dimension | **Overfitted Strategy** | **Robust Strategy** | **Underfit Strategy** |
|-----------|-------------------------|---------------------|-----------------------|
| **In-sample Sharpe** | 2.5+ (too good to be true) | 1.2–1.8 (realistic) | < 0.5 (poor) |
| **Out-of-sample Sharpe** | < 0.5 (fails) | 1.0–1.5 (validates) | < 0.5 (still poor) |
| **Parameter count** | 10+ free parameters | 2–5 parameters | 1–2 parameters (too simple) |
| **Parameter sensitivity** | High (small changes → large impact) | Low (stable across range) | N/A (few parameters) |
| **Walk-forward consistency** | Inconsistent OOS results | Consistent across periods | Consistently underperforms |
| **Signal source** | Historical noise mining | Economic rationale | Ignores available information |

**Key insight:** Optimal complexity balances capturing true signal vs. fitting noise. Occam's Razor: Simpler models with fewer parameters generalize better.

---

## Examples & Counterexamples

### Examples of Curve Fitting Risk

1. **Over-Parameterized Moving Average Strategy**  
   - Parameters optimized: MA1 length (10–200), MA2 length (50–300), entry threshold, exit threshold, stop-loss %, take-profit %, max holding period  
   - Grid search: 20 × 20 × 5 × 5 × 10 × 10 × 8 = 1.6 million combinations tested  
   - Best combo: MA1=73, MA2=184, entry=1.02, exit=0.98, stop=3.2%, target=5.7%, hold=13 days  
   - **Problem**: Specific to historical data; unlikely these precise values have economic meaning  

2. **Data Snooping in Factor Strategies**  
   - Researcher tests 100 different factors (P/E, momentum, volatility, etc.) on same historical dataset  
   - Finds 5 factors significant at 95% confidence (p < 0.05)  
   - **Expected by chance**: 100 × 0.05 = 5 false positives  
   - **Reality**: No true edge, just statistical noise  

3. **Equity Curve Fitting**  
   - Strategy optimized to minimize specific historical drawdown (e.g., 2008 crisis)  
   - Parameters chosen to exit equities precisely before crash  
   - **Problem**: Future crises have different triggers; tailored to past catastrophe, not future risks

### Non-Examples (or Strategies with Low Overfitting Risk)

- **Simple 60/40 portfolio**: No optimization; fixed allocation; minimal curve fitting.
- **Value investing with P/E < 15 rule**: Single parameter, economically justified (cheap stocks have expected mean reversion).
- **Reversion to moving average with fixed 2-sigma threshold**: Parsimonious; statistically motivated.

---

## Layer Breakdown

**Layer 1: Detecting Overfitting via IS/OOS Gap**  
Calculate:  
\[
\text{Degradation Ratio} = \frac{\text{Sharpe}_{\text{OOS}}}{\text{Sharpe}_{\text{IS}}}
\]  
**Thresholds:**  
- Ratio > 0.7: Healthy (minor degradation expected)  
- Ratio 0.4–0.7: Moderate overfitting (investigate parameter stability)  
- Ratio < 0.4: Severe overfitting (strategy likely unviable)

**Layer 2: Parameter Sensitivity Analysis**  
Vary each parameter ±10% around optimal value, observe performance change:  
\[
\text{Sensitivity} = \frac{\Delta \text{Sharpe}}{\Delta \text{Parameter}}
\]  
High sensitivity → fragile strategy; small estimation errors in live trading cause failure.  
**Solution:** Select parameter ranges with flat performance plateau (robust zone).

**Layer 3: Model Complexity Penalties**  
Use information criteria to balance fit quality vs. complexity:  
\[
\text{AIC} = -2 \ln(L) + 2k
\]  
\[
\text{BIC} = -2 \ln(L) + k \ln(n)
\]  
where \(L\) = likelihood, \(k\) = parameter count, \(n\) = sample size.  
Lower AIC/BIC preferred; penalizes adding parameters unless they significantly improve fit.

**Layer 4: Multiple Testing Correction**  
If testing \(m\) strategies, adjust significance threshold using Bonferroni correction:  
\[
\alpha_{\text{adjusted}} = \frac{\alpha}{m}
\]  
Example: Testing 20 strategies, require p < 0.05/20 = 0.0025 for significance.  
**Alternative:** False Discovery Rate (FDR) control via Benjamini-Hochberg procedure.

---

## Mini-Project: Detecting Overfitting with Parameter Sweeps

**Goal:** Optimize strategy parameters and test sensitivity to detect overfitting.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic price data with mean reversion
np.random.seed(777)
n_days = 1000
prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
prices = prices - np.linspace(0, 50, n_days)  # Inject mean reversion trend
df = pd.DataFrame({'price': prices})

# Split: In-sample (70%), Out-of-sample (30%)
split_idx = int(0.7 * n_days)
is_data = df.iloc[:split_idx]
oos_data = df.iloc[split_idx:]

# Parameter grid: lookback period for z-score calculation
lookback_range = range(10, 201, 10)
is_sharpes = []
oos_sharpes = []

for lookback in lookback_range:
    # In-sample optimization
    is_copy = is_data.copy()
    is_copy['rolling_mean'] = is_copy['price'].rolling(lookback).mean()
    is_copy['rolling_std'] = is_copy['price'].rolling(lookback).std()
    is_copy['z_score'] = (is_copy['price'] - is_copy['rolling_mean']) / is_copy['rolling_std']
    is_copy['signal'] = 0
    is_copy.loc[is_copy['z_score'] < -2, 'signal'] = 1  # Buy oversold
    is_copy.loc[is_copy['z_score'] > 2, 'signal'] = -1  # Sell overbought
    is_copy['returns'] = is_copy['price'].pct_change()
    is_copy['strategy_returns'] = is_copy['signal'].shift(1) * is_copy['returns']
    is_sharpe = is_copy['strategy_returns'].mean() / is_copy['strategy_returns'].std() * np.sqrt(252)
    is_sharpes.append(is_sharpe if not np.isnan(is_sharpe) else 0)
    
    # Out-of-sample test
    oos_copy = oos_data.copy()
    oos_copy['rolling_mean'] = oos_copy['price'].rolling(lookback).mean()
    oos_copy['rolling_std'] = oos_copy['price'].rolling(lookback).std()
    oos_copy['z_score'] = (oos_copy['price'] - oos_copy['rolling_mean']) / oos_copy['rolling_std']
    oos_copy['signal'] = 0
    oos_copy.loc[oos_copy['z_score'] < -2, 'signal'] = 1
    oos_copy.loc[oos_copy['z_score'] > 2, 'signal'] = -1
    oos_copy['returns'] = oos_copy['price'].pct_change()
    oos_copy['strategy_returns'] = oos_copy['signal'].shift(1) * oos_copy['returns']
    oos_sharpe = oos_copy['strategy_returns'].mean() / oos_copy['strategy_returns'].std() * np.sqrt(252)
    oos_sharpes.append(oos_sharpe if not np.isnan(oos_sharpe) else 0)

# Find optimal parameter
optimal_idx = np.argmax(is_sharpes)
optimal_lookback = list(lookback_range)[optimal_idx]
optimal_is_sharpe = is_sharpes[optimal_idx]
optimal_oos_sharpe = oos_sharpes[optimal_idx]

# Calculate degradation ratio
degradation_ratio = optimal_oos_sharpe / optimal_is_sharpe if optimal_is_sharpe != 0 else 0

print(f"Optimal Lookback: {optimal_lookback} days")
print(f"In-Sample Sharpe: {optimal_is_sharpe:.2f}")
print(f"Out-of-Sample Sharpe: {optimal_oos_sharpe:.2f}")
print(f"Degradation Ratio: {degradation_ratio:.2f}")

if degradation_ratio < 0.4:
    print("⚠️  WARNING: Severe overfitting detected!")
elif degradation_ratio < 0.7:
    print("⚠️  Moderate overfitting; review parameter stability")
else:
    print("✓ Acceptable OOS performance")

# Plot parameter sensitivity
plt.figure(figsize=(10, 5))
plt.plot(lookback_range, is_sharpes, label='In-Sample Sharpe', marker='o')
plt.plot(lookback_range, oos_sharpes, label='Out-of-Sample Sharpe', marker='s')
plt.axvline(optimal_lookback, color='red', linestyle='--', label=f'Optimal ({optimal_lookback})')
plt.xlabel('Lookback Period (days)')
plt.ylabel('Sharpe Ratio')
plt.title('Parameter Sensitivity: In-Sample vs Out-of-Sample')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Expected Output (illustrative):**
```
Optimal Lookback: 80 days
In-Sample Sharpe: 1.85
Out-of-Sample Sharpe: 0.95
Degradation Ratio: 0.51
⚠️  Moderate overfitting; review parameter stability
```

**Chart Interpretation:**  
- IS Sharpe peaks sharply at specific lookback (overfitting to noise).  
- OOS Sharpe flatter or negative around optimal IS parameter (fails to generalize).  
- **Robust parameter**: Choose lookback with stable OOS Sharpe across range (e.g., 60–120 days), not peak IS Sharpe.

---

## Challenge Round

1. **Data Snooping Bias**  
   Researcher tests 50 trading strategies on same dataset. One shows Sharpe ratio 1.8 with p-value 0.03. Is this statistically significant?

   <details><summary>Hint</summary>With 50 tests, expect 50 × 0.05 = 2.5 false positives at α=0.05. One significant result is consistent with chance. Apply Bonferroni correction: α_adjusted = 0.05/50 = 0.001. Strategy with p=0.03 > 0.001 → not significant after correction.</details>

2. **Parameter Plateau vs. Peak**  
   After optimization, two lookback periods emerge: (A) Lookback=100, IS Sharpe 1.5, OOS Sharpe 1.3; (B) Lookback=150, IS Sharpe 1.6, OOS Sharpe 1.4. Which is better?

   <details><summary>Solution</summary>
   Option B has higher IS *and* OOS Sharpe, suggesting true signal. But also check: (1) Degradation ratio: A = 1.3/1.5 = 0.87; B = 1.4/1.6 = 0.88 (both healthy). (2) Stability: If Sharpe remains >1.2 for lookback 120–180 (plateau), prefer B. If B's Sharpe drops sharply outside 145–155, it's overfitted peak → prefer A.
   </details>

3. **Equity Curve Overfitting**  
   A strategy is optimized to minimize max drawdown during 2008 financial crisis. Post-2008 OOS test shows max drawdown 5% (excellent), but 2015–2020 OOS shows max drawdown 25% (poor). What happened?

   <details><summary>Solution</summary>Parameters tailored to 2008 crisis characteristics (equity crash, VIX spike). Strategy learned to exit before *that specific* crash but lacks generalization to other regimes (e.g., 2015 volatility, 2020 pandemic). **Remedy:** Stress test across multiple historical crises; avoid optimizing to single event.</details>

4. **Complexity Penalty Trade-Off**  
   Model A: 2 parameters, IS Sharpe 1.2, OOS Sharpe 1.0. Model B: 10 parameters, IS Sharpe 1.8, OOS Sharpe 1.1. Which is preferable?

   <details><summary>Hint</summary>
   Model B has higher OOS Sharpe (1.1 vs. 1.0), but: (1) Degradation: A = 1.0/1.2 = 0.83 (healthy); B = 1.1/1.8 = 0.61 (moderate overfitting). (2) Stability: More parameters → higher risk of parameter drift in live trading. (3) Occam's Razor: If Model A's 1.0 Sharpe is acceptable, prefer simplicity. If seeking alpha, Model B's 1.1 OOS may justify complexity if walk-forward tested.
   </details>

---

## Key References

- **Bailey et al. (2014)**: "The Probability of Backtest Overfitting" ([Journal of Computational Finance](https://www.risk.net/))
- **López de Prado (2018)**: *Advances in Financial Machine Learning* (Chapter 11: Overfitting) ([Wiley](https://www.wiley.com/))
- **Harvey et al. (2016)**: "...and the Cross-Section of Expected Returns" (factor data-snooping) ([Review of Financial Studies](https://academic.oup.com/rfs))
- **Aronson (2007)**: *Evidence-Based Technical Analysis* (cognitive biases, data mining) ([Wiley](https://www.wiley.com/))

**Further Reading:**  
- Deflated Sharpe Ratio (adjusts for multiple testing and non-normal returns)  
- Combinatorial Purged Cross-Validation (CPCV) to reduce overfitting in time-series  
- White's Reality Check for data snooping in trading strategies
