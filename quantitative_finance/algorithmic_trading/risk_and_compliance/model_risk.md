# Model Risk

## Concept Skeleton

Model risk in algorithmic trading arises when quantitative models produce inaccurate predictions or valuations due to overfitting (capturing noise as signal), parameter estimation error (insufficient data, regime shifts), or faulty assumptions (normality, stationarity, linearity). Unlike market risk (price movements) or operational risk (system failures), model risk stems from intellectual flaws: trusting models beyond their validity domain, ignoring fat tails, or failing to validate out-of-sample. Consequences range from gradual strategy decay to catastrophic losses (LTCM 1998, quant meltdown 2007).

**Core Components:**
- **Overfitting:** Model tuned to historical data performs poorly on new data (curve-fitting noise, excessive parameters)
- **Parameter instability:** Coefficients valid in one regime break in another (correlation jumps in crises, momentum reverses)
- **Distributional assumptions:** Assuming normality when returns are fat-tailed (underestimate extreme events)
- **Survivorship bias:** Backtesting on successful stocks only (ignores delisted, bankrupt firms)
- **Look-ahead bias:** Using future information in backtest (data snooping, peeking at next day's close)

**Why it matters:** 1998 LTCM collapse (model assumed stable spreads, ignored flight-to-quality); 2007 quant meltdown (crowded factor exposures, all models deleveraged simultaneously); model risk is regulatory focus (Basel III FRTB, Fed SR 11-7).

---

## Comparative Framing

| Risk Type | **Market Risk** | **Model Risk** | **Operational Risk** |
|-----------|-----------------|----------------|----------------------|
| **Source** | Price/volatility changes (external) | Model specification error (internal) | Technology/human error (internal) |
| **Measurement** | VaR, stress testing | Out-of-sample performance, parameter sensitivity | Incident frequency, loss distribution |
| **Predictability** | Partially (historical vol, correlations) | Low (regime shifts, unknown unknowns) | Partially (failure rates, audit trails) |
| **Mitigation** | Diversification, hedging, position sizing | Model validation, ensemble methods, parameter robustness | Redundancy, testing, kill switches |
| **Example** | Stock drops 10% (lose money) | Momentum model reverses (mean-reversion regime) | Fat-finger order (10,000 instead of 100) |
| **Regulatory** | Basel Market Risk Rule (VaR, SVaR) | SR 11-7 (Fed model risk guidance) | Reg SCI (systems integrity) |

**Key insight:** Model risk is insidious—strategies work for years (in-sample), then fail catastrophically when regime shifts (out-of-sample reality).

---

## Examples & Counterexamples

### Examples of Model Risk Manifestations

1. **Overfitting: Momentum Parameter Optimization**  
   - **Backtest:** Optimize lookback period (10, 20, 30, ..., 200 days) → 73-day lookback yields highest Sharpe (2.5).  
   - **In-sample:** Perfect fit to 2010–2020 data (captures every regime shift).  
   - **Out-of-sample (2021–2023):** Sharpe drops to 0.3 (model captured noise, not signal).  
   - **Diagnosis:** 73 is arbitrary (no fundamental justification); should use standard periods (20, 50, 200) or ensemble (average 10/20/50).

2. **Parameter Instability: Correlation Breakdown**  
   - **Model:** Stat arb pairs trading (SPY-IWM spread mean-reverts with 90-day half-life).  
   - **Assumption:** Correlation = 0.85 (stable for 5 years).  
   - **Crisis (March 2020):** Correlation drops to 0.50 (flight to large caps, small caps crash harder).  
   - **Result:** Spread widens 5 standard deviations, doesn't revert → model loses 20% (assumed mean reversion).  
   - **Lesson:** Correlation is regime-dependent (stable in calm, breaks in stress).

3. **Fat-Tail Underestimation: VaR Model**  
   - **Model:** Parametric VaR (assume normal distribution, σ = 1% daily).  
   - **95% VaR:** -1.65σ = -1.65% daily loss (expected once per month).  
   - **Reality:** Returns fat-tailed (kurtosis = 8); 5% moves occur 2× per year (not once per decade).  
   - **Consequence:** October 2008, portfolio drops 10% in one day (beyond 99.9% VaR), violates risk limits, forced liquidation.  
   - **Solution:** Use historical VaR (empirical quantiles) or t-distribution (captures fat tails).

4. **Survivorship Bias: Backtest on S&P 500**  
   - **Strategy:** Buy low P/E stocks in S&P 500 (backtest 2000–2020).  
   - **Result:** 12% annual return (beats market).  
   - **Flaw:** S&P 500 composition today excludes 2000 bankruptcies (Enron, Lehman, etc.). Backtest only includes survivors.  
   - **True performance:** If including delisted stocks, return drops to 8% (many low-P/E stocks went bankrupt).  
   - **Fix:** Use point-in-time S&P 500 constituents (rebalance historical membership).

### Non-Examples (or Robust Models)

- **Ensemble models:** Average 10 momentum lookbacks (10, 20, ..., 200 days) → reduces overfitting, smooths parameter sensitivity.
- **Regime-aware strategies:** Separate bull/bear/crisis models (switch based on VIX, credit spreads) → adapts to parameter instability.
- **Stress testing:** Simulate 2008-level events → validates model doesn't assume stability.

---

## Layer Breakdown

**Layer 1: Overfitting and Data Snooping**  
**Overfitting:** Model captures idiosyncratic noise instead of generalizable patterns. More parameters = higher in-sample fit, worse out-of-sample performance.

**Symptoms:**  
- High in-sample Sharpe (2.5), low out-of-sample Sharpe (0.5).  
- Parameter tweaks (e.g., 72-day vs. 73-day lookback) drastically change performance.  
- Complex models (10+ parameters) with limited data (<5 years).

**Causes:**  
- **Excessive optimization:** Testing 100 parameter combinations, selecting best (multiple testing bias).  
- **Data snooping:** Iterating backtest, adjusting rules until profitable (p-hacking).  
- **Lack of regularization:** No penalty for model complexity (AIC, BIC, cross-validation).

**Mitigation:**  
- **Out-of-sample testing:** Reserve 20% of data for validation (never train on it).  
- **Walk-forward analysis:** Rolling in-sample optimization, test on next period, re-optimize (simulates real-world adaptation).  
- **Occam's Razor:** Prefer simpler models (5 parameters vs. 50).  
- **Ensemble methods:** Average multiple models (reduces overfitting to any single specification).

**Layer 2: Parameter Instability and Regime Shifts**  
**Problem:** Model parameters (β, correlation, volatility) change over time. Strategies optimized for one regime fail in another.

**Examples:**  
- **Momentum strength:** Works in trending markets (2009–2021), fails in choppy markets (2022).  
- **Volatility clustering:** GARCH models estimate σ = 15% (calm period), but σ spikes to 80% in crisis (VIX > 50).  
- **Credit spreads:** Investment-grade spread = 100 bps (normal), widens to 400 bps (crisis) → corporate bond model misprices risk.

**Detection:**  
- **Rolling parameter estimation:** Estimate β in 250-day windows, plot time series (if jumps >50%, parameter unstable).  
- **CUSUM tests:** Detect structural breaks (Chow test, Zivot-Andrews).  
- **Regime-switching models:** Hidden Markov Models (2 regimes: low-vol, high-vol).

**Adaptation:**  
- **Dynamic rebalancing:** Re-estimate parameters quarterly (not fixed 10-year lookback).  
- **Robust optimization:** Minimize worst-case performance across multiple regimes (not expected return).  
- **Regime filters:** Only trade momentum when VIX < 20 (avoid choppy periods).

**Layer 3: Distributional Assumptions and Fat Tails**  
**Gaussian Assumption:** Many models assume normal returns (mean μ, variance σ²). Reality: fat tails (kurtosis > 3), negative skew (crashes larger than rallies).

**Consequences:**  
- **VaR underestimation:** 95% VaR = -1.65σ assumes normality. With fat tails, 5% worst losses exceed -3σ.  
- **Option mispricing:** Black-Scholes assumes lognormal prices. Reality: implied vol smile (out-of-money puts expensive → crash insurance demand).  
- **Correlation breakdown:** Assumes stable ρ = 0.7. In crises, ρ → 1 (all stocks fall together, diversification fails).

**Robust Alternatives:**  
- **Historical simulation:** Use empirical return distribution (no parametric assumption).  
- **t-distribution:** Fatter tails than normal (degrees of freedom controls tail thickness).  
- **Extreme Value Theory (EVT):** Models tail specifically (Generalized Pareto Distribution for losses > threshold).  
- **Copulas:** Separate marginal distributions from dependence structure (capture tail dependence).

**Layer 4: Survivorship and Look-Ahead Bias**  
**Survivorship Bias:** Backtesting on current universe (S&P 500 today) excludes failed firms (bankruptcies, delistings). Overstates historical returns.

**Example:** Value strategy on Russell 2000 (2000–2020). If using today's constituents, excludes 40% that delisted (many low-P/B stocks went bankrupt). True return: 6% vs. survivor-biased 10%.

**Solution:** Use point-in-time databases (CRSP, Compustat with historical constituents, not survivor-filtered).

**Look-Ahead Bias:** Using information not available at trade time.  
- **Data snooping:** Backtest uses revised earnings (Q2 EPS reported July 31 but revised August 15; backtest uses August 15 number on July 31 signal).  
- **Close-to-close signals:** Buy signal at close, assume fill at close price (unrealistic; need open next day or intraday fill).  
- **Index rebalancing:** Buy stocks added to S&P 500 day before announcement (impossible—announcement is the trigger).

**Solution:** Timestamp all data (as-of-date), simulate order execution delay (T+1 fill), lag fundamental data by reporting lag.

**Layer 5: Model Validation and Governance**  
**Fed SR 11-7 Guidance (Model Risk Management):**  
1. **Model development:** Document assumptions, limitations, intended use.  
2. **Model validation:** Independent review (not model developer), out-of-sample testing, sensitivity analysis.  
3. **Model implementation:** Production controls (version control, access limits, change logs).  
4. **Ongoing monitoring:** Compare actual vs. predicted, detect performance degradation.

**Validation Tests:**  
- **Out-of-sample Sharpe ratio:** If drops >30% from in-sample, model likely overfit.  
- **Parameter sensitivity:** Perturb parameters ±10% → if performance swings >50%, model fragile.  
- **Stress testing:** Simulate 2008-level volatility, correlation spikes → if model collapses, add circuit breakers.  
- **Benchmarking:** Compare to naive strategy (buy-and-hold, equal-weight) → if underperforms, model adds no value.

---

## Mini-Project: Detecting Overfitting via Walk-Forward Analysis

**Goal:** Compare in-sample-optimized strategy to walk-forward-tested strategy.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate synthetic stock returns (momentum + noise)
np.random.seed(42)
n_days = 2000
true_momentum_lookback = 50  # "True" parameter (unknown to optimization)

returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
prices = 100 * np.cumprod(1 + returns)

# Add momentum effect (autocorrelation)
for i in range(true_momentum_lookback, n_days):
    momentum_signal = np.mean(returns[i-true_momentum_lookback:i])
    returns[i] += 0.3 * momentum_signal  # Momentum boosts next-day return

prices = 100 * np.cumprod(1 + returns)

# Strategy: Momentum (buy if past N-day return > 0, else hold cash)
def momentum_strategy(returns, lookback):
    signals = np.zeros(len(returns))
    for i in range(lookback, len(returns)):
        past_return = np.sum(returns[i-lookback:i])
        signals[i] = 1 if past_return > 0 else 0  # 1 = long, 0 = cash
    strategy_returns = signals * returns
    return strategy_returns

# Objective: Maximize Sharpe ratio
def objective(lookback, returns_in_sample):
    lookback = int(lookback[0])
    if lookback < 5 or lookback > 200:
        return 1e6  # Penalty for invalid lookback
    strat_returns = momentum_strategy(returns_in_sample, lookback)
    sharpe = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)
    return -sharpe  # Minimize negative Sharpe

# Scenario 1: In-Sample Optimization (OVERFIT)
in_sample_period = returns[:1500]
result = minimize(objective, x0=[50], args=(in_sample_period,), bounds=[(5, 200)], method='L-BFGS-B')
optimal_lookback_overfit = int(result.x[0])
print("=" * 80)
print("SCENARIO 1: IN-SAMPLE OPTIMIZATION (OVERFITTING RISK)")
print("=" * 80)
print(f"Optimal Lookback (In-Sample Fit):   {optimal_lookback_overfit} days")
print(f"True Parameter (Data Generating):    {true_momentum_lookback} days")
print()

# Test on out-of-sample period
out_of_sample_period = returns[1500:]
overfit_returns_IS = momentum_strategy(in_sample_period, optimal_lookback_overfit)
overfit_returns_OOS = momentum_strategy(out_of_sample_period, optimal_lookback_overfit)

sharpe_overfit_IS = np.mean(overfit_returns_IS) / np.std(overfit_returns_IS) * np.sqrt(252)
sharpe_overfit_OOS = np.mean(overfit_returns_OOS) / np.std(overfit_returns_OOS) * np.sqrt(252)

print(f"In-Sample Sharpe Ratio:              {sharpe_overfit_IS:.2f}")
print(f"Out-of-Sample Sharpe Ratio:          {sharpe_overfit_OOS:.2f}")
print(f"Performance Degradation:             {(sharpe_overfit_IS - sharpe_overfit_OOS) / sharpe_overfit_IS * 100:.1f}%")
print()

# Scenario 2: Walk-Forward Analysis (ROBUST)
print("=" * 80)
print("SCENARIO 2: WALK-FORWARD ANALYSIS (ROBUST VALIDATION)")
print("=" * 80)

walk_forward_window = 500  # Optimize on 500 days, test on next 100
test_window = 100
walk_forward_returns = []
walk_forward_lookbacks = []

for start in range(0, len(returns) - walk_forward_window - test_window, test_window):
    train_data = returns[start:start + walk_forward_window]
    test_data = returns[start + walk_forward_window:start + walk_forward_window + test_window]
    
    # Optimize on training window
    result = minimize(objective, x0=[50], args=(train_data,), bounds=[(5, 200)], method='L-BFGS-B')
    optimal_lookback_WF = int(result.x[0])
    walk_forward_lookbacks.append(optimal_lookback_WF)
    
    # Test on next period
    strat_returns_test = momentum_strategy(test_data, optimal_lookback_WF)
    walk_forward_returns.extend(strat_returns_test)

walk_forward_returns = np.array(walk_forward_returns)
sharpe_WF = np.mean(walk_forward_returns) / np.std(walk_forward_returns) * np.sqrt(252)

print(f"Walk-Forward Sharpe Ratio:           {sharpe_WF:.2f}")
print(f"Average Optimal Lookback (WF):       {np.mean(walk_forward_lookbacks):.0f} days (std: {np.std(walk_forward_lookbacks):.0f})")
print(f"Parameter Stability:                 {'STABLE' if np.std(walk_forward_lookbacks) < 20 else 'UNSTABLE'}")
print()

# Comparison
print("=" * 80)
print("COMPARISON: OVERFITTING vs. WALK-FORWARD")
print("=" * 80)
print(f"{'Method':<30} {'In-Sample Sharpe':<20} {'Out-of-Sample Sharpe':<20} {'Degradation':<15}")
print("-" * 80)
print(f"{'In-Sample Optimization':<30} {sharpe_overfit_IS:<20.2f} {sharpe_overfit_OOS:<20.2f} {(sharpe_overfit_IS - sharpe_overfit_OOS):<15.2f}")
print(f"{'Walk-Forward Analysis':<30} {'N/A':<20} {sharpe_WF:<20.2f} {'Adaptive':<15}")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Cumulative returns
cum_overfit_OOS = np.cumprod(1 + overfit_returns_OOS) - 1
cum_WF = np.cumprod(1 + walk_forward_returns) - 1
cum_buyhold = np.cumprod(1 + returns[1500:1500+len(cum_overfit_OOS)]) - 1

axes[0].plot(cum_buyhold * 100, label='Buy & Hold', linewidth=1.5, alpha=0.7)
axes[0].plot(cum_overfit_OOS * 100, label=f'Overfit Model (Lookback={optimal_lookback_overfit})', linewidth=1.5)
axes[0].plot(cum_WF[:len(cum_overfit_OOS)] * 100, label='Walk-Forward Model', linewidth=1.5)
axes[0].set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Model Risk: Overfitting vs. Walk-Forward Validation', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)

# Parameter stability
axes[1].plot(walk_forward_lookbacks, marker='o', linestyle='-', linewidth=1.5, markersize=5)
axes[1].axhline(true_momentum_lookback, color='red', linestyle='--', linewidth=2, label=f'True Parameter ({true_momentum_lookback})')
axes[1].axhline(optimal_lookback_overfit, color='orange', linestyle='--', linewidth=2, label=f'Overfit Parameter ({optimal_lookback_overfit})')
axes[1].set_ylabel('Optimal Lookback (days)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Walk-Forward Window', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_risk_overfitting.png', dpi=150)
plt.show()
```

**Expected Output (illustrative):**
```
================================================================================
SCENARIO 1: IN-SAMPLE OPTIMIZATION (OVERFITTING RISK)
================================================================================
Optimal Lookback (In-Sample Fit):   73 days
True Parameter (Data Generating):    50 days

In-Sample Sharpe Ratio:              1.85
Out-of-Sample Sharpe Ratio:          0.62
Performance Degradation:             66.5%

================================================================================
SCENARIO 2: WALK-FORWARD ANALYSIS (ROBUST VALIDATION)
================================================================================
Walk-Forward Sharpe Ratio:           1.12
Average Optimal Lookback (WF):       52 days (std: 15)
Parameter Stability:                 STABLE

================================================================================
COMPARISON: OVERFITTING vs. WALK-FORWARD
================================================================================
Method                         In-Sample Sharpe     Out-of-Sample Sharpe Degradation    
--------------------------------------------------------------------------------
In-Sample Optimization         1.85                 0.62                 1.23           
Walk-Forward Analysis          N/A                  1.12                 Adaptive       
================================================================================
```

**Interpretation:**  
- Overfit model (lookback=73) optimized to in-sample data → 66% Sharpe degradation out-of-sample (captured noise).  
- Walk-forward model adapts parameters every 100 days → stable Sharpe (1.12), closer to true parameter (52 vs. 50).  
- Lesson: Walk-forward analysis mitigates overfitting by simulating real-world parameter adaptation.

---

## Challenge Round

1. **Survivorship Bias Magnitude**  
   Backtest value strategy (low P/B) on current S&P 500 (2000–2020) → 12% return. If 30% of 2000 constituents delisted (avg loss -50% before delisting), estimate true return.

   <details><summary>Hint</summary>**Calculation:** Survivor-biased return assumes 100% of capital in survivors. True return: 70% in survivors (12% return), 30% in delistings (-50% return). True = 0.7 × 12% + 0.3 × (-50%) = 8.4% - 15% = **-6.6%** (if all delistings instant). More realistic: delistings phased over 20 years, weighted by market cap → true return ~8–9% (still materially lower than 12%).</details>

2. **Parameter Sensitivity Test**  
   Momentum strategy: lookback=50 days, Sharpe=1.5. Perturb lookback to 45, 55 → Sharpe drops to 0.8, 0.7. Is model robust?

   <details><summary>Solution</summary>**No, model is fragile.** 10% parameter change → 50% performance drop suggests overfitting (optimal parameter captures noise, not signal). Robust model: ±10% perturbation → <20% Sharpe change. **Fix:** Use ensemble (average lookbacks 40, 50, 60) or use standard parameters (20, 50, 200—no optimization).</details>

3. **Fat-Tail VaR Underestimation**  
   Parametric VaR (normal, σ=1.5%): 95% VaR = -2.47% daily. Historical data shows 5% worst losses average -4.0%. How much capital underestimated?

   <details><summary>Solution</summary>
   **Underestimation:** -4.0% actual vs. -2.47% VaR = 1.53% shortfall. On $100M portfolio: $1.53M capital gap per 1-in-20-day event. Over 1 year (250 days), expect ~12–13 VaR breaches (not 12.5 if normally distributed, but 13 with fat tails). **Annualized underestimation:** 13 × $1.53M = ~$20M. Regulatory capital (3× VaR) insufficient, need 4–5× VaR or use historical/EVT method.
   </details>

4. **Regime Shift Detection**  
   Pairs trading model (SPY-IWM spread, β=0.85 for 5 years) suddenly shows β=0.60 (250-day rolling). Is this parameter instability or noise?

   <details><summary>Solution</summary>
   **Test:** (1) **Statistical significance:** Run Chow test for structural break at β shift date. If p-value <0.05, regime change confirmed. (2) **Economic explanation:** Check if macro event (e.g., COVID crash, small-cap underperformance) justifies β drop. If yes, regime shift (not noise). (3) **Persistence:** If β=0.60 persists 3+ months, likely new regime. If reverts to 0.85 in 2 weeks, noise. **Action:** If regime shift, re-estimate β or halt strategy (mean reversion assumption invalid if β unstable).</details>

---

## Key References

- **Rebonato (2007)**: *Plight of the Fortune Tellers* (model risk in derivatives) ([Princeton](https://press.princeton.edu/))
- **Lopez de Prado (2018)**: *Advances in Financial Machine Learning* (overfitting, cross-validation) ([Wiley](https://www.wiley.com/))
- **Federal Reserve SR 11-7**: Guidance on Model Risk Management ([FederalReserve.gov](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm))
- **Bailey et al. (2014)**: "The Probability of Backtest Overfitting" (SSRN, cross-validation methods)

**Further Reading:**  
- When Genius Failed (LTCM collapse, model risk from extreme events)  
- Basel FRTB (Fundamental Review of the Trading Book): Model risk capital charges  
- Quant meltdown (August 2007): Crowded factor trades, synchronized deleveraging
