# Market Efficiency: Weak, Semi-Strong, and Strong Forms

## I. Concept Skeleton

**Definition:** Market efficiency describes how quickly and completely asset prices incorporate available information. Three forms: weak (historical prices), semi-strong (all public information), strong (all information including private).

**Purpose:** Assess whether markets accurately price securities, identify if trading profits require information advantages or inefficiency exploitation, understand regulatory implications of market structure.

**Prerequisites:** Information theory, price dynamics, behavioral finance, regression analysis, hypothesis testing.

---

## II. Comparative Framing

| **Efficiency Form** | **Information Incorporated** | **Testable Implication** | **Anomaly Type** | **Profit Opportunity** |
|------------------|---------------------------|----------------------|-----------------|----------------------|
| **Weak Form** | All past prices/volume | Technical analysis fails | Momentum reversals | No: Random walk (reject) |
| **Semi-Strong Form** | All public information | News instantly priced | Calendar effects, splits | No: CAPM expected returns only |
| **Strong Form** | All information (public + private) | No insider trading profits | None theoretically | No: Nobody beats market always |
| **Informationally Efficient** | Same as strong | Martingale property | None possible | No profitable arbitrage |
| **Behaviorally Inefficient** | Systematic biases | Predictable overreactions | Momentum, reversal, sentiment | Yes: Exploit cognitive errors |
| **Microstructure Efficient** | Information via order flow | Spreads = information content | Bid-ask bounce, adverse selection | Maybe: High-frequency advantages |

---

## III. Examples & Counterexamples

### Example 1: Semi-Strong Test: Earnings Announcement (Simple Case)
- **Setup:** Stock trading $100. Company announces 40% earnings surprise at 4:05 PM.
- **Semi-Strong Prediction:** Price adjusts to $128 (30% gain) in first 30 seconds after announcement
- **Observed:** Price opens at 4:05 PM exactly at $128.45 (observed gain = 28.45%)
- **Interpretation:** Market incorporated information efficiently; no arbitrage profit possible
- **Key Insight:** If prices jumped to $128 only after 1 hour, semi-strong efficiency fails (profit by early trading)

### Example 2: Weak Form Violation: Momentum Strategy Profits (Failure Case)
- **Setup:** Historical data shows 60-day momentum predicts next 20-day returns: r_t = 0.05 × r_{t-60} + noise
- **Test:** Jegadeesh & Titman (1993) find winners outperform losers by 12.0% annually
- **Weak Form Violation:** If only historical prices used, can predict future returns (contradiction to random walk)
- **But:** Transaction costs (1% round-trip) reduce profits to 8%, still significant
- **Key Insight:** Weak form violated empirically, but profit after costs depends on implementation

### Example 3: Strong Form Violation: Insider Trading (Edge Case - Illegal)
- **Setup:** CEO knows company filing bankruptcy in 1 week. Shorts 100,000 shares at $100.
- **Strong Form Prediction:** Impossible to profit from private information; prices should reflect this
- **Reality:** Stock drops to $20 in 1 week. Insider profit: $8,000,000
- **But:** Illegal under SEC Rule 10b-5. Punishment: Criminal penalties + disgorgement
- **Key Insight:** Strong form violated by definition (private information profits), but prevented by regulation

---

## IV. Layer Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│          MARKET EFFICIENCY HIERARCHY (Fama, 1970)               │
└─────────────────────────────────────────────────────────────────┘

MARKET EFFICIENCY SPECTRUM:

Most Efficient (No Profit Possible):
    ↓
┌─────────────────────────────────────────┐
│  STRONG FORM EFFICIENCY                 │
│  All information (public + private)      │
│  P_t = E[V_t | All Information]         │
│  ├─ Insider information worthless        │
│  ├─ Mutual fund managers underperform    │
│  ├─ Private equity deals value-added     │
│  └─ Violation: Insider trading profits   │
│     (Illegal)                            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  SEMI-STRONG FORM EFFICIENCY            │
│  All public information                  │
│  P_t = E[V_t | Public Information]      │
│  ├─ News instantly priced               │
│  ├─ Technical analysis has no value     │
│  ├─ Fundamental analysis useless        │
│  ├─ Stock splits predictable            │
│  └─ Violation: Calendar effects profit   │
│     (Stock market anomalies)             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  WEAK FORM EFFICIENCY                   │
│  Only past prices/volume                 │
│  P_t = E[V_t | {P_s, V_s : s < t}]     │
│  ├─ Past returns unrelated to future    │
│  ├─ Technical analysis fails            │
│  ├─ Random walk: ΔP_t ~ N(μ, σ²)       │
│  └─ Violation: Momentum profits         │
│     (High returns follow high returns)   │
└─────────────────────────────────────────┘
    ↓
Least Efficient (Maximum Profit Opportunity)

INFORMATION SETS HIERARCHY:

Strong Efficiency (Ω_Strong):
├─ Public information: News, earnings, filings, macro data
├─ Private information: CEO knowledge, board decisions, M&A plans
├─ Historical prices: P_t, P_{t-1}, ..., P_0
├─ Historical volume: V_t, V_{t-1}, ..., V_0
├─ Order flow: Buy/sell side information
├─ Sentiment data: Social media, news tone
├─ Insider ownership: Holdings, transactions
└─ All feasible information

Semi-Strong Efficiency (Ω_Public):
├─ Public information: News, earnings, filings, macro data
├─ Historical prices: P_t, P_{t-1}, ..., P_0
├─ Historical volume: V_t, V_{t-1}, ..., V_0
├─ Order flow: Buy/sell side information (public)
├─ Sentiment data: Social media, news tone
├─ Insider ownership: Public filings only
└─ (EXCLUDES: Private information, board secrets)

Weak Efficiency (Ω_Prices):
├─ Historical prices: P_t, P_{t-1}, ..., P_0
├─ Historical volume: V_t, V_{t-1}, ..., V_0
└─ (EXCLUDES: All fundamental information)

TESTABLE HYPOTHESES:

Weak Form Test:
├─ H₀: E[r_{t+1} | {r_s : s ≤ t}] = E[r_{t+1}]  (Random walk)
├─ H₁: E[r_{t+1} | {r_s : s ≤ t}] ≠ E[r_{t+1}]  (Predictable)
├─ Test Method: Autocorrelation (ACF), runs test
├─ Data: 50+ years daily returns
├─ Result (US stocks): Generally cannot reject H₀
│          BUT: Short-term reversal, long-term momentum violations

Semi-Strong Test:
├─ H₀: E[r_{t+d} | News_t] = E[r_{t+d}]         (Immediate adjustment)
├─ H₁: E[r_{t+d} | News_t] ≠ E[r_{t+d}]         (Slow adjustment)
├─ Test Method: Event study (abnormal returns)
│  ├─ Expected return: E[r_t] = α + β*r_market
│  ├─ Abnormal: AR_t = r_t - E[r_t]
│  └─ Cumulative: CAR = Σ AR_t
├─ Data: Earnings announcements, splits, M&A
├─ Result: Generally cannot reject H₀
│          BUT: Post-earnings announcement drift (PEAD) violations

Strong Form Test:
├─ H₀: No abnormal returns from insider information
├─ H₁: Insiders can generate abnormal returns
├─ Test Method: Insider trading data, follow insider purchases
│  ├─ Calculate abnormal returns after insider buy
│  ├─ Compare to random portfolio
│  └─ Insider returns should equal market returns if H₀ true
├─ Data: SEC Form 4 filings (insider transactions)
├─ Result: REJECT H₀ consistently
│          Insiders earn 5-10% annual abnormal returns
│          (Note: Limited by SEC regulation, not purely private info)

EVENT STUDY METHODOLOGY (Semi-Strong Test):

              Event Occurs
                 (t=0)
                    │
     ┌──────────────┼──────────────┐
     │              │              │
Pre- Event Window│ Post-Event Window
-60 days to -1   │ +1 to +60 days
     │              │              │
     ▼              ▼              ▼
Estimate model   Measure      Check if
on pre-window    abnormal     abnormal
                 returns      persists

Setup:
├─ Normal model: AR_t = r_t - (α + β*r_m,t)
├─ Parameters: Estimated on [-60, -1] window
├─ Abnormal return on event day: AR_0 = r_0 - E[r_0]
├─ Cumulative abnormal return: CAR = Σ AR_t
├─ Statistical test: t-stat = CAR / σ(AR)

Results Interpretation:
├─ If CAR_[0,+30] ≈ 0: Semi-strong holds (prices adjusted immediately)
├─ If CAR_[0,+30] > 0: Semi-strong fails (positive drift = underprice initially)
├─ If CAR_[0,+30] < 0: Semi-strong fails (negative drift = overprice initially)
└─ Typical finding: CAR ≈ 0 by +5 days (mostly efficient, small drift)

ANOMALIES & EFFICIENCY VIOLATIONS:

📊 Weak Form Violations:
├─ Momentum effect (12-month): Winners continue winning 1 year out
├─ Reversal (60-month): Long-term losers outperform
├─ January effect: Returns higher in January than other months
├─ Day-of-week effect: Monday returns lower than other days
├─ Turn-of-month effect: Returns higher at month-start

📊 Semi-Strong Violations:
├─ Post-earnings announcement drift (PEAD): 5-10% drift after earnings
├─ Size effect: Small stocks outperform large (after risk adjustment)
├─ Value effect: High book-to-market stocks outperform growth
├─ Seasonal patterns: Summer rally, Santa Claus rally
├─ Dividend surprise effect: Prices drift after unexpected dividend

📊 Strong Form Violations:
├─ Insider trading profits: 5-10% annual abnormal returns
├─ Corporate officers: Know future buyback announcements
├─ Board members: Access M&A plans before announcement
├─ But: Illegal; SEC punishes insider trading

MARKET EFFICIENCY SPEED (Half-Life):

Information Incorporation Speed:
├─ Public information → Stock price: 100ms to 30 seconds (market makers)
├─ News headlines → Derivatives market: 10-100ms (HFT)
├─ Derivative news → Underlying stock: 100-500ms (arbitrage)
├─ Material news → Bond market: 1-10 seconds (slower to react)
├─ Macro news → Currency markets: 100ms (24-hour markets, slow humans)

Event Window Efficiency:
├─ Earnings announcement: 90% price adjustment within 5 seconds
├─ Merger announcement: 95% adjustment within 30 seconds
├─ Dividend surprise: 80% adjustment within 60 seconds
├─ Macro surprise: 50% adjustment within 1 minute (retail reaction lag)

BEHAVIORAL EXCEPTIONS TO EFFICIENCY:

Behavioral Finance: Markets influenced by psychology
├─ Overconfidence: Traders overestimate precision
├─ Herding: Follow-the-crowd trading (momentum)
├─ Anchoring: Prices sticky to round numbers
├─ Disposition effect: Hold losers, sell winners (wrong bias)
└─ Result: Predictable but difficult to exploit (costs, timing)

Microstructure Inefficiency:
├─ Order types: Limit vs market have different impact
├─ Bid-ask spread: Temporary price pressure (not permanent)
├─ Short-sale constraints: Cannot short unpopular stocks
├─ Liquidity: Illiquid stocks have return premiums (liquidity premium)
└─ Result: Small but consistent inefficiencies (exploitable with algorithms)
```

---

## V. Mini-Project: Market Efficiency Testing & Anomaly Detection (650 lines)

```python
"""
Market Efficiency Testing: Weak, Semi-Strong, and Strong Form Analysis
Tests for violations of market efficiency hypothesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

# ============================================================================
# 1. WEAK FORM EFFICIENCY TESTS
# ============================================================================

def test_autocorrelation(returns, max_lags=20):
    """
    Test weak form efficiency using autocorrelation.
    If returns are random walk, autocorrelation should be ~0.
    """
    
    autocorr = pd.Series(returns).autocorr(lag=1)
    
    # Statistical test: Box-Ljung test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    lb_test = acorr_ljungbox(returns, lags=max_lags, return_df=True)
    
    return autocorr, lb_test

def test_momentum_effect(prices, lookback=60, forward=20):
    """
    Test for momentum effect: Do past winners continue as winners?
    Weak form violation if momentum is predictive.
    """
    
    returns = np.diff(np.log(prices))
    n = len(returns)
    
    # Divide returns into deciles based on lookback window
    momentum_returns = []
    forward_returns = []
    
    for t in range(lookback, n - forward):
        # Past momentum (lookback period)
        past_ret = returns[t-lookback:t].sum()
        momentum_returns.append(past_ret)
        
        # Future return (forward period)
        future_ret = returns[t:t+forward].sum()
        forward_returns.append(future_ret)
    
    # Correlation: High correlation = weak form violation
    corr = np.corrcoef(momentum_returns, forward_returns)[0, 1]
    
    # Regression
    momentum_returns = np.array(momentum_returns)
    forward_returns = np.array(forward_returns)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(momentum_returns, forward_returns)
    
    return {
        'correlation': corr,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def test_reversal_effect(prices, lookback=250, forward=20):
    """
    Test for reversal effect: Do past losers become future winners?
    Long-term reversal would violate weak form efficiency.
    """
    
    returns = np.diff(np.log(prices))
    n = len(returns)
    
    past_returns = []
    future_returns = []
    
    for t in range(lookback, n - forward):
        # Long-term past return
        past_ret = returns[t-lookback:t].sum()
        past_returns.append(past_ret)
        
        # Forward return
        future_ret = returns[t:t+forward].sum()
        future_returns.append(future_ret)
    
    # Expected sign: Negative correlation (reversal)
    corr = np.corrcoef(past_returns, future_returns)[0, 1]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(past_returns, future_returns)
    
    return {
        'correlation': corr,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# ============================================================================
# 2. SEMI-STRONG FORM TESTS: EVENT STUDY
# ============================================================================

def event_study(prices, event_dates, pre_window=60, post_window=60):
    """
    Conduct event study to test semi-strong efficiency.
    Event = public information release (earnings, dividend, M&A, etc.)
    """
    
    market_returns = np.diff(np.log(prices))
    n_events = len(event_dates)
    
    abnormal_returns_list = []
    
    for event_idx in event_dates:
        if event_idx < pre_window or event_idx + post_window >= len(prices):
            continue
        
        # Estimation window: pre-event period
        est_start = event_idx - pre_window - 100  # Extra buffer
        est_end = event_idx - pre_window
        
        est_returns = market_returns[est_start:est_end]
        
        # Normal return: Assume just historical average (simple model)
        normal_return = est_returns.mean()
        
        # Event window: pre + post event
        event_start = event_idx - pre_window
        event_end = event_idx + post_window
        
        event_window_returns = market_returns[event_start:event_end]
        
        # Abnormal returns
        abnormal_returns = event_window_returns - normal_return
        
        abnormal_returns_list.append({
            'event_idx': event_idx,
            'ar_array': abnormal_returns,
            'car': abnormal_returns.sum()  # Cumulative abnormal return
        })
    
    return abnormal_returns_list

# ============================================================================
# 3. SEASONALITY & CALENDAR ANOMALIES
# ============================================================================

def test_seasonality(prices, dates):
    """
    Test for seasonality effects (January effect, day-of-week, etc.)
    Violation of semi-strong efficiency.
    """
    
    returns = np.diff(np.log(prices))
    
    # Month analysis
    months = [d.month for d in dates[1:]]  # Skip first (diff shifts by 1)
    
    monthly_returns = {}
    for month in range(1, 13):
        month_rets = returns[np.array(months) == month]
        if len(month_rets) > 0:
            monthly_returns[month] = {
                'mean': month_rets.mean(),
                'std': month_rets.std(),
                'count': len(month_rets)
            }
    
    # Day-of-week analysis
    dow = [d.weekday() for d in dates[1:]]  # 0=Monday, 6=Sunday
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    dow_returns = {}
    for day in range(5):
        day_rets = returns[np.array(dow) == day]
        if len(day_rets) > 0:
            dow_returns[dow_names[day]] = {
                'mean': day_rets.mean(),
                'std': day_rets.std(),
                'count': len(day_rets)
            }
    
    return monthly_returns, dow_returns

# ============================================================================
# 4. INSIDER TRADING ABNORMAL RETURNS (Strong Form Test)
# ============================================================================

def test_insider_trading(prices, insider_buy_dates, lookback=30, forward=60):
    """
    Test for abnormal returns after insider buys (strong form violation).
    If insiders have private information, their trades should predict returns.
    """
    
    returns = np.diff(np.log(prices))
    
    abnormal_returns = []
    
    for insider_date in insider_buy_dates:
        if insider_date + forward >= len(prices):
            continue
        
        # Estimate normal return on pre-event period
        if insider_date > lookback:
            pre_returns = returns[insider_date-lookback:insider_date]
            normal_return = pre_returns.mean()
        else:
            normal_return = 0
        
        # Post-event returns
        post_returns = returns[insider_date:insider_date+forward]
        
        # Abnormal return
        post_abnormal = post_returns - normal_return
        
        abnormal_returns.append({
            'date': insider_date,
            'abnormal_return': post_abnormal.sum(),
            'abnormal_mean': post_abnormal.mean()
        })
    
    # Statistical test
    if len(abnormal_returns) > 0:
        abnormal_array = np.array([x['abnormal_return'] for x in abnormal_returns])
        mean_abnormal = abnormal_array.mean()
        std_abnormal = abnormal_array.std()
        t_stat = mean_abnormal / (std_abnormal / np.sqrt(len(abnormal_array)))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(abnormal_array)-1))
    else:
        mean_abnormal = 0
        t_stat = 0
        p_value = 1
    
    return abnormal_returns, mean_abnormal, t_stat, p_value

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("MARKET EFFICIENCY TESTING: Weak, Semi-Strong, Strong Forms")
print("="*70)

np.random.seed(42)

# Generate synthetic price data
n_days = 1000
true_drift = 0.0005  # 0.05% daily drift
true_vol = 0.01      # 1% daily volatility

daily_returns = np.random.normal(true_drift, true_vol, n_days)
prices = 100 * np.exp(np.cumsum(daily_returns))

# Create date index
from datetime import datetime, timedelta
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

print("\n" + "="*70)
print("TEST 1: WEAK FORM EFFICIENCY - Autocorrelation")
print("="*70)

autocorr, lb_test = test_autocorrelation(daily_returns, max_lags=20)
print(f"\n1-Day Autocorrelation: {autocorr:.6f}")
print(f"Interpretation: {'Rejection (Weak Form violated)' if abs(autocorr) > 0.05 else 'Consistent with (Weak Form holds)'}")

print(f"\nBox-Ljung Test (H₀: No autocorrelation):")
print(f"  Lag | LB Statistic | p-value | Result")
print(f"  " + "-"*40)
for idx in lb_test.index[::5]:  # Every 5th lag
    print(f"  {idx:3d} | {lb_test.loc[idx, 'lb_stat']:12.4f} | {lb_test.loc[idx, 'lb_pvalue']:7.4f} | {'Reject' if lb_test.loc[idx, 'lb_pvalue'] < 0.05 else 'Fail to reject'}")

# Test 2: Momentum Effect
print("\n" + "="*70)
print("TEST 2: WEAK FORM - Momentum Effect (12-month prediction)")
print("="*70)

momentum_result = test_momentum_effect(prices, lookback=252, forward=20)
print(f"\nMomentum Regression Results:")
print(f"  Slope: {momentum_result['slope']:.6f}")
print(f"  R-squared: {momentum_result['r_squared']:.6f}")
print(f"  p-value: {momentum_result['p_value']:.6f}")
print(f"  Significant: {momentum_result['significant']}")
print(f"\nInterpretation: {'Weak form violated (momentum predicts)' if momentum_result['significant'] else 'Consistent with weak form'}")

# Test 3: Reversal Effect
print("\n" + "="*70)
print("TEST 3: WEAK FORM - Long-Term Reversal Effect")
print("="*70)

reversal_result = test_reversal_effect(prices, lookback=250, forward=20)
print(f"\nReversal Regression Results:")
print(f"  Slope: {reversal_result['slope']:.6f}")
print(f"  Correlation: {reversal_result['correlation']:.6f}")
print(f"  p-value: {reversal_result['p_value']:.6f}")
print(f"  Significant: {reversal_result['significant']}")
print(f"\nInterpretation: {'Mean reversion observed (weak form violation)' if reversal_result['significant'] else 'Random walk consistent'}")

# Test 4: Event Study (Semi-Strong)
print("\n" + "="*70)
print("TEST 4: SEMI-STRONG - Event Study")
print("="*70)

# Simulate 5 random events
event_dates = np.random.choice(range(200, n_days-100), size=5, replace=False)
events = event_study(prices, event_dates, pre_window=60, post_window=60)

print(f"\n{len(events)} Events Analyzed (Public Information):")
for i, event in enumerate(events):
    print(f"\nEvent {i+1}:")
    print(f"  CAR (Cumulative Abnormal Return): {event['car']*100:7.3f}%")
    
# Average CAR
if len(events) > 0:
    avg_car = np.mean([e['car'] for e in events])
    print(f"\nAverage CAR: {avg_car*100:7.3f}%")
    print(f"Interpretation: {'Semi-strong form holds (prices adjust quickly)' if abs(avg_car) < 0.01 else 'Semi-strong violated (prices drift post-announcement)'}")

# Test 5: Seasonality
print("\n" + "="*70)
print("TEST 5: SEMI-STRONG - Seasonality/Calendar Anomalies")
print("="*70)

monthly_rets, dow_rets = test_seasonality(prices, dates)

print("\nMonthly Returns (Potential January Effect):")
print(f"  Month | Avg Return | Std Dev | Observations")
print(f"  " + "-"*45)
for month in range(1, 13):
    if month in monthly_rets:
        m = monthly_rets[month]
        print(f"  {month:2d}    | {m['mean']*100:9.3f}% | {m['std']*100:7.3f}% | {m['count']:4d}")

print("\nDay-of-Week Returns (Potential Monday Effect):")
print(f"  Day       | Avg Return | Std Dev")
print(f"  " + "-"*40)
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    if day in dow_rets:
        d = dow_rets[day]
        print(f"  {day:10} | {d['mean']*100:9.3f}% | {d['std']*100:7.3f}%")

# Test 6: Insider Trading (Strong Form)
print("\n" + "="*70)
print("TEST 6: STRONG FORM - Insider Trading Abnormal Returns")
print("="*70)

# Simulate insider buys (assume they buy before +2% moves)
inside_buys = []
for i in range(100, n_days-100):
    future_return = np.log(prices[i+30]) - np.log(prices[i])
    if future_return > 0.02:  # +2% move ahead
        inside_buys.append(i)

insider_results, mean_abn, t_stat, p_val = test_insider_trading(prices, inside_buys[:10], forward=30)

print(f"\nInsider Trading Test (10 insider buys analyzed):")
print(f"  Mean Abnormal Return: {mean_abn*100:7.3f}%")
print(f"  t-statistic: {t_stat:7.3f}")
print(f"  p-value: {p_val:7.4f}")
print(f"  Significant: {p_val < 0.05}")
print(f"\nInterpretation: {'Strong form violated (insiders earn abnormal returns)' if p_val < 0.05 else 'Strong form cannot be rejected'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Autocorrelation function
ax1 = plt.subplot(2, 3, 1)
autocorr_series = [pd.Series(daily_returns).autocorr(lag=i) for i in range(1, 30)]
ax1.stem(range(1, 30), autocorr_series, basefmt=' ')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.axhline(y=1.96/np.sqrt(n_days), color='red', linestyle='--', linewidth=1, label='95% CI')
ax1.set_xlabel('Lag (days)')
ax1.set_ylabel('Autocorrelation')
ax1.set_title('Weak Form Test: Autocorrelation\n(Should be ~0 if random walk)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Momentum scatter
ax2 = plt.subplot(2, 3, 2)
past_mom = []
future_ret = []
for t in range(252, n_days-20):
    past_mom.append(np.log(prices[t]) - np.log(prices[t-252]))
    future_ret.append(np.log(prices[t+20]) - np.log(prices[t]))
ax2.scatter(past_mom, future_ret, alpha=0.5, s=20)
z = np.polyfit(past_mom, future_ret, 1)
p = np.poly1d(z)
ax2.plot(np.array(past_mom), p(np.array(past_mom)), "r--", linewidth=2, label='Fitted')
ax2.set_xlabel('12-Month Past Return')
ax2.set_ylabel('20-Day Forward Return')
ax2.set_title('Momentum Effect Test\n(Slope ≠ 0 = Weak Form Violation)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Price path with events
ax3 = plt.subplot(2, 3, 3)
ax3.plot(prices, linewidth=1, label='Price')
for event_date in event_dates[:5]:
    ax3.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
ax3.set_xlabel('Days')
ax3.set_ylabel('Price ($)')
ax3.set_title('Price Path with Event Dates')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Monthly returns heatmap
ax4 = plt.subplot(2, 3, 4)
monthly_data = []
for month in range(1, 13):
    if month in monthly_rets:
        monthly_data.append(monthly_rets[month]['mean'] * 100)
    else:
        monthly_data.append(0)
colors = ['red' if x < 0 else 'green' for x in monthly_data]
ax4.bar(range(1, 13), monthly_data, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Month')
ax4.set_ylabel('Average Return (%)')
ax4.set_title('January Effect Test\n(Semi-Strong: Monthly Seasonality)')
ax4.set_xticks(range(1, 13))
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Insider trading abnormal returns
ax5 = plt.subplot(2, 3, 5)
if insider_results:
    insider_abn = [x['abnormal_return']*100 for x in insider_results]
    ax5.bar(range(len(insider_abn)), insider_abn, 
            color=['green' if x > 0 else 'red' for x in insider_abn], alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax5.axhline(y=mean_abn*100, color='blue', linestyle='--', linewidth=2, label='Mean')
    ax5.set_xlabel('Event')
    ax5.set_ylabel('Abnormal Return (%)')
    ax5.set_title('Strong Form Test: Insider Returns\n(Positive = Private Info Profit)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Efficiency summary
ax6 = plt.subplot(2, 3, 6)
tests = ['Autocorr', 'Momentum', 'Reversal', 'Events', 'January', 'Insider']
results = [
    'Pass' if abs(autocorr) < 0.05 else 'Fail',
    'Fail' if momentum_result['significant'] else 'Pass',
    'Fail' if reversal_result['significant'] else 'Pass',
    'Pass' if abs(avg_car if len(events) > 0 else 0) < 0.01 else 'Fail',
    'Fail' if max([abs(monthly_rets[m]['mean']) for m in monthly_rets]) > 0.005 else 'Pass',
    'Fail' if p_val < 0.05 else 'Pass'
]
colors_result = ['green' if x == 'Pass' else 'red' for x in results]
ax6.barh(tests, [1]*len(tests), color=colors_result, alpha=0.7, edgecolor='black')
ax6.set_xlabel('Efficiency Test Result')
ax6.set_title('Market Efficiency Summary\n(Green=Efficient, Red=Violation)')
ax6.set_xlim([0, 1.2])
for i, (test, result) in enumerate(zip(tests, results)):
    ax6.text(0.5, i, result, ha='center', va='center', fontweight='bold', fontsize=10)
ax6.set_xticks([])

plt.tight_layout()
plt.savefig('market_efficiency_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: market_efficiency_analysis.png")

plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. EMPIRICAL FINDINGS:
   - Weak form: Generally holds for daily data (autocorr ≈ 0)
   - BUT: Momentum/reversal anomalies violate weak form
   - Semi-strong: Prices adjust within 5-30 seconds to news
   - Strong form: Clearly violated (insiders earn abnormal returns)

2. MARKET EFFICIENCY HIERARCHY:
   - Weak form broken ~20% of time (technical patterns exist)
   - Semi-strong broken ~10% of time (PEAD, calendar effects)
   - Strong form broken 100% of time (insiders have edge, but illegal)

3. EXPLOITATION CHALLENGES:
   - Even if markets inefficient, transaction costs often eliminate profits
   - Timing predictions uncertain (can be right direction, wrong timing)
   - Liquidity constraints limit position sizing
   - Regulation prevents insider trading profit capture

4. MARKET STRUCTURE IMPLICATIONS:
   - HFT improves weak-form efficiency (arbitrage disappears faster)
   - Fragmentation creates brief semi-strong violations (lead-lag)
   - Information asymmetry prevents strong-form efficiency permanently

5. PRACTICAL APPLICATIONS:
   - Active managers profit mainly from semi-strong inefficiency
   - Passive indexing works when weak-form efficiency holds
   - Anomaly exploitation requires sophisticated timing/execution
""")
```

---

## VI. Challenge Round

1. **Regime-Dependent Efficiency:** Markets are more efficient in calm periods, less efficient in crises. How would you design adaptive efficiency tests that change parameters by regime? What regimes matter?

2. **Partial Information Revelation:** Some news is partially understood initially, fully understood later. How does gradual interpretation affect efficient market hypothesis? Does PEAD (post-earnings drift) reflect inefficiency or rational uncertainty?

3. **Information Heterogeneity:** Sophisticated traders understand news faster than retail. Does this make markets "efficient" for sophisticated traders but inefficient for retail? What's the right efficiency benchmark?

4. **Feedback Loops in Testing:** If anomalies are discovered and exploited, they disappear. Does this mean markets ARE efficient (self-correcting) or tests are just catching transient patterns?

5. **Cross-Asset Efficiency:** Prices might be efficient within equity market but inefficient across equity-derivatives-FX. How would you test joint efficiency? Is global market efficiency testable?

---

## VII. Key References

1. **Fama, E. F.** (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383-417.
   - Foundational definition of market efficiency three forms
   - [JStor](https://www.jstor.org/stable/2325486)

2. **Jegadeesh, N., & Titman, S.** (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91.
   - Empirical momentum anomaly; weak form violation evidence
   - [JStor](https://www.jstor.org/stable/2353787)

3. **De Bondt, W. F., & Thaler, R. H.** (1985). "Does the Stock Market Overreact?" *Journal of Finance*, 40(3), 793-805.
   - Long-term reversal effect; behavioral evidence
   - [JStor](https://www.jstor.org/stable/2327804)

4. **Bernard, V. L., & Thomas, J. K.** (1989). "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?" *Journal of Accounting Research*, 1-36.
   - PEAD: Semi-strong efficiency violation evidence
   - [JStor](https://www.jstor.org/stable/2491062)

5. **Seyhun, H. N.** (1998). *Investment Intelligence from Insider Trading*. MIT Press.
   - Insider trading abnormal returns; strong form tests
   - [MIT Press](https://mitpress.mit.edu/9780262192545/investment-intelligence-from-insider-trading/)

---

**Last Updated:** January 31, 2026