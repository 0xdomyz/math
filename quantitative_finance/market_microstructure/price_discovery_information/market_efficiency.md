# Market Efficiency

## 1. Concept Skeleton
**Definition:** Extent to which asset prices fully reflect all available information at every point in time  
**Purpose:** Determines whether active management can generate alpha, guides regulatory policy, measures market quality  
**Prerequisites:** Information theory, price discovery, rational expectations, statistical testing methods

## 2. Comparative Framing
| Form | Information Set | Predictability | Implication | Testable |
|------|----------------|----------------|-------------|----------|
| **Weak** | Past prices | Technical analysis fails | Random walk | Yes (autocorr) |
| **Semi-Strong** | All public info | Fundamental analysis fails | Event study | Yes (CAR) |
| **Strong** | Public + private | Insider trading fails | No alpha possible | Hard (joint hypothesis) |
| **Adaptive** | Time-varying | Sometimes efficient | Depends on conditions | Emerging |

## 3. Examples + Counterexamples

**Efficient:**  
Apple earnings announced → stock jumps 5% in 30 seconds → all information incorporated → no predictable drift. Semi-strong efficiency holds

**Inefficient:**  
Post-earnings announcement drift: Small-cap earnings surprise → price continues drifting 8% over next 60 days → underreaction → exploitable

**False Inefficiency:**  
January effect: Small-cap outperformance → but driven by tax-loss selling (rational) + risk (higher beta) → not true anomaly after risk adjustment

## 4. Layer Breakdown
```
Market Efficiency Framework:
├─ Efficient Markets Hypothesis (Fama 1970):
│   ├─ Core Principle:
│   │   - Prices fully reflect all available information
│   │   - No free lunch: Can't consistently beat market
│   │   - Implies: P_t = E[V | Info_t] (rational pricing)
│   │   - Requires: Rational investors, costless info, no frictions
│   ├─ Three Forms:
│   │   - Weak: Prices reflect past prices/returns
│   │   - Semi-Strong: Prices reflect all public information
│   │   - Strong: Prices reflect all information (incl. private)
│   ├─ Implications:
│   │   - Weak: Technical analysis useless
│   │   - Semi-Strong: Fundamental analysis useless
│   │   - Strong: Even insider trading can't profit
│   └─ Assumptions:
│       - Rational agents maximize expected utility
│       - Information freely available
│       - No transaction costs
│       - Competitive markets
├─ Weak-Form Efficiency:
│   ├─ Tests:
│   │   - Autocorrelation: Corr(R_t, R_{t-k}) = 0?
│   │   - Runs test: Randomness of + / - returns
│   │   - Variance ratio: VR(k) = Var(R_t→t+k) / (k × Var(R_t)) = 1?
│   │   - Filter rules: Buy if up X%, sell if down X%
│   ├─ Evidence:
│   │   - Short-term (daily): Near zero autocorrelation (≈efficient)
│   │   - Momentum (6-12 months): Positive autocorr (inefficiency)
│   │   - Mean reversion (3-5 years): Negative autocorr (inefficiency)
│   │   - Bid-ask bounce: Artificial negative autocorr (noise)
│   ├─ Random Walk Hypothesis:
│   │   - R_t = μ + ε_t, ε_t ~ i.i.d.
│   │   - Best predictor of P_{t+1} is P_t (martingale)
│   │   - Implies unpredictability
│   └─ Violations:
│       - Short-term reversal (1 week): Microstructure noise
│       - Medium-term momentum (6-12 mo): Underreaction
│       - Long-term reversal (3-5 yr): Overreaction
├─ Semi-Strong Form Efficiency:
│   ├─ Event Studies:
│   │   - Measure abnormal returns: AR_t = R_t - E[R_t]
│   │   - Cumulative: CAR = Σ AR_t
│   │   - Test: CAR = 0 after adjustment period?
│   │   - Adjustment speed: Minutes, hours, days?
│   ├─ Corporate Events:
│   │   - Earnings announcements: 90% incorporated in 5 min
│   │   - M&A announcements: Immediate jump, then drift
│   │   - Stock splits: No lasting impact (cosmetic)
│   │   - Dividend initiations: Positive AR (~3%)
│   ├─ Post-Earnings Announcement Drift (PEAD):
│   │   - Surprise → price continues moving 60 days
│   │   - SUE (standardized unexpected earnings) strategy
│   │   - Underreaction to earnings info
│   │   - Risk premium or mispricing? (Debate)
│   ├─ Anomalies:
│   │   - Size effect: Small-caps outperform (Jan)
│   │   - Value effect: High B/M outperforms
│   │   - Momentum: Past winners continue (6-12 mo)
│   │   - Low volatility: Low-vol stocks outperform
│   └─ Risk Adjustment:
│       - CAPM: α = R_p - [R_f + β(R_m - R_f)]
│       - Fama-French 3-factor: + SMB + HML
│       - Carhart 4-factor: + MOM
│       - Fama-French 5-factor: + RMW + CMA
├─ Strong-Form Efficiency:
│   ├─ Definition:
│   │   - Even insiders can't consistently profit
│   │   - All private information already in prices
│   │   - Impossible in practice (insiders do profit)
│   ├─ Tests:
│   │   - Insider trading returns: Do insiders beat market?
│   │   - Mutual fund performance: Do managers add value?
│   │   - Analyst recommendations: Profitable to follow?
│   ├─ Evidence:
│   │   - Corporate insiders: Earn abnormal returns (5-10% annual)
│   │   - Mutual funds: 95% underperform after fees
│   │   - Hedge funds: Mixed (survivorship bias)
│   │   - Analysts: Short-term impact, then reversion
│   └─ Conclusion:
│       - Strong-form violated (private info has value)
│       - But hard to exploit (legal, access barriers)
│       - Semi-strong approximately holds for public info
├─ Joint Hypothesis Problem (Fama 1970):
│   ├─ Issue:
│   │   - Testing efficiency requires model of expected returns
│   │   - Rejection: Is market inefficient OR model wrong?
│   │   - Can't separate "true" efficiency from model error
│   ├─ Example:
│   │   - CAPM α ≠ 0: Inefficiency or bad risk model?
│   │   - Momentum profits: Mispricing or missing risk factor?
│   │   - Value premium: Behavioral or distress risk?
│   ├─ Resolution Attempts:
│   │   - Better risk models (Fama-French, etc.)
│   │   - Behavioral explanations (limits to arbitrage)
│   │   - Microstructure frictions (transaction costs)
│   └─ Philosophical:
│       - Can never prove efficiency, only fail to reject
│       - "All models are wrong, some are useful"
├─ Speed of Adjustment:
│   ├─ Information Dissemination:
│   │   - Pre-1990s: Minutes to hours (phone, newswires)
│   │   - 2000s: Seconds (Internet, Bloomberg terminals)
│   │   - 2010s: Microseconds (HFT, co-location)
│   ├─ Empirical:
│   │   - Earnings: 90% incorporated in 5 minutes (large-caps)
│   │   - Macro news: 50% in 1 second, 90% in 5 seconds
│   │   - Small-caps: Slower (hours to days)
│   │   - Complex info: Slower (10-Ks take days/weeks)
│   ├─ Factors:
│   │   - Liquidity: Deep markets adjust faster
│   │   - Analyst coverage: More coverage → faster
│   │   - Information complexity: Simple → fast, complex → slow
│   │   - Trading costs: High costs → slow adjustment
│   └─ HFT Impact:
│       - Controversial: Speed race socially wasteful?
│       - Benefits: Faster price discovery, narrower spreads
│       - Costs: Arms race, flash crashes, fairness
├─ Limits to Arbitrage (Shleifer-Vishny 1997):
│   ├─ Why Inefficiencies Persist:
│   │   - Transaction costs: Spreads, fees, market impact
│   │   - Short-sale constraints: Hard to exploit overpricing
│   │   - Risk: Fundamental risk, noise trader risk
│   │   - Capital constraints: Margin requirements, redemptions
│   ├─ Examples:
│   │   - Twin shares (Royal Dutch/Shell): 15% mispricing persists
│   │   - Closed-end fund discounts: -10% to -20% stable
│   │   - Index inclusion effects: Permanent price increase
│   ├─ Behavioral:
│   │   - Overconfidence: Excess trading
│   │   - Herding: Momentum
│   │   - Anchoring: Slow adjustment
│   │   - Loss aversion: Disposition effect
│   └─ Implications:
│       - Inefficiencies can persist if arbitrage costly/risky
│       - "Market can stay irrational longer than you can stay solvent"
│       - Room for active management (skilled)
├─ Adaptive Markets Hypothesis (Lo 2004):
│   ├─ Core Idea:
│   │   - Efficiency varies over time
│   │   - Depends on: Competition, information environment, regulation
│   │   - Evolutionary perspective: Strategies compete
│   ├─ Predictions:
│   │   - Efficiency higher in liquid, transparent markets
│   │   - Anomalies discovered → exploited → disappear (learning)
│   │   - Crises → temporary inefficiency (adaptation lags)
│   ├─ Evidence:
│   │   - Momentum profits declined post-1990s (crowding)
│   │   - Market efficiency improved with electronic trading
│   │   - COVID crash: Temporary inefficiency, then recovered
│   └─ Implications:
│       - Not always efficient or inefficient
│       - Active management can add value (situationally)
│       - Regulatory interventions matter
└─ Microstructure and Efficiency:
    ├─ Frictions:
    │   - Bid-ask spread: Immediate cost (0.01-0.5%)
    │   - Market impact: Large orders (0.1-3% for blocks)
    │   - Short-sale costs: Borrow fees (0.1-10%+)
    │   - Taxes: Capital gains (0-37%), transaction taxes
    ├─ Effect on Efficiency:
    │   - High frictions → slow adjustment
    │   - Informational inefficiencies persist
    │   - Limits to arbitrage bind
    ├─ HFT and Efficiency:
    │   - Pro: Faster incorporation, tighter spreads
    │   - Con: Predatory (latency arbitrage), fragility (flash crashes)
    │   - Debate: Net positive or negative?
    └─ Regulation:
        - Disclosure (Reg FD): Levels playing field
        - Trading halts: Prevent panic, allow digestion
        - Short-sale rules: Debate (restrict overpricing correction)
        - Transaction taxes: Reduce noise trading or harm liquidity?
```

**Interaction:** Information release → trading → price adjustment → efficiency measured by speed and completeness

## 5. Mini-Project
Test weak-form efficiency and momentum:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Market Efficiency Testing
def generate_returns(n_periods=1000, model='random_walk', momentum_strength=0):
    """
    Generate return series under different efficiency assumptions
    
    model: 'random_walk', 'momentum', 'mean_reversion'
    """
    returns = np.zeros(n_periods)
    
    if model == 'random_walk':
        # Efficient market: i.i.d. returns
        returns = np.random.normal(0.0005, 0.02, n_periods)
    
    elif model == 'momentum':
        # Momentum: Positive autocorrelation
        returns[0] = np.random.normal(0.0005, 0.02)
        for t in range(1, n_periods):
            returns[t] = momentum_strength * returns[t-1] + \
                        np.random.normal(0.0005, 0.02)
    
    elif model == 'mean_reversion':
        # Mean reversion: Negative autocorrelation
        returns[0] = np.random.normal(0.0005, 0.02)
        for t in range(1, n_periods):
            returns[t] = -momentum_strength * returns[t-1] + \
                        np.random.normal(0.0005, 0.02)
    
    return returns

def variance_ratio_test(returns, k):
    """
    Variance ratio test: VR(k) = Var(k-period return) / (k * Var(1-period))
    Efficient market: VR = 1
    """
    n = len(returns)
    
    # k-period returns (non-overlapping)
    k_returns = []
    for i in range(0, n - k + 1, k):
        k_ret = np.sum(returns[i:i+k])
        k_returns.append(k_ret)
    
    if len(k_returns) < 2:
        return np.nan
    
    var_k = np.var(k_returns, ddof=1)
    var_1 = np.var(returns, ddof=1)
    
    if var_1 > 0:
        vr = var_k / (k * var_1)
    else:
        vr = np.nan
    
    return vr

def autocorrelation_test(returns, max_lag=20):
    """Calculate autocorrelations"""
    n = len(returns)
    mean_r = np.mean(returns)
    var_r = np.var(returns)
    
    autocorrs = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        
        cov = np.mean((returns[lag:] - mean_r) * (returns[:-lag] - mean_r))
        
        if var_r > 0:
            autocorrs.append(cov / var_r)
        else:
            autocorrs.append(0)
    
    return autocorrs

# Simulate different market types
print("Market Efficiency Tests")
print("=" * 70)

n_periods = 1000

# Three scenarios
scenarios = {
    'Efficient (Random Walk)': generate_returns(n_periods, 'random_walk', 0),
    'Momentum (Inefficient)': generate_returns(n_periods, 'momentum', 0.15),
    'Mean Reversion': generate_returns(n_periods, 'mean_reversion', 0.15)
}

# Test each scenario
results = {}

for name, returns in scenarios.items():
    print(f"\n{name}:")
    print("-" * 70)
    
    # Summary statistics
    print(f"Mean Return: {returns.mean()*252:.2f}% annual")
    print(f"Volatility: {returns.std()*np.sqrt(252):.2f}% annual")
    print(f"Sharpe Ratio: {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")
    
    # Autocorrelation
    autocorrs = autocorrelation_test(returns, max_lag=20)
    print(f"\nAutocorrelation lag-1: {autocorrs[0]:.4f}")
    
    # Test significance
    if abs(autocorrs[0]) > 1.96 / np.sqrt(n_periods):
        print(f"  → Significantly different from 0 (inefficient)")
    else:
        print(f"  → Not significant (consistent with efficiency)")
    
    # Variance ratio tests
    print(f"\nVariance Ratio Tests:")
    for k in [2, 5, 10]:
        vr = variance_ratio_test(returns, k)
        print(f"  VR({k}) = {vr:.3f}", end="")
        
        if abs(vr - 1.0) < 0.1:
            print(" [Random Walk]")
        elif vr > 1.0:
            print(" [Momentum/Slow Adjustment]")
        else:
            print(" [Mean Reversion]")
    
    results[name] = {
        'returns': returns,
        'autocorrs': autocorrs
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Cumulative returns
for name, data in results.items():
    returns = data['returns']
    cum_returns = np.cumprod(1 + returns) - 1
    axes[0, 0].plot(cum_returns * 100, label=name, linewidth=2, alpha=0.7)

axes[0, 0].set_xlabel('Trading Days')
axes[0, 0].set_ylabel('Cumulative Return (%)')
axes[0, 0].set_title('Price Paths Under Different Efficiency Assumptions')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)

# Plot 2: Autocorrelation functions
lags = range(1, 21)
for name, data in results.items():
    autocorrs = data['autocorrs']
    axes[0, 1].plot(lags, autocorrs, marker='o', label=name, linewidth=2, alpha=0.7)

# 95% confidence bands
axes[0, 1].axhline(1.96 / np.sqrt(n_periods), color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label='95% CI')
axes[0, 1].axhline(-1.96 / np.sqrt(n_periods), color='red', linestyle='--', 
                  linewidth=1, alpha=0.5)
axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_xlabel('Lag (days)')
axes[0, 1].set_ylabel('Autocorrelation')
axes[0, 1].set_title('Autocorrelation Function (ACF)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Return distributions
for name, data in results.items():
    returns = data['returns']
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.5, label=name, edgecolor='black')

axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Return Distributions')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Variance ratios
vr_periods = [2, 3, 5, 10, 20]
for name, data in results.items():
    returns = data['returns']
    vrs = [variance_ratio_test(returns, k) for k in vr_periods]
    axes[1, 1].plot(vr_periods, vrs, marker='o', label=name, linewidth=2, alpha=0.7)

axes[1, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, 
                  label='Random Walk (VR=1)')
axes[1, 1].set_xlabel('Period Length (k)')
axes[1, 1].set_ylabel('Variance Ratio VR(k)')
axes[1, 1].set_title('Variance Ratio Test')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Momentum trading strategy test
print(f"\n\nMomentum Trading Strategy (6-month)")
print("=" * 70)

# Test on momentum scenario
returns_mom = scenarios['Momentum (Inefficient)']
lookback = 120  # 6 months

strategy_returns = []
positions = []

for t in range(lookback, len(returns_mom)):
    # Calculate past 6-month return
    past_return = np.sum(returns_mom[t-lookback:t])
    
    # Signal: 1 if past positive, -1 if negative
    if past_return > 0:
        position = 1  # Long
    else:
        position = -1  # Short
    
    # Strategy return
    strategy_return = position * returns_mom[t]
    
    strategy_returns.append(strategy_return)
    positions.append(position)

strategy_returns = np.array(strategy_returns)
market_returns = returns_mom[lookback:]

print(f"Market Buy-and-Hold:")
print(f"  Mean Return: {market_returns.mean()*252:.2f}% annual")
print(f"  Volatility: {market_returns.std()*np.sqrt(252):.2f}%")
print(f"  Sharpe: {(market_returns.mean()/market_returns.std())*np.sqrt(252):.2f}")

print(f"\nMomentum Strategy:")
print(f"  Mean Return: {strategy_returns.mean()*252:.2f}% annual")
print(f"  Volatility: {strategy_returns.std()*np.sqrt(252):.2f}%")
print(f"  Sharpe: {(strategy_returns.mean()/strategy_returns.std())*np.sqrt(252):.2f}")

# Statistical test
excess_returns = strategy_returns - market_returns
t_stat, p_value = stats.ttest_1samp(excess_returns, 0)

print(f"\nExcess Return Test:")
print(f"  Mean Excess: {excess_returns.mean()*252:.2f}% annual")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  → Strategy significantly outperforms (market inefficient)")
else:
    print("  → No significant outperformance (consistent with efficiency)")

# Cumulative performance
cum_market = np.cumprod(1 + market_returns) - 1
cum_strategy = np.cumprod(1 + strategy_returns) - 1

print(f"\nCumulative Performance:")
print(f"  Market: {cum_market[-1]*100:.1f}%")
print(f"  Strategy: {cum_strategy[-1]*100:.1f}%")
print(f"  Outperformance: {(cum_strategy[-1] - cum_market[-1])*100:.1f}%")
```

## 6. Challenge Round
Why does post-earnings announcement drift persist despite being widely known?
- **Limits to arbitrage**: Short-sale constraints (30%+ borrow cost for small-caps), holding costs accumulate over 60-day drift period → arbitrage unprofitable after costs
- **Institutional constraints**: Mutual funds can't short, risk limits prevent concentration → can't fully exploit even when identified
- **Complexity**: Requires earnings surprise calculation (analyst forecasts), portfolio rebalancing costs, execution skill → barriers to retail/passive investors
- **Behavioral**: Anchoring (investors stick to pre-earnings price), underreaction (gradual information diffusion), limited attention (small-caps ignored)
- **Risk**: Fama-French argue it's rational risk premium for small/value stocks, not true inefficiency (joint hypothesis problem)

## 7. Key References
- [Fama (1970) - Efficient Capital Markets: A Review of Theory and Empirical Work](https://www.jstor.org/stable/2325486)
- [Fama (1991) - Efficient Capital Markets: II](https://www.jstor.org/stable/2328565)
- [Shleifer & Vishny (1997) - The Limits of Arbitrage](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03807.x)
- [Lo (2004) - The Adaptive Markets Hypothesis](https://www.jstor.org/stable/4126697)

---
**Status:** EMH tests and implications | **Complements:** Price Discovery, Information Asymmetry, Anomalies
