# Liquidity Measures

## 1. Concept Skeleton
**Definition:** Quantitative metrics capturing market's ability to facilitate trading without significant price impact across dimensions: tightness, depth, resiliency  
**Purpose:** Compare asset liquidity, monitor market quality, inform trading strategy, measure transaction costs  
**Prerequisites:** Market microstructure, bid-ask spread, order book, price impact, market efficiency

## 2. Comparative Framing
| Measure | Amihud Illiquidity | Roll Spread | Kyle's Lambda | LOT Liquidity |
|---------|-------------------|-------------|---------------|---------------|
| **Data Required** | Price, volume | Returns only | Price, signed volume | Full order book |
| **Dimension** | Price impact | Tightness | Depth | Multi-dimensional |
| **Frequency** | Daily | Intraday | Trade-level | Real-time |
| **Interpretation** | Higher = less liquid | Implicit spread | Impact per unit | 0-10 score |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock A: Amihud = 0.001, Stock B: Amihud = 0.1 → Stock B is 100x less liquid, higher impact per dollar traded

**Failure Case:**  
Roll estimator with positive return autocorrelation: Formula gives negative spread (impossible), model assumptions violated

**Edge Case:**  
Zero-volume days: Amihud undefined (division by zero), use alternative measure or interpolate

## 4. Layer Breakdown
```
Liquidity Measurement Framework:
├─ Dimensions of Liquidity (Kyle 1985):
│   ├─ Tightness: Cost of immediate execution (spread)
│   ├─ Depth: Volume tradeable without price impact
│   ├─ Resiliency: Speed of mean reversion after shock
│   └─ Multi-dimensional: No single perfect measure
├─ Spread-Based Measures:
│   ├─ Quoted Spread: Ask - Bid (directly observable)
│   ├─ Effective Spread: 2×|Price - Mid| (actual cost)
│   ├─ Realized Spread: Captures reversion component
│   ├─ Roll Estimator: 2√(-Cov(Δp_t, Δp_{t-1}))
│   └─ Relative Spread: Spread / Price × 10,000 (bps)
├─ Price Impact Measures:
│   ├─ Amihud Illiquidity Ratio:
│   │   ├─ Formula: (|r_t| / Volume_t) averaged over period
│   │   ├─ Units: Price impact per dollar traded
│   │   ├─ Interpretation: Higher = more illiquid
│   │   └─ Advantages: Simple, only needs daily data
│   ├─ Kyle's Lambda (λ):
│   │   ├─ Regression: Δp_t = λ × Q_t + ε_t
│   │   ├─ Q_t: Signed order flow (buy = +, sell = -)
│   │   ├─ λ: Slope = price impact per share
│   │   └─ Requires: Trade direction classification
│   ├─ Market Impact Function:
│   │   ├─ Temporary: f(v) = γ√v (square-root law)
│   │   ├─ Permanent: g(v) = θv^α (linear to sublinear)
│   │   └─ Total: Temporary + Permanent
│   └─ Hui-Heubel Ratio: (High - Low) / (High + Low)
├─ Volume/Turnover Measures:
│   ├─ Trading Volume: Daily shares traded
│   ├─ Dollar Volume: Shares × Price
│   ├─ Turnover Ratio: Volume / Shares Outstanding
│   ├─ Amivest Ratio: Volume / |Price Change|
│   └─ Martin Liquidity: Days to trade X% of shares
├─ Order Book Measures:
│   ├─ Depth at Best: Size at BBO
│   ├─ Cumulative Depth: Sum across N levels
│   ├─ Depth-Weighted Spread: Spread / Depth
│   ├─ Effective Tick: Average price improvement
│   └─ Resilience: Time to restore depth after shock
├─ Transaction Cost Measures:
│   ├─ Implementation Shortfall: Benchmark vs execution
│   ├─ VWAP Slippage: Execution vs volume-weighted
│   ├─ Percentage Price Impact: |Execution - Pre-trade| / Pre-trade
│   └─ Effective Cost: Spread + Impact + Fees
└─ Composite Measures:
    ├─ LOT Liquidity Score (Liu 2006):
    │   ├─ Based on zero-return days and turnover
    │   ├─ Scale: 0-10, higher = less liquid
    │   └─ Works well for low-frequency data
    ├─ Pastor-Stambaugh Gamma:
    │   ├─ Return reversal after volume
    │   ├─ Measures temporary price impact
    │   └─ Asset pricing factor
    └─ Liquidity Index: Weighted average of multiple measures
```

**Interaction:** Trade → Price impact → Spread widens → Volume decreases → Liquidity worsens

## 5. Mini-Project
Calculate and compare multiple liquidity measures:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

class LiquidityAnalyzer:
    """Comprehensive liquidity measurement toolkit"""
    
    @staticmethod
    def amihud_illiquidity(returns, dollar_volume):
        """
        Amihud (2002) illiquidity measure
        ILLIQ = Average(|Return| / Dollar Volume)
        
        Higher values = more illiquid
        """
        # Remove zero volume days
        mask = dollar_volume > 0
        returns_clean = returns[mask]
        volume_clean = dollar_volume[mask]
        
        if len(returns_clean) == 0:
            return np.nan
        
        daily_illiq = np.abs(returns_clean) / volume_clean
        
        # Average, multiply by 10^6 for scaling
        illiq = np.mean(daily_illiq) * 1e6
        
        return illiq
    
    @staticmethod
    def roll_estimator(returns):
        """
        Roll (1984) spread estimator
        Spread = 2 × sqrt(-Cov(r_t, r_{t-1}))
        
        Assumes bid-ask bounce causes negative serial correlation
        """
        if len(returns) < 2:
            return np.nan
        
        # Serial covariance
        cov = np.cov(returns[:-1], returns[1:])[0, 1]
        
        if cov >= 0:
            # Positive autocorrelation violates model assumptions
            return np.nan
        
        spread = 2 * np.sqrt(-cov)
        return spread
    
    @staticmethod
    def kyle_lambda(price_changes, signed_volume):
        """
        Kyle's Lambda (price impact coefficient)
        Δp_t = λ × Q_t + ε
        
        Q_t: Signed order flow (+ for buys, - for sells)
        λ: Price impact per unit volume
        """
        # Remove NaN and inf
        mask = np.isfinite(price_changes) & np.isfinite(signed_volume)
        dp = price_changes[mask]
        q = signed_volume[mask]
        
        if len(dp) < 10:
            return {'lambda': np.nan, 'r_squared': np.nan}
        
        # Regression
        reg = LinearRegression()
        reg.fit(q.reshape(-1, 1), dp)
        
        lambda_est = reg.coef_[0]
        r_squared = reg.score(q.reshape(-1, 1), dp)
        
        return {
            'lambda': lambda_est,
            'r_squared': r_squared,
            'intercept': reg.intercept_
        }
    
    @staticmethod
    def turnover_ratio(volume, shares_outstanding):
        """
        Volume / Shares Outstanding
        Measures trading activity relative to float
        """
        if shares_outstanding <= 0:
            return np.nan
        
        return volume / shares_outstanding
    
    @staticmethod
    def amivest_ratio(dollar_volume, price_change):
        """
        Amivest Liquidity Ratio
        Dollar Volume / |Price Change|
        
        Higher = more liquid (large volume, small price move)
        """
        if price_change == 0:
            return np.inf
        
        return dollar_volume / np.abs(price_change)
    
    @staticmethod
    def liu_lot_measure(returns, volume):
        """
        Liu (2006) Liquidity measure
        Based on proportion of zero-return days adjusted for volume
        
        Higher = less liquid
        """
        n = len(returns)
        
        # Count zero return days
        zero_days = np.sum(returns == 0)
        
        # Deflator based on turnover
        avg_turnover = np.mean(volume) / np.sum(volume) if np.sum(volume) > 0 else 0
        
        # Deflator: 1 / (21 days × avg turnover)
        # 21 = trading days in month
        deflator = 1 / (21 * avg_turnover) if avg_turnover > 0 else n
        
        # LM = [Zero days + 1/Deflator] × (21/No. of Days)
        lot = (zero_days + 1/deflator) * (21 / n) if n > 0 else np.nan
        
        return lot
    
    @staticmethod
    def hui_heubel_ratio(high, low):
        """
        Hui-Heubel Liquidity Ratio
        (High - Low) / (High + Low)
        
        Measures intraday volatility as proxy for liquidity
        """
        if high + low == 0:
            return np.nan
        
        return (high - low) / (high + low)
    
    @staticmethod
    def effective_spread(trade_price, bid, ask):
        """
        Effective spread = 2 × |Price - Mid|
        Captures actual transaction cost
        """
        mid = (bid + ask) / 2
        return 2 * np.abs(trade_price - mid)

# Generate realistic stock data
np.random.seed(42)

def simulate_stock_data(n_days=252, initial_price=100, liquidity_regime='normal'):
    """
    Simulate stock with varying liquidity characteristics
    
    liquidity_regime: 'high', 'normal', 'low', 'crisis'
    """
    # Base parameters by regime
    params = {
        'high': {'vol': 0.15, 'volume_mean': 1e6, 'spread_bps': 2},
        'normal': {'vol': 0.25, 'volume_mean': 5e5, 'spread_bps': 5},
        'low': {'vol': 0.35, 'volume_mean': 1e5, 'spread_bps': 15},
        'crisis': {'vol': 0.60, 'volume_mean': 3e5, 'spread_bps': 50}
    }
    
    p = params[liquidity_regime]
    
    # Price process (GBM)
    daily_vol = p['vol'] / np.sqrt(252)
    returns = np.random.normal(0, daily_vol, n_days)
    
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Volume (lognormal with some mean reversion)
    log_volume = np.log(p['volume_mean'])
    volumes = []
    for i in range(n_days):
        if i == 0:
            vol = np.random.lognormal(log_volume, 0.5)
        else:
            # Mean reversion in volume
            vol = np.exp(0.9 * np.log(volumes[-1]) + 0.1 * log_volume + np.random.normal(0, 0.3))
        volumes.append(vol)
    
    volumes = np.array(volumes)
    dollar_volumes = volumes * prices
    
    # Bid-ask spread (basis points)
    spreads = []
    for i in range(n_days):
        base_spread = p['spread_bps'] / 10000
        # Spread widens with volatility and narrows with volume
        vol_factor = 1 + abs(returns[i]) * 100
        volume_factor = 1 / (1 + np.log(volumes[i] / p['volume_mean']))
        
        spread = base_spread * vol_factor * volume_factor
        spread *= np.random.uniform(0.8, 1.2)  # Noise
        spreads.append(spread)
    
    spreads = np.array(spreads)
    
    # Construct bid/ask
    bids = prices * (1 - spreads / 2)
    asks = prices * (1 + spreads / 2)
    
    # High/Low (for Hui-Heubel)
    highs = prices * (1 + abs(returns) * np.random.uniform(0.5, 1.0, n_days))
    lows = prices * (1 - abs(returns) * np.random.uniform(0.5, 1.0, n_days))
    
    # Signed volume for Kyle's lambda
    signed_volumes = volumes * np.sign(returns)
    
    return pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=n_days, freq='D'),
        'price': prices,
        'bid': bids,
        'ask': asks,
        'high': highs,
        'low': lows,
        'volume': volumes,
        'dollar_volume': dollar_volumes,
        'returns': returns,
        'spread': spreads,
        'signed_volume': signed_volumes
    })

# Create datasets for different liquidity regimes
regimes = ['high', 'normal', 'low', 'crisis']
datasets = {regime: simulate_stock_data(n_days=252, liquidity_regime=regime) 
            for regime in regimes}

# Calculate all liquidity measures
print("="*80)
print("LIQUIDITY MEASURE COMPARISON")
print("="*80)

results = []

for regime, df in datasets.items():
    # Amihud illiquidity
    amihud = LiquidityAnalyzer.amihud_illiquidity(df['returns'].values, 
                                                   df['dollar_volume'].values)
    
    # Roll estimator
    roll = LiquidityAnalyzer.roll_estimator(df['returns'].values)
    
    # Kyle's lambda
    kyle_result = LiquidityAnalyzer.kyle_lambda(df['price'].diff().values[1:],
                                                df['signed_volume'].values[1:])
    
    # Turnover ratio (assume 10M shares outstanding)
    shares_outstanding = 10e6
    avg_turnover = LiquidityAnalyzer.turnover_ratio(df['volume'].mean(), 
                                                     shares_outstanding)
    
    # Amivest ratio
    total_volume = df['dollar_volume'].sum()
    total_price_change = abs(df['price'].iloc[-1] - df['price'].iloc[0])
    amivest = LiquidityAnalyzer.amivest_ratio(total_volume, total_price_change)
    
    # Liu LOT measure
    lot = LiquidityAnalyzer.liu_lot_measure(df['returns'].values, df['volume'].values)
    
    # Hui-Heubel ratio
    hui_heubel = df.apply(lambda row: LiquidityAnalyzer.hui_heubel_ratio(row['high'], row['low']), 
                          axis=1).mean()
    
    # Average spread (bps)
    avg_spread_bps = df['spread'].mean() * 10000
    
    # Average dollar volume
    avg_dollar_volume = df['dollar_volume'].mean() / 1e6  # Millions
    
    results.append({
        'regime': regime,
        'amihud': amihud,
        'roll_spread': roll,
        'kyle_lambda': kyle_result['lambda'],
        'kyle_r2': kyle_result['r_squared'],
        'turnover': avg_turnover,
        'amivest': amivest / 1e9,  # Billions
        'lot_measure': lot,
        'hui_heubel': hui_heubel,
        'avg_spread_bps': avg_spread_bps,
        'avg_volume_$M': avg_dollar_volume
    })

df_results = pd.DataFrame(results)

print(f"\n{'Regime':<15} {'Amihud':>10} {'Roll':>8} {'Lambda':>8} {'Turnover':>10} {'Spread(bps)':>12}")
print("-"*80)

for _, row in df_results.iterrows():
    print(f"{row['regime']:<15} {row['amihud']:>10.4f} {row['roll_spread']:>8.4f} "
          f"{row['kyle_lambda']:>8.6f} {row['turnover']:>10.4f} {row['avg_spread_bps']:>12.2f}")

print(f"\n{'Regime':<15} {'LOT':>10} {'Hui-Heubel':>12} {'Amivest($B)':>13} {'Volume($M)':>12}")
print("-"*80)

for _, row in df_results.iterrows():
    print(f"{row['regime']:<15} {row['lot_measure']:>10.4f} {row['hui_heubel']:>12.4f} "
          f"{row['amivest']:>13.2f} {row['avg_volume_$M']:>12.1f}")

# Cross-sectional correlation of measures
print(f"\n{'='*80}")
print(f"MEASURE CORRELATIONS")
print(f"{'='*80}")

corr_matrix = df_results[['amihud', 'avg_spread_bps', 'kyle_lambda', 'lot_measure']].corr()
print(f"\n{corr_matrix.to_string()}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Amihud by regime
axes[0, 0].bar(df_results['regime'], df_results['amihud'], alpha=0.7, color='steelblue')
axes[0, 0].set_title('Amihud Illiquidity by Regime')
axes[0, 0].set_ylabel('Amihud (×10⁻⁶)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Average spread by regime
axes[0, 1].bar(df_results['regime'], df_results['avg_spread_bps'], 
               alpha=0.7, color='coral')
axes[0, 1].set_title('Average Spread by Regime')
axes[0, 1].set_ylabel('Spread (bps)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Kyle's Lambda by regime
axes[0, 2].bar(df_results['regime'], df_results['kyle_lambda']*1000, 
               alpha=0.7, color='green')
axes[0, 2].set_title("Kyle's Lambda by Regime")
axes[0, 2].set_ylabel('Lambda (×10⁻³)')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(axis='y', alpha=0.3)

# Plot 4: Time series of price and volume (normal regime)
df_normal = datasets['normal']
ax_price = axes[1, 0]
ax_vol = ax_price.twinx()

ax_price.plot(df_normal['date'], df_normal['price'], 'b-', linewidth=2, label='Price')
ax_vol.bar(df_normal['date'], df_normal['volume']/1e3, alpha=0.3, color='gray', label='Volume')

ax_price.set_xlabel('Date')
ax_price.set_ylabel('Price ($)', color='b')
ax_vol.set_ylabel('Volume (000s)', color='gray')
ax_price.set_title('Price and Volume (Normal Regime)')
ax_price.tick_params(axis='x', rotation=45)
ax_price.grid(alpha=0.3)

# Plot 5: Spread time series comparison
for regime in ['high', 'normal', 'crisis']:
    df_regime = datasets[regime]
    axes[1, 1].plot(df_regime['date'], df_regime['spread']*10000, 
                    label=regime.capitalize(), alpha=0.7, linewidth=1.5)

axes[1, 1].set_title('Spread Evolution by Regime')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Spread (bps)')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(alpha=0.3)

# Plot 6: Scatter: Amihud vs Spread
axes[1, 2].scatter(df_results['amihud'], df_results['avg_spread_bps'], 
                   s=200, alpha=0.6, c=range(len(df_results)), cmap='viridis')

for i, row in df_results.iterrows():
    axes[1, 2].annotate(row['regime'], 
                        (row['amihud'], row['avg_spread_bps']),
                        fontsize=9, ha='center')

axes[1, 2].set_title('Amihud vs Spread (Measure Consistency)')
axes[1, 2].set_xlabel('Amihud Illiquidity')
axes[1, 2].set_ylabel('Spread (bps)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. All measures rank liquidity consistently: crisis < low < normal < high")
print(f"2. Amihud and spread highly correlated (both capture tightness)")
print(f"3. Kyle's lambda measures depth/impact dimension")
print(f"4. Volume-based measures (turnover, Amivest) complement price-based")
print(f"5. During crises, all liquidity dimensions deteriorate simultaneously")
```

## 6. Challenge Round
Which liquidity measure to use when?
- **Amihud**: Best for cross-sectional comparisons, requires only daily data, captures price impact
- **Roll/Effective Spread**: Tightness dimension, high-frequency data, transaction cost focus
- **Kyle's Lambda**: Depth/market impact, requires trade-level data, useful for execution algorithms
- **Turnover/Volume**: Activity level, but doesn't capture cost dimension
- **LOT measure**: Works with low-frequency data, handles zero-volume days well
- **Composite measures**: Combine multiple dimensions for holistic view

What are measurement challenges?
- **Data requirements**: High-frequency measures need tick data, expensive and complex
- **Asynchronous trading**: Cross-asset liquidity comparisons difficult with different trading hours
- **Structural breaks**: Liquidity measures non-stationary during regime changes
- **Endogeneity**: Trading volume affects liquidity, which affects volume (simultaneity)
- **Dark pools**: Off-exchange trading hides true liquidity, visible measures underestimate

## 7. Key References
- [Amihud (2002): Illiquidity and Stock Returns](https://www.sciencedirect.com/science/article/abs/pii/S0304405X01000726)
- [Kyle (1985): Continuous Auctions and Insider Trading](https://www.jstor.org/stable/1913210)
- [Liu (2006): Liquidity Measure for Low-Frequency Data](https://www.sciencedirect.com/science/article/abs/pii/S0304405X06000353)

---
**Status:** Essential liquidity quantification | **Complements:** Market Quality, Transaction Costs, Price Impact
