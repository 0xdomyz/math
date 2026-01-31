# Bid-Ask Spread

## 1. Concept Skeleton
**Definition:** Difference between highest price buyer willing to pay (bid) and lowest price seller willing to accept (ask)  
**Purpose:** Cost of immediate liquidity, compensation to market makers for inventory risk and adverse selection  
**Prerequisites:** Order book structure, market makers, liquidity provision, transaction costs

## 2. Comparative Framing
| Spread Type | Quoted Spread | Effective Spread | Realized Spread |
|-------------|---------------|------------------|-----------------|
| **Definition** | Ask - Bid at best quotes | 2×|trade price - mid| | Captures price reversion |
| **Observability** | Directly visible | Calculated post-trade | Requires future price |
| **Information** | Cost of liquidity | Actual execution cost | Market maker profit |
| **Usage** | Market quality measure | TCA benchmark | Adverse selection test |

## 3. Examples + Counterexamples

**Simple Example:**  
Bid $99.98, Ask $100.02: Spread = $0.04 (4 cents or 4 basis points), buy costs 2 bps, sell loses 2 bps

**Failure Case:**  
Quoted spread $0.01, but market order walks book to $0.05 higher: Effective spread >> quoted spread in thin markets

**Edge Case:**  
Flash crash: Spreads widen from $0.01 to $5.00 in seconds, liquidity evaporates, returns to normal after circuit breaker

## 4. Layer Breakdown
```
Bid-Ask Spread Components:
├─ Economic Sources:
│   ├─ Order Processing Costs: Fixed overhead (technology, clearing, labor)
│   ├─ Inventory Holding Costs: Price risk from maintaining positions
│   ├─ Adverse Selection Costs: Trading with informed counterparties
│   └─ Monopoly Rents: Market power in illiquid securities
├─ Measurement Approaches:
│   ├─ Quoted Spread: Ask₀ - Bid₀ (time-series of BBO)
│   ├─ Relative Spread: (Ask - Bid) / Midpoint × 100%
│   ├─ Effective Spread: 2 × D × (P - M) where D ∈ {-1,1} direction
│   └─ Realized Spread: 2 × D × (P - M_t+τ) captures reversion
├─ Determinants:
│   ├─ Trading Volume: Higher volume → narrower spreads
│   ├─ Volatility: Higher volatility → wider spreads (risk)
│   ├─ Information Asymmetry: Informed trading → wider spreads
│   ├─ Competition: More market makers → narrower spreads
│   ├─ Tick Size: Minimum increment constrains spread
│   └─ Market Structure: Dealer vs order-driven affects spreads
├─ Decomposition (Stoll, Glosten-Harris):
│   ├─ Transitory Component: Inventory effects, mean-reverts
│   ├─ Permanent Component: Information revelation, persists
│   └─ Fixed Component: Order processing overhead
└─ Market Impact:
    ├─ Retail Traders: Pay spread as liquidity demanders
    ├─ Market Makers: Earn spread as liquidity suppliers
    ├─ Portfolio Turnover: Annual cost = Spread × Turnover
    └─ Market Quality: Tighter spreads = better liquidity
```

**Interaction:** Information → Adverse selection → Wider spreads → Higher trading costs → Reduced volume

## 5. Mini-Project
Analyze bid-ask spreads and decompose into components:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# Generate realistic intraday spread data
np.random.seed(42)

def simulate_intraday_spreads(n_minutes=390, base_spread=0.01):
    """
    Simulate U-shaped intraday spread pattern
    - Wider at open/close (uncertainty, inventory management)
    - Narrower during mid-day (high liquidity)
    """
    minutes = np.arange(n_minutes)
    
    # U-shaped pattern: higher at start/end of day
    u_pattern = 1.5 * (np.abs(minutes - n_minutes/2) / (n_minutes/2)) ** 0.5
    
    # Add volatility clustering
    garch_vol = np.zeros(n_minutes)
    garch_vol[0] = 1.0
    for i in range(1, n_minutes):
        garch_vol[i] = 0.95 * garch_vol[i-1] + 0.05 * np.abs(np.random.randn())
    
    # Combine patterns with noise
    spreads = base_spread * (1 + 0.3 * u_pattern + 0.2 * garch_vol + 0.1 * np.random.randn(n_minutes))
    spreads = np.maximum(spreads, 0.005)  # Minimum spread (half tick)
    
    # Add occasional spread spikes (news events)
    spike_times = np.random.choice(n_minutes, size=5, replace=False)
    for t in spike_times:
        spreads[t:t+3] *= np.random.uniform(2, 5)
    
    return spreads

def calculate_roll_estimator(returns):
    """
    Roll (1984) spread estimator from return serial covariance
    Spread = 2 × sqrt(-Cov(r_t, r_{t-1}))
    """
    cov = np.cov(returns[:-1], returns[1:])[0, 1]
    if cov < 0:
        return 2 * np.sqrt(-cov)
    else:
        return 0  # Model assumes negative serial correlation

def decompose_spread_stoll(trade_data):
    """
    Stoll (1989) spread decomposition into components
    Δp_t = c + ψ·Q_t + u_t
    where ψ = adverse selection + inventory costs
    """
    # Trade direction indicator
    Q = trade_data['direction'].values  # +1 buy, -1 sell
    delta_p = trade_data['price'].diff().values[1:]
    Q = Q[:-1]
    
    # OLS regression: price change on order flow
    from sklearn.linear_model import LinearRegression
    
    X = Q.reshape(-1, 1)
    y = delta_p
    
    reg = LinearRegression().fit(X, y)
    psi = reg.coef_[0]  # Impact coefficient
    
    # Spread = 2 × ψ (round-trip cost)
    spread_estimate = 2 * abs(psi)
    
    return {
        'spread_estimate': spread_estimate,
        'price_impact': psi,
        'r_squared': reg.score(X, y)
    }

# Simulation 1: Intraday spread pattern
print("="*60)
print("INTRADAY SPREAD ANALYSIS")
print("="*60)

spreads = simulate_intraday_spreads(n_minutes=390, base_spread=0.01)
time_index = pd.date_range('2026-01-31 09:30', periods=390, freq='1min')

df_spreads = pd.DataFrame({
    'time': time_index,
    'spread': spreads,
    'spread_bps': spreads * 100 / 100  # Assuming $100 stock
})

# Calculate statistics by period
morning = df_spreads.iloc[:60]  # First hour
midday = df_spreads.iloc[180:240]  # 12:00-1:00
afternoon = df_spreads.iloc[-60:]  # Last hour

print(f"\nSpread Statistics (basis points):")
print(f"  Morning (9:30-10:30):   {morning['spread_bps'].mean():.2f} ± {morning['spread_bps'].std():.2f}")
print(f"  Midday (12:00-1:00):    {midday['spread_bps'].mean():.2f} ± {midday['spread_bps'].std():.2f}")
print(f"  Afternoon (3:00-4:00):  {afternoon['spread_bps'].mean():.2f} ± {afternoon['spread_bps'].std():.2f}")

# Simulation 2: Trade-level data for spread decomposition
n_trades = 500
trade_prices = np.zeros(n_trades)
trade_directions = np.zeros(n_trades)

# True mid price follows random walk
true_mid = 100.0
spread = 0.02
informed_prob = 0.3

for i in range(n_trades):
    # Determine if trade is informed
    is_informed = np.random.random() < informed_prob
    
    if is_informed:
        # Informed trader knows direction of next move
        if np.random.random() < 0.5:
            # Positive information: buy
            trade_directions[i] = 1
            trade_prices[i] = true_mid + spread/2
            true_mid += 0.01  # Permanent price impact
        else:
            # Negative information: sell
            trade_directions[i] = -1
            trade_prices[i] = true_mid - spread/2
            true_mid -= 0.01
    else:
        # Uninformed (liquidity) trade
        trade_directions[i] = np.random.choice([-1, 1])
        if trade_directions[i] == 1:
            trade_prices[i] = true_mid + spread/2
        else:
            trade_prices[i] = true_mid - spread/2
    
    # Random walk component
    if np.random.random() < 0.1:
        true_mid += np.random.randn() * 0.02

trade_df = pd.DataFrame({
    'price': trade_prices,
    'direction': trade_directions,
    'returns': np.diff(np.concatenate([[100], trade_prices]))
})

# Apply Roll estimator
roll_spread = calculate_roll_estimator(trade_df['price'].values)
print(f"\n{'='*60}")
print(f"SPREAD ESTIMATION METHODS")
print(f"{'='*60}")
print(f"\nRoll (1984) Estimator:")
print(f"  Estimated spread: ${roll_spread:.4f} ({roll_spread*100:.2f} bps)")
print(f"  True spread: ${spread:.4f} ({spread*100:.2f} bps)")

# Stoll decomposition
stoll_results = decompose_spread_stoll(trade_df)
print(f"\nStoll (1989) Decomposition:")
print(f"  Spread estimate: ${stoll_results['spread_estimate']:.4f}")
print(f"  Price impact (ψ): {stoll_results['price_impact']:.4f}")
print(f"  R²: {stoll_results['r_squared']:.3f}")

# Simulation 3: Impact of market conditions on spreads
def spread_regression_analysis():
    """Analyze determinants of bid-ask spreads"""
    n_days = 100
    
    data = {
        'spread': [],
        'volume': [],
        'volatility': [],
        'price': [],
        'num_trades': []
    }
    
    for _ in range(n_days):
        # Simulate market conditions
        volume = np.random.lognormal(15, 1)  # Daily volume
        volatility = np.random.uniform(0.01, 0.05)  # Daily volatility
        price = np.random.uniform(20, 200)  # Stock price
        num_trades = np.random.poisson(500)
        
        # Spread model: inversely related to volume, positively to volatility
        base_spread = 0.01
        spread = base_spread * (1 / np.log(volume)) * (1 + 10*volatility) * (100/price)**0.3
        spread += np.random.normal(0, 0.001)
        
        data['spread'].append(spread)
        data['volume'].append(volume)
        data['volatility'].append(volatility)
        data['price'].append(price)
        data['num_trades'].append(num_trades)
    
    df = pd.DataFrame(data)
    
    # Multiple regression
    from sklearn.linear_model import LinearRegression
    
    X = df[['volume', 'volatility', 'price', 'num_trades']].values
    X = np.column_stack([
        np.log(df['volume']),
        df['volatility'],
        np.log(df['price']),
        np.log(df['num_trades'])
    ])
    y = np.log(df['spread'])
    
    reg = LinearRegression().fit(X, y)
    
    return df, reg

df_regression, reg_model = spread_regression_analysis()

print(f"\n{'='*60}")
print(f"SPREAD DETERMINANTS REGRESSION")
print(f"{'='*60}")
print(f"\nLog(Spread) = β₀ + β₁·Log(Volume) + β₂·Volatility + β₃·Log(Price) + β₄·Log(Trades)")
print(f"\nCoefficients:")
print(f"  Log(Volume):     {reg_model.coef_[0]:>7.3f}  (expect negative)")
print(f"  Volatility:      {reg_model.coef_[1]:>7.3f}  (expect positive)")
print(f"  Log(Price):      {reg_model.coef_[2]:>7.3f}  (expect negative)")
print(f"  Log(Trades):     {reg_model.coef_[3]:>7.3f}  (expect negative)")
print(f"  R²: {reg_model.score(np.column_stack([np.log(df_regression['volume']), df_regression['volatility'], np.log(df_regression['price']), np.log(df_regression['num_trades'])]), np.log(df_regression['spread'])):.3f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Intraday spread pattern
axes[0, 0].plot(df_spreads['time'], df_spreads['spread_bps'], linewidth=1, alpha=0.7)
axes[0, 0].set_title('Intraday Spread Pattern (U-Shape)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Spread (bps)')
axes[0, 0].axhline(df_spreads['spread_bps'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_spreads["spread_bps"].mean():.2f} bps')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Spread distribution
axes[0, 1].hist(df_spreads['spread_bps'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].axvline(df_spreads['spread_bps'].median(), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {df_spreads["spread_bps"].median():.2f} bps')
axes[0, 1].set_title('Spread Distribution')
axes[0, 1].set_xlabel('Spread (bps)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Plot 3: Trade prices showing bid-ask bounce
axes[0, 2].plot(trade_df['price'][:100], 'o-', markersize=3, linewidth=0.5, alpha=0.7)
axes[0, 2].set_title('Bid-Ask Bounce (First 100 Trades)')
axes[0, 2].set_xlabel('Trade Number')
axes[0, 2].set_ylabel('Price ($)')
axes[0, 2].grid(alpha=0.3)

# Plot 4: Return autocorrelation (Roll estimator diagnostic)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(trade_df['price'].values, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('Price Autocorrelation (Roll Estimator)')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# Plot 5: Spread vs Volume relationship
axes[1, 1].scatter(np.log(df_regression['volume']), df_regression['spread'], 
                   alpha=0.5, s=30)
axes[1, 1].set_title('Spread vs Trading Volume')
axes[1, 1].set_xlabel('Log(Volume)')
axes[1, 1].set_ylabel('Spread ($)')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Spread vs Volatility relationship
axes[1, 2].scatter(df_regression['volatility'], df_regression['spread'], 
                   alpha=0.5, s=30, color='coral')
axes[1, 2].set_title('Spread vs Volatility')
axes[1, 2].set_xlabel('Daily Volatility')
axes[1, 2].set_ylabel('Spread ($)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Portfolio impact analysis
print(f"\n{'='*60}")
print(f"PORTFOLIO TURNOVER COST ANALYSIS")
print(f"{'='*60}")

portfolio_value = 1_000_000
annual_turnover = 2.0  # 200% turnover
avg_spread_bps = df_spreads['spread_bps'].mean()

annual_cost = portfolio_value * annual_turnover * (avg_spread_bps / 10000)

print(f"\nPortfolio: ${portfolio_value:,}")
print(f"Annual turnover: {annual_turnover*100:.0f}%")
print(f"Average spread: {avg_spread_bps:.2f} bps")
print(f"Annual spread cost: ${annual_cost:,.2f}")
print(f"Performance drag: {(annual_cost/portfolio_value)*100:.2f}%")
```

## 6. Challenge Round
How do spreads vary across market conditions and structures?
- **Stock characteristics**: Large-cap (1-2 bps) vs small-cap (10-50 bps) vs penny stocks (100+ bps)
- **Market events**: Spreads widen 10-100x during flash crashes, earnings announcements, market stress
- **Time of day**: U-shaped intraday pattern - wide at open/close, narrow midday
- **Market structure**: Dealer markets (wider) vs order-driven (narrower) vs hybrid
- **Competition**: More market makers → tighter spreads, but predatory HFT can widen spreads
- **Regulation**: Tick size rules constrain minimum spread, sub-penny pricing debate

What are optimal spread policies for market makers?
- **Inventory management**: Widen spread when holding large position (risk)
- **Adverse selection protection**: Widen during news events, narrow during pure liquidity trading
- **Queue dynamics**: Penny-jump to gain time priority vs maintain wider spread
- **Fee optimization**: Maker-taker economics influence optimal quotes
- **Volatility adjustment**: Dynamic spreads proportional to recent realized volatility

## 7. Key References
- [Roll (1984): Simple Implicit Measure of Bid-Ask Spread](https://www.jstor.org/stable/2327617)
- [Stoll (1989): Inferring Components of Bid-Ask Spread](https://www.jstor.org/stable/2352946)
- [Glosten & Harris (1988): Estimating Components of Bid-Ask Spread](https://www.jstor.org/stable/2328704)

---
**Status:** Fundamental liquidity cost measure | **Complements:** Effective Spread, Market Impact, Transaction Costs
