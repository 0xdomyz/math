# Adverse Selection Costs

## 1. Concept Skeleton
**Definition:** Cost incurred by liquidity providers from trading with informed counterparties who possess private information  
**Purpose:** Compensate market makers for risk of information asymmetry, component of bid-ask spread, inform quoting strategy  
**Prerequisites:** Information asymmetry, market microstructure, bid-ask spread decomposition, informed vs uninformed trading

## 2. Comparative Framing
| Cost Type | Adverse Selection | Inventory Cost | Order Processing |
|-----------|-------------------|----------------|------------------|
| **Nature** | Informational risk | Price risk | Fixed overhead |
| **Persistence** | Permanent impact | Temporary | Constant |
| **Mitigation** | Wider spreads, fast updates | Mean reversion hedging | Economies of scale |
| **Measurement** | Realized spread, PIN | Volatility exposure | Direct observation |

## 3. Examples + Counterexamples

**Simple Example:**  
Informed trader buys at ask $100.02 knowing positive news imminent; price jumps to $100.10, market maker loses $0.08

**Failure Case:**  
Market maker widens spread to avoid adverse selection, but loses market share to aggressive competitors, inventory builds

**Edge Case:**  
Coordinated informed trading: Multiple HFTs detect same signal, all buy simultaneously, market maker hit with large adverse flow

## 4. Layer Breakdown
```
Adverse Selection Framework:
├─ Information Types:
│   ├─ Private Information: Non-public, material information (illegal insider)
│   ├─ Superior Analysis: Legal edge from research, models, speed
│   ├─ Order Flow Information: See large order coming, front-run
│   └─ Latency Advantage: Faster market data, stale quote arbitrage
├─ Trader Classification:
│   ├─ Informed Traders:
│   │   ├─ Characteristics: Trade on information, directional, aggressive
│   │   ├─ Behavior: Market orders, large size, clustered timing
│   │   └─ Impact: Permanent price change, dealer losses
│   ├─ Uninformed (Liquidity) Traders:
│   │   ├─ Characteristics: Portfolio rebalancing, no edge
│   │   ├─ Behavior: Random direction, predictable patterns
│   │   └─ Impact: Temporary price pressure, mean reversion
│   └─ Noise Traders:
│       ├─ Characteristics: Irrational, sentiment-driven
│       ├─ Behavior: Contrarian indicators
│       └─ Impact: Liquidity provision opportunity
├─ Detection Methods:
│   ├─ PIN (Probability of Informed Trading):
│   │   ├─ EKOP Model: Easley, Kiefer, O'Hara, Paperman
│   │   ├─ Estimate: PIN = α·μ / (α·μ + 2ε)
│   │   ├─ Inputs: Buy/sell arrival rates, information event probability
│   │   └─ Output: Fraction of trades that are informed
│   ├─ VPIN (Volume-Synchronized PIN):
│   │   ├─ High-frequency version, no MLE needed
│   │   ├─ Order flow imbalance in volume buckets
│   │   └─ Real-time toxicity measure
│   ├─ Kyle's Lambda (λ):
│   │   ├─ Price impact per unit order flow
│   │   ├─ Regression: ΔP = λ·Q + ε
│   │   └─ Higher λ = more adverse selection
│   └─ Realized Spread Decomposition:
│       ├─ Permanent component = adverse selection
│       ├─ Temporary component = inventory + processing
│       └─ Compare trade price to price 5-30 min later
├─ Market Maker Response:
│   ├─ Spread Widening: Increase buffer against informed trades
│   ├─ Fast Quote Updates: Reduce stale quote exposure
│   ├─ Size Limits: Cap exposure per trade (icebergs)
│   ├─ Selective Quoting: Withdraw during high-risk periods (news)
│   └─ Adverse Selection Premium: Charge higher spread in toxic flow
└─ Economic Consequences:
    ├─ Welfare Loss: Informed traders extract rents from liquidity providers
    ├─ Spread Component: ~50% of spread in many markets
    ├─ Price Discovery: Informed trades move prices toward true value
    └─ Liquidity Impact: High adverse selection → wider spreads, less liquidity
```

**Interaction:** Information event → Informed trading → Price impact → Market maker loss → Wider spreads

## 5. Mini-Project
Estimate probability of informed trading and simulate adverse selection:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson

class AdverseSelectionModel:
    """Implement PIN and adverse selection analysis"""
    
    @staticmethod
    def simulate_informed_trading(n_days=100, alpha=0.3, mu=200, epsilon_b=150, epsilon_s=150):
        """
        Simulate EKOP model for PIN estimation
        
        Parameters:
        - alpha: Probability of information event
        - mu: Arrival rate of informed trades
        - epsilon_b: Arrival rate of uninformed buys
        - epsilon_s: Arrival rate of uninformed sells
        """
        data = []
        
        for day in range(n_days):
            # Information event
            has_info_event = np.random.random() < alpha
            
            if has_info_event:
                # Good or bad news (equal probability)
                is_good_news = np.random.random() < 0.5
                
                if is_good_news:
                    # Informed buying pressure
                    buys = poisson.rvs(epsilon_b + mu)
                    sells = poisson.rvs(epsilon_s)
                else:
                    # Informed selling pressure
                    buys = poisson.rvs(epsilon_b)
                    sells = poisson.rvs(epsilon_s + mu)
            else:
                # No information event, only uninformed trades
                buys = poisson.rvs(epsilon_b)
                sells = poisson.rvs(epsilon_s)
            
            data.append({
                'day': day,
                'buys': buys,
                'sells': sells,
                'has_info_event': has_info_event,
                'total_trades': buys + sells
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def estimate_pin(df, method='simple'):
        """
        Estimate PIN from buy/sell counts
        
        Simple method: Use sample moments
        MLE method: Maximum likelihood estimation (EKOP)
        """
        if method == 'simple':
            # Simplified PIN using order imbalance
            df['imbalance'] = abs(df['buys'] - df['sells']) / df['total_trades']
            pin_estimate = df['imbalance'].mean()
            
            return pin_estimate
        
        elif method == 'mle':
            # Maximum Likelihood Estimation (simplified version)
            buys = df['buys'].values
            sells = df['sells'].values
            
            def neg_log_likelihood(params):
                """Negative log-likelihood for EKOP model"""
                alpha, mu, eps_b, eps_s = params
                
                # Constrain parameters
                if alpha < 0 or alpha > 1 or mu < 0 or eps_b < 0 or eps_s < 0:
                    return 1e10
                
                ll = 0
                for b, s in zip(buys, sells):
                    # Probability of observing (b, s)
                    # P(no event) * P(b|no event) * P(s|no event)
                    p_no_event = (1 - alpha) * poisson.pmf(b, eps_b) * poisson.pmf(s, eps_s)
                    
                    # P(good news) * P(b|good) * P(s|good)
                    p_good = (alpha / 2) * poisson.pmf(b, eps_b + mu) * poisson.pmf(s, eps_s)
                    
                    # P(bad news) * P(b|bad) * P(s|bad)
                    p_bad = (alpha / 2) * poisson.pmf(b, eps_b) * poisson.pmf(s, eps_s + mu)
                    
                    p_total = p_no_event + p_good + p_bad
                    
                    if p_total > 0:
                        ll += np.log(p_total)
                    else:
                        ll += -1e10
                
                return -ll
            
            # Initial guess
            x0 = [0.3, 100, df['buys'].mean(), df['sells'].mean()]
            
            # Optimize
            result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                            options={'maxiter': 5000})
            
            if result.success:
                alpha_est, mu_est, eps_b_est, eps_s_est = result.x
                pin_est = (alpha_est * mu_est) / (alpha_est * mu_est + eps_b_est + eps_s_est)
                
                return {
                    'pin': pin_est,
                    'alpha': alpha_est,
                    'mu': mu_est,
                    'epsilon_b': eps_b_est,
                    'epsilon_s': eps_s_est
                }
            else:
                return None

def simulate_market_maker_losses(n_trades=1000, pin=0.3, spread=0.02):
    """
    Simulate market maker P&L under adverse selection
    
    Parameters:
    - n_trades: Number of trades
    - pin: Probability of informed trading
    - spread: Bid-ask spread (half-spread earned per trade)
    """
    trades = []
    cumulative_pnl = [0]
    true_value = 100.0
    
    for i in range(n_trades):
        # Determine if informed
        is_informed = np.random.random() < pin
        
        # Market maker earns half-spread
        spread_revenue = spread / 2
        
        if is_informed:
            # Informed trade: price moves against market maker
            # Good news → informed buys → market maker sells → price rises
            # Bad news → informed sells → market maker buys → price falls
            
            if np.random.random() < 0.5:
                # Good news: informed buy
                direction = 1
                trade_price = true_value + spread / 2
                
                # Price moves up (permanent impact)
                true_value += np.random.uniform(0.05, 0.15)
                
                # Market maker sold, now holds short position at loss
                adverse_loss = true_value - trade_price
                
            else:
                # Bad news: informed sell
                direction = -1
                trade_price = true_value - spread / 2
                
                # Price moves down
                true_value -= np.random.uniform(0.05, 0.15)
                
                # Market maker bought, now holds long position at loss
                adverse_loss = trade_price - true_value
        else:
            # Uninformed trade: no systematic price movement
            direction = np.random.choice([-1, 1])
            adverse_loss = 0
            
            # Small random walk
            true_value += np.random.randn() * 0.01
        
        # Net P&L: earn spread, lose to adverse selection
        trade_pnl = spread_revenue - adverse_loss
        
        cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl)
        
        trades.append({
            'trade_id': i,
            'is_informed': is_informed,
            'direction': direction,
            'spread_revenue': spread_revenue,
            'adverse_loss': adverse_loss,
            'trade_pnl': trade_pnl,
            'true_value': true_value
        })
    
    return pd.DataFrame(trades), cumulative_pnl

# Simulation 1: Estimate PIN from order flow
print("="*70)
print("PROBABILITY OF INFORMED TRADING (PIN) ESTIMATION")
print("="*70)

# True parameters
true_alpha = 0.3
true_mu = 200
true_epsilon_b = 150
true_epsilon_s = 150

true_pin = (true_alpha * true_mu) / (true_alpha * true_mu + true_epsilon_b + true_epsilon_s)

print(f"\nTrue Parameters:")
print(f"  α (info event probability): {true_alpha:.2f}")
print(f"  μ (informed arrival rate): {true_mu:.0f}")
print(f"  ε_b (uninformed buy rate): {true_epsilon_b:.0f}")
print(f"  ε_s (uninformed sell rate): {true_epsilon_s:.0f}")
print(f"  True PIN: {true_pin:.3f}")

# Generate data
df_trades = AdverseSelectionModel.simulate_informed_trading(
    n_days=100, alpha=true_alpha, mu=true_mu, 
    epsilon_b=true_epsilon_b, epsilon_s=true_epsilon_s)

# Estimate PIN
pin_simple = AdverseSelectionModel.estimate_pin(df_trades, method='simple')
print(f"\nSimple PIN estimate: {pin_simple:.3f}")

pin_mle = AdverseSelectionModel.estimate_pin(df_trades, method='mle')
if pin_mle:
    print(f"\nMLE Estimates:")
    print(f"  PIN: {pin_mle['pin']:.3f}")
    print(f"  α: {pin_mle['alpha']:.3f}")
    print(f"  μ: {pin_mle['mu']:.1f}")
    print(f"  ε_b: {pin_mle['epsilon_b']:.1f}")
    print(f"  ε_s: {pin_mle['epsilon_s']:.1f}")

# Simulation 2: Market maker P&L under adverse selection
print(f"\n{'='*70}")
print(f"MARKET MAKER ADVERSE SELECTION LOSSES")
print(f"{'='*70}")

scenarios = [
    {'name': 'Low Information (PIN=0.1)', 'pin': 0.1, 'spread': 0.02},
    {'name': 'Medium Information (PIN=0.3)', 'pin': 0.3, 'spread': 0.02},
    {'name': 'High Information (PIN=0.5)', 'pin': 0.5, 'spread': 0.02},
    {'name': 'High Info + Wide Spread', 'pin': 0.5, 'spread': 0.05},
]

results = {}

for scenario in scenarios:
    df_mm, cumulative_pnl = simulate_market_maker_losses(
        n_trades=1000, pin=scenario['pin'], spread=scenario['spread'])
    
    results[scenario['name']] = {
        'df': df_mm,
        'cumulative_pnl': cumulative_pnl
    }
    
    total_spread_revenue = df_mm['spread_revenue'].sum()
    total_adverse_loss = df_mm['adverse_loss'].sum()
    net_pnl = df_mm['trade_pnl'].sum()
    
    informed_trades = df_mm[df_mm['is_informed']]
    uninformed_trades = df_mm[~df_mm['is_informed']]
    
    print(f"\n{scenario['name']}:")
    print(f"  Spread revenue: ${total_spread_revenue:.2f}")
    print(f"  Adverse selection loss: ${total_adverse_loss:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f}")
    print(f"  Informed trade %: {len(informed_trades)/len(df_mm)*100:.1f}%")
    print(f"  Avg loss per informed trade: ${informed_trades['adverse_loss'].mean():.4f}")
    print(f"  Avg profit per uninformed: ${uninformed_trades['trade_pnl'].mean():.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Order flow imbalance distribution
axes[0, 0].scatter(df_trades['buys'], df_trades['sells'], alpha=0.5, s=30)
axes[0, 0].plot([0, df_trades['buys'].max()], [0, df_trades['sells'].max()],
                'r--', linewidth=2, label='Balanced')
axes[0, 0].set_title('Order Flow: Buys vs Sells')
axes[0, 0].set_xlabel('Buy Orders')
axes[0, 0].set_ylabel('Sell Orders')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Trade imbalance over time
axes[0, 1].plot(df_trades['day'], df_trades['imbalance'], alpha=0.7)
axes[0, 1].axhline(true_pin, color='red', linestyle='--', 
                   linewidth=2, label=f'True PIN: {true_pin:.3f}')
axes[0, 1].set_title('Daily Order Imbalance (Proxy for PIN)')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('|Buys - Sells| / Total')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Information event detection
info_days = df_trades[df_trades['has_info_event']]
no_info_days = df_trades[~df_trades['has_info_event']]

axes[0, 2].hist(info_days['imbalance'], bins=20, alpha=0.6, label='Info Event', color='red')
axes[0, 2].hist(no_info_days['imbalance'], bins=20, alpha=0.6, label='No Info', color='blue')
axes[0, 2].set_title('Imbalance Distribution by Info Event')
axes[0, 2].set_xlabel('Order Imbalance')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

# Plot 4: Cumulative P&L comparison
for scenario_name, result in results.items():
    axes[1, 0].plot(result['cumulative_pnl'], label=scenario_name, linewidth=2)

axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_title('Market Maker Cumulative P&L')
axes[1, 0].set_xlabel('Trade Number')
axes[1, 0].set_ylabel('Cumulative P&L ($)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 5: P&L per trade distribution (Medium PIN scenario)
df_medium = results['Medium Information (PIN=0.3)']['df']
informed = df_medium[df_medium['is_informed']]['trade_pnl']
uninformed = df_medium[~df_medium['is_informed']]['trade_pnl']

axes[1, 1].hist(informed, bins=30, alpha=0.6, label='Informed', color='red')
axes[1, 1].hist(uninformed, bins=30, alpha=0.6, label='Uninformed', color='green')
axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('P&L Distribution by Trade Type')
axes[1, 1].set_xlabel('Trade P&L ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

# Plot 6: Spread adequacy analysis
pin_range = np.linspace(0.1, 0.9, 20)
spreads_needed = []

for pin_val in pin_range:
    # Simulate to find breakeven spread
    df_test, _ = simulate_market_maker_losses(n_trades=500, pin=pin_val, spread=0.02)
    avg_adverse_loss = df_test['adverse_loss'].mean()
    
    # Breakeven spread: twice the average adverse loss
    breakeven_spread = 2 * avg_adverse_loss
    spreads_needed.append(breakeven_spread)

axes[1, 2].plot(pin_range, spreads_needed, 'o-', linewidth=2, markersize=6)
axes[1, 2].set_title('Required Spread vs Information Level')
axes[1, 2].set_xlabel('PIN (Probability of Informed Trading)')
axes[1, 2].set_ylabel('Breakeven Spread ($)')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*70}")
print(f"KEY INSIGHTS")
print(f"{'='*70}")
print(f"\n1. Higher PIN → Larger adverse selection losses")
print(f"2. Market makers must widen spreads when PIN increases")
print(f"3. Profits from uninformed trades must cover losses from informed")
print(f"4. Order flow imbalance signals information events")
print(f"5. Adverse selection is persistent (permanent price impact)")
```

## 6. Challenge Round
How can market makers mitigate adverse selection?
- **Fast quote updates**: Reduce stale quote exposure, cancel/reprice quickly after trades
- **Selective market making**: Withdraw liquidity during high-risk periods (news, volatility)
- **Order flow segmentation**: Charge wider spreads to toxic flow, rebate to retail
- **Information acquisition**: Invest in signals to detect informed trading patterns
- **Inventory management**: Keep positions small to limit directional exposure
- **Smart order routing avoidance**: Don't quote aggressively where HFTs hunt

What determines optimal spread width?
- **PIN estimate**: Higher informed probability → wider spread needed
- **Competition**: Can't charge too much or lose market share
- **Inventory risk**: Large positions require wider spreads for hedging
- **Volatility**: Higher uncertainty increases information asymmetry risk
- **Regulatory**: Tick size constraints, quote obligations

## 7. Key References
- [Glosten & Milgrom (1985): Bid-Ask Spread and Transaction Price](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443)
- [Easley et al. (1996): PIN Model](https://www.jstor.org/stable/2329394)
- [Kyle (1985): Continuous Auctions and Insider Trading](https://www.jstor.org/stable/1913210)

---
**Status:** Core information asymmetry concept | **Complements:** Bid-Ask Spread, Market Making, Informed Trading
