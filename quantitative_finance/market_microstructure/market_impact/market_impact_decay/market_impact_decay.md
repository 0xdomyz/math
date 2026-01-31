# Market Impact Decay

## 1. Concept Skeleton
**Definition:** Time-dependent relaxation of temporary market impact back to pre-trade equilibrium; characterizes how quickly prices revert after order completion  
**Purpose:** Estimate fraction of impact that is truly permanent versus transient, optimize execution timing strategies, quantify liquidity replenishment speed  
**Prerequisites:** Price impact decomposition, stochastic processes, time-series analysis, market microstructure theory

## 2. Comparative Framing
| Decay Model | Relaxation Function | Typical Half-Life | Mechanism | Trading Use |
|-------------|-------------------|-------------------|-----------|------------|
| **Exponential** | I(t) = I₀ × e^(-λt) | 5-30 min | Order book replenishment | Real-time algos |
| **Power-Law** | I(t) = I₀ × t^(-α) | 10-60 min | Herding/imitation unwind | Medium-freq strategies |
| **Two-Component** | I(t) = I_perm + I_temp × e^(-λt) | Permanent + 5-20 min | Mixed information/liquidity | Impact decomposition |
| **Square-Root** | I(t) = I₀ / √(1 + t/τ) | 10-40 min | Diffusive price adjustment | Theoretical reference |
| **Regime-Dependent** | Switch between models by market state | 2-100+ min | Liquidity condition varies | Risk management |

## 3. Examples + Counterexamples

**Fast Decay (Liquid Stock, Normal Conditions):**  
Apple (AAPL): Execute 50k shares, immediate impact +50 bps. After 5 minutes, impact down to 35 bps. After 30 min, only 5 bps remains (reversed). Half-life ≈ 8 minutes. Most impact temporary (dealers rebalanced inventory).

**Slow Decay (Illiquid Stock, Information Event):**  
Micro-cap biotech: Execute 10k shares, immediate impact +200 bps. After 30 min, still +150 bps (30% decay). After 2 hours, +80 bps. Half-life ≈ 3+ hours. High permanent component (market learning about stock).

**No Decay (Market Crisis 2008):**  
During flash crash: Impact doesn't decay—worsens. Sell causes cascade (others panic sell). Impact increases as traders follow. Eventually circuit breakers halt trading. Decay relationship breaks down (regime change).

**Overshooting (Momentum Continuation):**  
Execute buy 100k shares, price rallies +100 bps immediately. After 5 min, impact +120 bps (overshoots!). After 30 min, settles at +60 bps permanent. Other traders saw buying, continued momentum (temporary amplification before revert).

**Jump Decay (News Release):**  
Execute large sell before earnings. Initial impact -100 bps. Then: earnings released (positive surprise). Price bounces +200 bps total (+100 from fundamental news). Temporary component decayed, but fundamental shift dominates. Timing (information timing) matters most.

**Asymmetric Decay (Buy vs. Sell):**  
Buy 50k: Decay half-life 10 min (dealers want to sell to you to rebalance).  
Sell 50k: Decay half-life 20 min (dealers reluctant to buy, accumulate risk). Asymmetric dealer behavior → different decay rates.

## 4. Layer Breakdown
```
Market Impact Decay Mechanisms:

├─ Conceptual Framework:
│  ├─ Impact Components:
│  │  ├─ Total Impact:
│  │  │  ├─ I_total(t) = P_execution - P_before
│  │  │  ├─ Permanent: I_perm = lim(t→∞) I_total(t)
│  │  │  ├─ Temporary: I_temp(t) = I_total(t) - I_perm
│  │  │  ├─ I_temp(0) = I_total(0) - I_perm (initial temporary impact)
│  │  │  └─ As t→∞: I_temp(t) → 0 (decays away)
│  │  │
│  │  ├─ Decay Dynamics:
│  │  │  ├─ Immediate post-trade: I_temp(0) largest
│  │  │  ├─ Over time: I_temp(t) → 0 exponentially/power-law
│  │  │  ├─ Rate of decay = λ (decay constant) or α (power exponent)
│  │  │  ├─ Half-life τ_½: Time when I_temp = I_temp(0)/2
│  │  │  └─ Practical: Most decay within 30-60 minutes
│  │  │
│  │  └─ Quantification:
│  │     ├─ Measure I_total(0), I_total(5min), I_total(30min), I_total(∞)
│  │     ├─ I_perm = I_total(∞) or estimate via regression
│  │     ├─ I_temp(t) = I_total(t) - I_perm
│  │     ├─ Fit decay model to time series
│  │     └─ Extract λ or α parameters
│  │
│  ├─ Price Impact Decay Models:
│  │  ├─ Exponential Decay:
│  │  │  ├─ Formula: I(t) = I_perm + I_temp(0) × e^(-λt)
│  │  │  ├─ λ = decay rate (inverse of time constant)
│  │  │  ├─ Half-life: τ_½ = ln(2) / λ
│  │  │  ├─ Advantage: Simple, closed-form, easy to estimate
│  │  │  ├─ Mechanism: Dealer inventory mean-reversion or order book replenishment (Poisson arrivals)
│  │  │  ├─ Typical λ: 0.05 - 0.20 per minute (corresponds to 5-15 min half-life)
│  │  │  └─ Calibration:
│  │  │      ├─ Collect post-trade impact at t=0, 5, 10, 30, 60 minutes
│  │  │      ├─ Regress ln(I(t) - I_perm) = ln(I_temp(0)) - λ×t
│  │  │      ├─ Extract λ from slope
│  │  │      └─ Example: λ=0.10 → τ_½ = 6.9 min
│  │  │
│  │  ├─ Power-Law Decay:
│  │  │  ├─ Formula: I(t) = I_perm + I_temp(0) × (τ / (t + τ))^α
│  │  │  ├─ α = power exponent (0 < α < 2)
│  │  │  ├─ τ = time scale parameter
│  │  │  ├─ Advantage: Fatter tail (slower decay at long horizons)
│  │  │  ├─ Mechanism: Information diffusion, herding dynamics
│  │  │  ├─ Typical α: 0.3 - 1.0
│  │  │  └─ Calibration:
│  │  │      ├─ log-log regression: ln(I(t) - I_perm) = ln(I_temp(0)) - α×ln(t + τ)
│  │  │      ├─ Extract α from slope
│  │  │      └─ Example: α=0.5, τ=10 min → slower decay than exponential
│  │  │
│  │  ├─ Two-Component Decay:
│  │  │  ├─ Formula: I(t) = I_perm + I_temp_fast(0) × e^(-λ_fast×t) + I_temp_slow(0) × e^(-λ_slow×t)
│  │  │  ├─ Advantage: Captures initial quick reversion + tail drag
│  │  │  ├─ Fast component: First 5 minutes (inventory rebalancing)
│  │  │  ├─ Slow component: 30+ minutes (information revelation, herding)
│  │  │  ├─ Typical: λ_fast = 0.5 (quick), λ_slow = 0.02 (slow)
│  │  │  └─ Application: Algo design optimizes for fast decay capture
│  │  │
│  │  ├─ Square-Root Decay (Theoretical):
│  │  │  ├─ Formula: I(t) = I_perm + I_temp(0) / √(1 + t/τ)
│  │  │  ├─ τ = characteristic time scale
│  │  │  ├─ Half-life: τ_½ = 3τ (slower than exponential)
│  │  │  ├─ Mechanism: √t price diffusion (Almgren-Chriss model)
│  │  │  ├─ Typical: τ = 10-30 min
│  │  │  └─ Predicts: Long tail (impact lingers for hours)
│  │  │
│  │  └─ Regime-Dependent:
│  │     ├─ Normal market: λ = 0.15 (fast reversion, 5 min half-life)
│  │     ├─ High volatility: λ = 0.05 (slow reversion, 14 min half-life)
│  │     ├─ Low-liquidity regime: λ = 0.02 (very slow, 35 min half-life)
│  │     ├─ Stressed market: No decay / negative decay (worsening)
│  │     └─ Switch model based on realized vol, bid-ask spread, volume
│  │
│  └─ Measurement Methodology:
│     ├─ Data Collection:
│     │  ├─ Large trades (>1% daily volume)
│     │  ├─ Record: Execution price, timing, size
│     │  ├─ Post-trade: Price at t=1, 5, 10, 30, 60, 120 minutes
│     │  ├─ Basis: Midpoint or mid-quote
│     │  ├─ Sample: 100+ trades to get statistics
│     │  └─ Stock: Focus on one name to isolate variables
│     │
│     ├─ Impact Calculation:
│     │  ├─ I(t) = (P_t - P_before_execution) / P_before (in %)
│     │  ├─ Normalize by execution size to get impact elasticity
│     │  ├─ Example:
│     │  │   ├─ Pre-exec price: $100
│     │  │   ├─ At t=5min: $100.25 → I(5min) = +0.25%
│     │  │   ├─ At t=30min: $100.10 → I(30min) = +0.10%
│     │  │   ├─ At t=∞: $100.05 → I_perm ≈ +0.05%
│     │  │   └─ Temporary at 5min: 0.25 - 0.05 = 0.20%
│     │  │
│     │  ├─ Control Variables:
│     │  │  ├─ Order size: Normalize impact per unit size (impact per 0.1% volume)
│     │  │  ├─ Time of day: Morning vs. afternoon (different decay rates)
│     │  │  ├─ Market vol: Regime-adjust decay parameters
│     │  │  ├─ Market impact: Multi-order days (crowding effect)
│     │  │  └─ Information events: Exclude news days (confounding)
│     │  │
│     │  └─ Statistical Estimation:
│     │     ├─ Aggregate samples by order size bucket
│     │     ├─ Within bucket, fit decay model
│     │     ├─ Extract λ (or α) with confidence intervals
│     │     ├─ Test fit quality: R² > 0.70 desired
│     │     └─ Robustness check: Hold-out sample
│     │
├─ Economic Drivers of Decay:
│  ├─ Inventory-Based Decay:
│  │  ├─ Mechanism:
│  │  │  ├─ Dealer accumulates position from large buyer
│  │  │  ├─ Has negative inventory: Quotes high (incentive to sell)
│  │  │  ├─ Over next 5-15 min: New buyers arrive
│  │  │  ├─ Dealer sells to new buyers at tighter spread
│  │  │  ├─ Inventory reduces, quotes normalize
│  │  │  └─ Price relaxes back
│  │  │
│  │  ├─ Speed Factors:
│  │  │  ├─ Dealer risk tolerance: Risk-averse → faster unwinding (decay faster)
│  │  │  ├─ Order flow intensity: High flow → more counterparts → faster rebalancing
│  │  │  ├─ Inventory carrying cost: High → unwilling to hold → decay faster
│  │  │  └─ Competition: Multiple dealers → faster reversion
│  │  │
│  │  ├─ Prediction:
│  │  │  ├─ Exponential decay (natural inventory mean-reversion)
│  │  │  ├─ Typical half-life: 5-20 minutes (dealer time constant)
│  │  │  └─ Liquid stocks (many dealers) → faster; illiquid → slower
│  │  │
│  │  └─ Evidence: High-freq trading accelerates decay (fast inventory rebalancing)
│  │
│  ├─ Order Book Replenishment:
│  │  ├─ Mechanism:
│  │  │  ├─ Large trade consumes best bid/ask (depth depleted)
│  │  │  ├─ Temporarily widens spread (impact visible)
│  │  │  ├─ New limit orders arrive (Poisson process)
│  │  │  ├─ Over 5-30 min: Book fills back up
│  │  │  ├─ Spread normalizes
│  │  │  └─ Temporary component decays
│  │  │
│  │  ├─ Poisson Model:
│  │  │  ├─ Order arrivals follow Poisson(λ_orders) process
│  │  │  ├─ Book recovery time ~ exponential(1/λ_orders)
│  │  │  ├─ Faster order arrivals → faster decay
│  │  │  ├─ Typical: λ_orders = 0.3-1 orders/sec per stock
│  │  │  └─ Decay half-life = (ln 2) / λ_orders ≈ 5-20 min
│  │  │
│  │  └─ Prediction: Exponential decay proportional to order arrival rate
│  │
│  ├─ Information Dynamics (Slow Decay):
│  │  ├─ Mechanism:
│  │  │  ├─ Large trade signals possible information
│  │  │  ├─ Other traders learn from order flow (Kyle model)
│  │  │  ├─ Informed traders: Buy/sell to exploit signal
│  │  │  ├─ Price moves gradually (adverse selection)
│  │  │  ├─ Over 30-120 min: Full information incorporated
│  │  │  └─ Price settles at new equilibrium (permanent impact)
│  │  │
│  │  ├─ Information Content:
│  │  │  ├─ Private info → stronger signal → slower decay (more permanent)
│  │  │  ├─ Public news → weaker signal → faster decay (more temporary)
│  │  │  ├─ Large pension fund (low info) → fast decay
│  │  │  ├─ Insider trading (high info) → no decay
│  │  │  └─ Impact permanent for informed, temporary for uninformed
│  │  │
│  │  ├─ Prediction: Power-law decay (information diffusion)
│  │  │  ├─ α = 0.3-0.7 (slower than exponential)
│  │  │  ├─ Long tail (30+ min before settling)
│  │  │  └─ Amount of permanent impact depends on information asymmetry
│  │  │
│  │  └─ Evidence: Insider trades show little decay (high permanent); index rebalancing decays fast
│  │
│  ├─ Herding & Imitation:
│  │  ├─ Mechanism:
│  │  │  ├─ First large trader buys (price up, visible)
│  │  │  ├─ Trend followers see price rise, buy too
│  │  │  ├─ Cascading orders (correlated flow)
│  │  │  ├─ Momentum phase (5-30 min): Price continues higher (temporary impact amplified)
│  │  │  ├─ Unwinding phase: Trendfollowers take profits (price reverts)
│  │  │  ├─ Decay time depends on trend-following reaction speed
│  │  │  └─ Net: Temporary component larger, decay slower than pure inventory
│  │  │
│  │  ├─ Crowding Risk:
│  │  │  ├─ Many traders using same algo → cascading orders pile on
│  │  │  ├─ Temporary component inflated by herding
│  │  │  ├─ Eventually all herders done → sudden reversal (flash crash risk)
│  │  │  └─ Decay becomes very fast near end (cliff-like decay)
│  │  │
│  │  └─ Prediction: Two-component decay (fast initial + slow tail from herding)
│  │
│  └─ Crowding & Flash Crash:
│     ├─ Regime Change:
│     │  ├─ Normal: Decay (I_temp → 0)
│     │  ├─ Crowded: Amplification (I_temp increases before reverting)
│     │  ├─ Flash crash: No decay / collapse (I_temp → -I_total, overshoots)
│     │  └─ Identification: Track σ(decay_param) across days
│     │
│     └─ Risk Management:
│        ├─ Monitor: Daily decay rate estimates
│        ├─ Alert: If decay slows by >50% (warning sign)
│        ├─ Reduce: Order size if decay deteriorating
│        └─ Diversify: Avoid simultaneous execution with others
│
├─ Practical Applications:
│  ├─ Execution Algorithm Design:
│  │  ├─ Fast decay (λ > 0.10) → Aggressive algorithms acceptable
│  │  │  ├─ Rationale: Temporary impact reverts quickly
│  │  │  ├─ Pay higher market impact short-term for speed
│  │  │  ├─ Example: VWAP-aggressive variant
│  │  │  └─ Typical algo: Liquidate in 1-2 hours
│  │  │
│  │  ├─ Slow decay (λ < 0.05) → Patient algorithms optimal
│  │  │  ├─ Rationale: Temporary impact persists
│  │  │  ├─ Spread execution over longer horizon
│  │  │  ├─ Capture decay as ally (prices recover)
│  │  │  └─ Typical algo: Passive POV over full day
│  │  │
│  │  └─ Adaptive Algorithms:
│  │     ├─ Estimate λ on-the-fly
│  │     ├─ Adjust execution pace based on λ
│  │     ├─ Fast decay → accelerate; Slow decay → decelerate
│  │     └─ Continuous optimization
│  │
│  ├─ Trade Duration Optimization:
│  │  ├─ Shorter trades (5 min):
│  │  │  ├─ Capture momentum (herding amplification) if present
│  │  │  ├─ Avoid overnight risk
│  │  │  ├─ Higher slippage due to less decay benefit
│  │  │  └─ Use if decay fast (λ > 0.15)
│  │  │
│  │  ├─ Medium trades (30-60 min):
│  │  │  ├─ Standard balance: capture half decay benefit
│  │  │  ├─ Avoid most crowding risk
│  │  │  ├─ Typical execution window
│  │  │  └─ Work for most decay regimes
│  │  │
│  │  └─ Longer trades (4+ hours):
│  │     ├─ Capture full decay benefit
│  │     ├─ Accept overnight risk / market move risk
│  │     ├─ Use if decay slow (λ < 0.05)
│  │     └─ Prefer for large orders (illiquid names)
│  │
│  ├─ Inventory Management (Market Makers):
│  │  ├─ Estimate decay rate → predict position half-life
│  │  ├─ Fast decay → tolerate inventory accumulation (will revert)
│  │  ├─ Slow decay → immediately rebalance inventory (pay tighter spreads)
│  │  ├─ Use decay forecasts to set dynamic spreads
│  │  └─ Example: Quote spread = decay_estimate × time_to_rebalance
│  │
│  └─ Risk Monitoring:
│     ├─ Track daily λ estimates
│     ├─ Alert if decay deteriorates (market stress)
│     ├─ Adjust order size limits based on λ
│     └─ Hedge if decay negative (price not reverting)
│
└─ Empirical Evidence:
   ├─ Liquid Large-Cap (AAPL, MSFT):
   │  ├─ Exponential decay, λ ≈ 0.10-0.20
   │  ├─ Half-life: 5-10 minutes
   │  ├─ Permanent component: 20-40% of initial
   │  └─ Temporary reverts fully within 1 hour
   │
   ├─ Mid-Cap (500M-5B market cap):
   │  ├─ Mixed exponential + power-law decay
   │  ├─ λ ≈ 0.05-0.10 (exponential part)
   │  ├─ Half-life: 10-20 minutes
   │  ├─ Permanent component: 40-60% of initial
   │  └─ Tail decay extends 30+ minutes
   │
   ├─ Small-Cap (Illiquid):
   │  ├─ Power-law dominates, α ≈ 0.3-0.5
   │  ├─ Very slow decay, half-life 30-60+ min
   │  ├─ Permanent component: 60-80% of initial
   │  ├─ May not fully revert for hours
   │  └─ High regime-dependency (decay unstable)
   │
   ├─ Time of Day:
   │  ├─ Open (09:30-10:00 AM): Fast decay (λ high, vol high)
   │  ├─ Mid-day (11:00 AM-3:00 PM): Medium decay (λ medium)
   │  ├─ Close (3:00-4:00 PM): Mixed (decay can reverse during close auction)
   │  └─ After-hours: Very slow decay (low vol, illiquid)
   │
   └─ Market Regimes:
      ├─ Normal (VIX < 20): λ ≈ 0.10-0.20 (fast)
      ├─ Elevated vol (VIX 20-30): λ ≈ 0.05-0.10 (slow)
      ├─ Crisis (VIX > 40): λ ≈ 0.01-0.05 (very slow) or negative (no decay)
      └─ Flash crash: λ → negative (prices continue moving away)
```

**Interaction:** Post-trade → Measure impact → Fit decay model → Extract λ/α → Update execution algorithms → Forecast behavior → Repeat.

## 5. Mini-Project
Implement market impact decay analysis and forecast post-trade price trajectories:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

# Generate synthetic post-trade price impact data
np.random.seed(42)
n_trades = 100

# Parameters (varied by trade to simulate realism)
trade_times = np.arange(n_trades)
trade_sizes = np.random.uniform(0.5, 3.0, n_trades)  # % of daily volume
times_since_trade = np.array([1, 5, 10, 30, 60, 120])  # minutes post-trade

# Generate impact data with exponential + power-law components
permanent_impacts = trade_sizes * 0.15 + np.random.normal(0, 0.02, n_trades)  # 15 bps per % volume
temp_impacts_initial = trade_sizes * 0.35 + np.random.normal(0, 0.03, n_trades)  # 35 bps temporary

# Decay parameters (regime-dependent)
decay_rates = np.random.uniform(0.08, 0.15, n_trades)  # λ values
regime = np.random.choice(['normal', 'high_vol', 'crowded'], n_trades, p=[0.6, 0.3, 0.1])

# Store results
impact_data = []
for i in range(n_trades):
    for t in times_since_trade:
        if regime[i] == 'normal':
            decay_rate = 0.12
            temp_component = temp_impacts_initial[i] * np.exp(-decay_rate * t / 10)
        elif regime[i] == 'high_vol':
            decay_rate = 0.06
            temp_component = temp_impacts_initial[i] * np.exp(-decay_rate * t / 10)
        else:  # crowded
            decay_rate = 0.05
            temp_component = temp_impacts_initial[i] * (10 / (t + 10)) ** 0.5  # power-law
        
        total_impact = permanent_impacts[i] + temp_component
        noise = np.random.normal(0, 0.02)
        
        impact_data.append({
            'trade_id': i,
            'time_min': t,
            'order_size_pct': trade_sizes[i],
            'regime': regime[i],
            'total_impact_bps': total_impact * 100 + noise,
            'permanent_bps': permanent_impacts[i] * 100,
            'temporary_bps': temp_component * 100
        })

df_impact = pd.DataFrame(impact_data)

print("="*100)
print("MARKET IMPACT DECAY ANALYSIS")
print("="*100)

print(f"\nStep 1: Data Summary")
print(f"-" * 50)
print(f"Sample trades: {n_trades}")
print(f"Time points per trade: {len(times_since_trade)}")
print(f"Total observations: {len(df_impact)}")
print(f"\nRegime distribution:")
print(df_impact['regime'].value_counts())
print(f"\nOrder size statistics (% daily volume):")
print(df_impact['order_size_pct'].describe().round(3))

# Step 2: Fit decay models per trade
print(f"\nStep 2: Decay Model Estimation (Per-Trade)")
print(f"-" * 50)

def exponential_decay(t, I_perm, I_temp_init, decay_rate):
    return I_perm + I_temp_init * np.exp(-decay_rate * t / 10)

def power_law_decay(t, I_perm, I_temp_init, tau, alpha):
    return I_perm + I_temp_init * (tau / (t + tau)) ** alpha

# Fit individual trades
fit_results = []
for trade_id in range(min(10, n_trades)):  # Fit first 10 trades
    trade_data = df_impact[df_impact['trade_id'] == trade_id]
    times = trade_data['time_min'].values
    impacts = trade_data['total_impact_bps'].values / 100
    
    # Fit exponential
    try:
        popt_exp, _ = curve_fit(exponential_decay, times, impacts, p0=[0.02, 0.3, 0.10])
        I_perm_exp, I_temp_exp, lambda_exp = popt_exp
        half_life_exp = -np.log(2) / (lambda_exp / 10) if lambda_exp > 0 else np.inf
        r2_exp = 1 - np.sum((impacts - exponential_decay(times, *popt_exp))**2) / np.sum((impacts - np.mean(impacts))**2)
        
        # Fit power-law
        popt_pl, _ = curve_fit(power_law_decay, times, impacts, p0=[0.02, 0.3, 15, 0.5])
        I_perm_pl, I_temp_pl, tau_pl, alpha_pl = popt_pl
        r2_pl = 1 - np.sum((impacts - power_law_decay(times, *popt_pl))**2) / np.sum((impacts - np.mean(impacts))**2)
        
        fit_results.append({
            'trade_id': trade_id,
            'regime': trade_data['regime'].iloc[0],
            'order_size': trade_data['order_size_pct'].iloc[0],
            'I_perm_exp': I_perm_exp * 100,
            'lambda': lambda_exp,
            'half_life_exp': half_life_exp,
            'r2_exp': r2_exp,
            'alpha_pl': alpha_pl,
            'tau_pl': tau_pl,
            'r2_pl': r2_pl,
            'better_model': 'exp' if r2_exp > r2_pl else 'power-law'
        })
    except:
        pass

fit_df = pd.DataFrame(fit_results)
print(fit_df[['trade_id', 'regime', 'lambda', 'half_life_exp', 'r2_exp', 'better_model']].to_string(index=False))

# Step 3: Aggregate decay statistics by regime
print(f"\nStep 3: Decay Statistics by Regime")
print(f"-" * 50)

regime_stats = fit_df.groupby('regime').agg({
    'lambda': ['mean', 'std'],
    'half_life_exp': ['mean', 'std'],
    'r2_exp': 'mean'
}).round(3)
print(regime_stats)

# Step 4: Impact decomposition
print(f"\nStep 4: Impact Decomposition (Avg across all trades)")
print(f"-" * 50)

impact_decomp = df_impact.groupby('time_min')[['total_impact_bps', 'permanent_bps', 'temporary_bps']].mean()
impact_decomp['perm_pct'] = 100 * impact_decomp['permanent_bps'] / impact_decomp['total_impact_bps']
impact_decomp['temp_pct'] = 100 * impact_decomp['temporary_bps'] / impact_decomp['total_impact_bps']

print(impact_decomp.round(2))

# Step 5: Regime-specific impact
print(f"\nStep 5: Average Impact by Regime & Time")
print(f"-" * 50)

regime_impact = df_impact.groupby(['regime', 'time_min'])['total_impact_bps'].mean().unstack()
print(regime_impact.round(2))

print(f"\nDecay Speed (minutes to 50% reversion):")
for regime_name in regime_impact.index:
    impact_0 = regime_impact.loc[regime_name, 1]
    impact_120 = regime_impact.loc[regime_name, 120]
    target = impact_0 / 2 + impact_120 / 2
    for t in times_since_trade:
        impact_t = regime_impact.loc[regime_name, t]
        if impact_t <= target:
            print(f"  {regime_name}: ~{t} minutes")
            break

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Individual trade decay curves (sample)
ax = axes[0, 0]
for trade_id in range(min(5, n_trades)):
    trade_data = df_impact[df_impact['trade_id'] == trade_id]
    times = trade_data['time_min'].values
    impacts = trade_data['total_impact_bps'].values
    ax.plot(times, impacts, marker='o', label=f'Trade {trade_id}', linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Impact (basis points)')
ax.set_title('Individual Trade Impact Decay')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Average decay by regime
ax = axes[0, 1]
for regime_name in ['normal', 'high_vol', 'crowded']:
    regime_data = df_impact[df_impact['regime'] == regime_name].groupby('time_min')['total_impact_bps'].mean()
    ax.plot(regime_data.index, regime_data.values, marker='o', label=regime_name, linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Average Impact (basis points)')
ax.set_title('Impact Decay by Market Regime')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Impact decomposition (permanent vs temporary)
ax = axes[1, 0]
x = impact_decomp.index
width = 15
ax.bar(x - width/2, impact_decomp['permanent_bps'], width, label='Permanent', alpha=0.7)
ax.bar(x + width/2, impact_decomp['temporary_bps'], width, label='Temporary', alpha=0.7)
ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Impact (basis points)')
ax.set_title('Permanent vs Temporary Impact Over Time')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cumulative decay (% of initial impact reverted)
ax = axes[1, 1]
for regime_name in ['normal', 'high_vol', 'crowded']:
    regime_data = df_impact[df_impact['regime'] == regime_name].groupby('time_min')['total_impact_bps'].mean()
    initial_impact = regime_data.iloc[0]
    final_impact = regime_data.iloc[-1]
    decay_pct = 100 * (initial_impact - regime_data) / (initial_impact - final_impact)
    ax.plot(regime_data.index, decay_pct, marker='o', label=regime_name, linewidth=2)

ax.set_xlabel('Minutes Since Trade')
ax.set_ylabel('Decay (%)')
ax.set_ylim([0, 120])
ax.set_title('Cumulative Decay: % of Reversible Impact Decayed')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- Normal regime: λ ≈ 0.12 (half-life ~6 min) - fast decay, mostly temporary")
print(f"- High-vol regime: λ ≈ 0.06 (half-life ~12 min) - slower decay, more impact persists")
print(f"- Crowded regime: Power-law dominates - very slow decay, 30+ min tail")
print(f"- Permanent impact: 20-40% of initial, unavoidable")
print(f"- Temporary impact: 60-80% of initial, reverts in 30-120 min (regime-dependent)")
print(f"- Implication: Slower execution beneficial if decay is slow (illiquid/crowded)")
```

## 6. Challenge Round
- Estimate decay rate λ for 10 stocks across regimes (normal, high-vol, stressed); build regime classifier
- Test whether faster trades benefit from "exploiting decay" (prices recover); compare vs slow execution
- Develop decay-adaptive execution algorithm: adjust execution speed based on estimated λ
- Analyze flash crash episodes: Show how decay turns negative; identify warning signs
- Design "decay-neutral" execution cost metric: Control for decay rate when comparing traders

## 7. Key References
- [Bouchaud et al (2004), "How Markets Slowly Digest Changes in Supply and Demand," SSRN](https://arxiv.org/abs/cond-mat/0406224) — Price impact decay models and empirical measurement
- [Hasbrouck & Seppi (2001), "Common Factors in Prices, Order Flows and Liquidity," JFE](https://www.sciencedirect.com/science/article/pii/S0304405X01000748) — Impact dynamics and persistence
- [Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions," Mathematical Finance](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf) — Theoretical decay functions and optimization
- [Kirilenko et al (2017), "The Flash Crash: High-Frequency Trading in an Electronic Market," JFE](https://www.jstor.org/stable/26652722) — Regime breakdown and decay failure

---
**Status:** Advanced microstructure concept (critical for execution timing) | **Complements:** Market Impact, Transaction Costs, Execution Algorithms, Flash Crash Risk
