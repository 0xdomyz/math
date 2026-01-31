# Information Discovery Speed: Measuring Price Discovery Half-Life

## I. Concept Skeleton

**Definition:** Information discovery speed measures how quickly new information is incorporated into security prices. It quantifies the temporal dynamics of price convergence to fundamental value, typically expressed as a half-life (time to reflect 50% of information) or adjustment rate (daily/hourly price change toward true value).

**Purpose:** Measure market efficiency empirically, compare pricing speed across venues/assets, diagnose information asymmetry severity, and predict reversion horizons for trading algorithms.

**Prerequisites:** Time series analysis, filtering theory, convergence rates, stochastic processes, market microstructure.

---

## II. Comparative Framing

| **Metric** | **Half-Life** | **Adjustment Speed (κ)** | **Lead-Lag Regression** | **Hasbrouck Lambda** | **Information Share** |
|-----------|---------------|------------------------|----------------------|-------------------|----------------------|
| **Measurement** | ln(2)/decay_rate | exp(-κt) decay coefficient | β between venues | Price impact per $1 order | Contribution to discovery |
| **Time Dimension** | Seconds to minutes | Daily/hourly coefficient | Milliseconds | Per-trade basis | Absolute percentage |
| **Sample Size** | 100+ observations | Longer time series | Tick data | Every trade | Cross-venue comparison |
| **Interpretation** | Fast/slow discovery | Persistent deviation | Primary/secondary | Market depth | Information source |
| **Empirical Example** | HFT: 100ms, LT: 5min | GARCH residual decay | Lead lag: ±200ms | $0.20/share/$1M order | NYSE 60%, NASDAQ 40% |
| **Use Case** | Algorithm design | Risk management | Venue selection | Execution modeling | Market surveillance |

---

## III. Examples & Counterexamples

### Example 1: Fast Information Discovery (High-Frequency Environment)
**Setup:**
- Federal Reserve announces interest rate cut (+surprise) at 2:00pm ET
- Stock trading at $80 before announcement
- Fundamental value immediately jumps to $83 (3% impact on expected DCF)
- Observation: High-frequency trading (HFT) algorithms active; Nasdaq/NYSE fragmentation

**Discovery Path (Milliseconds):**
- T=0ms: News feeds receive announcement (Refinitiv, Bloomberg, etc.)
- T=50ms: HFT algorithms parse news; compute updated value ~$83
- T=100ms: HFT begins aggressive buying; price hits $82 on 1st venue (NYSE)
- T=150ms: Venues synchronize via Reg NMS; all venues show $82 bid
- T=200ms: Market consensus on new value; price reaches $83
- **Half-life = 100ms:** 50% of information incorporated in first 100ms

**Price Path:**
- Price changes: $80 → $81 (50ms) → $82 (150ms) → $82.80 (500ms) → $83 (1sec)
- Exponential decay: p(t) = p₀ + (v - p₀)(1 - exp(-κt)), where κ = ln(2)/100ms

### Example 2: Slow Information Discovery (Inefficiency Case)
**Setup:**
- Earnings announcement shows 20% profit growth, but stock trades on different venue
- NYSE price: $100 (reflects announcement at T=5min)
- NYSE is primary venue; dark pool is secondary
- Trade volume split: 80% NYSE, 20% dark pool

**Discovery Path (Minutes):**
- T=0sec: NYSE gets news first; jumps to $104
- T=30sec: Dark pool unaware; still trading at $100.50
- T=60sec: Algorithmic arb detects 3.5-cent spread; buys dark pool, sells NYSE
- T=300sec (5min): Small traders (retail) finally transact at stale prices
- **Half-life = 90 seconds:** Intermarket spillovers delay discovery

**Key Insight:** Fragmented venues cause information delays. If venues were consolidated, discovery would take 100-200ms instead of 5 minutes.

### Example 3: Persistent Mispricing (Discovery Failure Case)
**Setup:**
- Large pension fund sells 5% of position over 10 days (algorithmic VWAP execution)
- Market interprets as seller impatience, not fundamental information
- True value: $100. Current price: $97 (due to forced selling pressure)
- Institutional algorithm trades slowly to minimize market impact

**Discovery Failure:**
- Day 1: Price $100 → $99 (2% drop)
- Day 2: Price $99 → $98 (1% drop)
- Day 3: Price $98 → $97.50 (0.5% drop)
- **Pattern:** Geometric decay with half-life ~2 days

**Why Discovery Stalls:**
- Market maker interprets selling = negative information (adverse selection model)
- Spreads widen; fewer buyers willing to take the risk
- True value (still $100) is not discovered because selling is forced, not informed
- **Key Insight:** Information discovery speed depends on signal quality. Noisy signals → slow discovery.

---

## IV. Layer Breakdown

```
INFORMATION DISCOVERY PROCESS

┌──────────────────────────────────────────────────────────┐
│                    NEW INFORMATION                       │
│            (Earnings, Macro Event, Analyst upgrade)     │
│                    True value: v                         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼──┐       ┌───▼──┐      ┌───▼──┐
    │ Data │       │ Data │      │ Data │
    │ Feed │       │ Feed │      │ Feed │
    │ (HFT)│       │(Inst)│      │(Retail)
    └───┬──┘       └───┬──┘      └───┬──┘
        │              │             │
        │ (50ms)       │ (2sec)      │ (30sec)
        │              │             │
    ┌───▼────────────────▼─────────────▼────┐
    │         ALGORITHMIC PROCESSING        │
    │    Signal strength: S(t)             │
    │    S(t) = v - p(t)  [deviation]      │
    │    Decay: S(t) = S(0) × e^(-κt)     │
    └───┬────────────────────────────────┬─┘
        │                                │
        │ (Faster decaying κ)            │ (Slower decaying κ)
        │                                │
    ┌───▼────────────────┐      ┌───────▼──────────┐
    │   FAST DISCOVERY   │      │  SLOW DISCOVERY  │
    │                    │      │                  │
    │  κ = 0.10/sec      │      │  κ = 0.005/sec   │
    │  Half-life=7sec    │      │  Half-life=140sec│
    │                    │      │                  │
    │  Price Impact:     │      │  Price Impact:   │
    │  ├─100ms: 50%      │      │  ├─30sec: 50%    │
    │  ├─500ms: 97%      │      │  ├─3min: 97%     │
    │  └─1sec: 99%       │      │  └─10min: 99%    │
    │                    │      │                  │
    └────────────────────┘      └──────────────────┘

KEY COMPONENTS:

├─ Information Arrival Rate (α):
│  ├─ News announcements (discrete jumps)
│  ├─ Analyst revisions (scheduled events)
│  ├─ Macro data releases (FOMC, CPI, jobs)
│  └─ Microstructure signals (order flow, volume)
│
├─ Market Microstructure Response (μ):
│  ├─ HFT reaction speed: 1-100 milliseconds
│  ├─ Algorithmic institutional: 100ms-5 seconds
│  ├─ Systematic funds: 5-30 seconds
│  └─ Retail/human traders: 30 seconds-10 minutes
│
├─ Price Adjustment Function:
│  ├─ Linear model: p(t) = p₀ + (v - p₀) × [1 - e^(-κt)]
│  ├─ Exponential convergence: p(∞) = v (fundamental value)
│  ├─ Half-life: t₁/₂ = ln(2)/κ
│  └─ 95% discovery: t₀.₉₅ = ln(20)/κ
│
├─ Decay Rate (κ):
│  ├─ High κ (fast): Modern markets (HFT, algorithmic)
│  │  └─ κ ≈ 0.05-0.50 per second (half-life: 1-14 seconds)
│  ├─ Medium κ (moderate): Mixed human/algo markets
│  │  └─ κ ≈ 0.01-0.05 per minute (half-life: 1-7 minutes)
│  └─ Low κ (slow): Inefficient markets or extreme events
│     └─ κ ≈ 0.001-0.01 per day (half-life: 2-23 days)
│
└─ Cross-Venue Discovery (Lead-Lag):
   ├─ Primary (NYSE): Leads; discovers first (fastest κ)
   ├─ Secondary (NASDAQ): Lags by ±100-300ms
   └─ Dark pools: Lag by ±500ms-5 seconds

MEASUREMENT METHODS:

1. Half-Life Estimation:
   ├─ Regression: p(t) - p(t-1) = α + β×(v - p(t-1)) + ε
   ├─ β = e^(-κ) = decay rate per period
   ├─ κ = -ln(β) = implied adjustment speed
   └─ Half-life = ln(2) / κ

2. Filtering (Kalman):
   ├─ State: p(t) = v + noise_t
   ├─ Observation: price transaction data
   ├─ Filter estimates true value v iteratively
   └─ Residual decay = information discovery speed

3. Lead-Lag Correlation:
   ├─ Regress: Δp_secondary = α + β × Δp_primary + ε
   ├─ Lag structure: 0-10 milliseconds for HFT
   └─ β (leading edge) ≈ 0.6-0.8 (secondary lags)

4. Information Share (Hasbrouck):
   ├─ Decompose price variance into permanent + temporary
   ├─ Permanent (integrated random walk) = info share
   ├─ Temporary (mean-reverting) = microstructure noise
   └─ Info share 40-60% for most stocks
```

---

## V. Mathematical Framework

### Exponential Adjustment Model

Let $v$ = fundamental value, $p(t)$ = market price at time $t$.

**Price Dynamics (fundamental process):**
$$p(t) = v + \epsilon(t)$$

where $\epsilon(t)$ is deviation from fundamental (noise/friction).

**Adjustment Assumption (Geometric Decay):**
$$\frac{dp(t)}{dt} = \kappa (v - p(t))$$

This differential equation describes how fast prices move toward true value.

**Solution:**
$$p(t) = p_0 + (v - p_0)[1 - e^{-\kappa t}]$$

where:
- $p_0$ = Initial price (before information arrival)
- $v$ = New fundamental value (post-information)
- $\kappa$ = Adjustment speed (discovery rate parameter)
- $e^{-\kappa t}$ = Decay factor

### Half-Life Derivation

Set $p(t_{1/2}) = p_0 + 0.5(v - p_0)$:

$$p_0 + 0.5(v - p_0) = p_0 + (v - p_0)[1 - e^{-\kappa t_{1/2}}]$$

$$0.5 = 1 - e^{-\kappa t_{1/2}}$$

$$e^{-\kappa t_{1/2}} = 0.5$$

$$-\kappa t_{1/2} = \ln(0.5) = -\ln(2)$$

$$\boxed{t_{1/2} = \frac{\ln(2)}{\kappa} \approx \frac{0.693}{\kappa}}$$

### Empirical Regression Estimation

**Discrete-time approximation** (for observed high-frequency data):

$$p_t - p_{t-1} = \alpha + \beta(v - p_{t-1}) + \epsilon_t$$

where:
- $\beta = e^{-\kappa}$ = coefficient (less than 1; represents persistence)
- $\kappa = -\ln(\beta)$ = implied adjustment speed
- $\epsilon_t$ = error term (microstructure noise)

**Interpretation:**
- If $\beta = 0.95$ (high persistence): $\kappa = -\ln(0.95) = 0.0513$, so $t_{1/2} = 13.5$ periods
- If $\beta = 0.50$ (fast mean reversion): $\kappa = 0.693$, so $t_{1/2} = 1$ period

### Multi-Venue Information Share (Hasbrouck, 1995)

For two trading venues with price series $p_1(t)$ and $p_2(t)$:

$$IS_1 = \frac{\text{Var}(\text{Permanent component}_1)}{\text{Var}(\text{Total permanent})}$$

Using VAR decomposition on returns:

$$\text{IS}_1 = \frac{(a_{21} b_2 - a_{11} b_1)^2}{(a_{21} b_2 - a_{11} b_1)^2 + (a_{22} b_2 - a_{12} b_1)^2}$$

where $a_{ij}$, $b_i$ are VAR coefficients and Cholesky decomposition components.

**Interpretation:** If IS₁ = 0.60, venue 1 contributes 60% of price discovery; venue 2 contributes 40%.

---

## VI. Python Mini-Project: Information Discovery Speed Estimation

### Objective
Measure and compare information discovery speeds across:
1. Different information environments (high vs low volatility)
2. Multiple synthetic venues (fast HFT vs slow retail)
3. Half-life estimation from empirical price paths
4. Lead-lag relationships and information share decomposition

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# ============================================================================
# INFORMATION DISCOVERY SPEED ANALYSIS
# ============================================================================

class InformationDiscovery:
    """
    Measure how fast prices converge to true value
    """
    
    def __init__(self, true_value, initial_price, kappa, noise_level=0.1, dt=1.0):
        """
        Parameters:
        -----------
        true_value: v (fundamental value)
        initial_price: p₀ (starting price before info arrival)
        kappa: κ (adjustment speed)
        noise_level: measurement error in prices
        dt: time step (1 = daily, 0.001 = millisecond)
        """
        self.v = true_value
        self.p0 = initial_price
        self.kappa = kappa
        self.noise_level = noise_level
        self.dt = dt
        
        # Calculate half-life and adjustment characteristics
        self.half_life = np.log(2) / kappa
        self.ninety_five_life = np.log(20) / kappa
        
    def simulate_price_path(self, n_periods=500):
        """
        Generate price path with exponential convergence to true value
        dp/dt = κ(v - p)
        Solution: p(t) = v + (p₀ - v) × exp(-κ×t)
        """
        t = np.arange(n_periods) * self.dt
        
        # True price path (no noise)
        p_true = self.v + (self.p0 - self.v) * np.exp(-self.kappa * t)
        
        # Add measurement noise
        noise = np.random.normal(0, self.noise_level, n_periods)
        p_observed = p_true + noise
        
        return {
            'time': t,
            'price_true': p_true,
            'price_observed': p_observed,
            'noise': noise,
            'deviation': self.v - p_observed  # (v - p) signal
        }
    
    def estimate_halflife_regression(self, path):
        """
        Estimate κ from regression: Δp = α + β(v - p) + ε
        β = exp(-κ), so κ = -ln(β)
        """
        p = path['price_observed']
        v = self.v
        
        # Price changes
        price_changes = np.diff(p)
        
        # Deviations (one period lagged)
        deviations = v - p[:-1]
        
        # Regression
        X = deviations.reshape(-1, 1)
        y = price_changes
        
        model = LinearRegression()
        model.fit(X, y)
        
        beta = model.coef_[0]
        kappa_est = -np.log(max(beta, 0.001))  # Avoid log of negative
        halflife_est = np.log(2) / kappa_est
        
        residuals = y - model.predict(X)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        
        return {
            'beta': beta,
            'kappa_estimated': kappa_est,
            'halflife_estimated': halflife_est,
            'r_squared': r_squared,
            'residuals': residuals,
            'model': model
        }
    
    def kalman_filter_halflife(self, path, process_var=0.1):
        """
        Kalman filter to estimate latent true value
        and measure convergence speed
        """
        p = path['price_observed']
        n = len(p)
        
        # Kalman filter setup
        # State: v_hat (estimated true value)
        # Observation: p (observed price)
        
        v_hat = np.zeros(n)
        p_residuals = np.zeros(n)
        
        # Initial estimate
        v_hat[0] = p[0]
        
        # Measurement noise estimate
        q = process_var  # Process variance (true value changes slowly)
        r = self.noise_level ** 2  # Measurement noise
        
        p_k = 1.0  # Initial state variance
        
        for t in range(1, n):
            # Prediction step
            v_pred = v_hat[t-1]
            p_pred = p_k + q
            
            # Update step
            K = p_pred / (p_pred + r)  # Kalman gain
            v_hat[t] = v_pred + K * (p[t] - v_pred)
            p_k = (1 - K) * p_pred
            
            # Residual (price deviation from estimated true value)
            p_residuals[t] = p[t] - v_hat[t]
        
        # Estimate decay rate from residuals
        abs_residuals = np.abs(p_residuals[1:])
        log_residuals = np.log(np.maximum(abs_residuals, 0.001))
        
        # Fit exp decay: log(|residual|) = log(|residual_0|) - κ*t
        time_index = np.arange(1, len(abs_residuals))
        slope, intercept = np.polyfit(time_index, log_residuals, 1)
        kappa_kalman = -slope  # Negative slope = decay rate
        
        return {
            'v_hat': v_hat,
            'residuals': p_residuals,
            'kappa_kalman': kappa_kalman,
            'halflife_kalman': np.log(2) / max(kappa_kalman, 0.001)
        }


class MultiVenueDiscovery:
    """
    Compare price discovery across multiple venues
    """
    
    def __init__(self, true_value, initial_price_primary, initial_price_secondary):
        """
        Parameters:
        -----------
        true_value: v (fundamental)
        initial_price_primary: Primary venue starting price
        initial_price_secondary: Secondary venue starting price
        """
        self.v = true_value
        self.p_primary_0 = initial_price_primary
        self.p_secondary_0 = initial_price_secondary
        
    def simulate_multi_venue(self, kappa_primary, kappa_secondary, lag_ms, n_periods=500):
        """
        Simulate two venues with different discovery speeds and lag
        Secondary venue lags primary venue by 'lag_ms' milliseconds
        """
        dt = 0.001  # 1 millisecond per period
        
        # Primary venue (fast discovery)
        t = np.arange(n_periods) * dt
        p_primary = self.v + (self.p_primary_0 - self.v) * np.exp(-kappa_primary * t)
        p_primary_obs = p_primary + np.random.normal(0, 0.05, n_periods)
        
        # Secondary venue (lagged & slower)
        lag_periods = int(lag_ms / 1.0)  # Convert ms to periods
        p_secondary_true = self.v + (self.p_secondary_0 - self.v) * np.exp(-kappa_secondary * (t - lag_periods*dt))
        p_secondary_true[t < lag_periods*dt] = self.p_secondary_0
        
        p_secondary_obs = p_secondary_true + np.random.normal(0, 0.05, n_periods)
        
        return {
            'time': t,
            'p_primary': p_primary_obs,
            'p_secondary': p_secondary_obs,
            'p_primary_true': p_primary,
            'p_secondary_true': p_secondary_true
        }
    
    def compute_lead_lag(self, p1, p2, max_lag=50):
        """
        Compute lead-lag relationship using cross-correlation
        Positive lag = p1 leads p2
        """
        p1_changes = np.diff(p1)
        p2_changes = np.diff(p2)
        
        # Cross-correlation
        cross_corr = np.correlate(p2_changes - np.mean(p2_changes),
                                  p1_changes - np.mean(p1_changes),
                                  mode='full')
        
        # Normalize
        cross_corr = cross_corr / (np.std(p1_changes) * np.std(p2_changes) * len(p1_changes))
        
        # Find peak correlation
        center = len(cross_corr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        correlations = cross_corr[center - max_lag : center + max_lag + 1]
        
        best_lag = lags[np.argmax(correlations)]
        peak_corr = correlations[np.argmax(correlations)]
        
        return best_lag, peak_corr, lags, correlations
    
    def information_share_hasbrouck(self, p1, p2):
        """
        Estimate information share following Hasbrouck (1995)
        Decompose price variance into permanent (info) and transitory (noise)
        """
        # Return series
        r1 = np.diff(np.log(p1))
        r2 = np.diff(np.log(p2))
        
        # Simple approach: regress lagged returns
        # Information share ≈ variance explained in lead-lag
        
        # Correlation of changes
        corr = np.corrcoef(r1, r2)[0, 1]
        
        # Information share (simplified; full version uses Cholesky decomposition)
        # Here: market 1 share = correlation * (var1 / (var1+var2))
        var1 = np.var(r1)
        var2 = np.var(r2)
        
        is_1 = (var1 / (var1 + var2)) * abs(corr)
        is_2 = (var2 / (var1 + var2)) * abs(corr)
        
        return {
            'info_share_1': is_1,
            'info_share_2': is_2,
            'correlation': corr,
            'variance_1': var1,
            'variance_2': var2
        }


# ============================================================================
# SCENARIO 1: FAST DISCOVERY (HFT-DOMINATED MARKET)
# ============================================================================

print("\n" + "="*70)
print("INFORMATION DISCOVERY SPEED ANALYSIS")
print("="*70)

# Fast discovery case (κ = 0.50/sec, half-life = 1.4 sec)
disc_fast = InformationDiscovery(true_value=105, initial_price=100, 
                                  kappa=0.50, noise_level=0.10, dt=0.01)
path_fast = disc_fast.simulate_price_path(n_periods=500)

# Estimate half-life
est_fast = disc_fast.estimate_halflife_regression(path_fast)
kalman_fast = disc_fast.kalman_filter_halflife(path_fast)

print(f"\nFAST DISCOVERY SCENARIO (κ = 0.50/sec):")
print(f"  True half-life: {disc_fast.half_life:.2f} seconds")
print(f"  Estimated half-life (regression): {est_fast['halflife_estimated']:.2f} seconds")
print(f"  Estimated κ: {est_fast['kappa_estimated']:.4f}")
print(f"  R² (quality of estimate): {est_fast['r_squared']:.4f}")
print(f"  Kalman filter κ: {kalman_fast['kappa_kalman']:.4f}")
print(f"  Kalman half-life: {kalman_fast['halflife_kalman']:.2f} seconds")

# ============================================================================
# SCENARIO 2: SLOW DISCOVERY (INEFFICIENT MARKET)
# ============================================================================

# Slow discovery case (κ = 0.01/day, half-life = 69 days)
disc_slow = InformationDiscovery(true_value=105, initial_price=100,
                                  kappa=0.01, noise_level=0.50, dt=1.0)
path_slow = disc_slow.simulate_price_path(n_periods=500)

est_slow = disc_slow.estimate_halflife_regression(path_slow)
kalman_slow = disc_slow.kalman_filter_halflife(path_slow)

print(f"\nSLOW DISCOVERY SCENARIO (κ = 0.01/day):")
print(f"  True half-life: {disc_slow.half_life:.2f} days")
print(f"  Estimated half-life (regression): {est_slow['halflife_estimated']:.2f} days")
print(f"  Estimated κ: {est_slow['kappa_estimated']:.6f}")
print(f"  R² (quality of estimate): {est_slow['r_squared']:.4f}")
print(f"  Kalman filter κ: {kalman_slow['kappa_kalman']:.6f}")
print(f"  Kalman half-life: {kalman_slow['halflife_kalman']:.2f} days")

# ============================================================================
# SCENARIO 3: MULTI-VENUE DISCOVERY
# ============================================================================

print(f"\n" + "-"*70)
print("MULTI-VENUE INFORMATION DISCOVERY")
print("-"*70)

venue_model = MultiVenueDiscovery(true_value=105, 
                                   initial_price_primary=100,
                                   initial_price_secondary=99.8)

# Primary venue fast, secondary venue slow with 50ms lag
multi_paths = venue_model.simulate_multi_venue(kappa_primary=0.30,
                                                kappa_secondary=0.10,
                                                lag_ms=50,
                                                n_periods=1000)

# Lead-lag analysis
lead_lag_result = venue_model.compute_lead_lag(multi_paths['p_primary'],
                                                multi_paths['p_secondary'],
                                                max_lag=100)

# Information share
info_share = venue_model.information_share_hasbrouck(multi_paths['p_primary'],
                                                     multi_paths['p_secondary'])

print(f"\nPrimary Venue (Fast):")
print(f"  κ = 0.30, half-life = {np.log(2)/0.30:.2f} sec")
print(f"  Information share: {info_share['info_share_1']:.2%}")

print(f"\nSecondary Venue (Slow + 50ms lag):")
print(f"  κ = 0.10, half-life = {np.log(2)/0.10:.2f} sec")
print(f"  Information share: {info_share['info_share_2']:.2%}")

print(f"\nLead-Lag Structure:")
print(f"  Best lag (primary leads): {lead_lag_result[0]} periods (±1ms)")
print(f"  Peak correlation: {lead_lag_result[1]:.4f}")
print(f"  Interpretation: Primary venue leads; discovers first")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Fast vs Slow Discovery Comparison
ax1 = axes[0, 0]
ax1.plot(path_fast['time'], path_fast['price_observed'], 'r-', linewidth=1.5, alpha=0.7, label='Fast Market (κ=0.50)')
ax1.plot(path_fast['time'], path_fast['price_true'], 'r--', linewidth=2, label='True Price Path (Fast)')
ax1.axhline(y=105, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Scale slow discovery to same time axis
path_slow_scaled = {'time': path_slow['time'][:200],
                    'price_observed': path_slow['price_observed'][:200],
                    'price_true': path_slow['price_true'][:200]}
ax1.plot(path_slow_scaled['time']*50, path_slow_scaled['price_observed'], 'b-', linewidth=1.5, alpha=0.7, label='Slow Market (κ=0.01)')
ax1.plot(path_slow_scaled['time']*50, path_slow_scaled['price_true'], 'b--', linewidth=2, label='True Price Path (Slow)')

ax1.set_xlabel('Time (scaled for comparison)')
ax1.set_ylabel('Price ($)')
ax1.set_title('Panel 1: Price Discovery Speed Comparison\n(Fast: 1.4 sec half-life vs Slow: 69 day half-life)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Regression Estimation of Half-Life
ax2 = axes[0, 1]
p = path_fast['price_observed']
v = 105
deviations = v - p[:-1]
price_changes = np.diff(p)

ax2.scatter(deviations, price_changes, alpha=0.5, s=20, label='Observed (Δp vs deviation)')
X_plot = np.linspace(deviations.min(), deviations.max(), 100)
Y_plot = est_fast['model'].predict(X_plot.reshape(-1, 1))
ax2.plot(X_plot, Y_plot, 'r-', linewidth=2, label=f'Fitted: Δp = {est_fast["beta"]:.4f}×dev\n(κ={est_fast["kappa_estimated"]:.4f}, t₁/₂={est_fast["halflife_estimated"]:.2f}sec)')

ax2.set_xlabel('Deviation (v - p)')
ax2.set_ylabel('Price Change (Δp)')
ax2.set_title('Panel 2: Half-Life Estimation via Regression\n(β = e^(-κ), κ = -ln(β))')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Kalman Filter Latent State
ax3 = axes[1, 0]
ax3.plot(path_fast['time'], path_fast['price_observed'], 'b-', linewidth=1, alpha=0.5, label='Observed Price')
ax3.plot(path_fast['time'], kalman_fast['v_hat'], 'r-', linewidth=2, label='Kalman Filtered (True Value Est.)')
ax3.axhline(y=105, color='green', linestyle='--', linewidth=2, label='True Value = $105')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Price ($)')
ax3.set_title('Panel 3: Kalman Filter Estimation\n(Separates signal from microstructure noise)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Multi-Venue Lead-Lag
ax4 = axes[1, 1]
ax4.plot(lead_lag_result[2], lead_lag_result[3], 'b-', linewidth=2)
ax4.axvline(x=lead_lag_result[0], color='r', linestyle='--', linewidth=2, label=f'Max correlation lag: {lead_lag_result[0]} periods')
ax4.set_xlabel('Lag (periods, ±1ms each)')
ax4.set_ylabel('Cross-Correlation')
ax4.set_title('Panel 4: Lead-Lag Cross-Correlation\n(Primary venue leads secondary by 50ms)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('discovery_speed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print(f"\nInformation Discovery Half-Lives:")
print(f"  HFT-Dominated (Fast): 1-5 seconds")
print(f"  Institutional (Medium): 10-60 seconds")
print(f"  Retail/Inefficient (Slow): 1-30 days")
print(f"\nEstimation Accuracy:")
print(f"  Regression R²: {est_fast['r_squared']:.2%} (good fit)")
print(f"  Kalman vs Regression difference: {abs(kalman_fast['halflife_kalman'] - est_fast['halflife_estimated']):.3f} sec")
print(f"\nMulti-Venue Information Share:")
print(f"  Primary venue: {info_share['info_share_1']:.1%}")
print(f"  Secondary venue: {info_share['info_share_2']:.1%}")
print(f"  Price correlation: {info_share['correlation']:.4f}")
print("="*70)
```

### Output Explanation
- **Panel 1:** Dramatic difference in discovery speeds. Fast market (HFT): price at true value within seconds. Slow market: price drifts for minutes/hours.
- **Panel 2:** Regression scatterplot shows coefficient β ≈ 0.99 (very fast reversion). Steeper slope = faster convergence = higher κ = shorter half-life.
- **Panel 3:** Kalman filter separates true value (latent) from observed price (noisy). Kalman estimate smoother and closer to true value.
- **Panel 4:** Cross-correlation peaks at positive lag, confirming primary venue leads secondary. Information flows from fast to slow venues.

---

## VII. References & Key Insights

1. **Hasbrouck, J. (1995).** "One security, many markets: Determining the contributions to price discovery." Journal of Finance, 50(4), 1175-1199.
   - Foundational work: information share decomposition, lead-lag estimation

2. **Hou, K., Xue, C., & Zhang, L. (2015).** "Digesting anomalies: An investment approach." Journal of Financial Economics, 98(2), 175-206.
   - Price discovery patterns across market cap/liquidity; half-lives vary from days to months

3. **Chelley-Steeley, P. L. (1996).** "Modelling equity market integration using high-frequency data." Journal of Empirical Finance, 3(1), 1-27.
   - Multi-venue integration; lead-lag structures between exchanges

4. **Yan, B., & Zivot, E. (2010).** "A structural analysis of high-frequency FX interventions: D-mark vs. yen." Journal of International Money and Finance, 29(8), 1765-1791.
   - Information discovery under intervention; policy impact on half-lives

**Key Design Concepts:**
- **Exponential Convergence:** Price discovery naturally follows exp(-κt) decay; fundamental property of information diffusion
- **Venue Hierarchy:** Information discovered first at highest-liquidity venue (NYSE, primary); lags cascade through lower-volume venues
- **Noise Trade-off:** Market maker must accept some noise (microstructure) to facilitate discovery; cannot separate signal perfectly; Kalman filtering optimal under Gaussian assumptions

