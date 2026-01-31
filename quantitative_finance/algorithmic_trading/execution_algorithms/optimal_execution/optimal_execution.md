# Optimal Execution Algorithms

## 1. Concept Skeleton
**Definition:** Techniques to minimize execution cost when liquidating/acquiring large positions; trade off speed (market impact) vs patience (timing risk); dynamic programming and optimal control formulations; real-time adaptation to market conditions  
**Purpose:** Maximize execution performance (minimize total cost = realized price - target price); reduce market impact costs; adapt to market microstructure changes; manage information leakage; balance urgency vs opportunity  
**Prerequisites:** Market microstructure, market impact models, continuous-time finance, stochastic optimization, dynamic programming, liquidity metrics, transaction costs

## 2. Comparative Framing
| Algorithm | Flexibility | Market Impact | Execution Risk | Complexity | When to Use |
|-----------|------------|---------------|----------------|-----------|------------|
| **TWAP** | Fixed schedule | Linear in time | High (back-loaded) | Low | Passive, small orders |
| **VWAP** | Volume-dependent | Nonlinear, market-aware | Medium | Medium | High-volume venues |
| **ALMM** | Fully adaptive | Minimized (optimal control) | Low (front-loaded) | High | Large institutional orders |
| **Implementation Shortfall** | Time-dependent | Dynamic | Low-Medium | High | Minimize total deviation |
| **Participation Rate** | POV-based | Dynamic | Low (volume tracking) | Medium | Follow natural flow |
| **Market Microstructure** | Tick/level aware | Level-optimal | Medium | Very High | HFT, competitive execution |

| Execution Pressure | Time Remaining | Liquidity Environment | Optimal Strategy | Expected Cost |
|-------------------|----------------|----------------------|------------------|---------------|
| **High** | Minutes to hours | Normal | Front-load (urgency premium) | 5-15 bps |
| **High** | Minutes to hours | Low | Smaller size, spread trades | 10-25 bps |
| **Medium** | Hours to days | Normal | VWAP/Adaptive blend | 2-8 bps |
| **Low** | Days to weeks | Normal | Spread orders, limit orders | 1-3 bps |
| **Low** | Days to weeks | Volatile | Patient execution, limit ladder | 1-5 bps |
| **Urgent** | Minutes | Illiquid | Full information leak, accept 30+ bps | Variable |

## 3. Examples + Counterexamples

**Simple Example:**  
Sell 10M shares (5% of daily volume). TWAP over 60 min: 167k/min constant. VWAP tracks market volume: front-load when volume surges. ALMM dynamically adapts: larger size when spreads narrow (low impact), pause when spreads widen. VWAP: expected cost 4 bps. ALMM: 2-3 bps.

**Perfect Fit:**  
Hedge fund liquidates $50M equity position (0.5% market cap) over 5 days. ALMM with intraday market microstructure model: participates when book depth > median, retreats when spreads spike. Balances urgency (avoid momentum reversal) vs patience (capture mean reversion). Execution at VWAP - 3 bps.

**Poor Fit:**  
Apply VWAP to micro-cap stock (illiquid, 1% daily volume). Algorithm demands 100% of volume in each minute-bar → forces market orders → price impact ∝ 1000 bps! Use patient limit orders instead, accept longer execution.

**Over-Optimization:**  
Fit optimal execution model to historical volatility and volume using parametric optimization. Test on same period: 1 bp outperformance vs VWAP. Forward test on new data: worse than VWAP (-2 bps). Overfitting to noise, not capturing regime changes.

**Hidden Costs:**  
Algorithm executes too fast → leaves unexecuted shares, rolling order recalculation, information leakage signaling "still selling". Market front-runs remaining order. Total cost higher than slower, more transparent execution.

**Counterexample:**  
Very patient execution (low daily participation, e.g., 2% of volume). Stretches liquidation over 1 month. Capture mean reversion and reduce temporary impact. BUT: position marks move 50 bps adverse between execution start/end. Opportunity cost (price slippage) exceeds market impact savings.

## 4. Layer Breakdown
```
Optimal Execution Algorithms Framework:

├─ Problem Formulation:
│  ├─ Goal: Minimize Total Execution Cost
│  │   Total Cost = Realized Price - Target Price + Opportunity Cost
│  │   Target Price: Market price at decision time
│  │   Realized Price: Average execution price (weighted average)
│  │   Opportunity Cost: Price move due to fundamental shifts during execution
│  ├─ Execution Cost Decomposition:
│  │   ├─ Temporary Impact:
│  │   │   Bid-ask spread crossing
│  │   │   Price pressure (pushes quotes away)
│  │   │   Recovers within seconds/minutes
│  │   │   Proportional to trade size and urgency
│  │   ├─ Permanent Impact:
│  │   │   Information content of trade
│  │   │   Market updates fundamental price
│  │   │   Persists over longer horizon
│  │   │   Proportional to trade size and predictability
│  │   ├─ Spread Cost:
│  │   │   Immediate cost of crossing spread
│  │   │   S_t: Bid-ask spread at time t
│  │   │   Linear in number of shares
│  │   ├─ Timing Risk/Opportunity Cost:
│  │   │   Fundamental price drift during execution
│  │   │   σ: Volatility of price
│  │   │   Grows with execution time
│  │   └─ Participation/Leakage Cost:
│  │       Market front-running (knowing algorithm behavior)
│  │       Adverse selection (information leakage)
│  │       Proportional to information content
│  ├─ Single-Period Model:
│  │   │   Execute N shares, receive price P_t
│  │   │   Impact model: P_t = P_t^0 - f(N_t) - S_t/2
│  │   │   P_t^0: Unperturbed price
│  │   │   f(N_t): Temporary impact (concave)
│  │   │   S_t/2: Spread cost
│  │   │   Optimal N_t: Trade off size (reduce N_t → lower impact) vs time (increase N_t → finish faster)
│  │   └─ Closed Form (linear model):
│  │       f(N) = κN (linear impact coefficient)
│  │       Realize price difference: ΔP = κN + S/2
│  │       Cost per share: κN + S/2
│  ├─ Multi-Period Model (Almgren-Chriss):
│  │   ├─ State:
│  │   │   X_t: Shares remaining (cumulative from 0)
│  │   │   Price: S_t = S_0 + σW_t (random walk, drift 0)
│  │   ├─ Execution path: x_t = dX_t (shares traded at time t)
│  │   │   ∫_0^T x_t dt = X (total shares)
│  │   │   x_t ≥ 0 (no sales then buys)
│  │   ├─ Realized price:
│  │   │   Ŝ = S_0 - ∫_0^T f(x_t) dt - h(X_t)
│  │   │   f(x_t): Temporary impact (immediate recovery)
│  │   │   h(X_t): Permanent impact (accumulates)
│  │   ├─ Impact Functions (Almgren-Chriss):
│  │   │   Temporary: f(x) = τx (linear, recovers instantly)
│  │   │   Permanent: h(X) = γX (proportional to cumulative)
│  │   │   τ, γ: Market impact parameters (empirically calibrated)
│  │   ├─ Objective (Expected Shortfall):
│  │   │   E[Implementation Shortfall] = γX² + τ∫x_t² dt + λσ√(∫x_t² dt)
│  │   │   First term: Permanent impact cost
│  │   │   Second term: Temporary impact cost (sum of squared trades)
│  │   │   Third term: Timing risk (volatility × execution trajectory squared)
│  │   │   λ: Risk aversion parameter (trader preference for speed)
│  │   ├─ Optimal Solution (Linear Impact):
│  │   │   Piecewise linear execution path
│  │   │   If λ=0 (ignore timing risk): Uniform execution (TWAP-like)
│  │   │   If λ=∞ (maximize speed): Front-load all shares immediately
│  │   │   Optimal λ balances: faster execution vs timing risk growth
│  │   └─ Convexity:
│  │       Accelerating impact: f(x) ~ x^α, α > 1
│  │       Deceleration not realistic (implies benefit to bunching trades)
│  ├─ Volume Participation Models:
│  │   ├─ VWAP (Volume-Weighted Average Price):
│  │   │   Execute proportion of volume each period
│  │   │   x_t = V_t × POV (Participation of Volume)
│  │   │   V_t: Market volume in period t
│  │   │   POV: Fixed fraction (e.g., 15% of market volume)
│  │   │   Benchmark: VWAP (volume-weighted market price)
│  │   │   Execution cost: Spread + small impact (hidden orders)
│  │   │   Risk: If V_t drops unexpectedly, can't hit target, forced urgency
│  │   ├─ TWAP (Time-Weighted Average Price):
│  │   │   Execute same size each period
│  │   │   x_t = X / T (uniform)
│  │   │   Benchmark: TWAP (time-weighted market price)
│  │   │   Execution cost: Spread + permanent impact
│  │   │   Risk: Front-loaded by volume watchers (predictable schedule)
│  │   ├─ Participation Adjusted:
│  │   │   x_t = α V_t (POV-based)
│  │   │   Adapt α based on market liquidity (e.g., α = 15% ± 5% variation)
│  │   │   Less predictable than fixed TWAP
│  │   │   Faster when volume available, slower during dry periods
│  │   └─ Information-Optimal:
│  │       Hide order in natural flow (mimic random participant)
│  │       x_t ~ Distribution of natural order sizes
│  │       Minimizes price impact through disguise (information hiding)
│  ├─ Implementation Shortfall (Perold):
│  │   ├─ Definition:
│  │   │   IS = (Price at decision - Realized execution price) × Quantity
│  │   │   Measures total cost (spread + impact + timing)
│  │   ├─ Decomposition:
│  │   │   IS = Delay Cost + Trading Cost + Opportunity Cost
│  │   │   Delay Cost: Price move before first trade (wait cost)
│  │   │   Trading Cost: Market impact + spread during execution
│  │   │   Opportunity Cost: Price moves during unexecuted portion (adverse)
│  │   ├─ Relation to Almgren-Chriss:
│  │   │   IS ≈ γX² + τ∫x_t² dt + (Unexecuted × Price Move)
│  │   ├─ Optimal Strategy:
│  │   │   Execute faster when price moves favorably (capitalize)
│  │   │   Execute slower when price moves adversely (wait for reversion)
│  │   │   Conditional on current price relative to initial target
│  │   └─ Calculation:
│  │       IS = (Decision Price - Execution Price) × Shares
│  │       Decision Price: Market price at order initiation
│  │       Execution Price: Weighted average of fills
│  │       Include fees, commissions explicitly
│  ├─ Market Microstructure Optimization:
│  │   ├─ Limit Order vs Market Order Decision:
│  │   │   Limit orders: Lower cost (get paid spread), high execution risk
│  │   │   Market orders: Guaranteed execution, pay full spread
│  │   │   Optimal mix depends on:
│  │   │     - Urgency (time pressure)
│  │   │     - Queue position (probability of limit fill)
│  │   │     - Spread width (cost differential)
│  │   │     - Adverse selection (probability of mid-move against order)
│  │   ├─ Price Ladder Strategy:
│  │   │   Place multiple limit orders at different price levels
│  │   │   E.g., 10% @ limit, 20% @ limit-1bp, 30% @ limit-2bp, 40% market
│  │   │   Captures depth levels, fallback to market if urgent
│  │   │   Dynamic: adjust levels as volume/spread changes
│  │   ├─ Dark Pool Routing:
│  │   │   Route portion to dark pools (no market impact signal)
│  │   │   Trade off: Reduced market impact vs execution uncertainty
│  │   │   Some dark pools have higher spreads (adverse selection)
│  │   ├─ Iceberg Orders:
│  │   │   Hide true order size (e.g., show 1M, actual 10M)
│  │   │   Reduces information leakage but may violate fair access rules
│  │   │   Market recognizes icebergs, may front-run replenishment
│  │   └─ Execution Tactics by Microstructure Regime:
│  │       │ Tight spread, deep book → More limit orders
│  │       │ Wide spread, shallow book → More market orders (accept cost)
│  │       │ High volume → Aggressive participation rate
│  │       │ Low volume → Patient execution, more limit orders
│  │       │ Volatile → Mix of strategies, reduce size during spikes
│  └─ Dynamic Optimal Execution:
│      ├─ Stochastic Control Formulation:
│      │   State: (X_t, S_t, r_t)
│      │   X_t: Remaining shares
│      │   S_t: Current price
│      │   r_t: Liquidity/market conditions (regime)
│      │   Value function: V_t(X, S, r)
│      │   = E[min execution cost | state (X,S,r)]
│      │   Optimal trade size x_t minimizes V_t forward
│      ├─ Feedback Control:
│      │   x_t*(X_t, S_t, r_t) = f(X_t, S_t, r_t)
│      │   Increases if:
│      │     - X_t large (urgency, less time)
│      │     - S_t moves favorably (capture opportunity)
│      │     - r_t indicates high liquidity (low impact environment)
│      │   Decreases if:
│      │     - X_t small (less urgency)
│      │     - S_t moves adversely (opportunity cost growing)
│      │     - r_t indicates low liquidity (high impact environment)
│      ├─ Regime Switching:
│      │   Distinguish: Normal ↔ Stressed markets
│      │   Normal: Higher volume, tighter spreads, lower volatility
│      │   Stressed: Lower volume, wider spreads, higher volatility
│      │   Execution strategy shifts (more patient in stressed)
│      │   Estimate regime via market indicators (VIX, bid-ask, volume)
│      └─ Example (Two-State Regime):
│          High Liquidity: Execute 20% of remaining shares each period
│          Low Liquidity: Execute 10% of remaining shares each period
│          Transition probabilities estimated from historical data
│          Value > static schedule (adapt to market conditions)
├─ Algorithmic Trading Execution Venues:
│  ├─ Public Exchanges:
│  │   Full transparency, regulated, all participants see order
│  │   Liquidity: Generally high, especially for liquid stocks
│  │   Execution: By price-time priority (FIFO at each level)
│  │   Cost: Spread + clearing fee
│  │   Impact: Immediate market impact visible to all
│  ├─ Dark Pools:
│  │   Block trades executed without public visibility
│  │   Liquidity: Variable (depends on pool size, reputation)
│  │   Execution: Price improving or mid-point (depends on venue)
│  │   Cost: Spread + venue fee (often lower than exchange spread)
│  │   Impact: Hidden from market (information advantage)
│  │   Risk: Execution uncertain (may not fill)
│  ├─ Block Desks:
│  │   Manual negotiation with broker
│  │   Liquidity: Can customize price, size, timing
│  │   Execution: Negotiated (not FIFO)
│  │   Cost: Negotiated spread + commission (sometimes high)
│  │   Impact: Broker absorbs or manages into market
│  ├─ Crossing Networks:
│  │   Match buyers and sellers off-market
│  │   Liquidity: Depends on order flow (variable)
│  │   Execution: Mid-point or agreed price
│  │   Cost: Low (minimal spread)
│  │   Impact: Hidden until execution
│  └─ Algorithmic Brokers:
│      Third-party execution algorithms
│      Liquidity: Aggregate across venues
│      Execution: Algorithmic (TWAP, VWAP, Implementation Shortfall, etc.)
│      Cost: Commission + mark-up on liquidity sources
│      Impact: Depends on algorithm chosen and parameters
├─ Parameter Calibration:
│  ├─ Temporary Impact (τ):
│  │   Estimates slope of price impact
│  │   ΔP = τ × (shares executed)
│  │   Empirical: Regress price change on order size
│  │   Typical range: 0.1-1.0 basis points per 1% of daily volume
│  │   Varies by:
│  │     - Liquidity (tighter spreads → lower τ)
│  │     - Volatility (higher vol → higher τ, less predictable)
│  │     - Venue (exchange vs dark pool differences)
│  │     - Time of day (morning vol > afternoon)
│  ├─ Permanent Impact (γ):
│  │   Estimates information content
│  │   Δμ = γ × (cumulative order size)
│  │   Long-term price shift after execution
│  │   Empirical: Run-up + reversal analysis
│  │   Typical range: 0.5-5 basis points per 1% of market cap
│  │   Varies by:
│  │     - Asset class (stocks > bonds > commodities)
│  │     - Liquidity (liquid → lower γ)
│  │     - Information content (known reason → lower γ)
│  ├─ Volatility (σ):
│  │   Realized or predicted volatility
│  │   Determines timing risk magnitude
│  │   Intraday volatility higher than daily
│  │   Can use GARCH, realized variance, or option prices
│  ├─ Calibration Data:
│  │   Sample large block trades (institutional orders)
│  │   Track execution price vs VWAP, TWAP benchmarks
│  │   Regress cost on order characteristics
│  │   Include controls: time of day, volatility, spreads
│  │   Use cross-validation to avoid overfitting
│  └─ Robustness:
│      Test parameters on out-of-sample data
│      Check sensitivity to parameter misspecification
│      Forward-test on recent data (market regime changes)
├─ Constraints and Practical Issues:
│  ├─ Execution Constraints:
│  │   ├─ Minimum Order Size (MOQ):
│  │   │   Many venues have MOQ (e.g., 100 share lots)
│  │   │   Algorithm must round up to MOQ
│  │   │   Additional rounding cost (1-2 shares per MOQ)
│  │   ├─ Venue Participation Limits:
│  │   │   Regulatory caps (e.g., 20% of volume in 5 min)
│  │   │   Prevents market manipulation
│  │   │   Algorithm must respect limits or route to multiple venues
│  │   ├─ Queue Position:
│  │   │   Limit orders placed at back of queue
│  │   │   Must wait for fills ahead in queue
│  │   │   Mid-move can cause non-execution
│  │   ├─ Latency:
│  │   │   Order submission, network, matching engine delays
│  │   │   Millisecond delays can cause adverse price moves
│  │   │   Co-location, direct feeds reduce latency
│  │   └─ Circuit Breakers:
│  │       Exchanges halt trading during large moves
│  │       Algorithm must pause during halt
│  │       Execution cannot complete if halt triggered
│  ├─ Market Conditions:
│  │   ├─ Liquidity Dry-Ups:
│  │   │   Market event reduces volume (earnings, data releases)
│  │   │   Algorithm must adapt (reduce execution pace)
│  │   ├─ Volatility Spikes:
│  │   │   Large price moves increase impact and timing risk
│  │   │   Algorithm should use smaller trades, more patient execution
│  │   ├─ Spread Widening:
│  │   │   Cost of execution increases
│  │   │   Algorithm may retreat to limit orders
│  │   └─ Block Trades:
│  │       Large competing trades may absorb liquidity
│  │       Algorithm must estimate depth, adjust accordingly
│  └─ Operational Constraints:
│      ├─ Funding Limits:
│      │   May not have cash to buy entire position upfront
│      │   Financing costs (borrowing, repo)
│      │   Algorithm constrained by available capital
│      ├─ Risk Limits:
│      │   Portfolio limits (max position size)
│      │   Sector/strategy limits
│      │   Algorithm must respect limits
│      ├─ Compliance:
│      │   Fair access (no preferential order routing)
│      │   Best execution obligations
│      │   Execution quality monitoring (SEC requirements)
│      └─ System Reliability:
│          Algorithm crashes must be handled
│          Fallback to manual execution
│          Cost of system failure substantial
├─ Performance Evaluation:
│  ├─ Benchmark Selection:
│  │   VWAP: Volume-weighted average market price
│  │   TWAP: Time-weighted average market price
│  │   Arrival price: Price at order initiation
│  │   Close price: End-of-day price
│  │   Next-day VWAP: VWAP of next trading day (for overnight orders)
│  ├─ Cost Metrics:
│  │   Absolute cost: Realized price - Benchmark price
│  │   Relative cost: Absolute cost / Benchmark price (basis points)
│  │   Execution cost: Cost per share
│  │   Opportunity cost: Included in IS but separate from direct cost
│  ├─ Statistical Tests:
│  │   Compare algorithm to benchmark (null: zero difference)
│  │   t-test on costs (multiple executions)
│  │   Is outperformance statistically significant?
│  │   Account for multiple comparisons
│  ├─ Attribution Analysis:
│  │   Decompose cost into components:
│  │     - Spread paid
│  │     - Market impact (temporary + permanent)
│  │     - Timing (delay to execution)
│  │   Which component drives cost?
│  │   Opportunities for improvement?
│  ├─ Slippage Analysis:
│  │   Unexecuted shares at decision time vs end of execution
│  │   If prices move against order, incur slippage
│  │   Track: execution speed, market moves, realized slippage
│  └─ Reporting:
│      Daily/weekly execution summaries
│      Compare to benchmarks
│      Identify outliers (unusually high/low cost executions)
│      Continuous improvement feedback loop
└─ Software and Implementation:
   ├─ Execution Algorithms Libraries:
   │   Bloomberg AIM (Arrival Price, VWAP, TWAP, Implementation Shortfall)
   │   SunGard Portia (Implementation Shortfall focus)
   │   Broker algorithms (Morgan Stanley, Goldman Sachs)
   │   Open-source: Backtrader, Zipline (limited execution algorithms)
   ├─ Venue Connectivity:
   │   FIX protocol (standard market connectivity)
   │   Vendor APIs (Exchange APIs, venue-specific)
   │   Smart order routers (determine venue allocation)
   ├─ Risk Management:
   │   Pre-trade: Validate order (size, price reasonableness)
   │   Intra-trade: Monitor execution pace vs plan
   │   Post-trade: Analysis, reporting
   ├─ Data Requirements:
   │   Real-time market data (quotes, trades, volume)
   │   Historical data (for calibration)
   │   Execution data (fills, prices, timing)
   │   Risk limits and parameters
   └─ Latency Optimization:
      Co-location with exchanges (reduce network latency)
      Direct exchange feeds (skip market data vendor)
      FPGA implementations (ultra-low latency)
```

**Interaction:** Given order → Select algorithm (TWAP/VWAP/ALMM/IS) based on urgency + liquidity → Calibrate parameters → Real-time adapt to market conditions (spreads, volume, volatility) → Execute according to algorithm → Monitor cost vs benchmark → Adjust parameters for next execution.

## 5. Mini-Project
Implement optimal execution algorithms and parameter calibration:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("OPTIMAL EXECUTION ALGORITHMS")
print("="*80)

class MarketSimulator:
    """Simulate market dynamics during execution"""
    
    def __init__(self, S0=100, sigma=0.02, dt=1/252/6.5/60):  # 1-minute bars
        self.S0 = S0
        self.sigma = sigma
        self.dt = dt
    
    def generate_path(self, n_steps, seed=None):
        """Generate GBM price path"""
        if seed is not None:
            np.random.seed(seed)
        
        returns = np.random.normal(0, self.sigma * np.sqrt(self.dt), n_steps)
        prices = self.S0 * np.exp(np.cumsum(returns))
        
        return prices
    
    def generate_volume_path(self, n_steps, avg_volume=1000, seasonality=True):
        """Generate intraday volume pattern"""
        t = np.arange(n_steps)
        
        if seasonality:
            # U-shaped volume pattern (high open/close, low midday)
            factor = 1.5 + 0.5 * np.sin(np.pi * t / n_steps)
        else:
            factor = np.ones(n_steps)
        
        volume = avg_volume * factor * np.random.gamma(2, 1, n_steps)
        
        return np.maximum(volume, 10)  # Min 10 shares

class ExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    def __init__(self, total_shares, T, impact_params=None):
        self.total_shares = total_shares
        self.T = T  # Total periods
        self.tau = impact_params.get('tau', 0.001) if impact_params else 0.001  # Temporary impact
        self.gamma = impact_params.get('gamma', 0.0001) if impact_params else 0.0001  # Permanent impact
        self.sigma = impact_params.get('sigma', 0.02) if impact_params else 0.02  # Volatility
    
    def get_execution_path(self, volumes=None):
        """Return execution quantities at each period"""
        raise NotImplementedError

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution"""
    
    def get_execution_path(self, volumes=None):
        """Execute uniformly over time"""
        x = np.ones(self.T) * self.total_shares / self.T
        return x

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution"""
    
    def __init__(self, total_shares, T, pov=0.15, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.pov = pov  # Participation of volume
    
    def get_execution_path(self, volumes):
        """Execute proportional to market volume"""
        if volumes is None:
            volumes = np.ones(self.T) * np.mean(volumes or 1000)
        
        # Start with POV-based execution
        x = self.pov * volumes
        
        # Ensure total execution matches target
        remaining = self.total_shares - np.sum(x)
        if remaining > 0:
            # Catch up in later periods
            scale = self.total_shares / np.sum(x)
            x = scale * x
        
        return np.minimum(x, self.total_shares)  # Cap at total shares

class OptimalExecutionAlmgrenChriss(ExecutionAlgorithm):
    """Almgren-Chriss optimal execution (linear impact)"""
    
    def __init__(self, total_shares, T, lambda_risk=1e-6, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.lambda_risk = lambda_risk  # Risk aversion parameter
    
    def get_execution_path(self):
        """
        Optimal piecewise-linear execution path
        Solves: min E[IS] = γX² + τ∫x_t² dt + λ*σ²*∫x_t² dt
        """
        # Almgren-Chriss solution for linear impact
        # x_t = X * sqrt(kappa/T) * sinh(g*t) / sinh(g*T)
        # where g = sqrt(kappa) * sqrt(lambda_risk * sigma² / tau)
        
        # Simplified: use gradient descent optimization
        def objective(x_path):
            """Objective function to minimize"""
            # Ensure sum = total shares
            x_path = np.abs(x_path)  # Ensure positive
            x_path = x_path / np.sum(x_path) * self.total_shares
            
            X_cumsum = np.cumsum(x_path)  # Cumulative executed
            
            # Permanent impact cost: γ × (cumulative executed)²
            permanent_cost = self.gamma * np.sum(X_cumsum**2)
            
            # Temporary impact cost: τ × (sum of squared trades)
            temporary_cost = self.tau * np.sum(x_path**2)
            
            # Timing risk: λ × σ² × (sum of squared trades) × (time to completion)
            timing_risk = self.lambda_risk * (self.sigma**2) * np.sum(x_path**2 * (self.T - np.arange(self.T)))
            
            return permanent_cost + temporary_cost + timing_risk
        
        # Initial guess: uniform (TWAP)
        x0 = np.ones(self.T) * self.total_shares / self.T
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=[(0, self.total_shares) for _ in range(self.T)],
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_shares}
        )
        
        x_opt = result.x
        x_opt = np.maximum(x_opt, 0)  # Ensure non-negative
        x_opt = x_opt / np.sum(x_opt) * self.total_shares  # Re-normalize
        
        return x_opt

class LimitOrderExecutor(ExecutionAlgorithm):
    """Execution using limit orders at multiple levels"""
    
    def __init__(self, total_shares, T, limit_levels=5, impact_params=None):
        super().__init__(total_shares, T, impact_params)
        self.limit_levels = limit_levels
    
    def get_execution_path(self, prices):
        """
        Execute via limit orders at different price levels
        Front-load with limit orders, use market orders for remainder
        """
        x_path = np.zeros(self.T)
        executed = 0
        
        for t in range(self.T):
            remaining = self.total_shares - executed
            
            if remaining <= 0:
                break
            
            # Probability of limit order fill decreases with depth
            # Allocate: 20% limit (best bid - 1bp), 30% limit (- 2bp), 30% limit (- 3bp), 20% market
            fill_probs = [0.8, 0.5, 0.2, 0.05, 1.0]
            
            for level, prob in enumerate(fill_probs):
                if level < self.limit_levels:
                    size = remaining * (1 - level * 0.15)
                else:
                    size = remaining
                
                if np.random.random() < prob:
                    x_path[t] += size
                    executed += size
                    break
        
        # Ensure total is hit (force market orders if needed)
        if executed < self.total_shares:
            x_path[self.T - 1] += self.total_shares - executed
        
        return np.minimum(x_path, self.total_shares)

class ExecutionCostCalculator:
    """Calculate execution costs given an execution path"""
    
    def __init__(self, prices, volumes, tau=0.001, gamma=0.0001):
        self.prices = prices
        self.volumes = volumes
        self.tau = tau
        self.gamma = gamma
        self.S0 = prices[0]
    
    def calculate_execution_cost(self, x_path):
        """
        Calculate total execution cost (Implementation Shortfall)
        
        IS = Σ[x_t * (temporary_impact + permanent_impact)]
        """
        n = len(x_path)
        costs = np.zeros(n)
        
        X_cumsum = np.cumsum(x_path)  # Cumulative executed
        
        for t in range(n):
            # Temporary impact: recovers after execution
            temp_impact = self.tau * x_path[t]
            
            # Permanent impact: reflects in all future prices
            perm_impact = self.gamma * X_cumsum[t]
            
            # Total impact cost at time t
            impact_cost = temp_impact + perm_impact
            
            # Execution price = market price - impact
            execution_price = self.prices[t] - impact_cost
            
            # Cost for shares executed at time t
            costs[t] = x_path[t] * (self.prices[t] - execution_price)
        
        total_cost = np.sum(costs)
        cost_basis_points = (total_cost / (self.S0 * self.total_shares)) * 10000 if self.total_shares > 0 else 0
        
        return {
            'total_cost': total_cost,
            'cost_bps': cost_basis_points,
            'avg_execution_price': (self.S0 * self.total_shares - total_cost) / self.total_shares if self.total_shares > 0 else self.S0,
            'costs_by_period': costs
        }

# Simulation setup
total_shares = 100000  # 100k shares
total_days = 5
periods_per_day = 6.5 * 60  # 1-minute bars
T = total_days  # 5 periods for simplicity

# Generate market data
simulator = MarketSimulator(S0=100, sigma=0.015, dt=1/252)
prices = simulator.generate_path(n_steps=T, seed=42)
volumes = simulator.generate_volume_path(n_steps=T, avg_volume=total_shares / T, seasonality=True)

print(f"\nInitial price: ${prices[0]:.2f}")
print(f"Final price: ${prices[-1]:.2f}")
print(f"Total shares to execute: {total_shares:,}")
print(f"Average daily volume: {np.mean(volumes):.0f} shares")

# Impact parameters (calibrated to typical market)
impact_params = {
    'tau': 0.0001,      # Temporary impact: 1 bp per 1% of daily volume
    'gamma': 0.00001,   # Permanent impact: 0.1 bp per 1% of daily volume
    'sigma': 0.015      # Daily volatility
}

# Test different algorithms
algorithms = {
    'TWAP': TWAPAlgorithm(total_shares, T, impact_params),
    'VWAP': VWAPAlgorithm(total_shares, T, pov=0.3, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-6)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-6, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-5)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-5, impact_params=impact_params),
    'Almgren-Chriss (λ=1e-4)': OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=1e-4, impact_params=impact_params),
}

print("\n" + "="*80)
print("EXECUTION ALGORITHM COMPARISON")
print("="*80)

results = {}

for name, algo in algorithms.items():
    if 'VWAP' in name:
        x_path = algo.get_execution_path(volumes)
    else:
        x_path = algo.get_execution_path()
    
    # Normalize to ensure we execute all shares
    x_path = x_path / np.sum(x_path) * total_shares
    
    # Calculate costs
    calculator = ExecutionCostCalculator(prices, volumes, tau=impact_params['tau'], gamma=impact_params['gamma'])
    calculator.total_shares = total_shares
    cost_result = calculator.calculate_execution_cost(x_path)
    
    results[name] = {
        'x_path': x_path,
        'cost_result': cost_result,
        'total_executed': np.sum(x_path)
    }
    
    print(f"\n{name}:")
    print(f"  Execution path (shares/period): {x_path}")
    print(f"  Total execution cost: ${cost_result['total_cost']:,.2f}")
    print(f"  Cost (basis points): {cost_result['cost_bps']:.2f} bps")
    print(f"  Average execution price: ${cost_result['avg_execution_price']:.4f}")
    print(f"  vs Arrival price difference: ${(prices[0] - cost_result['avg_execution_price']):.4f}")

# Compare to benchmarks
print("\n" + "="*80)
print("BENCHMARK COMPARISON")
print("="*80)

# VWAP benchmark
vwap = np.average(prices, weights=volumes)
twap = np.mean(prices)
arrival_price = prices[0]

print(f"\nVWAP: ${vwap:.4f}")
print(f"TWAP: ${twap:.4f}")
print(f"Arrival Price: ${arrival_price:.4f}")

# Compare algorithm execution prices
print(f"\n{'Algorithm':<30} {'Execution Price':<15} {'vs VWAP (bps)':<15} {'vs Arrival (bps)':<15}")
print("-" * 75)

for name, result in results.items():
    exec_price = result['cost_result']['avg_execution_price']
    vs_vwap = (exec_price - vwap) / vwap * 10000
    vs_arrival = (exec_price - arrival_price) / arrival_price * 10000
    
    print(f"{name:<30} ${exec_price:<14.4f} {vs_vwap:<14.2f} {vs_arrival:<14.2f}")

# Sensitivity analysis: varying risk aversion
print("\n" + "="*80)
print("SENSITIVITY: RISK AVERSION PARAMETER (λ)")
print("="*80)

lambda_values = np.logspace(-8, -3, 10)
sensitivity_results = []

for lambda_risk in lambda_values:
    algo = OptimalExecutionAlmgrenChriss(total_shares, T, lambda_risk=lambda_risk, impact_params=impact_params)
    x_path = algo.get_execution_path()
    x_path = x_path / np.sum(x_path) * total_shares
    
    calculator = ExecutionCostCalculator(prices, volumes, tau=impact_params['tau'], gamma=impact_params['gamma'])
    calculator.total_shares = total_shares
    cost_result = calculator.calculate_execution_cost(x_path)
    
    sensitivity_results.append({
        'lambda': lambda_risk,
        'cost_bps': cost_result['cost_bps'],
        'max_execution_size': np.max(x_path),
        'avg_execution_size': np.mean(x_path)
    })

df_sensitivity = pd.DataFrame(sensitivity_results)
print(f"\n{'λ':<15} {'Cost (bps)':<15} {'Max Size':<15} {'Avg Size':<15}")
print("-" * 60)
for _, row in df_sensitivity.iterrows():
    print(f"{row['lambda']:<15.2e} {row['cost_bps']:<15.2f} {row['max_execution_size']:<15,.0f} {row['avg_execution_size']:<15,.0f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Price path
ax = axes[0, 0]
ax.plot(prices, 'b-', linewidth=2, label='Market Price')
ax.axhline(vwap, color='g', linestyle='--', linewidth=1.5, label=f'VWAP: ${vwap:.2f}')
ax.axhline(twap, color='r', linestyle=':', linewidth=1.5, label=f'TWAP: ${twap:.2f}')
ax.set_title('Price Path During Execution')
ax.set_xlabel('Period')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volume path
ax = axes[0, 1]
ax.bar(range(len(volumes)), volumes, alpha=0.6, color='blue')
ax.set_title('Market Volume Pattern')
ax.set_xlabel('Period')
ax.set_ylabel('Volume (shares)')
ax.grid(alpha=0.3)

# Plot 3: Execution paths comparison
ax = axes[0, 2]
for name, result in results.items():
    ax.plot(result['x_path'], marker='o', label=name, linewidth=1.5)
ax.set_title('Execution Paths by Algorithm')
ax.set_xlabel('Period')
ax.set_ylabel('Execution Size (shares)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Cumulative execution
ax = axes[1, 0]
for name, result in results.items():
    cumsum = np.cumsum(result['x_path'])
    ax.plot(cumsum, marker='s', label=name, linewidth=1.5)
ax.axhline(total_shares, color='k', linestyle='--', alpha=0.5, label='Target')
ax.set_title('Cumulative Execution')
ax.set_xlabel('Period')
ax.set_ylabel('Cumulative Shares')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 5: Cost comparison (bar chart)
ax = axes[1, 1]
costs_bps = [results[name]['cost_result']['cost_bps'] for name in results.keys()]
colors = ['green' if c < np.mean(costs_bps) else 'red' for c in costs_bps]
ax.bar(range(len(results)), costs_bps, color=colors, alpha=0.7)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(list(results.keys()), rotation=45, ha='right', fontsize=8)
ax.set_title('Execution Cost Comparison')
ax.set_ylabel('Cost (basis points)')
ax.axhline(np.mean(costs_bps), color='k', linestyle='--', alpha=0.5, label='Mean')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Sensitivity to λ
ax = axes[1, 2]
ax.semilogx(df_sensitivity['lambda'], df_sensitivity['cost_bps'], marker='o', linewidth=2, markersize=6)
ax.set_title('Sensitivity: Execution Cost vs Risk Aversion (λ)')
ax.set_xlabel('Risk Aversion Parameter (λ)')
ax.set_ylabel('Cost (basis points)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*80)
print("EXECUTION SUMMARY STATISTICS")
print("="*80)

print(f"\nCost Statistics (basis points):")
costs = [r['cost_result']['cost_bps'] for r in results.values()]
print(f"  Mean cost: {np.mean(costs):.2f} bps")
print(f"  Min cost: {np.min(costs):.2f} bps (best algorithm)")
print(f"  Max cost: {np.max(costs):.2f} bps (worst algorithm)")
print(f"  Std dev: {np.std(costs):.2f} bps")
print(f"  Spread: {np.max(costs) - np.min(costs):.2f} bps")

best_algo = min(results.items(), key=lambda x: x[1]['cost_result']['cost_bps'])
print(f"\nBest performing algorithm: {best_algo[0]}")
print(f"  Cost: {best_algo[1]['cost_result']['cost_bps']:.2f} bps")
print(f"  Savings vs worst: {np.max(costs) - best_algo[1]['cost_result']['cost_bps']:.2f} bps")

# Trade-off: speed vs cost
print(f"\nExecution Speed Trade-offs:")
for name, result in results.items():
    x_path = result['x_path']
    # Measure concentration (lower = more spread out)
    concentration = np.max(x_path) / np.mean(x_path)
    cost_bps = result['cost_result']['cost_bps']
    print(f"  {name}: Concentration={concentration:.2f}, Cost={cost_bps:.2f} bps")
```

## 6. Challenge Round
1. **Parameter Estimation:** Given execution trade data (timestamps, prices, quantities, benchmarks), calibrate τ (temporary impact) and γ (permanent impact) via nonlinear regression. Validate on hold-out period. Does model capture realized costs?

2. **Regime Detection:** Simulate 3 liquidity regimes (normal, stressed, volatile). Train Markov switching model on volumes/spreads. Implement adaptive algorithm that switches execution pace based on detected regime. Does it outperform fixed algorithm?

3. **Market Impact Decay:** Implement permanent impact with exponential recovery: γX(1 - e^{-θt}). Estimate θ (decay rate) from block trade data. Compare to constant permanent impact model.

4. **Latency Optimization:** Simulate venue with latency (order submission → fill with 10-100ms delay). Measure: what order size can execute before mid moves? How does latency affect optimal path?

5. **Dark Pool Routing:** Allocate execution across exchange + 3 dark pools with different fees, fills, spreads. Optimize allocation to minimize total cost. Compare single-venue vs multi-venue execution.

## 7. Key References
- [Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2001)](https://www.math.nyu.edu/~almgren/papers/optliq.pdf) - canonical optimal execution framework with linear impact
- [Perold, "The Implementation Shortfall: Paper Losses and Transaction Costs" (1988)](https://finance.zicklin.baruch.cuny.edu/research/seminars/almgren-chriss-optliq/) - Implementation Shortfall as execution cost metric
- [Bouchaud et al., "Market Microstructure Facing the Challenge of the Electronification" (2004)](https://arxiv.org/abs/cond-mat/0410072) - empirical market impact models and microstructure

---
**Status:** Core institutional trading infrastructure | **Complements:** Market Microstructure, Execution Algorithms, TWAP/VWAP Benchmarks, Trading Signals, Portfolio Management
