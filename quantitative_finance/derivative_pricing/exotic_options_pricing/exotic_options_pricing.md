# Exotic Options Pricing

## 1. Concept Skeleton
**Definition:** Non-standard derivatives with payoffs depending on path, multiple assets, or complex conditions beyond vanilla call/put structures  
**Purpose:** Tailor risk exposure to specific market views; reduce hedging costs; exploit market inefficiencies; enhance yield or protection  
**Prerequisites:** Black-Scholes framework, risk-neutral valuation, Monte Carlo methods, numerical PDE, path-dependent option mechanics

## 2. Comparative Framing
| Option Type | Vanilla (European) | Barrier | Asian | Lookback | Digital |
|-------------|-------------------|---------|-------|----------|---------|
| **Payoff Depends On** | Terminal price | Path + barrier | Path average | Path extreme | Binary outcome |
| **Complexity** | Simple | Moderate | Moderate | High | Discontinuous |
| **Price vs Vanilla** | Benchmark | Cheaper | Cheaper | More expensive | Varies |
| **Hedging** | Delta-Gamma | Jump risk at barrier | Continuous | Requires path | Infinite gamma |
| **Closed-Form** | Yes (BS) | Sometimes | Geometric only | Rarely | Yes |

| Pricing Method | Monte Carlo | PDE/Finite Diff | Tree Methods | Analytical | Closed-Form Approx |
|----------------|-------------|-----------------|--------------|------------|--------------------|
| **Path-Dependent** | Excellent | Difficult | Non-recombining | Rare | Limited |
| **Multi-Asset** | Excellent | Curse of dim. | Infeasible | Very rare | Very limited |
| **Barriers** | Good (monitoring) | Good | Moderate | Some cases | Some cases |
| **Accuracy** | O(1/√n) | O(h²) | O(1/n) | Exact | Approximate |

## 3. Examples + Counterexamples

**Simple Example:**  
Barrier knock-out call (barrier=$110, strike=$100, spot=$100): If price hits $110 before expiry, option worthless. Cheaper than vanilla since might knock out.

**Perfect Fit:**  
Asian option for commodity hedger: Payoff based on average oil price over quarter matches physical delivery pattern. Reduces manipulation risk at single fixing.

**Digital/Binary:**  
All-or-nothing call: Pays $100 if S_T>K, else $0. Used in structured notes. Infinite gamma near K at expiry → difficult to hedge.

**Lookback Call:**  
Payoff = S_T - min(S_t over [0,T]). Always ITM at expiry. Expensive (guarantees best execution). Popular in FX for importers/exporters.

**Basket Option:**  
Call on weighted portfolio of 5 stocks: Payoff = max(Σw_i S_i - K, 0). Cheaper than sum of individual calls due to diversification/correlation.

**Poor Fit:**  
Using Black-Scholes for barrier: Continuous monitoring assumption vs reality of discrete checks. Can misprice by 5-10% depending on frequency.

## 4. Layer Breakdown
```
Exotic Options Framework:

├─ Path-Dependent Options:
│  ├─ Asian Options (Average Price):
│  │   ├─ Payoff Structures:
│  │   │   ├─ Arithmetic average: Payoff = max((1/n)Σ S_ti - K, 0)
│  │   │   ├─ Geometric average: Payoff = max(∏S_ti^(1/n) - K, 0)
│  │   │   ├─ Fixed strike: K predetermined
│  │   │   └─ Floating strike: K = average, payoff on terminal S_T
│  │   ├─ Pricing:
│  │   │   ├─ Geometric: Closed-form (adjusted BS)
│  │   │   │   Parameters: σ_geo = σ/√3, r_geo adjusted
│  │   │   ├─ Arithmetic: No closed-form, use Monte Carlo
│  │   │   ├─ Control variate: Use geometric as control
│  │   │   └─ PDE: Requires state variable for running average
│  │   ├─ Advantages:
│  │   │   ├─ Lower volatility → cheaper than vanilla
│  │   │   ├─ Manipulation-resistant (average vs single fixing)
│  │   │   └─ Matches cash flows for physical delivery
│  │   └─ Uses:
│  │       Commodities, currencies, equity compensation
│  ├─ Barrier Options:
│  │   ├─ Types:
│  │   │   ├─ Knock-out: Dies if barrier hit
│  │   │   │   ├─ Down-and-out: Lower barrier
│  │   │   │   └─ Up-and-out: Upper barrier
│  │   │   ├─ Knock-in: Activates if barrier hit
│  │   │   │   ├─ Down-and-in: Lower barrier
│  │   │   │   └─ Up-and-in: Upper barrier
│  │   │   ├─ Double barrier: Two barriers (in or out)
│  │   │   └─ Partial barriers: Active only during period
│  │   ├─ In-Out Parity:
│  │   │   Knock-in + Knock-out = Vanilla
│  │   │   Arbitrage relationship
│  │   ├─ Pricing:
│  │   │   ├─ Closed-form: Some cases (Merton, Reiner-Rubinstein)
│  │   │   ├─ Reflection principle: Mirror image method
│  │   │   ├─ Monte Carlo: Track barrier breaches
│  │   │   │   Continuous monitoring: Brownian bridge
│  │   │   │   Discrete monitoring: Actual path checks
│  │   │   └─ PDE: Boundary condition at barrier (value=0 or rebate)
│  │   ├─ Greeks:
│  │   │   ├─ Delta: Discontinuous at barrier
│  │   │   ├─ Gamma: Spikes near barrier
│  │   │   └─ Vega: Different behavior vs vanilla
│  │   └─ Practical Considerations:
│  │       ├─ Monitoring frequency: Daily, continuous, specific times
│  │       ├─ Rebate: Payment if knocked out
│  │       ├─ Hedging difficulty: Jump risk at barrier
│  │       └─ Used to cheapen vanilla (OTM barrier less likely)
│  ├─ Lookback Options:
│  │   ├─ Fixed Strike Lookback:
│  │   │   ├─ Call: max(max(S_t) - K, 0)
│  │   │   └─ Put: max(K - min(S_t), 0)
│  │   ├─ Floating Strike Lookback:
│  │   │   ├─ Call: S_T - min(S_t) (always ITM)
│  │   │   └─ Put: max(S_t) - S_T (always ITM)
│  │   ├─ Pricing:
│  │   │   ├─ Closed-form exists (Goldman et al.)
│  │   │   ├─ Involves cumulative normal integrals
│  │   │   ├─ Monte Carlo: Track running max/min
│  │   │   └─ PDE: Two state variables (S and max/min)
│  │   ├─ Value:
│  │   │   Expensive (guarantees best execution)
│  │   │   Floating strike: Worth more than fixed
│  │   └─ Uses:
│  │       FX (best rate), performance measurement
│  └─ Ladder Options:
│      ├─ Lock in profits at rungs (price levels)
│      ├─ Payoff = max of (locked gains, terminal payoff)
│      └─ Path-dependent with discrete memory points
├─ Multi-Asset Options:
│  ├─ Basket Options:
│  │   ├─ Payoff: max(Σ w_i S_i - K, 0)
│  │   │   Weighted sum of assets
│  │   ├─ Pricing:
│  │   │   ├─ No closed-form (non-lognormal sum)
│  │   │   ├─ Monte Carlo: Simulate correlated assets
│  │   │   │   Use Cholesky decomposition for correlation
│  │   │   ├─ Approximations: Moment-matching to lognormal
│  │   │   └─ Tree: Tensor product (infeasible for many assets)
│  │   ├─ Correlation Impact:
│  │   │   ├─ Higher correlation → closer to single asset
│  │   │   ├─ Lower correlation → diversification benefit
│  │   │   └─ Dispersion trade: Long basket, short components
│  │   └─ Uses:
│  │       Index options (custom), portfolio hedging
│  ├─ Rainbow Options:
│  │   ├─ Best-of / Worst-of:
│  │   │   ├─ Best-of call: max(max(S₁, S₂, ...) - K, 0)
│  │   │   ├─ Worst-of put: max(K - min(S₁, S₂, ...), 0)
│  │   │   └─ Best/worst of multiple assets
│  │   ├─ Pricing:
│  │   │   ├─ 2-asset: Closed-form (Stulz)
│  │   │   ├─ n-asset: Monte Carlo
│  │   │   └─ Correlation crucial: Determines spread
│  │   ├─ Value:
│  │   │   ├─ Best-of: More valuable than individual
│  │   │   ├─ Worst-of: Less valuable
│  │   │   └─ Correlation effect opposite for calls vs puts
│  │   └─ Uses:
│  │       Employee stock options (best of company/index)
│  │       Currency hedging (best rate of multiple pairs)
│  ├─ Spread Options:
│  │   ├─ Payoff: max(S₁ - S₂ - K, 0)
│  │   │   Difference between two assets
│  │   ├─ Exchange Option (Margrabe):
│  │   │   K=0: max(S₁ - S₂, 0)
│  │   │   Closed-form solution
│  │   ├─ Pricing:
│  │   │   ├─ Margrabe formula (K=0)
│  │   │   ├─ Kirk approximation (K>0)
│  │   │   └─ Monte Carlo for general case
│  │   └─ Uses:
│  │       Commodities (crack spreads), pairs trading
│  ├─ Quanto Options:
│  │   ├─ Payoff in different currency from underlying
│  │   ├─ Example: Nikkei option paying in USD
│  │   ├─ Pricing: Adjust drift for correlation
│  │   │   μ_quanto = μ - ρ σ_asset σ_FX
│  │   └─ Uses:
│  │       International investments without FX risk
│  └─ Correlation Options:
│      Direct bets on correlation between assets
│      Dispersion trading, correlation swaps
├─ Digital / Binary Options:
│  ├─ Cash-or-Nothing:
│  │   ├─ Call: Pays fixed amount C if S_T > K, else 0
│  │   ├─ Put: Pays C if S_T < K, else 0
│  │   ├─ Pricing: C × e^(-rT) × N(±d₂)
│  │   └─ Derivative of vanilla call w.r.t. K
│  ├─ Asset-or-Nothing:
│  │   ├─ Pays S_T if S_T > K (call) or S_T < K (put)
│  │   ├─ Pricing: S₀ × N(±d₁)
│  │   └─ Building block for vanillas
│  ├─ Greeks:
│  │   ├─ Delta: Spikes near strike at expiry
│  │   ├─ Gamma: Dirac delta function (infinite at K)
│  │   └─ Vega: Also spikes, changes sign near K
│  ├─ Hedging:
│  │   ├─ Extremely difficult near expiry
│  │   ├─ Small move → large delta change
│  │   └─ Often hedged with vanilla spreads
│  └─ Uses:
│      Structured products, binary bets, FX barriers
├─ Chooser / Compound Options:
│  ├─ Chooser:
│  │   ├─ Holder chooses call or put at future date
│  │   ├─ Simple chooser: Same K, T for both
│  │   ├─ Complex chooser: Different parameters
│  │   └─ Pricing: Closed-form for simple (Rubinstein)
│  ├─ Compound Options:
│  │   ├─ Option on an option
│  │   ├─ Call-on-call, put-on-put, call-on-put, put-on-call
│  │   ├─ Two strikes, two expiries (T₁ < T₂)
│  │   ├─ Pricing: Nested expectations, bivariate normal
│  │   └─ Uses: Real options (staged investment), volatility bets
│  └─ Value:
│      Optionality to wait → time value premium
├─ Variance / Volatility Products:
│  ├─ Variance Swaps:
│  │   ├─ Payoff: N × (σ²_realized - K_var)
│  │   │   N = notional per variance point
│  │   ├─ Realized variance: σ²_real = (252/n) Σ ln²(S_t/S_{t-1})
│  │   ├─ Fair strike: K_var = E[σ²_realized]
│  │   ├─ Pricing: Replication with log-contract
│  │   │   K_var = (2/T) ∫ C(K)/K² dK + put integral
│  │   │   Model-free using strip of options
│  │   └─ Properties:
│  │       ├─ Pure volatility exposure (convex in vol)
│  │       ├─ Vega: Constant across strikes
│  │       └─ Path-dependent (realized vol over period)
│  ├─ Volatility Swaps:
│  │   ├─ Payoff: N × (σ_realized - K_vol)
│  │   │   Linear in vol, not variance
│  │   ├─ Approximation: K_vol ≈ K_var - σ³/(8×K_var)
│  │   │   Convexity adjustment
│  │   └─ Less liquid than variance swaps
│  ├─ VIX Options:
│  │   ├─ Underlying: VIX index (30-day implied vol)
│  │   ├─ Pricing: Not lognormal (mean-reverting)
│  │   │   Use VIX futures as forward
│  │   └─ Hedging: Tail risk, vol spike protection
│  └─ Corridor Variance Swaps:
│      Only accrues when spot in corridor [L, H]
│      Reduces cost, targets specific scenarios
├─ Forward-Start / Cliquet Options:
│  ├─ Forward-Start:
│  │   ├─ Option granted now, strike set at future date
│  │   ├─ Typically K = S_T1 (at-the-money forward)
│  │   ├─ Pricing: Closed-form (homogeneity property)
│  │   │   V = S₀ × BS(1, 1, r, T₂-T₁, σ) / B(0,T₁)
│  │   └─ Uses: Employee stock options (ESO)
│  ├─ Cliquet (Ratchet):
│  │   ├─ Series of forward-start options
│  │   ├─ Locks in periodic gains (sum of returns)
│  │   ├─ Payoff: Σ max(α × return_i, floor)
│  │   │   α = participation rate, may have caps/floors
│  │   └─ Pricing: Sum of forward-starts with caps/floors
│  └─ Value:
│      Protection against vol spikes in future
│      Popular in structured products
├─ Other Exotic Structures:
│  ├─ Himalaya Options:
│  │   ├─ Basket with best performer removed each period
│  │   ├─ Payoff = sum of best assets at each date
│  │   └─ Reduces concentration risk
│  ├─ Napoleon Options:
│  │   Like Himalaya but worst performer removed
│  ├─ Shout Options:
│  │   Holder can "shout" once to lock in intrinsic value
│  │   Combines lookback and call features
│  ├─ Parisian Options:
│  │   Barrier triggered only if breached for continuous period
│  │   Less sensitive to brief spikes than standard barriers
│  └─ Power Options:
│      Payoff = (S_T)^α - K
│      Non-linear exposure, higher moments matter
└─ Pricing Considerations:
   ├─ Model Selection:
   │   ├─ GBM: Standard, may underprice barriers/digitals
   │   ├─ Jump-diffusion: Better for discontinuous payoffs
   │   ├─ Stochastic vol: Smile/skew dependent payoffs
   │   └─ Local vol: Path-dependent, barriers
   ├─ Numerical Methods:
   │   ├─ Monte Carlo: Path-dependent, multi-asset
   │   │   ├─ Variance reduction crucial (antithetic, control)
   │   │   ├─ Barriers: Brownian bridge for continuous monitoring
   │   │   └─ Discretization error: Euler vs Milstein
   │   ├─ PDE/Finite Difference:
   │   │   ├─ Low-dimensional (<3 assets)
   │   │   ├─ Barriers natural as boundary conditions
   │   │   └─ Stability, convergence issues for discontinuous payoffs
   │   ├─ Trees:
   │   │   ├─ Path-dependent: Non-recombining (exponential)
   │   │   └─ Better for American-style exotics
   │   └─ Semi-Analytical:
   │       Fourier methods, Laplace transforms for special cases
   ├─ Hedging Challenges:
   │   ├─ Path-dependence: Greeks change with history
   │   ├─ Barriers: Jump risk, discontinuous deltas
   │   ├─ Digitals: Infinite gamma at strike
   │   ├─ Multi-asset: Correlation risk (vega, vanna)
   │   └─ Dynamic replication often imperfect
   ├─ Market Practices:
   │   ├─ Bid-ask spreads: Wider than vanilla (illiquidity)
   │   ├─ Valuation adjustments: Model risk, liquidity
   │   ├─ Hedging costs: Built into price
   │   └─ Regulatory capital: Higher risk weights
   └─ Applications:
      ├─ Structured products: Tailored payoffs for retail
      ├─ Corporate hedging: Match cash flow patterns
      ├─ Trading strategies: Express specific views
      └─ Cost reduction: Barriers cheaper than vanillas
```

**Interaction:** Exotic payoff structure → Select pricing method → Model calibration → Risk management (Greeks, scenarios) → Dynamic hedging strategy.

## 5. Mini-Project
Implement pricing for major exotic option types:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EXOTIC OPTIONS PRICING")
print("="*60)

class ExoticOptionPricer:
    """Pricing engine for exotic options"""
    
    def __init__(self, S0, r, sigma, T):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
    
    def generate_path(self, n_steps):
        """Generate single GBM path"""
        dt = self.T / n_steps
        path = np.zeros(n_steps + 1)
        path[0] = self.S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1)
            path[t] = path[t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            )
        
        return path
    
    def asian_arithmetic_mc(self, K, n_paths=50000, n_steps=252):
        """Price arithmetic average Asian option via Monte Carlo"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            avg_price = np.mean(path)
            payoff = max(avg_price - K, 0)
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def asian_geometric_closed_form(self, K):
        """Price geometric average Asian option (closed-form)"""
        # Adjusted parameters for geometric average
        sigma_geo = self.sigma / np.sqrt(3)
        r_geo = 0.5 * (self.r - 0.5*self.sigma**2 + sigma_geo**2)
        
        # Black-Scholes with adjusted parameters
        d1 = (np.log(self.S0/K) + (r_geo + 0.5*sigma_geo**2)*self.T) / (sigma_geo*np.sqrt(self.T))
        d2 = d1 - sigma_geo*np.sqrt(self.T)
        
        price = np.exp(-self.r*self.T) * (self.S0*np.exp(r_geo*self.T)*norm.cdf(d1) - K*norm.cdf(d2))
        
        return price
    
    def barrier_down_out_call(self, K, H, n_paths=50000, n_steps=252):
        """Price down-and-out barrier call"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            
            # Check if barrier breached
            if np.min(path) > H:  # Not knocked out
                payoff = max(path[-1] - K, 0)
            else:
                payoff = 0
            
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def lookback_floating_call(self, n_paths=50000, n_steps=252):
        """Price floating strike lookback call: S_T - min(S_t)"""
        payoffs = []
        
        for _ in range(n_paths):
            path = self.generate_path(n_steps)
            payoff = path[-1] - np.min(path)  # Always positive
            payoffs.append(payoff)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def digital_call(self, K, cash_payoff=1.0):
        """Price cash-or-nothing digital call"""
        d2 = (np.log(self.S0/K) + (self.r - 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        price = cash_payoff * np.exp(-self.r*self.T) * norm.cdf(d2)
        return price

class MultiAssetExotics:
    """Multi-asset exotic options"""
    
    def __init__(self, S0_list, r, sigma_list, corr_matrix, T):
        self.S0 = np.array(S0_list)
        self.r = r
        self.sigma = np.array(sigma_list)
        self.corr_matrix = corr_matrix
        self.T = T
        self.n_assets = len(S0_list)
        
        # Cholesky decomposition for correlation
        self.chol = cholesky(corr_matrix, lower=True)
    
    def generate_paths(self, n_paths):
        """Generate correlated terminal prices"""
        # Independent normal draws
        Z = np.random.normal(0, 1, (n_paths, self.n_assets))
        
        # Correlate
        Z_corr = Z @ self.chol.T
        
        # Terminal prices
        S_T = self.S0 * np.exp(
            (self.r - 0.5*self.sigma**2)*self.T + 
            self.sigma*np.sqrt(self.T)*Z_corr
        )
        
        return S_T
    
    def basket_option(self, K, weights, n_paths=50000):
        """Price basket call option"""
        S_T = self.generate_paths(n_paths)
        
        # Weighted basket value
        basket_values = S_T @ weights
        payoffs = np.maximum(basket_values - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def best_of_call(self, K, n_paths=50000):
        """Price best-of (rainbow) call option"""
        S_T = self.generate_paths(n_paths)
        
        # Maximum of all assets
        best_prices = np.max(S_T, axis=1)
        payoffs = np.maximum(best_prices - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def worst_of_put(self, K, n_paths=50000):
        """Price worst-of put option"""
        S_T = self.generate_paths(n_paths)
        
        # Minimum of all assets
        worst_prices = np.min(S_T, axis=1)
        payoffs = np.maximum(K - worst_prices, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def spread_option(self, K, n_paths=50000):
        """Price spread option on first two assets"""
        S_T = self.generate_paths(n_paths)
        
        # Spread: S1 - S2
        spreads = S_T[:, 0] - S_T[:, 1]
        payoffs = np.maximum(spreads - K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se

# Black-Scholes for vanilla comparison
def black_scholes(S, K, r, T, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Scenario 1: Asian options
print("\n" + "="*60)
print("SCENARIO 1: Asian Options (Path-Dependent)")
print("="*60)

S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20

pricer = ExoticOptionPricer(S0, r, sigma, T)

# Arithmetic Asian
price_arith, se_arith = pricer.asian_arithmetic_mc(K, n_paths=50000)

# Geometric Asian (closed-form)
price_geom = pricer.asian_geometric_closed_form(K)

# Vanilla for comparison
vanilla_call = black_scholes(S0, K, r, T, sigma, 'call')

print(f"\nParameters: S=${S0}, K=${K}, r={r:.1%}, T={T}yr, σ={sigma:.1%}")

print(f"\nArithmetic Average Asian Call:")
print(f"  Price: ${price_arith:.4f} ± ${se_arith:.4f}")
print(f"  Discount vs Vanilla: {(vanilla_call - price_arith)/vanilla_call*100:.1f}%")

print(f"\nGeometric Average Asian Call:")
print(f"  Price: ${price_geom:.4f} (closed-form)")
print(f"  Discount vs Vanilla: {(vanilla_call - price_geom)/vanilla_call*100:.1f}%")

print(f"\nVanilla European Call: ${vanilla_call:.4f}")
print(f"\nAsian options cheaper due to reduced volatility from averaging")

# Scenario 2: Barrier options
print("\n" + "="*60)
print("SCENARIO 2: Barrier Options")
print("="*60)

barriers = [85, 90, 95]

print(f"\nDown-and-Out Call Options (K=${K}):")
print(f"{'Barrier':<12} {'Price':<12} {'Vanilla':<12} {'Discount %':<12}")
print("-" * 48)

for H in barriers:
    price_barrier, se_barrier = pricer.barrier_down_out_call(K, H, n_paths=50000)
    discount_pct = (vanilla_call - price_barrier) / vanilla_call * 100
    
    print(f"${H:<11} ${price_barrier:<11.4f} ${vanilla_call:<11.4f} {discount_pct:<11.1f}%")

print(f"\nLower barrier → higher knock-out probability → cheaper option")

# In-out parity check
H_test = 90
price_out, _ = pricer.barrier_down_out_call(K, H_test, n_paths=50000)

# Simulate down-and-in
payoffs_in = []
for _ in range(50000):
    path = pricer.generate_path(252)
    if np.min(path) <= H_test:  # Knocked in
        payoff = max(path[-1] - K, 0)
    else:
        payoff = 0
    payoffs_in.append(payoff)

price_in = np.exp(-r*T) * np.mean(payoffs_in)

print(f"\nIn-Out Parity Check (Barrier=${H_test}):")
print(f"  Down-and-Out: ${price_out:.4f}")
print(f"  Down-and-In: ${price_in:.4f}")
print(f"  Sum: ${price_out + price_in:.4f}")
print(f"  Vanilla: ${vanilla_call:.4f}")
print(f"  Difference: ${abs(price_out + price_in - vanilla_call):.4f}")

# Scenario 3: Lookback options
print("\n" + "="*60)
print("SCENARIO 3: Lookback Options")
print("="*60)

price_lookback, se_lookback = pricer.lookback_floating_call(n_paths=50000)

print(f"\nFloating Strike Lookback Call:")
print(f"  Payoff: S_T - min(S_t)")
print(f"  Price: ${price_lookback:.4f} ± ${se_lookback:.4f}")
print(f"  Vanilla Call: ${vanilla_call:.4f}")
print(f"  Premium: ${price_lookback - vanilla_call:.4f} ({(price_lookback/vanilla_call - 1)*100:.1f}%)")

print(f"\nLookback guarantees best execution → always ITM → expensive")

# Scenario 4: Digital options
print("\n" + "="*60)
print("SCENARIO 4: Digital (Binary) Options")
print("="*60)

strikes_digital = np.linspace(90, 110, 9)
cash_payoff = 10.0

print(f"\nCash-or-Nothing Digital Call (pays ${cash_payoff} if ITM):")
print(f"{'Strike':<12} {'Digital Price':<15} {'Risk-Neutral Prob':<20}")
print("-" * 47)

for K_dig in strikes_digital:
    price_dig = pricer.digital_call(K_dig, cash_payoff)
    prob = price_dig / (cash_payoff * np.exp(-r*T))
    
    if K_dig in [90, 100, 110]:
        print(f"${K_dig:<11} ${price_dig:<14.4f} {prob*100:<19.2f}%")

print(f"\nDigital price = Discounted probability × Cash payoff")

# Scenario 5: Multi-asset options
print("\n" + "="*60)
print("SCENARIO 5: Multi-Asset (Rainbow) Options")
print("="*60)

S0_multi = [100, 100, 100]
sigma_multi = [0.20, 0.25, 0.30]
corr_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
weights = np.array([0.4, 0.3, 0.3])

multi_pricer = MultiAssetExotics(S0_multi, r, sigma_multi, corr_matrix, T)

# Basket
price_basket, se_basket = multi_pricer.basket_option(K, weights, n_paths=50000)

# Best-of
price_best, se_best = multi_pricer.best_of_call(K, n_paths=50000)

# Worst-of
price_worst, se_worst = multi_pricer.worst_of_put(K, n_paths=50000)

# Spread
price_spread, se_spread = multi_pricer.spread_option(0, n_paths=50000)

# Individual calls for comparison
vanilla_calls_sum = sum([black_scholes(S0_multi[i], K, r, T, sigma_multi[i], 'call') 
                         for i in range(3)])

print(f"\n3-Asset Options (S=[{S0_multi[0]}, {S0_multi[1]}, {S0_multi[2]}]):")
print(f"Correlations: ρ₁₂={corr_matrix[0,1]:.1f}, ρ₁₃={corr_matrix[0,2]:.1f}, ρ₂₃={corr_matrix[1,2]:.1f}")

print(f"\nBasket Call (weights={weights}):")
print(f"  Price: ${price_basket:.4f} ± ${se_basket:.4f}")
print(f"  Sum of individual calls: ${vanilla_calls_sum:.4f}")
print(f"  Diversification benefit: ${vanilla_calls_sum - price_basket:.4f}")

print(f"\nBest-of Call (max of 3 assets):")
print(f"  Price: ${price_best:.4f} ± ${se_best:.4f}")
print(f"  Premium over single: {price_best/vanilla_call - 1:.1%}")

print(f"\nWorst-of Put (min of 3 assets):")
print(f"  Price: ${price_worst:.4f} ± ${se_worst:.4f}")

print(f"\nSpread Option (S₁ - S₂):")
print(f"  Price: ${price_spread:.4f} ± ${se_spread:.4f}")

# Scenario 6: Correlation impact on basket
print("\n" + "="*60)
print("SCENARIO 6: Correlation Impact on Basket Options")
print("="*60)

correlations_test = [0.0, 0.3, 0.6, 0.9]

print(f"\nBasket Call Sensitivity to Correlation:")
print(f"{'Correlation':<15} {'Price':<12} {'vs ρ=0':<15}")
print("-" * 42)

prices_by_corr = []

for rho in correlations_test:
    # Uniform correlation matrix
    corr_test = np.eye(3) + (1 - np.eye(3)) * rho
    
    pricer_corr = MultiAssetExotics(S0_multi, r, sigma_multi, corr_test, T)
    price_corr, _ = pricer_corr.basket_option(K, weights, n_paths=30000)
    prices_by_corr.append(price_corr)
    
    if rho == 0.0:
        base_price = price_corr
        diff_str = "baseline"
    else:
        diff_str = f"+${price_corr - base_price:.4f}"
    
    print(f"ρ={rho:<13.1f} ${price_corr:<11.4f} {diff_str:<15}")

print(f"\nHigher correlation → less diversification → higher basket value")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Sample paths with Asian averaging
ax = axes[0, 0]
n_sample_paths = 10
n_steps_viz = 252

for _ in range(n_sample_paths):
    path = pricer.generate_path(n_steps_viz)
    times = np.linspace(0, T, n_steps_viz + 1)
    ax.plot(times, path, 'b-', alpha=0.5, linewidth=1)
    
    # Show average
    avg = np.mean(path)
    ax.axhline(avg, color='r', linestyle='--', alpha=0.3, linewidth=1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Sample Paths (Asian Average in Red)')
ax.grid(alpha=0.3)

# Plot 2: Barrier knock-out illustration
ax = axes[0, 1]
np.random.seed(123)

for i in range(15):
    path = pricer.generate_path(n_steps_viz)
    times = np.linspace(0, T, n_steps_viz + 1)
    
    barrier = 90
    knocked = np.min(path) <= barrier
    color = 'red' if knocked else 'green'
    alpha = 0.3 if knocked else 0.7
    
    ax.plot(times, path, color=color, alpha=alpha, linewidth=1.5)

ax.axhline(barrier, color='black', linestyle='--', linewidth=2, label=f'Barrier ${barrier}')
ax.axhline(K, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Down-and-Out Paths (Red=Knocked Out)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Asian vs Vanilla price comparison
ax = axes[0, 2]
strikes_range = np.linspace(85, 115, 15)
asian_prices = []
vanilla_prices = []

for K_test in strikes_range:
    pricer_test = ExoticOptionPricer(S0, r, sigma, T)
    p_asian, _ = pricer_test.asian_arithmetic_mc(K_test, n_paths=10000, n_steps=100)
    p_vanilla = black_scholes(S0, K_test, r, T, sigma, 'call')
    asian_prices.append(p_asian)
    vanilla_prices.append(p_vanilla)

ax.plot(strikes_range, vanilla_prices, 'b-', linewidth=2.5, marker='o', label='Vanilla Call')
ax.plot(strikes_range, asian_prices, 'r-', linewidth=2.5, marker='s', label='Asian Call')
ax.axvline(S0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Strike')
ax.set_ylabel('Option Price')
ax.set_title('Asian vs Vanilla Call Prices')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Digital call delta profile
ax = axes[1, 0]
spots_digital = np.linspace(80, 120, 100)
digital_prices = []

for S_test in spots_digital:
    pricer_dig = ExoticOptionPricer(S_test, r, sigma, T)
    p_dig = pricer_dig.digital_call(K, cash_payoff=1.0)
    digital_prices.append(p_dig)

ax.plot(spots_digital, digital_prices, 'purple', linewidth=2.5)
ax.axvline(K, color='r', linestyle='--', linewidth=2, label=f'Strike ${K}')
ax.set_xlabel('Spot Price')
ax.set_ylabel('Digital Call Price')
ax.set_title('Digital Option: Discontinuous Payoff')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Lookback payoff distribution
ax = axes[1, 1]
lookback_payoffs = []

for _ in range(5000):
    path = pricer.generate_path(252)
    payoff = path[-1] - np.min(path)
    lookback_payoffs.append(payoff)

ax.hist(lookback_payoffs, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(np.mean(lookback_payoffs), color='r', linestyle='--', linewidth=2, 
           label=f'Mean: ${np.mean(lookback_payoffs):.2f}')
ax.set_xlabel('Payoff (S_T - min)')
ax.set_ylabel('Density')
ax.set_title('Lookback Option Payoff Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Basket price vs correlation
ax = axes[1, 2]
ax.plot(correlations_test, prices_by_corr, 'go-', linewidth=2.5, markersize=10)
ax.set_xlabel('Correlation')
ax.set_ylabel('Basket Option Price')
ax.set_title('Basket Call Price vs Correlation')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Parisian Barrier:** Implement Parisian option (barrier triggered only after continuous breach for time τ). How does window size τ affect price vs standard barrier?

2. **Cliquet Ratchet:** Price cliquet with annual resets, 100% participation, 0% floor, 10% cap per period (5 years). How sensitive to forward volatility term structure?

3. **Variance Swap Replication:** Replicate variance swap using strip of OTM calls and puts. Calculate fair strike. Compare to realized variance. Why difference?

4. **Himalaya Option:** Price 3-asset Himalaya (best performer removed each year). Use nested Monte Carlo for dynamic selection. Compare to sum of individual lookbacks.

5. **Smile Impact on Digitals:** Price digital call using flat vol vs volatility smile. How much difference? Explain via risk-neutral density impact.

## 7. Key References
- [Haug, The Complete Guide to Option Pricing Formulas (Part II)](https://www.mhprofessional.com/the-complete-guide-to-option-pricing-formulas-9780071389976-usa)
- [Gatheral, The Volatility Surface (Chapter 6 - Exotic Options)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [Wilmott, Paul Wilmott Introduces Quantitative Finance (Chapter 13)](https://www.wiley.com/en-us/Paul+Wilmott+Introduces+Quantitative+Finance-p-9780470319581)
- [Joshi, The Concepts and Practice of Mathematical Finance (Chapter 16)](https://www.cambridge.org/core/books/concepts-and-practice-of-mathematical-finance/2B5B6F1C2B3D0F8E5E7E7F8E9E9E9E9E)

---
**Status:** Advanced derivative structures | **Complements:** Monte Carlo, PDE Methods, Greeks, Volatility Surface, Multi-Asset Models
