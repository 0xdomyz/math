# Volatility Surface and Skew

## 1. Concept Skeleton
**Definition:** Three-dimensional structure of implied volatility varying across strike and maturity; skew is asymmetric smile pattern reflecting non-lognormal returns and crash risk  
**Purpose:** Model market's volatility expectations across all option strikes/maturities; price exotic options consistently; capture tail risk and jump dynamics  
**Prerequisites:** Implied volatility calculation, option pricing, probability distributions, arbitrage-free constraints, local/stochastic volatility models

## 2. Comparative Framing
| Feature | Volatility Smile | Volatility Skew | Flat Vol Surface | Term Structure |
|---------|------------------|-----------------|------------------|----------------|
| **Shape** | U-shaped (symmetric) | Downward sloping | Constant across K | IV varies by T |
| **Market** | FX, commodities | Equity indices | Theory (BS) | All markets |
| **Cause** | Fat tails both sides | Leverage, crashes | Perfect model | Mean reversion |
| **Risk** | Straddle expensive | OTM puts pricey | No skew risk | Calendar spreads |

| Model | Local Volatility | Stochastic Volatility | Implied Vol | Jump-Diffusion |
|-------|------------------|----------------------|-------------|----------------|
| **Calibration** | Fit surface exactly | Approximate fit | Direct from market | Add jump terms |
| **Dynamics** | Deterministic σ(S,t) | Random volatility | Static snapshot | Discrete jumps |
| **Smile Dynamics** | Sticky strike | Sticky delta | No model | Mixed behavior |
| **Complexity** | Moderate | High | None (data) | Moderate |

## 3. Examples + Counterexamples

**Simple Example:**  
SPX options: 95% strike IV=22%, 100% ATM IV=18%, 105% strike IV=17%. Negative skew reflects left-tail crash fear premium.

**Perfect Fit:**  
Black Monday aftermath: OTM put IVs spike to 40% while ATM=25%. Market prices insurance against crashes → persistent downward skew pattern.

**Volatility Smile (FX):**  
EUR/USD: 90% strike IV=12%, 100% ATM IV=10%, 110% strike IV=12%. Symmetric smile reflects currency can spike either direction (risk-reversal patterns).

**Term Structure Interaction:**  
Front-month ATM IV=30% (earnings), 3-month IV=20%, 1-year IV=18%. Combine with strike skew → full 3D surface shows event risk fading over time.

**Arbitrage Violation:**  
Calendar spread: σ₁²T₁ > σ₂²T₂ with T₁ > T₂ → negative variance increment → arbitrage. Surface must respect increasing total variance.

**Poor Fit:**  
Using flat volatility for barrier options: Market skew means OTM barriers hit more frequently than BS predicts → significant mispricing (~10-30%).

## 4. Layer Breakdown
```
Volatility Surface Framework:

├─ Market Observation:
│  ├─ Raw Data: Option prices across strikes and maturities
│  ├─ Implied Vol Calculation: Invert BS for each (K, T) pair
│  ├─ Moneyness Measures:
│  │   ├─ Absolute: K (strike level)
│  │   ├─ Relative: K/S or K/F (moneyness ratio)
│  │   ├─ Log-moneyness: ln(K/S) or ln(K/F)
│  │   └─ Delta: Option's hedge ratio (standardized)
│  ├─ Data Quality Issues:
│  │   ├─ Illiquid strikes: Wide bid-ask, stale quotes
│  │   ├─ Missing points: Sparse data away from ATM
│  │   ├─ Outliers: Fat-finger trades, illiquidity
│  │   └─ Asynchronous quotes: Time stamps differ
│  └─ Preprocessing:
│      ├─ Filter by bid-ask spread < threshold
│      ├─ Remove arbitrage violations
│      ├─ Interpolate missing points
│      └─ Smooth outliers
├─ Volatility Patterns:
│  ├─ Volatility Smile (Symmetric):
│  │   ├─ Shape: U-shaped, minimum at ATM
│  │   ├─ Markets: FX, commodities, rates
│  │   ├─ Interpretation: Fat tails (kurtosis > 3)
│  │   │   Both large up and down moves more likely than BS
│  │   ├─ Cause: Jump risk, stochastic volatility
│  │   └─ Trading: Straddles expensive, butterflies cheap
│  ├─ Volatility Skew (Asymmetric):
│  │   ├─ Negative skew (equity):
│  │   │   ├─ OTM put IV > ATM > OTM call IV
│  │   │   ├─ Downward sloping to the right
│  │   │   ├─ Crash protection premium
│  │   │   └─ Leverage effect: Falling S → higher σ
│  │   ├─ Positive skew (rare):
│  │   │   ├─ OTM call IV > ATM
│  │   │   ├─ Upside tail risk
│  │   │   └─ Example: Takeover targets
│  │   └─ Quantification:
│  │       ├─ Skew = IV(90% strike) - IV(110% strike)
│  │       ├─ Risk reversal: IV(25Δ put) - IV(25Δ call)
│  │       └─ Slope: ∂IV/∂K (per strike unit)
│  ├─ Volatility Term Structure:
│  │   ├─ Upward sloping: σ(T₁) < σ(T₂) for T₁ < T₂
│  │   │   Mean reversion: Low vol expected to rise
│  │   ├─ Downward sloping: σ(T₁) > σ(T₂)
│  │   │   Event risk: Near-term uncertainty, long-term calm
│  │   ├─ Humped: Peak at intermediate maturity
│  │   │   Specific event (earnings, election) in near future
│  │   └─ VIX term structure:
│  │       VIX, VIX3M, VIX6M quotes show market's vol expectations
│  └─ Full Surface (3D):
│      IV = IV(K, T) varies across both dimensions
│      Combines smile/skew (strike) with term structure (time)
├─ Parametric Models (Smile Interpolation):
│  ├─ SVI (Stochastic Volatility Inspired):
│  │   ├─ Formula: σ²(k) = a + b[ρ(k-m) + √((k-m)² + ξ²)]
│  │   │   where k = ln(K/F), 5 parameters (a,b,ρ,m,ξ)
│  │   ├─ Advantages: Flexible, no arbitrage with constraints
│  │   ├─ Calibration: Minimize (Model_IV - Market_IV)²
│  │   ├─ Constraints: Ensure no butterfly arbitrage
│  │   │   ∂²σ²/∂k² ≥ -2 (density stays positive)
│  │   └─ Extensions: SSVI (surface SVI) for term structure
│  ├─ SABR (Stochastic Alpha Beta Rho):
│  │   ├─ Model: dF = α F^β dW₁, dα = ν α dW₂, Cov(dW₁,dW₂) = ρ dt
│  │   ├─ Approximation: Analytical formula for IV(K)
│  │   │   σ_SABR(K) = function of (α, β, ρ, ν, F, K)
│  │   ├─ Parameters:
│  │   │   ├─ α: ATM volatility level
│  │   │   ├─ β: Backbone (0=normal, 1=lognormal)
│  │   │   ├─ ρ: Correlation (skew direction)
│  │   │   └─ ν: Vol-of-vol (smile curvature)
│  │   ├─ Market standard: FX, rates (swaptions)
│  │   └─ Limitations: Approximation breaks for extreme strikes
│  ├─ Polynomial Fits:
│  │   ├─ Quadratic: σ(k) = a + bk + ck²
│  │   ├─ Simple but inflexible
│  │   └─ Risk: Can violate arbitrage away from fit points
│  └─ Cubic Splines:
│      ├─ Piecewise polynomials with smooth joins
│      ├─ Advantages: Flexible, smooth
│      ├─ Disadvantages: No arbitrage guarantee
│      └─ Need additional constraints (monotone, convex)
├─ Arbitrage-Free Constraints:
│  ├─ Static Arbitrage:
│  │   ├─ Call prices: C(K₁) ≥ C(K₂) for K₁ < K₂ (monotone)
│  │   ├─ Convexity: ∂²C/∂K² ≥ 0
│  │   │   Equivalent to: Risk-neutral density ≥ 0
│  │   ├─ Butterfly spread: C(K-δ) - 2C(K) + C(K+δ) ≥ 0
│  │   └─ In vol terms: Complex constraint on ∂²σ²/∂k²
│  ├─ Calendar Arbitrage:
│  │   ├─ Total variance increasing: σ₁²T₁ ≤ σ₂²T₂ for T₁ < T₂
│  │   ├─ Forward variance positive:
│  │   │   σ²_fwd = (σ₂²T₂ - σ₁²T₁) / (T₂ - T₁) ≥ 0
│  │   └─ Equivalently: ∂(σ²T)/∂T ≥ 0
│  ├─ Call Spread Arbitrage:
│  │   (C(K₁) - C(K₂))/(K₂ - K₁) should be in [0, 1]
│  ├─ Put-Call Parity:
│  │   C - P = F e^(-rT) - K e^(-rT)
│  │   Ensures call IV = put IV at same strike
│  └─ Detection:
│      ├─ Numerical checks on fitted surface
│      ├─ Perturb surface, check arbitrage appears
│      └─ Use optimization constraints during calibration
├─ Surface Dynamics (How Surface Evolves):
│  ├─ Sticky Strike:
│  │   ├─ IV stays at strike level K
│  │   ├─ If spot moves, IV(K) unchanged
│  │   ├─ Used for: P&L attribution, scenario analysis
│  │   └─ Observed: Short-term moves, post-event
│  ├─ Sticky Delta:
│  │   ├─ IV stays at delta level
│  │   ├─ If spot moves, IV moves with option's new delta
│  │   ├─ Used for: Hedging, vega bucketing
│  │   └─ Observed: Medium-term, normal market conditions
│  ├─ Sticky Moneyness:
│  │   ├─ IV stays at K/S ratio
│  │   ├─ Hybrid between strike and delta
│  │   └─ Observed: Long-term, structural changes
│  ├─ Reality: Combination of all three
│  │   ├─ Short-term: More sticky strike
│  │   ├─ Medium-term: Mix of delta and strike
│  │   └─ Shocks: Can reset entire surface
│  └─ Vanna-Volga:
│      Cross-sensitivity: ∂Δ/∂σ captures surface dynamics
│      Important for hedging skew risk
├─ Local Volatility Model:
│  ├─ Dupire's Formula:
│  │   σ_local²(K,T) = [∂C/∂T + rK∂C/∂K] / [½K²∂²C/∂K²]
│  │   Extract local vol from option prices
│  ├─ Properties:
│  │   ├─ Fits any arbitrage-free surface exactly
│  │   ├─ Deterministic: σ = σ(S,t)
│  │   ├─ Forward smile: Generated by spot moves and local vol
│  │   └─ Implementation: Forward PDE or Monte Carlo
│  ├─ Limitations:
│  │   ├─ Sticky strike dynamics (unrealistic)
│  │   ├─ Forward smile too flat
│  │   ├─ Poor for exotics with vol exposure
│  │   └─ Calibration instability (numerical derivatives)
│  └─ Uses:
│      Barrier options, lookbacks, any path-dependent
│      Better than flat vol, worse than stochastic vol
├─ Stochastic Volatility Models:
│  ├─ Heston Model:
│  │   ├─ Dynamics:
│  │   │   dS = μS dt + √v S dW₁
│  │   │   dv = κ(θ - v)dt + ξ√v dW₂
│  │   │   Cov(dW₁, dW₂) = ρ dt
│  │   ├─ Parameters:
│  │   │   ├─ v₀: Initial variance
│  │   │   ├─ θ: Long-run variance (mean reversion level)
│  │   │   ├─ κ: Mean reversion speed
│  │   │   ├─ ξ: Vol-of-vol
│  │   │   └─ ρ: Spot-vol correlation (skew)
│  │   ├─ Calibration: Fit to option prices across strikes/maturities
│  │   ├─ Smile: Negative ρ creates skew, ξ creates curvature
│  │   └─ Forward smile: More realistic than local vol
│  ├─ SABR Model:
│  │   Already described above
│  │   Used directly for quoting (FX markets)
│  └─ Advantages:
│      ├─ Captures smile dynamics better
│      ├─ Vega risk more realistic
│      └─ Better for exotic options with vol exposure
├─ Market Conventions:
│  ├─ Quoting by Delta:
│  │   ├─ "25-delta put" refers to put with Δ=-0.25
│  │   ├─ Standardized across strikes/spots
│  │   ├─ Common: 10Δ, 25Δ, 50Δ (ATM)
│  │   └─ Risk-reversal: 25Δ call - 25Δ put (skew measure)
│  ├─ Butterfly (Smile Curvature):
│  │   Butterfly = (25Δ call + 25Δ put)/2 - 50Δ straddle
│  │   Measures smile width/convexity
│  ├─ ATM Definition:
│  │   ├─ ATM strike: K = S (spot)
│  │   ├─ ATM forward: K = F = S e^(rT) (forward)
│  │   ├─ ATM delta: K where Δ = 0.5 (delta-neutral)
│  │   └─ Market convention varies (FX vs equity)
│  └─ Variance Swap Strike:
│      Fair variance = ∫ IV(K)² × weight(K) dK
│      Model-free measure of expected variance
├─ Practical Applications:
│  ├─ Exotic Option Pricing:
│  │   ├─ Use calibrated surface, not flat vol
│  │   ├─ Local vol or stochastic vol model
│  │   └─ Critical for barriers, digitals, lookbacks
│  ├─ Risk Management:
│  │   ├─ Vega by strike: Greeks at each vol point
│  │   ├─ Skew risk: Exposure to skew steepening
│  │   ├─ Surface risk: Parallel shift vs twist vs skew change
│  │   └─ Scenario analysis: Shock different surface regions
│  ├─ Trading Strategies:
│  │   ├─ Skew trades: Buy low IV strikes, sell high IV
│  │   ├─ Calendar spreads: Term structure steepening/flattening
│  │   ├─ Butterfly spreads: Smile width expansion/contraction
│  │   └─ Dispersion: Trade realized vs implied correlation
│  ├─ Model Validation:
│  │   ├─ Mark-to-market: Reprice portfolio with new surface
│  │   ├─ P&L explain: Decompose into spot, vol, skew changes
│  │   └─ Backtesting: Historical surface accuracy
│  └─ Hedging:
│      Dynamic hedging considers surface moves, not just ATM vol
└─ Advanced Topics:
   ├─ Jump-Diffusion Models:
   │   Add discrete jumps to capture gap risk
   │   Merton, Kou models
   ├─ Rough Volatility:
   │   Fractional Brownian motion (H < 0.5)
   │   Better fits to high-frequency vol dynamics
   ├─ Smile Extrapolation:
   │   Far OTM/ITM wings behavior
   │   Power-law tails, exponential decay
   ├─ Multi-Asset Surfaces:
   │   Correlation surface: Implied correlations
   │   Basket options require vol surface + correlation
   └─ Machine Learning:
      Neural networks to interpolate/extrapolate surface
      Enforce arbitrage via constraints or loss function
```

**Interaction:** Market prices → IV extraction → Surface construction → Arbitrage checks → Model calibration → Exotic pricing; skew reflects non-BS dynamics and must be modeled consistently.

## 5. Mini-Project
Implement volatility surface construction and analysis:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, least_squares
from scipy.interpolate import CubicSpline, griddata
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("VOLATILITY SURFACE AND SKEW ANALYSIS")
print("="*60)

class VolatilitySurface:
    """Volatility surface construction and analysis"""
    
    def __init__(self, S0, r=0.05):
        self.S0 = S0
        self.r = r
        self.market_data = {}
    
    def add_market_data(self, K, T, market_price, option_type='call'):
        """Add market option data"""
        if T not in self.market_data:
            self.market_data[T] = []
        self.market_data[T].append({
            'K': K, 'price': market_price, 'type': option_type
        })
    
    def implied_vol_newton(self, K, T, market_price, option_type='call', 
                          initial_guess=0.2, tol=1e-6, max_iter=100):
        """Calculate implied volatility using Newton-Raphson"""
        sigma = initial_guess
        
        for i in range(max_iter):
            # BS price and vega
            d1 = (np.log(self.S0/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = self.S0*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-self.r*T)*norm.cdf(-d2) - self.S0*norm.cdf(-d1)
            
            vega = self.S0 * norm.pdf(d1) * np.sqrt(T)
            
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            
            if vega < 1e-10:
                return np.nan
            
            sigma = sigma - diff / vega
            sigma = max(0.001, min(sigma, 5.0))
        
        return np.nan
    
    def build_iv_surface(self):
        """Build IV surface from market data"""
        surface = {}
        
        for T in sorted(self.market_data.keys()):
            surface[T] = []
            for option in self.market_data[T]:
                iv = self.implied_vol_newton(option['K'], T, option['price'], option['type'])
                if not np.isnan(iv):
                    moneyness = option['K'] / self.S0
                    log_moneyness = np.log(moneyness)
                    surface[T].append({
                        'K': option['K'],
                        'moneyness': moneyness,
                        'log_moneyness': log_moneyness,
                        'iv': iv
                    })
        
        return surface
    
    def check_butterfly_arbitrage(self, surface, T):
        """Check for butterfly arbitrage violations"""
        if T not in surface or len(surface[T]) < 3:
            return True, []
        
        violations = []
        points = sorted(surface[T], key=lambda x: x['K'])
        
        for i in range(1, len(points) - 1):
            K_low = points[i-1]['K']
            K_mid = points[i]['K']
            K_high = points[i+1]['K']
            
            # Approximate butterfly value from IVs
            # Should be non-negative for no arbitrage
            iv_low = points[i-1]['iv']
            iv_mid = points[i]['iv']
            iv_high = points[i+1]['iv']
            
            # Simplified check: convexity in variance
            w1 = (K_high - K_mid) / (K_high - K_low)
            w2 = (K_mid - K_low) / (K_high - K_low)
            
            iv_interp = w1 * iv_low + w2 * iv_high
            
            # If actual IV significantly above interpolated, may violate
            if iv_mid < iv_interp * 0.8:  # Threshold
                violations.append((K_mid, iv_mid, iv_interp))
        
        return len(violations) == 0, violations
    
    def check_calendar_arbitrage(self, surface):
        """Check for calendar arbitrage (variance must increase)"""
        maturities = sorted(surface.keys())
        violations = []
        
        if len(maturities) < 2:
            return True, []
        
        # Check at ATM
        atm_variances = []
        for T in maturities:
            points = surface[T]
            # Find closest to ATM
            atm_point = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
            total_var = atm_point['iv']**2 * T
            atm_variances.append((T, total_var))
        
        for i in range(1, len(atm_variances)):
            if atm_variances[i][1] < atm_variances[i-1][1]:
                violations.append((atm_variances[i-1][0], atm_variances[i][0]))
        
        return len(violations) == 0, violations

class SVIModel:
    """SVI (Stochastic Volatility Inspired) volatility smile model"""
    
    @staticmethod
    def svi_variance(k, a, b, rho, m, sigma):
        """SVI raw parameterization: total variance as function of log-moneyness"""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def svi_implied_vol(k, T, a, b, rho, m, sigma):
        """Convert SVI variance to implied volatility"""
        var = SVIModel.svi_variance(k, a, b, rho, m, sigma)
        return np.sqrt(var / T) if var > 0 else 0.01
    
    @staticmethod
    def calibrate(log_moneyness, ivs, T, initial_guess=None):
        """Calibrate SVI to market IVs"""
        if initial_guess is None:
            # Initial guess
            a_init = np.mean(ivs)**2 * T
            b_init = 0.1
            rho_init = -0.3
            m_init = 0.0
            sigma_init = 0.2
            initial_guess = [a_init, b_init, rho_init, m_init, sigma_init]
        
        target_variances = ivs**2 * T
        
        def objective(params):
            a, b, rho, m, sigma = params
            model_vars = np.array([SVIModel.svi_variance(k, a, b, rho, m, sigma) 
                                   for k in log_moneyness])
            return np.sum((model_vars - target_variances)**2)
        
        # Constraints for no arbitrage (simplified)
        bounds = [
            (0.001, None),      # a > 0
            (0.001, None),      # b > 0
            (-0.999, 0.999),    # -1 < rho < 1
            (None, None),       # m
            (0.001, None)       # sigma > 0
        ]
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            return result.x, result.fun
        else:
            return initial_guess, np.inf

# Generate synthetic market with skew
def generate_skewed_market(S0, K_range, T_range, base_vol=0.20, skew=-0.15, curve=0.05):
    """Generate synthetic option prices with volatility skew"""
    market = []
    r = 0.05
    
    for T in T_range:
        for K in K_range:
            log_moneyness = np.log(K/S0)
            
            # Stylized skew: negative slope + some curvature
            # Term structure: slightly decreasing
            term_adj = 1.0 - 0.1 * (1 - np.exp(-T))
            iv = (base_vol + skew * log_moneyness + curve * log_moneyness**2) * term_adj
            iv = max(iv, 0.05)  # Floor
            
            # Calculate BS price
            d1 = (np.log(S0/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
            d2 = d1 - iv*np.sqrt(T)
            call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            
            market.append({
                'K': K, 'T': T, 'price': call_price, 
                'true_iv': iv, 'type': 'call'
            })
    
    return market

# Scenario 1: Build volatility surface from market data
print("\n" + "="*60)
print("SCENARIO 1: Volatility Surface Construction")
print("="*60)

S0 = 100
r = 0.05

# Generate market data with skew
strikes = np.linspace(85, 115, 13)
maturities = [0.25, 0.5, 1.0]

market_data = generate_skewed_market(S0, strikes, maturities)

# Build surface
vol_surface = VolatilitySurface(S0, r)

for data in market_data:
    vol_surface.add_market_data(data['K'], data['T'], data['price'], data['type'])

iv_surface = vol_surface.build_iv_surface()

print(f"\nMarket: S=${S0}, r={r:.1%}")
print(f"Strikes: {len(strikes)} from ${strikes[0]:.0f} to ${strikes[-1]:.0f}")
print(f"Maturities: {maturities}")

for T in sorted(iv_surface.keys()):
    print(f"\nMaturity T={T}yr:")
    print(f"{'Strike':<10} {'Moneyness':<12} {'Impl Vol':<12} {'True IV':<12}")
    print("-" * 46)
    
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    for i, point in enumerate(points):
        if i % 3 == 0:  # Show every 3rd point
            true_iv = next(d['true_iv'] for d in market_data 
                          if d['K'] == point['K'] and d['T'] == T)
            print(f"${point['K']:<9.0f} {point['moneyness']:<11.3f} "
                  f"{point['iv']*100:<11.2f}% {true_iv*100:<11.2f}%")

# Scenario 2: Analyze skew
print("\n" + "="*60)
print("SCENARIO 2: Volatility Skew Analysis")
print("="*60)

for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    
    # Find specific moneyness points
    otm_put = next((p for p in points if p['moneyness'] < 0.95), points[0])
    atm = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
    otm_call = next((p for p in reversed(points) if p['moneyness'] > 1.05), points[-1])
    
    skew_measure = otm_put['iv'] - otm_call['iv']
    slope = (points[-1]['iv'] - points[0]['iv']) / (points[-1]['log_moneyness'] - points[0]['log_moneyness'])
    
    print(f"\nMaturity T={T}yr:")
    print(f"  OTM Put (K=${otm_put['K']:.0f}): IV={otm_put['iv']*100:.2f}%")
    print(f"  ATM (K=${atm['K']:.0f}): IV={atm['iv']*100:.2f}%")
    print(f"  OTM Call (K=${otm_call['K']:.0f}): IV={otm_call['iv']*100:.2f}%")
    print(f"  Skew (Put-Call): {skew_measure*100:.2f}%")
    print(f"  Slope (∂IV/∂ln(K)): {slope:.4f}")

# Scenario 3: Arbitrage checks
print("\n" + "="*60)
print("SCENARIO 3: Arbitrage-Free Constraints")
print("="*60)

for T in sorted(iv_surface.keys()):
    butterfly_ok, butterfly_viol = vol_surface.check_butterfly_arbitrage(iv_surface, T)
    
    print(f"\nMaturity T={T}yr:")
    if butterfly_ok:
        print(f"  ✓ No butterfly arbitrage detected")
    else:
        print(f"  ✗ Butterfly violations: {len(butterfly_viol)} points")
        for K, iv_actual, iv_expected in butterfly_viol[:3]:
            print(f"    K=${K:.0f}: IV={iv_actual*100:.2f}% vs expected {iv_expected*100:.2f}%")

calendar_ok, calendar_viol = vol_surface.check_calendar_arbitrage(iv_surface)
print(f"\nCalendar Arbitrage:")
if calendar_ok:
    print(f"  ✓ Total variance increasing with maturity")
else:
    print(f"  ✗ Calendar violations between maturities:")
    for T1, T2 in calendar_viol:
        print(f"    T={T1}yr → T={T2}yr")

# Scenario 4: SVI model calibration
print("\n" + "="*60)
print("SCENARIO 4: SVI Model Calibration")
print("="*60)

for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    
    log_moneyness = np.array([p['log_moneyness'] for p in points])
    ivs = np.array([p['iv'] for p in points])
    
    params, error = SVIModel.calibrate(log_moneyness, ivs, T)
    a, b, rho, m, sigma = params
    
    # Calculate fitted IVs
    fitted_ivs = [SVIModel.svi_implied_vol(k, T, *params) for k in log_moneyness]
    rmse = np.sqrt(np.mean((np.array(fitted_ivs) - ivs)**2))
    
    print(f"\nMaturity T={T}yr:")
    print(f"  SVI Parameters:")
    print(f"    a={a:.6f}, b={b:.6f}, ρ={rho:.4f}, m={m:.4f}, σ={sigma:.4f}")
    print(f"  Fit Quality:")
    print(f"    RMSE: {rmse*10000:.2f} bps")
    print(f"    Max Error: {max(abs(np.array(fitted_ivs) - ivs))*10000:.2f} bps")

# Scenario 5: Term structure analysis
print("\n" + "="*60)
print("SCENARIO 5: Volatility Term Structure")
print("="*60)

atm_term_structure = []

for T in sorted(iv_surface.keys()):
    points = iv_surface[T]
    atm = min(points, key=lambda x: abs(x['moneyness'] - 1.0))
    atm_term_structure.append((T, atm['iv'], atm['iv']**2 * T))

print(f"\nATM Implied Volatility Term Structure:")
print(f"{'Maturity':<12} {'IV':<12} {'Total Var':<15} {'Fwd Var':<15}")
print("-" * 54)

for i, (T, iv, total_var) in enumerate(atm_term_structure):
    if i == 0:
        fwd_var_str = "N/A"
    else:
        T_prev, _, total_var_prev = atm_term_structure[i-1]
        fwd_var = (total_var - total_var_prev) / (T - T_prev)
        fwd_vol = np.sqrt(fwd_var)
        fwd_var_str = f"{fwd_vol*100:.2f}%"
    
    print(f"{T:<11.2f}yr {iv*100:<11.2f}% {total_var:<14.6f} {fwd_var_str:<15}")

# Check if term structure is upward/downward sloping
if len(atm_term_structure) >= 2:
    if atm_term_structure[-1][1] > atm_term_structure[0][1]:
        print(f"\nTerm structure: Upward sloping (mean reversion expected)")
    elif atm_term_structure[-1][1] < atm_term_structure[0][1]:
        print(f"\nTerm structure: Downward sloping (event risk near-term)")
    else:
        print(f"\nTerm structure: Flat")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Volatility smile by maturity
ax = axes[0, 0]
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['K'])
    strikes_plot = [p['K'] for p in points]
    ivs_plot = [p['iv']*100 for p in points]
    ax.plot(strikes_plot, ivs_plot, 'o-', linewidth=2.5, markersize=8, label=f'T={T}yr')

ax.axvline(S0, color='k', linestyle='--', alpha=0.3, label='ATM')
ax.set_xlabel('Strike')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile Across Maturities')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Smile by moneyness (normalized)
ax = axes[0, 1]
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['moneyness'])
    moneyness_plot = [p['moneyness'] for p in points]
    ivs_plot = [p['iv']*100 for p in points]
    ax.plot(moneyness_plot, ivs_plot, 'o-', linewidth=2.5, markersize=8, label=f'T={T}yr')

ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, label='ATM')
ax.set_xlabel('Moneyness (K/S)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title('Volatility Smile by Moneyness')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Term structure (ATM)
ax = axes[0, 2]
T_vals = [t[0] for t in atm_term_structure]
iv_vals = [t[1]*100 for t in atm_term_structure]
ax.plot(T_vals, iv_vals, 'ro-', linewidth=2.5, markersize=10)
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('ATM Implied Volatility (%)')
ax.set_title('ATM Volatility Term Structure')
ax.grid(alpha=0.3)

# Plot 4: 3D Surface
ax = axes[1, 0] = plt.subplot(2, 3, 4, projection='3d')
all_strikes = []
all_maturities = []
all_ivs = []

for T in sorted(iv_surface.keys()):
    for point in iv_surface[T]:
        all_strikes.append(point['K'])
        all_maturities.append(T)
        all_ivs.append(point['iv']*100)

ax.scatter(all_strikes, all_maturities, all_ivs, c=all_ivs, cmap='viridis', s=50)
ax.set_xlabel('Strike')
ax.set_ylabel('Maturity')
ax.set_zlabel('IV (%)')
ax.set_title('3D Volatility Surface')

# Plot 5: SVI fit for one maturity
ax = axes[1, 1]
T_fit = maturities[1]  # Middle maturity
points = sorted(iv_surface[T_fit], key=lambda x: x['K'])
log_moneyness_fit = np.array([p['log_moneyness'] for p in points])
ivs_market = np.array([p['iv']*100 for p in points])

# Calibrate SVI
params_fit, _ = SVIModel.calibrate(log_moneyness_fit, ivs_market/100, T_fit)
log_k_fine = np.linspace(log_moneyness_fit.min(), log_moneyness_fit.max(), 100)
ivs_svi = [SVIModel.svi_implied_vol(k, T_fit, *params_fit)*100 for k in log_k_fine]

ax.plot(log_moneyness_fit, ivs_market, 'ro', markersize=10, label='Market')
ax.plot(log_k_fine, ivs_svi, 'b-', linewidth=2.5, label='SVI Fit')
ax.set_xlabel('Log-Moneyness ln(K/S)')
ax.set_ylabel('Implied Volatility (%)')
ax.set_title(f'SVI Model Fit (T={T_fit}yr)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Skew across maturities
ax = axes[1, 2]
skew_values = []
for T in sorted(iv_surface.keys()):
    points = sorted(iv_surface[T], key=lambda x: x['log_moneyness'])
    log_m = [p['log_moneyness'] for p in points]
    ivs = [p['iv'] for p in points]
    
    # Linear fit to get slope
    slope = np.polyfit(log_m, ivs, 1)[0]
    skew_values.append((T, slope))

T_skew = [s[0] for s in skew_values]
slopes = [s[1] for s in skew_values]

ax.plot(T_skew, slopes, 'mo-', linewidth=2.5, markersize=10)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Maturity (years)')
ax.set_ylabel('Skew Slope (∂IV/∂ln(K))')
ax.set_title('Skew Evolution with Maturity')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **SSVI Calibration:** Implement Surface SVI (SSVI) for full surface parameterization. Ensure calendar arbitrage-free across all maturities. How many parameters needed?

2. **Local Vol Extraction:** Use Dupire's formula to extract local volatility σ_local(K,T) from implied vol surface. Compare forward smile to market. Why does it flatten?

3. **Sticky Delta Simulation:** Simulate spot move +10%. Update surface using sticky delta rule. Recalculate portfolio Greeks. How much does vega P&L differ from sticky strike?

4. **Arbitrage Detection:** Create surface with deliberate butterfly violation. Write algorithm to detect and fix (minimal perturbation). Use quadratic programming?

5. **Variance Swap:** Price variance swap using strip of options across strikes. Compare to ATM vol. Why is variance swap strike higher than ATM²?

## 7. Key References
- [Gatheral, The Volatility Surface (Chapters 3-5)](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792529)
- [Dupire (1994) - Pricing with a Smile](https://www.sciencedirect.com/science/article/abs/pii/0165188994900201)
- [Hagan et al (2002) - SABR Model](https://www.researchgate.net/publication/235622441_Managing_Smile_Risk)
- [Gatheral & Jacquier (2014) - Arbitrage-Free SVI](https://arxiv.org/abs/1204.0646)

---
**Status:** Core market microstructure | **Complements:** Implied Volatility, Local Volatility, Stochastic Volatility, Greeks, Exotic Options
