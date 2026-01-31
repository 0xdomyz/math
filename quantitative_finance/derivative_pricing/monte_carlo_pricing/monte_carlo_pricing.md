# Monte Carlo Pricing

## 1. Concept Skeleton
**Definition:** Stochastic simulation method generating random price paths under risk-neutral measure to estimate option value via discounted expected payoff  
**Purpose:** Price complex path-dependent and multi-dimensional derivatives where closed-forms don't exist; handle exotic features like barriers, lookbacks, Asians  
**Prerequisites:** Risk-neutral valuation, stochastic calculus, geometric Brownian motion, variance reduction techniques, law of large numbers

## 2. Comparative Framing
| Method | Monte Carlo | Binomial Tree | Finite Difference | Closed-Form |
|--------|-------------|---------------|-------------------|-------------|
| **Dimensionality** | Excellent (100+ assets) | Poor (d>3 infeasible) | Poor (curse of dim.) | Limited |
| **Path-Dependent** | Natural | Difficult | Very difficult | Impossible |
| **Accuracy** | O(1/√n) | O(1/n) | O(h²) | Exact |
| **Speed** | Slow (parallel) | Fast (low-dim) | Fast (low-dim) | Instant |
| **American** | Difficult (LSM) | Natural | Natural | N/A |
| **Memory** | O(1) per path | O(n²) | O(n^d) | O(1) |

| Variance Reduction | Antithetic | Control Variate | Importance Sampling | Quasi-Random |
|-------------------|------------|-----------------|---------------------|--------------|
| **Complexity** | Trivial | Moderate | Hard | Easy |
| **Reduction** | 2x | 10-100x | 10-1000x | 10x |
| **Robustness** | High | Medium | Low | High |
| **Implementation** | Flip signs | Need benchmark | Tune distribution | Sobol sequence |

## 3. Examples + Counterexamples

**Simple Example:**  
European call: Simulate 10,000 paths, calculate max(S_T - K, 0) each path, average and discount → converges to Black-Scholes price.

**Perfect Fit:**  
Asian option (average price): Simulate path, compute arithmetic average S̄, payoff = max(S̄ - K, 0). Tree methods require non-recombining tree (2^n nodes) → infeasible.

**Multi-Asset:**  
Rainbow option on 20 stocks (best-of): Generate 20 correlated GBM paths, payoff = max(S₁_T, ..., S₂₀_T) - K. Monte Carlo handles easily, other methods fail.

**Convergence:**  
10² paths: High standard error ~$2. 10⁴ paths: SE ~$0.20. 10⁶ paths: SE ~$0.02. Need 100x more samples for 10x precision.

**Poor Fit:**  
American options: Early exercise decision at each time step requires backward induction. Standard MC goes forward only → need Longstaff-Schwartz regression (complex).

**Bermudan Option:**  
Exercise on specific dates (not continuous). Can use MC with regression at each exercise date, but binomial tree much simpler for few dates.

## 4. Layer Breakdown
```
Monte Carlo Pricing Framework:

├─ Basic Algorithm:
│  ├─ Step 1: Generate random paths under risk-neutral measure
│  │   For each simulation i = 1, ..., N:
│  │   ├─ Draw random numbers Z ~ N(0,1)
│  │   ├─ Evolve: S(t+Δt) = S(t) exp((r - 0.5σ²)Δt + σ√(Δt)Z)
│  │   └─ Store path: {S₀, S₁, ..., S_T}
│  ├─ Step 2: Calculate payoff for each path
│  │   V_i = Payoff(Path_i)
│  ├─ Step 3: Average discounted payoffs
│  │   V̂ = (1/N) Σᵢ e^(-rT) V_i
│  └─ Step 4: Calculate standard error
│      SE = σ̂/√N where σ̂ = sample std dev of payoffs
├─ Path Generation (GBM):
│  ├─ Continuous monitoring (discretized):
│  │   S_{t+Δt} = S_t exp((r - q - 0.5σ²)Δt + σ√Δt Z_t)
│  │   where Z_t ~ N(0,1) i.i.d.
│  ├─ Time steps: Choose Δt small enough
│  │   ├─ Barrier monitoring: Δt < 1/252 (daily or finer)
│  │   ├─ Asian averaging: Match averaging dates
│  │   └─ Trade-off: Accuracy vs computation time
│  ├─ Exact simulation (single time point):
│  │   S_T = S_0 exp((r - q - 0.5σ²)T + σ√T Z)
│  │   No discretization error
│  ├─ Milstein scheme (higher accuracy):
│  │   Includes second-order terms for better convergence
│  └─ Jump-diffusion:
│      Add Poisson jumps: dS = μS dt + σS dW + (J-1)S dN
├─ Multi-Asset Simulation:
│  ├─ Correlated paths:
│  │   Generate independent Z₁, ..., Z_d ~ N(0,1)
│  │   Apply Cholesky: Z_corr = L × Z
│  │   where L L^T = Correlation matrix Σ
│  ├─ Correlation matrix:
│  │   Must be positive semi-definite
│  │   Check eigenvalues ≥ 0
│  ├─ Copula approach:
│  │   Separate marginals from dependence structure
│  │   More flexible than Gaussian correlation
│  └─ Dimension: MC scales linearly in d
│      Unlike PDE/tree methods (exponential)
├─ Payoff Calculation:
│  ├─ European: Only terminal value matters
│  │   Call: max(S_T - K, 0)
│  │   Put: max(K - S_T, 0)
│  ├─ Path-Dependent:
│  │   ├─ Asian (arithmetic average):
│  │   │   Payoff = max((1/n)Σ S_tᵢ - K, 0)
│  │   ├─ Asian (geometric average):
│  │   │   Payoff = max(exp((1/n)Σ ln(S_tᵢ)) - K, 0)
│  │   ├─ Lookback (floating strike):
│  │   │   Payoff = S_T - min(S_t over [0,T])
│  │   ├─ Lookback (fixed strike):
│  │   │   Payoff = max(S_t over [0,T]) - K
│  │   ├─ Barrier (knock-out):
│  │   │   Payoff = max(S_T - K, 0) × 𝟙(S_t < H for all t)
│  │   ├─ Barrier (knock-in):
│  │   │   Payoff = max(S_T - K, 0) × 𝟙(S_t ≥ H for some t)
│  │   └─ Digital (binary):
│  │       Payoff = 𝟙(S_T > K) (0 or 1)
│  └─ Multi-asset:
│      ├─ Basket: max(Σ wᵢ Sᵢ_T - K, 0)
│      ├─ Best-of: max(max(S₁_T, ..., S_d_T) - K, 0)
│      └─ Worst-of: max(min(S₁_T, ..., S_d_T) - K, 0)
├─ Convergence & Error:
│  ├─ Central Limit Theorem:
│  │   V̂ → N(V_true, σ²/N) as N→∞
│  ├─ Standard error:
│  │   SE = σ̂/√N
│  │   95% CI: [V̂ - 1.96×SE, V̂ + 1.96×SE]
│  ├─ Convergence rate: O(1/√N)
│  │   Need 100x paths for 10x precision
│  │   Slow compared to deterministic methods
│  ├─ Bias vs variance:
│  │   ├─ Discretization bias: Δt too large
│  │   ├─ Statistical variance: N too small
│  │   └─ Optimal: Balance both errors
│  └─ Stopping criteria:
│      SE < desired tolerance or max iterations
├─ Variance Reduction Techniques:
│  ├─ Antithetic Variates:
│  │   ├─ For each Z, also simulate -Z
│  │   ├─ Paths are negatively correlated
│  │   ├─ Average reduces variance
│  │   ├─ Effective variance reduction: ~50%
│  │   └─ Cost: None (same # random numbers)
│  ├─ Control Variates:
│  │   ├─ Use correlated instrument with known price
│  │   │   V̂_CV = V̂ - β(Ĉ - C_true)
│  │   │   where C is control (e.g., European call)
│  │   ├─ Optimal β: Cov(V,C) / Var(C)
│  │   ├─ Variance reduction: Proportional to correlation²
│  │   ├─ Example: Use vanilla call to price exotic call
│  │   └─ Can combine multiple controls
│  ├─ Importance Sampling:
│  │   ├─ Change probability measure to focus on critical region
│  │   ├─ Example: OTM option → shift drift to make ITM more likely
│  │   ├─ Reweight: E[f(X)] = E_Q[f(X) × dP/dQ]
│  │   ├─ Radon-Nikodym derivative adjusts for measure change
│  │   ├─ Huge reduction for rare events (barriers, deep OTM)
│  │   └─ Difficult to tune (need domain knowledge)
│  ├─ Stratified Sampling:
│  │   ├─ Divide sample space into strata
│  │   ├─ Sample proportionally from each
│  │   ├─ Ensures coverage of entire range
│  │   └─ Reduces variance within strata
│  ├─ Quasi-Random (Low-Discrepancy) Sequences:
│  │   ├─ Sobol, Halton sequences: Fill space uniformly
│  │   ├─ Avoid clustering of random points
│  │   ├─ Convergence: O(log(N)^d / N) better than O(1/√N)
│  │   ├─ Effective for smooth payoffs
│  │   └─ Degrades for discontinuous payoffs (digitals)
│  └─ Moment Matching:
│      Force sample paths to match theoretical moments
│      Reduces bias from finite samples
├─ Greeks Calculation:
│  ├─ Finite Differences:
│  │   ├─ Delta: (V(S+ε) - V(S-ε)) / (2ε)
│  │   ├─ Gamma: (V(S+ε) - 2V(S) + V(S-ε)) / ε²
│  │   ├─ Requires multiple MC runs → expensive
│  │   ├─ High variance (ratio of noisy estimates)
│  │   └─ Use same random numbers (common random numbers)
│  ├─ Pathwise Method (Infinitesimal Perturbation):
│  │   ├─ Delta: E[∂Payoff/∂S₀] directly
│  │   ├─ Differentiate payoff along each path
│  │   ├─ Single MC run → efficient
│  │   ├─ Requires smooth payoff (fails for digitals)
│  │   └─ Example: Call delta = E[𝟙(S_T > K) × ∂S_T/∂S₀]
│  └─ Likelihood Ratio Method (Score Function):
│      ├─ Delta: E[Payoff × ∂ln(f)/∂S₀]
│      │   where f is path density
│      ├─ Works for discontinuous payoffs
│      ├─ Higher variance than pathwise
│      └─ Useful for digitals, barriers
├─ American Options (Longstaff-Schwartz):
│  ├─ Challenge: Need backward induction in MC
│  ├─ LSM Algorithm:
│  │   ├─ Step 1: Generate all paths forward
│  │   ├─ Step 2: Backward induction at exercise dates
│  │   │   At each date t and path i:
│  │   │   ├─ Intrinsic value: V_intrinsic = Payoff(S_t^i)
│  │   │   ├─ Continuation value: E[V_{t+1} | S_t] via regression
│  │   │   │   Regress future values on basis functions of S_t
│  │   │   │   φ(S) = [1, S, S², ..., polynomials, Laguerre, etc.]
│  │   │   └─ Optimal: Exercise if V_intrinsic > V_continuation
│  │   └─ Step 3: Average optimal exercise values
│  ├─ Regression basis:
│  │   ├─ Polynomials: 1, S, S², S³
│  │   ├─ Laguerre polynomials: Better conditioning
│  │   └─ Need enough terms but avoid overfitting
│  ├─ Only ITM paths:
│  │   Regression only on paths where intrinsic > 0
│  │   (OTM paths never exercise → no information)
│  └─ Convergence: Slower than European
│      Need more paths and time steps
├─ Advanced Topics:
│  ├─ Stochastic Volatility:
│  │   ├─ Heston model: dσ² = κ(θ - σ²)dt + ξσdW
│  │   ├─ Simulate both S and σ jointly
│  │   └─ Captures vol smile/skew
│  ├─ Jump-Diffusion:
│  │   ├─ Merton model: Add Poisson jumps to GBM
│  │   ├─ Simulate: Poisson arrivals + jump sizes
│  │   └─ Captures gap risk
│  ├─ Local Volatility:
│  │   σ(S,t) depends on spot and time
│  │   Calibrate to match implied vol surface
│  ├─ Multifactor Models:
│  │   Interest rate models (HJM, LMM)
│  │   Stochastic dividends, stochastic vol
│  └─ Parallel Computing:
│      ├─ Embarrassingly parallel: Each path independent
│      ├─ GPU acceleration: 100x+ speedup
│      └─ Distributed computing: Split paths across nodes
└─ Practical Considerations:
   ├─ Random Number Generation:
   │   ├─ Quality matters: Use Mersenne Twister or better
   │   ├─ Seed management: Reproducibility vs independence
   │   └─ Avoid periodicity: RNG period >> number of draws
   ├─ Performance Optimization:
   │   ├─ Vectorization: Batch path generation
   │   ├─ Memory: Stream paths (don't store all)
   │   ├─ Early termination: Adaptive # paths based on SE
   │   └─ Precomputation: Cholesky decomposition, constants
   ├─ Numerical Stability:
   │   ├─ Exponentiation: Use log-space for small probabilities
   │   ├─ Overflow: Cap extreme paths (rare in practice)
   │   └─ Underflow: Careful with barrier monitoring
   ├─ Validation:
   │   ├─ Compare to analytical (when available)
   │   ├─ Convergence plots: SE vs √N
   │   ├─ Greeks consistency: Put-call parity, bounds
   │   └─ Sensitivity to time steps, # paths
   └─ Production Implementation:
      ├─ Error handling: Invalid inputs, market data
      ├─ Diagnostics: Return SE, # paths used, convergence flag
      ├─ Caching: Reuse paths for Greeks
      └─ Monitoring: Track computation time, accuracy
```

**Interaction:** Random paths → Payoff calculation → Averaging → Discounting; variance reduction techniques dramatically improve efficiency without changing algorithm structure.

## 5. Mini-Project
Implement Monte Carlo pricer with variance reduction techniques:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("MONTE CARLO OPTION PRICING")
print("="*60)

class MonteCarloPricer:
    """Monte Carlo option pricing with variance reduction"""
    
    def __init__(self, S0, K, r, T, sigma, option_type='call', n_paths=10000, n_steps=252):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps
    
    def generate_paths(self, antithetic=False):
        """Generate stock price paths using GBM"""
        # Random normal draws
        if antithetic:
            Z = np.random.normal(0, 1, (self.n_paths // 2, self.n_steps))
            Z = np.vstack([Z, -Z])  # Antithetic pairs
        else:
            Z = np.random.normal(0, 1, (self.n_paths, self.n_steps))
        
        # Initialize paths
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate paths
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*self.dt + 
                self.sigma*np.sqrt(self.dt)*Z[:, t-1]
            )
        
        return paths
    
    def price_european(self, antithetic=False):
        """Price European option"""
        paths = self.generate_paths(antithetic)
        S_T = paths[:, -1]
        
        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        # Discount and average
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se, payoffs
    
    def price_asian(self, average_type='arithmetic', antithetic=False):
        """Price Asian option"""
        paths = self.generate_paths(antithetic)
        
        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        if self.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_barrier(self, barrier, barrier_type='down-and-out', antithetic=False):
        """Price barrier option"""
        paths = self.generate_paths(antithetic)
        S_T = paths[:, -1]
        
        # Check barrier condition
        if barrier_type == 'down-and-out':
            knocked = np.any(paths <= barrier, axis=1)
        elif barrier_type == 'up-and-out':
            knocked = np.any(paths >= barrier, axis=1)
        elif barrier_type == 'down-and-in':
            knocked = ~np.any(paths <= barrier, axis=1)
        else:  # up-and-in
            knocked = ~np.any(paths >= barrier, axis=1)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        # Apply barrier condition
        if 'out' in barrier_type:
            payoffs = payoffs * (~knocked)
        else:  # in
            payoffs = payoffs * knocked
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_with_control_variate(self):
        """Price using European option as control variate"""
        paths = self.generate_paths(antithetic=False)
        S_T = paths[:, -1]
        
        # Target: Asian option
        avg_prices = np.mean(paths, axis=1)
        asian_payoffs = np.maximum(avg_prices - self.K, 0) if self.option_type == 'call' \
                        else np.maximum(self.K - avg_prices, 0)
        
        # Control: European option (known analytical price)
        euro_payoffs = np.maximum(S_T - self.K, 0) if self.option_type == 'call' \
                       else np.maximum(self.K - S_T, 0)
        
        # Analytical European price
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if self.option_type == 'call':
            euro_analytical = self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            euro_analytical = self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S0*norm.cdf(-d1)
        
        # Optimal beta
        cov = np.cov(asian_payoffs, euro_payoffs)[0, 1]
        var = np.var(euro_payoffs)
        beta = cov / var if var > 0 else 0
        
        # Control variate estimate
        euro_mc = np.mean(euro_payoffs)
        asian_cv = asian_payoffs - beta * (euro_payoffs - euro_analytical * np.exp(self.r*self.T))
        
        price = np.exp(-self.r*self.T) * np.mean(asian_cv)
        se = np.exp(-self.r*self.T) * np.std(asian_cv) / np.sqrt(self.n_paths)
        
        # Variance reduction factor
        var_ratio = np.var(asian_cv) / np.var(asian_payoffs) if np.var(asian_payoffs) > 0 else 1
        
        return price, se, beta, var_ratio

class MultiAssetMC:
    """Multi-asset Monte Carlo pricing"""
    
    def __init__(self, S0, K, r, T, sigma, corr_matrix, n_paths=10000):
        self.S0 = np.array(S0)
        self.K = K
        self.r = r
        self.T = T
        self.sigma = np.array(sigma)
        self.corr_matrix = corr_matrix
        self.n_paths = n_paths
        self.n_assets = len(S0)
        
        # Cholesky decomposition for correlated paths
        self.chol = cholesky(corr_matrix, lower=True)
    
    def generate_correlated_paths(self):
        """Generate correlated asset paths"""
        # Independent normal draws
        Z = np.random.normal(0, 1, (self.n_paths, self.n_assets))
        
        # Correlate using Cholesky
        Z_corr = Z @ self.chol.T
        
        # Generate terminal prices
        S_T = self.S0 * np.exp(
            (self.r - 0.5*self.sigma**2)*self.T + 
            self.sigma*np.sqrt(self.T)*Z_corr
        )
        
        return S_T
    
    def price_basket(self, weights):
        """Price basket option"""
        S_T = self.generate_correlated_paths()
        
        # Basket value
        basket = S_T @ weights
        payoffs = np.maximum(basket - self.K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se
    
    def price_rainbow(self, option_type='best-of'):
        """Price rainbow option"""
        S_T = self.generate_correlated_paths()
        
        if option_type == 'best-of':
            extreme = np.max(S_T, axis=1)
        else:  # worst-of
            extreme = np.min(S_T, axis=1)
        
        payoffs = np.maximum(extreme - self.K, 0)
        
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        se = np.exp(-self.r*self.T) * np.std(payoffs) / np.sqrt(self.n_paths)
        
        return price, se

# Black-Scholes for comparison
def black_scholes(S, K, r, T, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Scenario 1: European option convergence
print("\n" + "="*60)
print("SCENARIO 1: European Option - Convergence Analysis")
print("="*60)

S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.2
bs_price = black_scholes(S0, K, r, T, sigma, 'call')

print(f"\nParameters: S=${S0}, K=${K}, r={r:.1%}, T={T}yr, σ={sigma:.1%}")
print(f"Black-Scholes Price: ${bs_price:.4f}")

path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
mc_prices = []
mc_errors = []
mc_ses = []

print(f"\n{'Paths':<10} {'MC Price':<12} {'Std Error':<12} {'95% CI':<25} {'BS Error':<12}")
print("-" * 81)

for n in path_counts:
    pricer = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=n, n_steps=1)
    price, se, _ = pricer.price_european()
    
    mc_prices.append(price)
    mc_ses.append(se)
    mc_errors.append(abs(price - bs_price))
    
    ci_lower = price - 1.96*se
    ci_upper = price + 1.96*se
    
    if n in [1000, 10000, 100000]:
        print(f"{n:<10} ${price:<11.4f} ${se:<11.4f} [{ci_lower:.4f}, {ci_upper:.4f}] ${abs(price - bs_price):<11.4f}")

# Scenario 2: Antithetic variates
print("\n" + "="*60)
print("SCENARIO 2: Variance Reduction - Antithetic Variates")
print("="*60)

n_paths = 10000
pricer = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=n_paths, n_steps=1)

# Standard MC
price_std, se_std, payoffs_std = pricer.price_european(antithetic=False)

# Antithetic MC
price_anti, se_anti, payoffs_anti = pricer.price_european(antithetic=True)

variance_reduction = (se_std**2) / (se_anti**2) if se_anti > 0 else 1

print(f"\n{n_paths:,} paths simulation:")
print(f"\nStandard Monte Carlo:")
print(f"  Price: ${price_std:.4f}")
print(f"  Std Error: ${se_std:.4f}")
print(f"  95% CI: [${price_std - 1.96*se_std:.4f}, ${price_std + 1.96*se_std:.4f}]")

print(f"\nAntithetic Variates:")
print(f"  Price: ${price_anti:.4f}")
print(f"  Std Error: ${se_anti:.4f}")
print(f"  95% CI: [${price_anti - 1.96*se_anti:.4f}, ${price_anti + 1.96*se_anti:.4f}]")

print(f"\nVariance Reduction Factor: {variance_reduction:.2f}x")
print(f"Equivalent to {int(n_paths * variance_reduction):,} standard MC paths")

# Scenario 3: Asian option pricing
print("\n" + "="*60)
print("SCENARIO 3: Path-Dependent - Asian Option")
print("="*60)

n_paths = 50000
pricer = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=n_paths, n_steps=252)

price_asian_arith, se_asian_arith = pricer.price_asian('arithmetic', antithetic=True)
price_asian_geom, se_asian_geom = pricer.price_asian('geometric', antithetic=True)

print(f"\nAsian Call Option ({n_paths:,} paths, K=${K}):")
print(f"\nArithmetic Average:")
print(f"  Price: ${price_asian_arith:.4f} ± ${se_asian_arith:.4f}")
print(f"  95% CI: [${price_asian_arith - 1.96*se_asian_arith:.4f}, ${price_asian_arith + 1.96*se_asian_arith:.4f}]")

print(f"\nGeometric Average:")
print(f"  Price: ${price_asian_geom:.4f} ± ${se_asian_geom:.4f}")
print(f"  95% CI: [${price_asian_geom - 1.96*se_asian_geom:.4f}, ${price_asian_geom + 1.96*se_asian_geom:.4f}]")

print(f"\nEuropean Call (for comparison): ${bs_price:.4f}")
print(f"Asian discount: {(bs_price - price_asian_arith)/bs_price*100:.1f}% (averaging reduces volatility)")

# Scenario 4: Control variates
print("\n" + "="*60)
print("SCENARIO 4: Control Variate Method")
print("="*60)

pricer_cv = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=10000, n_steps=252)

# Standard Asian pricing
price_std_asian, se_std_asian = pricer_cv.price_asian('arithmetic', antithetic=False)

# With control variate
price_cv, se_cv, beta, var_ratio = pricer_cv.price_with_control_variate()

print(f"\nAsian Call with Control Variate (European call as control):")
print(f"\nStandard MC:")
print(f"  Price: ${price_std_asian:.4f} ± ${se_std_asian:.4f}")

print(f"\nControl Variate MC:")
print(f"  Price: ${price_cv:.4f} ± ${se_cv:.4f}")
print(f"  Optimal β: {beta:.4f}")
print(f"  Variance Ratio: {var_ratio:.4f}")
print(f"  Variance Reduction: {1/var_ratio:.2f}x")
print(f"  Effective paths: {int(10000/var_ratio):,}")

# Scenario 5: Barrier options
print("\n" + "="*60)
print("SCENARIO 5: Barrier Options")
print("="*60)

barrier_down = 85
barrier_up = 115
n_paths_barrier = 50000

pricer_barrier = MonteCarloPricer(S0, K, r, T, sigma, 'call', 
                                  n_paths=n_paths_barrier, n_steps=252)

price_down_out, se_down_out = pricer_barrier.price_barrier(barrier_down, 'down-and-out', antithetic=True)
price_down_in, se_down_in = pricer_barrier.price_barrier(barrier_down, 'down-and-in', antithetic=True)
price_up_out, se_up_out = pricer_barrier.price_barrier(barrier_up, 'up-and-out', antithetic=True)

print(f"\nBarrier Options ({n_paths_barrier:,} paths, K=${K}):")
print(f"European Call: ${bs_price:.4f}")

print(f"\nDown-and-Out (Barrier=${barrier_down}):")
print(f"  Price: ${price_down_out:.4f} ± ${se_down_out:.4f}")
print(f"  Discount: {(bs_price - price_down_out)/bs_price*100:.1f}%")

print(f"\nDown-and-In (Barrier=${barrier_down}):")
print(f"  Price: ${price_down_in:.4f} ± ${se_down_in:.4f}")

print(f"\nSum (should ≈ European): ${price_down_out + price_down_in:.4f}")

print(f"\nUp-and-Out (Barrier=${barrier_up}):")
print(f"  Price: ${price_up_out:.4f} ± ${se_up_out:.4f}")
print(f"  Discount: {(bs_price - price_up_out)/bs_price*100:.1f}%")

# Scenario 6: Multi-asset options
print("\n" + "="*60)
print("SCENARIO 6: Multi-Asset Options")
print("="*60)

S0_multi = [100, 100, 100]
sigma_multi = [0.2, 0.25, 0.3]
corr_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])
weights = np.array([0.4, 0.3, 0.3])

multi_pricer = MultiAssetMC(S0_multi, K, r, T, sigma_multi, corr_matrix, n_paths=50000)

price_basket, se_basket = multi_pricer.price_basket(weights)
price_best, se_best = multi_pricer.price_rainbow('best-of')
price_worst, se_worst = multi_pricer.price_rainbow('worst-of')

print(f"\n3-Asset Options (K=${K}, T={T}yr):")
print(f"Assets: S₁=${S0_multi[0]}, S₂=${S0_multi[1]}, S₃=${S0_multi[2]}")
print(f"Vols: σ₁={sigma_multi[0]:.0%}, σ₂={sigma_multi[1]:.0%}, σ₃={sigma_multi[2]:.0%}")

print(f"\nBasket Call (weights: {weights}):")
print(f"  Price: ${price_basket:.4f} ± ${se_basket:.4f}")

print(f"\nBest-of Call (max of 3 assets):")
print(f"  Price: ${price_best:.4f} ± ${se_best:.4f}")

print(f"\nWorst-of Call (min of 3 assets):")
print(f"  Price: ${price_worst:.4f} ± ${se_worst:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Convergence to BS
ax = axes[0, 0]
ax.plot(path_counts, mc_prices, 'bo-', linewidth=2, markersize=8, label='MC Price')
ax.axhline(bs_price, color='r', linestyle='--', linewidth=2, label='Black-Scholes')
ax.fill_between(path_counts, 
                [p - 1.96*se for p, se in zip(mc_prices, mc_ses)],
                [p + 1.96*se for p, se in zip(mc_prices, mc_ses)],
                alpha=0.3, label='95% CI')
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Option Price')
ax.set_title('MC Convergence to Black-Scholes')
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Standard error vs paths
ax = axes[0, 1]
theoretical_se = [mc_ses[0] * np.sqrt(path_counts[0]/n) for n in path_counts]
ax.plot(path_counts, mc_ses, 'go-', linewidth=2, markersize=8, label='Actual SE')
ax.plot(path_counts, theoretical_se, 'r--', linewidth=2, label='O(1/√n)')
ax.set_xlabel('Number of Paths')
ax.set_ylabel('Standard Error')
ax.set_title('Standard Error Convergence')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Sample paths
ax = axes[0, 2]
sample_pricer = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=50, n_steps=252)
sample_paths = sample_pricer.generate_paths()
times = np.linspace(0, T, sample_pricer.n_steps + 1)

for i in range(min(20, sample_paths.shape[0])):
    ax.plot(times, sample_paths[i, :], alpha=0.5, linewidth=1)

ax.axhline(K, color='r', linestyle='--', linewidth=2, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Sample GBM Paths')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Payoff distribution
ax = axes[1, 0]
_, _, payoffs = pricer.price_european(antithetic=False)
ax.hist(payoffs, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(payoffs), color='r', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(payoffs):.2f}')
ax.set_xlabel('Payoff')
ax.set_ylabel('Density')
ax.set_title('Call Option Payoff Distribution')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Barrier option paths
ax = axes[1, 1]
barrier_pricer = MonteCarloPricer(S0, K, r, T, sigma, 'call', n_paths=50, n_steps=252)
barrier_paths = barrier_pricer.generate_paths()

for i in range(min(20, barrier_paths.shape[0])):
    knocked = np.any(barrier_paths[i, :] <= barrier_down)
    color = 'red' if knocked else 'green'
    alpha = 0.3 if knocked else 0.7
    ax.plot(times, barrier_paths[i, :], color=color, alpha=alpha, linewidth=1)

ax.axhline(barrier_down, color='black', linestyle='--', linewidth=2, label=f'Barrier ${barrier_down}')
ax.axhline(K, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Down-and-Out Barrier Paths')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Multi-asset correlation
ax = axes[1, 2]
sample_multi = MultiAssetMC(S0_multi, K, r, T, sigma_multi, corr_matrix, n_paths=1000)
S_T_multi = sample_multi.generate_correlated_paths()

ax.scatter(S_T_multi[:, 0], S_T_multi[:, 1], alpha=0.5, s=20)
ax.set_xlabel('Asset 1 Price')
ax.set_ylabel('Asset 2 Price')
ax.set_title(f'Terminal Prices (ρ={corr_matrix[0,1]:.1f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Sobol Sequences:** Implement quasi-random Monte Carlo using Sobol sequence. Compare convergence to pseudo-random. When does it outperform?

2. **Longstaff-Schwartz:** Implement American put pricing using LSM regression. Use Laguerre polynomial basis. Compare to binomial tree pricing.

3. **Heston Model:** Add stochastic volatility (Heston dynamics). Price European call and compare to BS. How does vol-of-vol affect price?

4. **Importance Sampling:** Price deep OTM option (K=150, S=100) with/without importance sampling. Shift drift to focus on ITM region. Quantify variance reduction.

5. **Greeks via Pathwise:** Implement pathwise delta calculation. Compare variance to finite difference method. Why does pathwise fail for digital options?

## 7. Key References
- [Glasserman, Monte Carlo Methods in Financial Engineering (Chapters 2-4)](https://www.springer.com/gp/book/9780387004518)
- [Longstaff & Schwartz (2001) - LSM for American Options](https://www.jstor.org/stable/222481)
- [Boyle (1977) - Options: A Monte Carlo Approach](https://www.jstor.org/stable/2978788)
- [Jäckel, Monte Carlo Methods in Finance (Chapters 5-7)](https://www.wiley.com/en-us/Monte+Carlo+Methods+in+Finance-p-9780471497417)

---
**Status:** Essential for exotic/multi-asset derivatives | **Complements:** Binomial Trees, Finite Difference, Risk-Neutral Valuation, Variance Reduction Techniques
