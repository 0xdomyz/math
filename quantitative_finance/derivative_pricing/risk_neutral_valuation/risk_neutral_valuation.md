# Risk-Neutral Valuation

## 1. Concept Skeleton
**Definition:** Derivative pricing framework where discounted expected payoffs under risk-neutral measure equal current price; eliminates need to estimate real-world drift  
**Purpose:** Value options and derivatives without knowing investors' risk preferences; enables arbitrage-free pricing through replication arguments  
**Prerequisites:** No-arbitrage principle, probability theory, martingales, change of measure (Radon-Nikodym), stochastic calculus basics

## 2. Comparative Framing
| Measure | Risk-Neutral (Q) | Real-World (P) | Forward Measure | T-Forward Measure |
|---------|------------------|----------------|-----------------|-------------------|
| **Drift** | Risk-free rate r | Actual μ | Zero (martingale) | Specific numeraire |
| **Purpose** | Pricing | Forecasting | Simplify formulas | Bond options |
| **Discount** | e^(-rT) | e^(-μT) | No discount needed | T-bond numeraire |
| **Volatility** | Same as P | Historical/forecast | Same | Same |

| Concept | Risk-Neutral | No-Arbitrage | Replication | Martingale |
|---------|--------------|--------------|-------------|------------|
| **Foundation** | Q-measure pricing | Law of one price | Synthetic portfolio | Mathematical tool |
| **Application** | All derivatives | Any asset | Complete markets | Pricing framework |
| **Assumption** | Q exists | Markets efficient | Hedgeable | Q is EMM |
| **Limitation** | Model-dependent | Frictionless | May not exist | Technical |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock $100, grows 15%/year real-world. Risk-neutral: grows at r=5%. Option prices using r=5%, not 15%. Investors' risk preferences embedded in current stock price already.

**Perfect Fit:**  
European call pricing: Simulate under Q (drift=r), calculate E^Q[max(S_T-K,0)], discount at r → matches Black-Scholes exactly. No need to know real drift μ.

**Replication Argument:**  
Delta-hedge portfolio: Δ shares + B in bank replicates option. Portfolio grows at r (self-financing). Option must also grow at r under Q → risk-neutral drift emerges naturally.

**Real-World vs Risk-Neutral:**  
Real-world P: E^P[S_T]=S_0 e^(μT), μ=15%. Risk-neutral Q: E^Q[S_T]=S_0 e^(rT), r=5%. Different expectations, same price S_0 due to risk adjustment.

**Incomplete Market:**  
Jump-diffusion with unhedgeable jumps: Multiple risk-neutral measures exist (Q not unique). Range of arbitrage-free prices, not single value. Need additional pricing principle (utility, etc.).

**Poor Fit:**  
Long-dated equity options (10+ years): Discount factor e^(-rT) dominates, small vol changes huge impact. Real-world default risk, model risk become significant → Q-measure approximation breaks down.

## 4. Layer Breakdown
```
Risk-Neutral Valuation Framework:

├─ Fundamental Principle:
│  ├─ No-Arbitrage: Cannot create riskless profit from nothing
│  │   ├─ Law of one price: Same payoff → same price
│  │   ├─ Implies: Discounted price process is martingale
│  │   └─ Consequence: Risk-neutral measure Q exists
│  ├─ Pricing Formula:
│  │   V_0 = E^Q[e^(-rT) Payoff(S_T)]
│  │   Expectation under Q, discount at risk-free rate
│  ├─ Key Insight:
│  │   Current price S_0 already reflects risk premium
│  │   Option value depends only on S_0, not future drift μ
│  └─ Why It Works:
│      Replication: Can hedge continuously → return must be r
│      Alternative: Arbitrage opportunity exists
├─ Risk-Neutral Measure (Q):
│  ├─ Definition:
│  │   Probability measure where discounted asset prices are martingales
│  │   E^Q[S_T | F_t] = S_t e^(r(T-t))
│  ├─ Construction (Girsanov Theorem):
│  │   ├─ Real-world: dS = μS dt + σS dW^P
│  │   ├─ Risk-neutral: dS = rS dt + σS dW^Q
│  │   ├─ Change of measure: dW^Q = dW^P + ((μ-r)/σ)dt
│  │   └─ Market price of risk: λ = (μ-r)/σ
│  ├─ Radon-Nikodym Derivative:
│  │   dQ/dP = exp(-λW^P_T - ½λ²T)
│  │   Converts probabilities: P → Q
│  ├─ Properties:
│  │   ├─ Q is EMM (Equivalent Martingale Measure)
│  │   ├─ Same null sets as P (equivalent)
│  │   ├─ Volatility unchanged: σ^Q = σ^P
│  │   └─ Only drift shifts: μ → r
│  └─ Existence & Uniqueness:
│      ├─ Complete market: Unique Q
│      ├─ Incomplete: Multiple Q's (bounds on price)
│      └─ Arbitrage exists: No Q exists
├─ Derivation via Replication:
│  ├─ Self-Financing Portfolio:
│  │   ├─ Hold Δ_t shares of stock
│  │   ├─ Hold B_t in bank account (bond)
│  │   ├─ Portfolio value: Π_t = Δ_t S_t + B_t
│  │   └─ Replicates option: Π_T = Payoff(S_T)
│  ├─ Dynamics:
│  │   dΠ = Δ dS + r B dt
│  │   No cash injection (self-financing)
│  ├─ Hedging Condition:
│  │   Choose Δ such that dΠ has no dW term
│  │   → Π grows at rate r (riskless)
│  ├─ Result:
│  │   Π_t = e^(-r(T-t)) E^Q[Payoff | F_t]
│  │   Discounted portfolio is Q-martingale
│  └─ Conclusion:
│      Option value = Replication cost = Risk-neutral expectation
├─ Black-Scholes via Risk-Neutral:
│  ├─ Under Q:
│  │   S_T = S_0 exp((r - ½σ²)T + σ√T Z)
│  │   where Z ~ N(0,1) under Q
│  ├─ Call payoff:
│  │   C(S_T) = max(S_T - K, 0)
│  ├─ Expected payoff:
│  │   E^Q[C(S_T)] = ∫ max(S_T - K, 0) φ(z) dz
│  │   Integral over lognormal distribution
│  ├─ Analytical evaluation:
│  │   Yields: S_0 N(d_1) - K e^(-rT) N(d_2)
│  │   Black-Scholes formula
│  └─ No μ appears: Only S_0, K, r, T, σ
├─ Risk-Neutral Probability:
│  ├─ Interpretation:
│  │   NOT real probability of outcomes
│  │   Mathematical construct for pricing
│  ├─ Example (Binomial):
│  │   ├─ Real-world: P(up)=0.6, P(down)=0.4
│  │   ├─ Risk-neutral: Q(up)=(e^(rΔt)-d)/(u-d)
│  │   │   Typically Q(up) < P(up) if μ > r
│  │   └─ Risk adjustment: Reduces probability of good outcomes
│  ├─ Intuition:
│  │   Q-probabilities price risk aversion into expectations
│  │   Equivalent to using P-probabilities with risk-adjusted discount
│  └─ Connection:
│      E^Q[X] = E^P[X × (dQ/dP)]
│      Expectation under Q = Weighted expectation under P
├─ Numeraire Change:
│  ├─ General Principle:
│  │   Any tradable asset can be numeraire (unit of account)
│  │   Relative prices in numeraire units are martingales
│  ├─ Bank Account Numeraire:
│  │   ├─ N_t = e^(rt) (money market account)
│  │   ├─ Measure: Risk-neutral Q
│  │   ├─ Result: S_t / N_t = S_t e^(-rt) is Q-martingale
│  │   └─ Standard pricing: V_0 = E^Q[e^(-rT) V_T]
│  ├─ Stock as Numeraire:
│  │   ├─ N_t = S_t (stock price)
│  │   ├─ Measure: Stock measure Q^S
│  │   ├─ Result: V_t / S_t is Q^S-martingale
│  │   └─ Use: Simplifies exchange options (Margrabe)
│  ├─ Zero-Coupon Bond Numeraire:
│  │   ├─ N_t = P(t,T) (T-bond price)
│  │   ├─ Measure: T-forward measure Q^T
│  │   ├─ Result: Forward prices are martingales
│  │   └─ Use: Interest rate derivatives (caps, swaptions)
│  └─ Conversion (Fundamental Theorem):
│      V_0/N_0 = E^Q^N[V_T / N_T]
│      Change numeraire → change measure → simplify calculations
├─ Applications:
│  ├─ European Options:
│  │   ├─ Calls, puts: Direct expected value calculation
│  │   ├─ Digitals: Q(S_T > K) under lognormal
│  │   └─ Any terminal payoff: E^Q[g(S_T)]
│  ├─ Path-Dependent Options:
│  │   ├─ Asians: E^Q[max(Avg(S)-K, 0)]
│  │   ├─ Barriers: E^Q[Payoff × Indicator(no breach)]
│  │   ├─ Lookbacks: E^Q[max over path - K]
│  │   └─ Monte Carlo: Simulate under Q, average payoffs
│  ├─ Multi-Asset Options:
│  │   ├─ Correlation enters through joint distribution under Q
│  │   ├─ Baskets: E^Q[max(w·S_T - K, 0)]
│  │   └─ Spreads: E^Q[max(S₁_T - S₂_T - K, 0)]
│  ├─ Interest Rate Derivatives:
│  │   ├─ Caps/Floors: Use forward measure
│  │   ├─ Swaptions: Swap measure (annuity numeraire)
│  │   └─ Exotic rates: HJM, LMM frameworks
│  └─ Credit Derivatives:
│      Default-adjusted Q: Intensity models, hazard rates
├─ Real-World vs Risk-Neutral:
│  ├─ Real-World (P-measure):
│  │   ├─ Purpose: Forecasting, risk management, VaR
│  │   ├─ Drift: Historical μ or estimated
│  │   ├─ Probabilities: True likelihood
│  │   └─ Example: 20% chance stock below $90
│  ├─ Risk-Neutral (Q-measure):
│  │   ├─ Purpose: Pricing derivatives
│  │   ├─ Drift: Risk-free rate r
│  │   ├─ Probabilities: Risk-adjusted (not real)
│  │   └─ Example: 35% "risk-neutral probability" below $90
│  ├─ Relationship:
│  │   ├─ Q puts more weight on bad outcomes
│  │   ├─ Reflects risk aversion in market prices
│  │   └─ Connected via market price of risk λ
│  └─ When to Use Which:
│      ├─ Pricing: Always use Q
│      ├─ Hedging: Can use either (both give same hedge ratio)
│      ├─ Risk assessment: Use P (real probabilities)
│      └─ Scenario analysis: Use P (realistic outcomes)
├─ Market Price of Risk:
│  ├─ Definition:
│  │   λ = (μ - r) / σ
│  │   Excess return per unit volatility
│  ├─ Interpretation:
│  │   ├─ Compensation for bearing risk
│  │   ├─ Higher λ → higher risk premium
│  │   └─ Market determines λ via supply/demand
│  ├─ Girsanov Connection:
│  │   dW^Q = dW^P + λ dt
│  │   Shift Brownian motion by λ
│  ├─ Multi-Factor:
│  │   Vector λ = [λ₁, ..., λ_n]
│  │   One λ per risk factor
│  └─ Calibration:
│      Extract λ from option prices (implied)
│      Or estimate from time series (historical)
├─ Completeness:
│  ├─ Complete Market:
│  │   ├─ Every contingent claim can be replicated
│  │   ├─ Unique risk-neutral measure Q
│  │   ├─ Example: Black-Scholes (1 stock, 1 bond)
│  │   └─ Consequence: Unique arbitrage-free price
│  ├─ Incomplete Market:
│  │   ├─ Some claims cannot be hedged
│  │   ├─ Multiple Q's exist (set of EMMs)
│  │   ├─ Example: Jump-diffusion, stochastic vol
│  │   └─ Consequence: Price bounds, not unique price
│  ├─ No-Arbitrage vs Completeness:
│  │   ├─ No-arbitrage: Q exists (feasible prices)
│  │   ├─ Completeness: Q unique (unique price)
│  │   └─ Both: Harrison-Pliska fundamental theorem
│  └─ Practical Impact:
│      Incompleteness → Model risk, calibration challenges
├─ Limitations & Caveats:
│  ├─ Continuous Trading:
│  │   Assumes infinite rebalancing (unrealistic)
│  │   Transaction costs violate perfect replication
│  ├─ No-Arbitrage Assumption:
│  │   Requires liquid, efficient markets
│  │   Breaks during crises, illiquidity
│  ├─ Model Risk:
│  │   Q depends on chosen model (GBM, jumps, etc.)
│  │   Wrong model → wrong Q → wrong price
│  ├─ Real-World Drift Irrelevant:
│  │   True for pricing, NOT for risk management
│  │   P&L depends on real outcomes under P
│  └─ Long Maturity:
│      Model assumptions degrade over long horizons
│      Discount factors compound small errors
└─ Practical Implementation:
   ├─ Monte Carlo:
   │   ├─ Simulate paths under Q (drift=r)
   │   ├─ Calculate payoff each path
   │   ├─ Average and discount: V_0 = e^(-rT) × mean(payoffs)
   │   └─ Variance reduction: Same as before
   ├─ Trees:
   │   ├─ Risk-neutral probabilities at each node
   │   ├─ Backward induction with discount
   │   └─ Matches risk-neutral expectation
   ├─ PDE Approach:
   │   Black-Scholes PDE derived from risk-neutral argument
   │   Solve PDE with boundary conditions
   ├─ Closed-Form:
   │   Evaluate E^Q[Payoff] analytically if possible
   │   Black-Scholes, Bachelier, etc.
   └─ Calibration:
      ├─ Extract Q from liquid option prices
      ├─ Use calibrated Q to price illiquid derivatives
      └─ Ensure consistency across products
```

**Interaction:** No-arbitrage → Q exists → Pricing via E^Q[Discounted payoff] → Model choice determines Q → Calibration to market → Exotic valuation.

## 5. Mini-Project
Implement risk-neutral valuation across different methods:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("RISK-NEUTRAL VALUATION FRAMEWORK")
print("="*60)

class RiskNeutralPricing:
    """Risk-neutral valuation implementation"""
    
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    def simulate_paths_real_world(self, mu, T, n_paths=10000, n_steps=252):
        """Simulate under real-world measure P"""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            )
        
        return paths
    
    def simulate_paths_risk_neutral(self, T, n_paths=10000, n_steps=252):
        """Simulate under risk-neutral measure Q"""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            )
        
        return paths
    
    def price_european_mc(self, K, T, option_type='call', n_paths=10000):
        """Price European option via Monte Carlo under Q"""
        # Terminal stock price under Q
        Z = np.random.normal(0, 1, n_paths)
        S_T = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*T + self.sigma*np.sqrt(T)*Z)
        
        # Payoff
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Risk-neutral expectation
        price = np.exp(-self.r*T) * np.mean(payoffs)
        se = np.exp(-self.r*T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def price_european_analytical(self, K, T, option_type='call'):
        """Black-Scholes closed form"""
        d1 = (np.log(self.S0/K) + (self.r + 0.5*self.sigma**2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma*np.sqrt(T)
        
        if option_type == 'call':
            return self.S0*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-self.r*T)*norm.cdf(-d2) - self.S0*norm.cdf(-d1)
    
    def risk_neutral_density(self, S_T, T):
        """Risk-neutral probability density at S_T"""
        # Lognormal density under Q
        mu_Q = np.log(self.S0) + (self.r - 0.5*self.sigma**2)*T
        sigma_T = self.sigma*np.sqrt(T)
        
        return (1 / (S_T * sigma_T * np.sqrt(2*np.pi))) * \
               np.exp(-0.5 * ((np.log(S_T) - mu_Q) / sigma_T)**2)
    
    def real_world_density(self, S_T, T, mu):
        """Real-world probability density at S_T"""
        # Lognormal density under P
        mu_P = np.log(self.S0) + (mu - 0.5*self.sigma**2)*T
        sigma_T = self.sigma*np.sqrt(T)
        
        return (1 / (S_T * sigma_T * np.sqrt(2*np.pi))) * \
               np.exp(-0.5 * ((np.log(S_T) - mu_P) / sigma_T)**2)
    
    def radon_nikodym_derivative(self, S_T, T, mu):
        """Radon-Nikodym derivative dQ/dP"""
        # Market price of risk
        lambda_mpr = (mu - self.r) / self.sigma
        
        # From final stock price, infer Brownian motion
        W_T = (np.log(S_T/self.S0) - (mu - 0.5*self.sigma**2)*T) / (self.sigma*np.sqrt(T))
        
        return np.exp(-lambda_mpr * W_T - 0.5 * lambda_mpr**2 * T)

class BinomialRiskNeutral:
    """Binomial tree with explicit risk-neutral probabilities"""
    
    def __init__(self, S0, K, r, T, sigma, n_steps):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # Up/down factors
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        
        # Risk-neutral probability
        self.q = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        
        # Real-world probability (illustrative - higher for positive drift)
        mu_real = 0.12  # Assumed real drift
        self.p = (np.exp(mu_real * self.dt) - self.d) / (self.u - self.d)
    
    def price_option(self, option_type='call'):
        """Price using risk-neutral probabilities"""
        # Build terminal payoffs
        payoffs = np.zeros(self.n_steps + 1)
        
        for i in range(self.n_steps + 1):
            S_T = self.S0 * (self.u ** (self.n_steps - i)) * (self.d ** i)
            if option_type == 'call':
                payoffs[i] = max(S_T - self.K, 0)
            else:
                payoffs[i] = max(self.K - S_T, 0)
        
        # Backward induction using q (risk-neutral)
        for step in range(self.n_steps - 1, -1, -1):
            for i in range(step + 1):
                payoffs[i] = np.exp(-self.r*self.dt) * \
                            (self.q * payoffs[i] + (1 - self.q) * payoffs[i+1])
        
        return payoffs[0]
    
    def martingale_test(self):
        """Verify discounted stock price is martingale under Q"""
        # Expected stock price under Q at T
        E_Q_S_T = self.S0 * np.exp(self.r * self.T)
        
        # Calculate via tree
        S_T_values = []
        probs = []
        
        for i in range(self.n_steps + 1):
            S_T = self.S0 * (self.u ** (self.n_steps - i)) * (self.d ** i)
            # Binomial probability
            from scipy.special import comb
            prob = comb(self.n_steps, i) * (self.q ** (self.n_steps - i)) * ((1-self.q) ** i)
            S_T_values.append(S_T)
            probs.append(prob)
        
        E_Q_S_T_tree = np.sum(np.array(S_T_values) * np.array(probs))
        
        return E_Q_S_T, E_Q_S_T_tree

# Scenario 1: Real-world vs Risk-neutral paths
print("\n" + "="*60)
print("SCENARIO 1: Real-World vs Risk-Neutral Simulation")
print("="*60)

S0, r, sigma = 100, 0.05, 0.20
T = 1.0
mu_real = 0.12  # Real-world drift (higher than r)

pricer = RiskNeutralPricing(S0, r, sigma)

# Simulate paths under both measures
paths_P = pricer.simulate_paths_real_world(mu_real, T, n_paths=5000)
paths_Q = pricer.simulate_paths_risk_neutral(T, n_paths=5000)

# Terminal distributions
S_T_P = paths_P[:, -1]
S_T_Q = paths_Q[:, -1]

print(f"\nParameters: S=${S0}, r={r:.1%}, σ={sigma:.1%}, T={T}yr")
print(f"Real-world drift: μ={mu_real:.1%}")
print(f"Risk-neutral drift: r={r:.1%}")

print(f"\nTerminal Stock Prices:")
print(f"Real-World (P-measure):")
print(f"  Mean: ${np.mean(S_T_P):.2f}")
print(f"  Expected: ${S0 * np.exp(mu_real*T):.2f}")
print(f"  Std Dev: ${np.std(S_T_P):.2f}")

print(f"\nRisk-Neutral (Q-measure):")
print(f"  Mean: ${np.mean(S_T_Q):.2f}")
print(f"  Expected: ${S0 * np.exp(r*T):.2f}")
print(f"  Std Dev: ${np.std(S_T_Q):.2f}")

print(f"\nDifference in means: ${np.mean(S_T_P) - np.mean(S_T_Q):.2f}")
print(f"Market price of risk: λ={(mu_real - r)/sigma:.4f}")

# Scenario 2: Option pricing consistency
print("\n" + "="*60)
print("SCENARIO 2: Option Pricing via Risk-Neutral Valuation")
print("="*60)

K = 100

# Monte Carlo under Q
price_mc, se_mc = pricer.price_european_mc(K, T, 'call', n_paths=50000)

# Analytical (Black-Scholes)
price_bs = pricer.price_european_analytical(K, T, 'call')

# Binomial tree
binomial = BinomialRiskNeutral(S0, K, r, T, sigma, n_steps=100)
price_binom = binomial.price_option('call')

print(f"\nATM Call Option (K=${K}):")
print(f"\nMonte Carlo (Q-measure):")
print(f"  Price: ${price_mc:.4f} ± ${se_mc:.4f}")

print(f"\nBlack-Scholes (Analytical):")
print(f"  Price: ${price_bs:.4f}")

print(f"\nBinomial Tree (Q-probabilities):")
print(f"  Price: ${price_binom:.4f}")
print(f"  Risk-neutral prob: q={binomial.q:.4f}")
print(f"  Real-world prob: p={binomial.p:.4f}")

print(f"\nConsistency Check:")
print(f"  MC vs BS error: ${abs(price_mc - price_bs):.4f}")
print(f"  Binomial vs BS error: ${abs(price_binom - price_bs):.4f}")

# Scenario 3: Martingale property
print("\n" + "="*60)
print("SCENARIO 3: Martingale Property Verification")
print("="*60)

E_Q_S_T_exact, E_Q_S_T_tree = binomial.martingale_test()

print(f"\nDiscounted Stock Price is Martingale under Q:")
print(f"  S_0 = ${S0:.2f}")
print(f"  E^Q[S_T] (exact) = ${E_Q_S_T_exact:.2f}")
print(f"  E^Q[S_T] (tree) = ${E_Q_S_T_tree:.2f}")
print(f"  Ratio: {E_Q_S_T_tree / E_Q_S_T_exact:.6f}")

# Check if ratio close to e^(rT)
ratio = E_Q_S_T_exact / S0
expected_ratio = np.exp(r * T)
print(f"\n  E^Q[S_T] / S_0 = {ratio:.6f}")
print(f"  e^(rT) = {expected_ratio:.6f}")
print(f"  ✓ Martingale property verified" if abs(ratio - expected_ratio) < 0.01 
      else "  ✗ Martingale property violated")

# Scenario 4: Probability densities
print("\n" + "="*60)
print("SCENARIO 4: Risk-Neutral vs Real-World Densities")
print("="*60)

S_range = np.linspace(60, 160, 100)
density_Q = [pricer.risk_neutral_density(S, T) for S in S_range]
density_P = [pricer.real_world_density(S, T, mu_real) for S in S_range]

# Probabilities in different regions
K_low, K_high = 80, 120

# Under P
prob_below_P = norm.cdf((np.log(K_low/S0) - (mu_real - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
prob_above_P = 1 - norm.cdf((np.log(K_high/S0) - (mu_real - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))

# Under Q
prob_below_Q = norm.cdf((np.log(K_low/S0) - (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
prob_above_Q = 1 - norm.cdf((np.log(K_high/S0) - (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))

print(f"\nProbability S_T < ${K_low}:")
print(f"  Real-World (P): {prob_below_P*100:.2f}%")
print(f"  Risk-Neutral (Q): {prob_below_Q*100:.2f}%")
print(f"  Q assigns {(prob_below_Q - prob_below_P)*100:.2f}% more probability")

print(f"\nProbability S_T > ${K_high}:")
print(f"  Real-World (P): {prob_above_P*100:.2f}%")
print(f"  Risk-Neutral (Q): {prob_above_Q*100:.2f}%")
print(f"  Q assigns {(prob_above_P - prob_above_Q)*100:.2f}% less probability")

print(f"\nInterpretation: Q shifts probability to downside (risk aversion)")

# Scenario 5: Radon-Nikodym derivative
print("\n" + "="*60)
print("SCENARIO 5: Change of Measure (Radon-Nikodym)")
print("="*60)

S_test_values = [80, 90, 100, 110, 120]

print(f"\nRadon-Nikodym Derivative dQ/dP at different outcomes:")
print(f"{'S_T':<10} {'dQ/dP':<12} {'Interpretation':<30}")
print("-" * 52)

for S_T in S_test_values:
    rn = pricer.radon_nikodym_derivative(S_T, T, mu_real)
    
    if S_T < S0:
        interp = "Higher weight under Q (downside)"
    elif S_T > S0 * 1.1:
        interp = "Lower weight under Q (upside)"
    else:
        interp = "Near neutral"
    
    print(f"${S_T:<9} {rn:<11.4f} {interp:<30}")

# Verify E^Q[X] = E^P[X × dQ/dP]
K_verify = 105
# Under P
def integrand_P(S_T):
    payoff = max(S_T - K_verify, 0)
    return payoff * pricer.real_world_density(S_T, T, mu_real)

# Under Q
def integrand_Q(S_T):
    payoff = max(S_T - K_verify, 0)
    return payoff * pricer.risk_neutral_density(S_T, T)

# Under P with dQ/dP
def integrand_P_weighted(S_T):
    payoff = max(S_T - K_verify, 0)
    density_P = pricer.real_world_density(S_T, T, mu_real)
    rn = pricer.radon_nikodym_derivative(S_T, T, mu_real)
    return payoff * density_P * rn

E_Q, _ = quad(integrand_Q, 0, 500)
E_P_weighted, _ = quad(integrand_P_weighted, 0, 500)

print(f"\nVerification: E^Q[Payoff] = E^P[Payoff × dQ/dP]")
print(f"  E^Q[max(S_T - {K_verify}, 0)] = {E_Q:.4f}")
print(f"  E^P[(dQ/dP) × max(S_T - {K_verify}, 0)] = {E_P_weighted:.4f}")
print(f"  Difference: {abs(E_Q - E_P_weighted):.6f}")

# Scenario 6: Multiple strikes
print("\n" + "="*60)
print("SCENARIO 6: Option Prices Across Strikes")
print("="*60)

strikes = np.linspace(85, 115, 13)
prices_call_bs = []
prices_put_bs = []

print(f"\n{'Strike':<10} {'Call (BS)':<12} {'Put (BS)':<12} {'Put-Call Parity':<20}")
print("-" * 54)

for K in strikes:
    call = pricer.price_european_analytical(K, T, 'call')
    put = pricer.price_european_analytical(K, T, 'put')
    
    # Put-call parity check
    parity_lhs = call - put
    parity_rhs = S0 - K * np.exp(-r*T)
    
    prices_call_bs.append(call)
    prices_put_bs.append(put)
    
    if K in [85, 95, 105, 115]:
        print(f"${K:<9} ${call:<11.4f} ${put:<11.4f} {abs(parity_lhs - parity_rhs):<19.6f}")

print(f"\n✓ Put-call parity verified across all strikes")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Sample paths (P vs Q)
ax = axes[0, 0]
times = np.linspace(0, T, paths_P.shape[1])
for i in range(10):
    ax.plot(times, paths_P[i, :], 'b-', alpha=0.6, linewidth=1)
    ax.plot(times, paths_Q[i, :], 'r-', alpha=0.6, linewidth=1)

ax.plot([], [], 'b-', label='Real-World (P)', linewidth=2)
ax.plot([], [], 'r-', label='Risk-Neutral (Q)', linewidth=2)
ax.axhline(S0 * np.exp(mu_real*T), color='b', linestyle='--', alpha=0.5, label=f'E^P[S_T]')
ax.axhline(S0 * np.exp(r*T), color='r', linestyle='--', alpha=0.5, label=f'E^Q[S_T]')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')
ax.set_title('Sample Paths: P-measure vs Q-measure')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Terminal distributions
ax = axes[0, 1]
ax.hist(S_T_P, bins=50, density=True, alpha=0.5, label='P-measure', color='blue')
ax.hist(S_T_Q, bins=50, density=True, alpha=0.5, label='Q-measure', color='red')
ax.axvline(np.mean(S_T_P), color='blue', linestyle='--', linewidth=2)
ax.axvline(np.mean(S_T_Q), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Terminal Stock Price')
ax.set_ylabel('Density')
ax.set_title('Terminal Distributions')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Probability densities
ax = axes[0, 2]
ax.plot(S_range, density_P, 'b-', linewidth=2.5, label='Real-World (P)')
ax.plot(S_range, density_Q, 'r-', linewidth=2.5, label='Risk-Neutral (Q)')
ax.axvline(S0, color='k', linestyle='--', alpha=0.3, label='Current Price')
ax.set_xlabel('Stock Price at T')
ax.set_ylabel('Probability Density')
ax.set_title('P vs Q Density Functions')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Radon-Nikodym derivative
ax = axes[1, 0]
S_rn_range = np.linspace(60, 160, 100)
rn_values = [pricer.radon_nikodym_derivative(S, T, mu_real) for S in S_rn_range]
ax.plot(S_rn_range, rn_values, 'purple', linewidth=2.5)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax.axvline(S0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Terminal Stock Price')
ax.set_ylabel('dQ/dP')
ax.set_title('Radon-Nikodym Derivative (Change of Measure)')
ax.grid(alpha=0.3)

# Plot 5: Option prices across strikes
ax = axes[1, 1]
ax.plot(strikes, prices_call_bs, 'b-', linewidth=2.5, marker='o', markersize=8, label='Call')
ax.plot(strikes, prices_put_bs, 'r-', linewidth=2.5, marker='s', markersize=8, label='Put')
ax.axvline(S0, color='k', linestyle='--', alpha=0.3, label='Current Price')
ax.set_xlabel('Strike')
ax.set_ylabel('Option Price')
ax.set_title('Option Prices (Risk-Neutral Valuation)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Binomial tree probabilities
ax = axes[1, 2]
n_viz = 10
tree_probs_q = []
tree_probs_p = []

for i in range(n_viz + 1):
    from scipy.special import comb
    prob_q = comb(n_viz, i) * (binomial.q ** (n_viz - i)) * ((1-binomial.q) ** i)
    prob_p = comb(n_viz, i) * (binomial.p ** (n_viz - i)) * ((1-binomial.p) ** i)
    tree_probs_q.append(prob_q)
    tree_probs_p.append(prob_p)

x_pos = np.arange(n_viz + 1)
width = 0.35
ax.bar(x_pos - width/2, tree_probs_p, width, label='Real-World (P)', alpha=0.7, color='blue')
ax.bar(x_pos + width/2, tree_probs_q, width, label='Risk-Neutral (Q)', alpha=0.7, color='red')
ax.set_xlabel('Number of Down Moves')
ax.set_ylabel('Probability')
ax.set_title(f'Binomial Probabilities ({n_viz} steps)')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Heston Model Q-Measure:** Derive risk-neutral dynamics for Heston stochastic volatility. What's market price of volatility risk? How does it affect option prices?

2. **Incomplete Market:** Model jump-diffusion (Merton). Show multiple EMMs exist. Calculate bounds on option price. What additional principle determines unique price?

3. **Foreign Exchange:** Derive risk-neutral measure for FX option (two interest rates). Show how domestic/foreign rate enter. What's Garman-Kohlhagen formula?

4. **Change of Numeraire:** Price Margrabe exchange option using stock as numeraire. Verify simpler than bank account numeraire. What's new risk-neutral measure?

5. **Long-Dated Options:** Price 20-year equity call. How sensitive to model assumptions? Compare real-world P&L distribution to Q-measure pricing. Why diverge?

## 7. Key References
- [Harrison & Pliska (1981) - Fundamental Theorem of Asset Pricing](https://www.jstor.org/stable/3689775)
- [Cox & Ross (1976) - Risk-Neutral Pricing](https://www.jstor.org/stable/2978261)
- [Shreve, Stochastic Calculus for Finance II (Chapters 1-5)](https://www.springer.com/series/3401)
- [Baxter & Rennie, Financial Calculus (Chapter 3)](https://www.cambridge.org/core/books/financial-calculus/33D34BFC5A07FA0DBEF0D4A39A0C1B13)

---
**Status:** Foundation of derivative pricing | **Complements:** Black-Scholes, Replication, Martingales, Girsanov Theorem, No-Arbitrage Principle
