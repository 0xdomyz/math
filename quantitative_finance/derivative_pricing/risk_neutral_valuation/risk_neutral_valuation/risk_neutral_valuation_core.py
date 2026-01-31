import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import warnings
            from scipy.special import comb

# Block 1
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