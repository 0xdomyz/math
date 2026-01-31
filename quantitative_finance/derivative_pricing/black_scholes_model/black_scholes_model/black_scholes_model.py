
# Block 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("BLACK-SCHOLES MODEL IMPLEMENTATION")
print("="*60)

class BlackScholes:
    """Black-Scholes European option pricing"""
    
    def __init__(self, S, K, r, T, sigma, q=0):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.q = q  # Dividend yield
        
        # Calculate d1, d2
        self.d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        
    def call_price(self):
        return (self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
    
    def put_price(self):
        return (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2) - 
                self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1))
    
    def delta_call(self):
        return np.exp(-self.q*self.T) * norm.cdf(self.d1)
    
    def delta_put(self):
        return -np.exp(-self.q*self.T) * norm.cdf(-self.d1)
    
    def gamma(self):
        return (np.exp(-self.q*self.T) * norm.pdf(self.d1) / 
                (self.S * self.sigma * np.sqrt(self.T)))
    
    def vega(self):
        return self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100
    
    def theta_call(self):
        term1 = -self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
        term3 = self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1)
        return (term1 + term2 + term3) / 365
    
    def theta_put(self):
        term1 = -self.S * np.exp(-self.q*self.T) * norm.pdf(self.d1) * self.sigma / (2*np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2)
        term3 = -self.q * self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1)
        return (term1 + term2 + term3) / 365
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(self.d2) / 100
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(-self.d2) / 100
    
    def implied_volatility(self, market_price, option_type='call'):
        """Find IV using Newton-Raphson"""
        def objective(sigma):
            bs = BlackScholes(self.S, self.K, self.r, self.T, sigma, self.q)
            if option_type == 'call':
                return bs.call_price() - market_price
            else:
                return bs.put_price() - market_price
        
        def vega_func(sigma):
            bs = BlackScholes(self.S, self.K, self.r, self.T, sigma, self.q)
            return bs.vega() * 100  # Revert from /100 to get sensitivity
        
        try:
            iv = brentq(objective, 0.001, 2.0)
            return iv
        except:
            return np.nan

# Scenario 1: Basic pricing
print("\n" + "="*60)
print("SCENARIO 1: Basic Black-Scholes Pricing")
print("="*60)

S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.2

bs = BlackScholes(S, K, r, T, sigma)
call = bs.call_price()
put = bs.put_price()

print(f"\nParameters:")
print(f"  S = ${S}, K = ${K}, r = {r:.1%}, T = {T} year, σ = {sigma:.1%}")
print(f"\nOption Prices:")
print(f"  Call: ${call:.2f}")
print(f"  Put: ${put:.2f}")
print(f"  Put-Call Parity: C - P = {call - put:.4f}, S - Ke^(-rT) = {S - K*np.exp(-r*T):.4f}")

# Scenario 2: Greeks calculation
print("\n" + "="*60)
print("SCENARIO 2: Greeks for ATM Option")
print("="*60)

delta_c = bs.delta_call()
delta_p = bs.delta_put()
gamma = bs.gamma()
vega = bs.vega()
theta_c = bs.theta_call()
theta_p = bs.theta_put()
rho_c = bs.rho_call()
rho_p = bs.rho_put()

print(f"\nCall Greeks:")
print(f"  Δ = {delta_c:.4f} (hedge by selling {delta_c:.4f} shares per call)")
print(f"  Γ = {gamma:.6f} (delta changes by {gamma:.6f} per $1 move)")
print(f"  Θ (daily) = ${theta_c:.4f} (loses ${abs(theta_c):.4f}/day)")
print(f"  ν (per 1% vol) = ${vega:.4f}")
print(f"  ρ (per 1% rate) = ${rho_c:.4f}")

print(f"\nPut Greeks:")
print(f"  Δ = {delta_p:.4f}")
print(f"  Γ = {gamma:.6f} (same as call)")
print(f"  Θ (daily) = ${theta_p:.4f}")
print(f"  ν = ${vega:.4f} (same as call)")
print(f"  ρ = ${rho_p:.4f}")

# Scenario 3: Sensitivity analysis
print("\n" + "="*60)
print("SCENARIO 3: Sensitivity Analysis")
print("="*60)

# Change each parameter by 1 unit and measure impact
S_bump = S + 1
r_bump = r + 0.01
T_bump = T - 1/365  # 1 day passes
sigma_bump = sigma + 0.01

bs_bump_s = BlackScholes(S_bump, K, r, T, sigma)
bs_bump_r = BlackScholes(S, K, r_bump, T, sigma)
bs_bump_t = BlackScholes(S, K, r, T_bump, sigma)
bs_bump_vol = BlackScholes(S, K, r, T, sigma_bump)

call_delta_approx = delta_c * 1
call_rho_approx = rho_c * 1
call_theta_approx = theta_c * 1
call_vega_approx = vega * 1

print(f"\nApprox vs Actual Change in Call Price:")
print(f"  S +$1:")
print(f"    Approximate (Delta): ${call_delta_approx:.4f}")
print(f"    Actual: ${bs_bump_s.call_price() - call:.4f}")
print(f"  r +1%:")
print(f"    Approximate (Rho): ${call_rho_approx:.4f}")
print(f"    Actual: ${bs_bump_r.call_price() - call:.4f}")
print(f"  T -1 day:")
print(f"    Approximate (Theta): ${call_theta_approx:.4f}")
print(f"    Actual: ${bs_bump_t.call_price() - call:.4f}")
print(f"  σ +1%:")
print(f"    Approximate (Vega): ${call_vega_approx:.4f}")
print(f"    Actual: ${bs_bump_vol.call_price() - call:.4f}")

# Scenario 4: Greeks surface across spot prices
print("\n" + "="*60)
print("SCENARIO 4: Greeks Across Moneyness")
print("="*60)

spot_range = np.linspace(80, 120, 9)
print(f"\n{'Spot':<8} {'Delta':<10} {'Gamma':<12} {'Theta/day':<12} {'Vega':<10}")
print("-" * 52)
for s in spot_range:
    bs_temp = BlackScholes(s, K, r, T, sigma)
    print(f"${s:>6.0f}  {bs_temp.delta_call():>8.4f}  {bs_temp.gamma():>10.6f}  ${bs_temp.theta_call():>10.4f}  ${bs_temp.vega():>8.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Option prices vs spot
ax = axes[0, 0]
spot_range_plot = np.linspace(70, 130, 100)
calls = [BlackScholes(s, K, r, T, sigma).call_price() for s in spot_range_plot]
puts = [BlackScholes(s, K, r, T, sigma).put_price() for s in spot_range_plot]
intrinsic_call = np.maximum(spot_range_plot - K, 0)
intrinsic_put = np.maximum(K - spot_range_plot, 0)

ax.plot(spot_range_plot, calls, 'b-', linewidth=2.5, label='Call Value')
ax.plot(spot_range_plot, puts, 'r-', linewidth=2.5, label='Put Value')
ax.plot(spot_range_plot, intrinsic_call, 'b--', alpha=0.5, label='Call Intrinsic')
ax.plot(spot_range_plot, intrinsic_put, 'r--', alpha=0.5, label='Put Intrinsic')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Option Value')
ax.set_title('BS Option Prices vs Spot')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Greeks vs spot
ax = axes[0, 1]
deltas = [BlackScholes(s, K, r, T, sigma).delta_call() for s in spot_range_plot]
gammas = [BlackScholes(s, K, r, T, sigma).gamma() for s in spot_range_plot]

ax_twin = ax.twinx()
ax.plot(spot_range_plot, deltas, 'b-', linewidth=2.5, label='Delta')
ax_twin.plot(spot_range_plot, gammas, 'g-', linewidth=2.5, label='Gamma')
ax.axvline(S, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Delta', color='b')
ax_twin.set_ylabel('Gamma', color='g')
ax.set_title('Delta & Gamma vs Spot')
ax.legend(loc='upper left')
ax_twin.legend(loc='upper right')
ax.grid(alpha=0.3)

# Plot 3: Theta decay
ax = axes[0, 2]
time_range = np.linspace(T, 0.01, 50)
theta_decay = [BlackScholes(S, K, r, t, sigma).call_price() for t in time_range]
ax.plot(time_range, theta_decay, 'b-', linewidth=2.5)
ax.fill_between(time_range, theta_decay, alpha=0.3)
ax.set_xlabel('Time to Expiry (years)')
ax.set_ylabel('Call Value')
ax.set_title('Theta Decay (ATM Call)')
ax.grid(alpha=0.3)

# Plot 4: Volatility sensitivity
ax = axes[1, 0]
vol_range = np.linspace(0.05, 0.5, 50)
calls_vol = [BlackScholes(S, K, r, T, v).call_price() for v in vol_range]
puts_vol = [BlackScholes(S, K, r, T, v).put_price() for v in vol_range]
ax.plot(vol_range*100, calls_vol, 'b-', linewidth=2.5, label='Call')
ax.plot(vol_range*100, puts_vol, 'r-', linewidth=2.5, label='Put')
ax.axvline(sigma*100, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Option Value')
ax.set_title('Option Value vs Volatility')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Greeks surface (Delta vs spot and time)
ax = axes[1, 1]
spot_fine = np.linspace(70, 130, 30)
time_fine = np.linspace(T, 0.1, 30)
delta_surface = np.zeros((len(time_fine), len(spot_fine)))

for i, t in enumerate(time_fine):
    for j, s in enumerate(spot_fine):
        delta_surface[i, j] = BlackScholes(s, K, r, t, sigma).delta_call()

contour = ax.contourf(spot_fine, time_fine, delta_surface, levels=15, cmap='RdYlGn')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Time to Expiry')
ax.set_title('Delta Surface')
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Delta')

# Plot 6: Put-Call Parity verification
ax = axes[1, 2]
spot_parity = np.linspace(80, 120, 50)
parity_lhs = []
parity_rhs = []

for s in spot_parity:
    bs_temp = BlackScholes(s, K, r, T, sigma)
    lhs = bs_temp.call_price() - bs_temp.put_price()
    rhs = s - K*np.exp(-r*T)
    parity_lhs.append(lhs)
    parity_rhs.append(rhs)

ax.plot(spot_parity, parity_lhs, 'b-', linewidth=2.5, label='C - P')
ax.plot(spot_parity, parity_rhs, 'r--', linewidth=2.5, label='S - Ke^(-rT)')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Stock Price')
ax.set_ylabel('Value Difference')
ax.set_title('Put-Call Parity Verification')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()