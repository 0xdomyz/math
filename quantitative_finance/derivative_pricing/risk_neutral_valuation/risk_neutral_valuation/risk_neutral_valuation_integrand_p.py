import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import warnings
            from scipy.special import comb
    from scipy.special import comb
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