import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from dataclasses import dataclass

@dataclass
class KyleParameters:
    """Kyle model parameters"""
    p0: float          # Prior mean (initial price)
    Sigma0: float      # Prior variance (uncertainty about value)
    sigma_u: float     # Noise trader volatility
    T: int = 1         # Number of trading periods
    
class KyleSinglePeriod:
    """Kyle (1985) single-period model"""
    
    def __init__(self, params: KyleParameters):
        self.params = params
        
        # Equilibrium coefficients
        self.compute_equilibrium()
    
    def compute_equilibrium(self):
        """
        Solve for linear equilibrium
        
        Insider: X = β(v - p₀)
        Market maker: p(y) = p₀ + λy
        """
        Sigma0 = self.params.Sigma0
        sigma_u = self.params.sigma_u
        
        # Kyle's lambda (price impact)
        self.lam = 0.5 * np.sqrt(Sigma0 / sigma_u**2)
        
        # Insider trading intensity
        self.beta = 0.5 * np.sqrt(sigma_u**2 / Sigma0)
        
        # Expected insider profit
        self.expected_profit = 0.5 * np.sqrt(Sigma0 * sigma_u**2)
        
        # Posterior variance (after observing order flow)
        self.Sigma1 = Sigma0 * (1 - self.lam * self.beta)
        
    def insider_optimal_trade(self, v):
        """Compute insider's optimal trade given value v"""
        return self.beta * (v - self.params.p0)
    
    def market_maker_price(self, y):
        """Market maker's pricing rule given order flow y"""
        return self.params.p0 + self.lam * y
    
    def simulate(self, n_simulations=1000):
        """Simulate Kyle equilibrium"""
        p0 = self.params.p0
        Sigma0 = self.params.Sigma0
        sigma_u = self.params.sigma_u
        
        results = []
        
        for i in range(n_simulations):
            # Draw true value
            v = np.random.normal(p0, np.sqrt(Sigma0))
            
            # Insider's optimal order
            X = self.insider_optimal_trade(v)
            
            # Noise trader order
            u = np.random.normal(0, sigma_u)
            
            # Total order flow
            y = X + u
            
            # Market maker's price
            p = self.market_maker_price(y)
            
            # Insider profit
            profit = (v - p) * X
            
            # Market maker profit (should be zero in expectation)
            mm_profit = (p - v) * y
            
            results.append({
                'true_value': v,
                'insider_order': X,
                'noise_order': u,
                'total_flow': y,
                'price': p,
                'insider_profit': profit,
                'mm_profit': mm_profit,
                'price_error': v - p
            })
        
        return pd.DataFrame(results)

class KyleMultiPeriod:
    """Kyle multi-period model"""
    
    def __init__(self, params: KyleParameters):
        self.params = params
        self.T = params.T
        
        # Equilibrium path
        self.compute_equilibrium_path()
    
    def compute_equilibrium_path(self):
        """Solve for equilibrium coefficients in each period"""
        T = self.T
        Sigma0 = self.params.Sigma0
        sigma_u = self.params.sigma_u
        
        # Initialize arrays
        self.lambdas = np.zeros(T)
        self.betas = np.zeros(T)
        self.Sigmas = np.zeros(T + 1)
        self.Sigmas[0] = Sigma0
        
        # Backward induction
        for t in range(T):
            remaining = T - t
            
            # Lambda in period t
            self.lambdas[t] = np.sqrt(self.Sigmas[t] / sigma_u**2) / remaining
            
            # Beta in period t
            self.betas[t] = 1 / (2 * self.lambdas[t])
            
            # Updated variance
            self.Sigmas[t + 1] = self.Sigmas[t] * (1 - self.lambdas[t] * self.betas[t])
    
    def simulate(self, n_simulations=100):
        """Simulate multi-period Kyle model"""
        p0 = self.params.p0
        Sigma0 = self.params.Sigma0
        sigma_u = self.params.sigma_u
        T = self.T
        
        simulations = []
        
        for sim in range(n_simulations):
            # Draw true value (fixed for this simulation)
            v = np.random.normal(p0, np.sqrt(Sigma0))
            
            # Initialize
            prices = [p0]
            posterior_means = [p0]
            insider_orders = []
            noise_orders = []
            total_flows = []
            
            # Simulate T periods
            for t in range(T):
                # Insider's order
                X = self.betas[t] * (v - posterior_means[-1])
                
                # Noise trader order
                u = np.random.normal(0, sigma_u)
                
                # Total flow
                y = X + u
                
                # Price update
                p = posterior_means[-1] + self.lambdas[t] * y
                
                # Store
                insider_orders.append(X)
                noise_orders.append(u)
                total_flows.append(y)
                prices.append(p)
                posterior_means.append(p)
            
            simulations.append({
                'sim_id': sim,
                'true_value': v,
                'prices': prices,
                'insider_orders': insider_orders,
                'noise_orders': noise_orders,
                'total_flows': total_flows,
                'final_price': prices[-1],
                'price_error': v - prices[-1]
            })
        
        return simulations

def comparative_statics_analysis():
    """Analyze how equilibrium changes with parameters"""
    
    # Baseline
    base_params = KyleParameters(p0=100, Sigma0=4, sigma_u=50)
    
    # Vary Sigma0 (uncertainty about value)
    Sigma0_values = np.linspace(1, 10, 20)
    results_sigma = []
    
    for Sigma0 in Sigma0_values:
        params = KyleParameters(p0=100, Sigma0=Sigma0, sigma_u=50)
        kyle = KyleSinglePeriod(params)
        
        results_sigma.append({
            'Sigma0': Sigma0,
            'lambda': kyle.lam,
            'beta': kyle.beta,
            'expected_profit': kyle.expected_profit
        })
    
    df_sigma = pd.DataFrame(results_sigma)
    
    # Vary sigma_u (noise trader volume)
    sigma_u_values = np.linspace(10, 100, 20)
    results_noise = []
    
    for sigma_u in sigma_u_values:
        params = KyleParameters(p0=100, Sigma0=4, sigma_u=sigma_u)
        kyle = KyleSinglePeriod(params)
        
        results_noise.append({
            'sigma_u': sigma_u,
            'lambda': kyle.lam,
            'beta': kyle.beta,
            'expected_profit': kyle.expected_profit
        })
    
    df_noise = pd.DataFrame(results_noise)
    
    return df_sigma, df_noise

# Run simulations
print("="*80)
print("KYLE MODEL SIMULATION")
print("="*80)

# Single period
print("\n" + "="*80)
print("SINGLE-PERIOD KYLE MODEL")
print("="*80)

params_single = KyleParameters(p0=100, Sigma0=4, sigma_u=50)
kyle_single = KyleSinglePeriod(params_single)

print(f"\nParameters:")
print(f"  Prior mean (p₀): ${params_single.p0:.2f}")
print(f"  Prior variance (Σ₀): {params_single.Sigma0:.2f}")
print(f"  Noise trader volatility (σᵤ): {params_single.sigma_u:.2f}")

print(f"\nEquilibrium Coefficients:")
print(f"  Kyle's lambda (λ): {kyle_single.lam:.4f}")
print(f"  Insider intensity (β): {kyle_single.beta:.4f}")
print(f"  Market depth (1/λ): {1/kyle_single.lam:.2f} shares per dollar")
print(f"  Expected insider profit: ${kyle_single.expected_profit:.2f}")

# Simulate
df_single = kyle_single.simulate(n_simulations=10000)

print(f"\nSimulation Results (N=10,000):")
print(f"  Mean insider profit: ${df_single['insider_profit'].mean():.2f}")
print(f"  Std insider profit: ${df_single['insider_profit'].std():.2f}")
print(f"  Mean MM profit: ${df_single['mm_profit'].mean():.2f} (should be ~0)")
print(f"  Price efficiency (RMSE): ${np.sqrt((df_single['price_error']**2).mean()):.2f}")
print(f"  Price informativeness: {1 - (df_single['price_error'].var() / params_single.Sigma0):.2%}")

# Multi-period
print("\n" + "="*80)
print("MULTI-PERIOD KYLE MODEL (T=5)")
print("="*80)

params_multi = KyleParameters(p0=100, Sigma0=4, sigma_u=50, T=5)
kyle_multi = KyleMultiPeriod(params_multi)

print(f"\nEquilibrium Path:")
for t in range(params_multi.T):
    print(f"  Period {t}: λ={kyle_multi.lambdas[t]:.4f}, β={kyle_multi.betas[t]:.4f}, Σ={kyle_multi.Sigmas[t+1]:.4f}")

simulations_multi = kyle_multi.simulate(n_simulations=100)

# Calculate statistics
final_errors = [s['price_error'] for s in simulations_multi]
print(f"\nFinal Price Errors:")
print(f"  Mean: ${np.mean(final_errors):.4f}")
print(f"  RMSE: ${np.sqrt(np.mean(np.array(final_errors)**2)):.4f}")
print(f"  Final informativeness: {1 - np.var(final_errors) / params_multi.Sigma0:.2%}")

# Comparative statics
print("\n" + "="*80)
print("COMPARATIVE STATICS")
print("="*80)

df_sigma, df_noise = comparative_statics_analysis()

print(f"\nEffect of Uncertainty (Σ₀):")
print(f"  When Σ₀ doubles: λ increases by {np.sqrt(2):.2f}x (√2)")
print(f"  Higher uncertainty → higher adverse selection → wider spreads")

print(f"\nEffect of Noise Trading (σᵤ):")
print(f"  When σᵤ doubles: λ decreases by {1/np.sqrt(2):.2f}x (1/√2)")
print(f"  More noise traders → deeper market → lower price impact")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# Plot 1: Price impact vs order flow
axes[0, 0].scatter(df_single['total_flow'], df_single['price'] - params_single.p0, 
                   alpha=0.1, s=5)
flow_range = np.linspace(df_single['total_flow'].min(), df_single['total_flow'].max(), 100)
axes[0, 0].plot(flow_range, kyle_single.lam * flow_range, 'r-', linewidth=2, 
                label=f'λ = {kyle_single.lam:.4f}')
axes[0, 0].set_title('Price Impact Function')
axes[0, 0].set_xlabel('Total Order Flow (y)')
axes[0, 0].set_ylabel('Price Change (p - p₀)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Insider profit distribution
axes[0, 1].hist(df_single['insider_profit'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(kyle_single.expected_profit, color='red', linestyle='--', linewidth=2,
                   label=f'Expected: ${kyle_single.expected_profit:.2f}')
axes[0, 1].set_title('Insider Profit Distribution')
axes[0, 1].set_xlabel('Profit ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price error vs true value
axes[1, 0].scatter(df_single['true_value'], df_single['price_error'], alpha=0.1, s=5)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Price Efficiency')
axes[1, 0].set_xlabel('True Value (v)')
axes[1, 0].set_ylabel('Price Error (v - p)')
axes[1, 0].grid(alpha=0.3)

# Plot 4: Multi-period price paths
for i, sim in enumerate(simulations_multi[:20]):
    axes[1, 1].plot(sim['prices'], alpha=0.3, color='blue')
    axes[1, 1].axhline(sim['true_value'], alpha=0.2, linestyle='--', color='red')

axes[1, 1].set_title('Price Discovery Paths (20 simulations)')
axes[1, 1].set_xlabel('Period')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].grid(alpha=0.3)

# Plot 5: Comparative statics - Sigma0
ax1 = axes[2, 0]
ax1.plot(df_sigma['Sigma0'], df_sigma['lambda'], 'b-', linewidth=2, label='λ')
ax1.set_xlabel('Prior Variance (Σ₀)')
ax1.set_ylabel('Lambda (λ)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df_sigma['Sigma0'], df_sigma['expected_profit'], 'r-', linewidth=2, label='E[Profit]')
ax2.set_ylabel('Expected Profit ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax1.set_title('Effect of Uncertainty')

# Plot 6: Comparative statics - sigma_u
ax3 = axes[2, 1]
ax3.plot(df_noise['sigma_u'], df_noise['lambda'], 'b-', linewidth=2, label='λ')
ax3.set_xlabel('Noise Trader Volatility (σᵤ)')
ax3.set_ylabel('Lambda (λ)', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.grid(alpha=0.3)

ax4 = ax3.twinx()
ax4.plot(df_noise['sigma_u'], df_noise['expected_profit'], 'r-', linewidth=2)
ax4.set_ylabel('Expected Profit ($)', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax3.set_title('Effect of Noise Trading')

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Kyle's λ measures adverse selection cost: price impact per unit order flow")
print(f"2. Market depth inversely proportional to information asymmetry (Σ₀)")
print(f"3. Noise traders provide liquidity, allowing informed profit extraction")
print(f"4. Multi-period model shows gradual price discovery over time")
print(f"5. Insider trades strategically to balance profit vs information leakage")
print(f"6. Equilibrium spreads compensate market maker for adverse selection risk")
