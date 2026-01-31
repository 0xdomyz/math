import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats

np.random.seed(42)

# Kyle Model Simulation
class KyleModel:
    def __init__(self, p0=100, sigma_V=5, sigma_u=1000):
        """
        p0: prior price
        sigma_V: standard deviation of fundamental value
        sigma_u: standard deviation of noise trading
        """
        self.p0 = p0
        self.sigma_V = sigma_V
        self.sigma_u = sigma_u
        
        # Kyle's lambda (equilibrium price impact)
        self.lambda_kyle = np.sqrt(sigma_V**2 / (4 * sigma_u**2))
        
        # Informed trader's beta (aggressiveness)
        self.beta = 1 / (2 * self.lambda_kyle)
        
    def simulate_single_period(self):
        """Simulate one trading period"""
        # Draw true value
        V = np.random.normal(self.p0, self.sigma_V)
        
        # Informed trader observes V, chooses order size
        x = self.beta * (V - self.p0)
        
        # Noise traders
        u = np.random.normal(0, self.sigma_u)
        
        # Total order flow
        y = x + u
        
        # Market maker sets price
        p = self.p0 + self.lambda_kyle * y
        
        # Profits
        informed_profit = (V - p) * x
        noise_loss = -(V - p) * u
        mm_profit = 0  # Breaks even in expectation
        
        return {
            'V': V,
            'x': x,
            'u': u,
            'y': y,
            'p': p,
            'informed_profit': informed_profit,
            'noise_loss': noise_loss
        }
    
    def simulate_multi_period(self, n_periods=100):
        """Simulate multiple trading periods"""
        results = []
        for _ in range(n_periods):
            results.append(self.simulate_single_period())
        return results

# Simulation
p0 = 100
sigma_V = 5  # Fundamental volatility
sigma_u = 1000  # Noise trading std dev

model = KyleModel(p0=p0, sigma_V=sigma_V, sigma_u=sigma_u)

print("Kyle Model Simulation")
print("=" * 70)
print(f"\nModel Parameters:")
print(f"Prior Price: ${p0}")
print(f"Fundamental Std Dev: ${sigma_V}")
print(f"Noise Trading Std Dev: {sigma_u} shares")

print(f"\nEquilibrium Values:")
print(f"Kyle's Lambda (λ): ${model.lambda_kyle:.6f} per share")
print(f"Beta (aggressiveness): {model.beta:.4f}")
print(f"Market Depth (1/λ): {1/model.lambda_kyle:.0f} shares per $1")

# Theoretical predictions
theoretical_informed_profit = 0.5 * model.lambda_kyle * sigma_u**2
theoretical_R_squared = 0.5  # 50% of uncertainty resolved

print(f"\nTheoretical Predictions:")
print(f"Expected Informed Profit: ${theoretical_informed_profit:.2f}")
print(f"R² (information revealed): {theoretical_R_squared*100:.0f}%")
print(f"Price Error Var: {(1 - theoretical_R_squared) * sigma_V**2:.2f}")

# Run simulation
n_simulations = 1000
results = model.simulate_multi_period(n_simulations)

# Extract results
V_values = np.array([r['V'] for r in results])
x_values = np.array([r['x'] for r in results])
u_values = np.array([r['u'] for r in results])
y_values = np.array([r['y'] for r in results])
p_values = np.array([r['p'] for r in results])
informed_profits = np.array([r['informed_profit'] for r in results])
noise_losses = np.array([r['noise_loss'] for r in results])

# Empirical results
print(f"\nEmpirical Results ({n_simulations} simulations):")
print(f"Mean Informed Profit: ${informed_profits.mean():.2f} (theoretical: ${theoretical_informed_profit:.2f})")
print(f"Std Dev Informed Profit: ${informed_profits.std():.2f}")
print(f"Mean Noise Trader Loss: ${noise_losses.mean():.2f}")

# Price efficiency
price_errors = V_values - p_values
var_price_error = np.var(price_errors)
R_squared_empirical = 1 - var_price_error / sigma_V**2

print(f"\nPrice Efficiency:")
print(f"Var(V - p): {var_price_error:.2f} (theoretical: {(1-theoretical_R_squared)*sigma_V**2:.2f})")
print(f"Empirical R²: {R_squared_empirical*100:.1f}% (theoretical: {theoretical_R_squared*100:.0f}%)")
print(f"Mean Price Error: ${price_errors.mean():.4f}")
print(f"Std Dev Price Error: ${price_errors.std():.2f}")

# Verify pricing rule
slope, intercept, r_value, p_value, std_err = stats.linregress(y_values, p_values - p0)
print(f"\nEmpirical Pricing Rule: p = {p0} + {slope:.6f} × y")
print(f"Theoretical λ: {model.lambda_kyle:.6f}")
print(f"R² of pricing rule: {r_value**2:.4f}")

# Verify informed trading rule
slope_x, intercept_x, r_value_x, _, _ = stats.linregress(V_values - p0, x_values)
print(f"\nEmpirical Trading Rule: x = {slope_x:.4f} × (V - p₀)")
print(f"Theoretical β: {model.beta:.4f}")
print(f"R² of trading rule: {r_value_x**2:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Informed trading strategy
axes[0, 0].scatter(V_values - p0, x_values, alpha=0.3, s=10)
V_range = np.linspace((V_values - p0).min(), (V_values - p0).max(), 100)
x_predicted = model.beta * V_range
axes[0, 0].plot(V_range, x_predicted, 'r-', linewidth=2, 
               label=f'x = {model.beta:.4f}(V - p₀)')
axes[0, 0].set_xlabel('V - p₀ ($)')
axes[0, 0].set_ylabel('Informed Order Size (shares)')
axes[0, 0].set_title('Informed Trading Strategy: x = β(V - p₀)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Pricing rule
axes[0, 1].scatter(y_values, p_values - p0, alpha=0.3, s=10)
y_range = np.linspace(y_values.min(), y_values.max(), 100)
p_predicted = model.lambda_kyle * y_range
axes[0, 1].plot(y_range, p_predicted, 'r-', linewidth=2,
               label=f'p - p₀ = {model.lambda_kyle:.6f} × y')
axes[0, 1].set_xlabel('Total Order Flow y = x + u (shares)')
axes[0, 1].set_ylabel('Price Impact p - p₀ ($)')
axes[0, 1].set_title('Market Maker Pricing Rule: p = p₀ + λy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Price efficiency
axes[1, 0].scatter(V_values, p_values, alpha=0.3, s=10)
V_range = np.linspace(V_values.min(), V_values.max(), 100)
axes[1, 0].plot(V_range, V_range, 'r--', linewidth=2, label='Perfect Efficiency (p=V)')
axes[1, 0].plot(V_range, p0 + 0.5 * (V_range - p0), 'b-', linewidth=2,
               label='Partial Efficiency (R²=50%)')
axes[1, 0].set_xlabel('True Value V ($)')
axes[1, 0].set_ylabel('Market Price p ($)')
axes[1, 0].set_title(f'Price Discovery (Empirical R²={R_squared_empirical*100:.1f}%)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Profit distributions
axes[1, 1].hist(informed_profits, bins=50, alpha=0.6, label='Informed Trader', 
               color='green', edgecolor='black')
axes[1, 1].hist(noise_losses, bins=50, alpha=0.6, label='Noise Traders', 
               color='red', edgecolor='black')
axes[1, 1].axvline(informed_profits.mean(), color='green', linestyle='--', 
                  linewidth=2, label=f'Informed Mean: ${informed_profits.mean():.2f}')
axes[1, 1].axvline(noise_losses.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Noise Mean: ${noise_losses.mean():.2f}')
axes[1, 1].set_xlabel('Profit ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Profit Distributions')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Test theoretical relationship: E[π_informed] = 0.5 * λ * σ_u²
t_stat, p_val = stats.ttest_1samp(informed_profits, theoretical_informed_profit)
print(f"\nHypothesis Test: H0: E[π] = {theoretical_informed_profit:.2f}")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_val:.4f}")
if p_val > 0.05:
    print("Result: Cannot reject H0 - consistent with theory")
else:
    print("Result: Reject H0")

# Comparative statics: vary noise trading
print(f"\n\nComparative Statics: Effect of Noise Trading")
print("=" * 70)

sigma_u_values = [500, 1000, 2000, 5000]
for sigma_u_test in sigma_u_values:
    model_test = KyleModel(p0=p0, sigma_V=sigma_V, sigma_u=sigma_u_test)
    theoretical_profit = 0.5 * model_test.lambda_kyle * sigma_u_test**2
    
    print(f"\nσ_u = {sigma_u_test}:")
    print(f"  λ = {model_test.lambda_kyle:.6f}")
    print(f"  β = {model_test.beta:.4f}")
    print(f"  Market Depth = {1/model_test.lambda_kyle:.0f} shares/$")
    print(f"  Expected Informed Profit = ${theoretical_profit:.2f}")

# Verify zero-sum property
total_pnl = informed_profits + noise_losses
print(f"\n\nZero-Sum Verification:")
print(f"Mean Total PnL: ${total_pnl.mean():.4f} (should be ≈ 0)")
print(f"Std Dev Total PnL: ${total_pnl.std():.2f}")

# Market maker breaks even
mm_pnl = -(informed_profits + noise_losses)
print(f"Market Maker Mean PnL: ${mm_pnl.mean():.4f} (should be ≈ 0)")

# Information content of order flow
# Regress V on y to see how much information order flow contains
slope_info, intercept_info, r_value_info, _, _ = stats.linregress(y_values, V_values)
print(f"\n\nInformation Content of Order Flow:")
print(f"Regression: V = {intercept_info:.2f} + {slope_info:.6f} × y")
print(f"R²: {r_value_info**2:.3f}")
print(f"Interpretation: Order flow explains {r_value_info**2*100:.1f}% of fundamental value variation")
