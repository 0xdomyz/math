import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)

class PriceImpactModeler:
    def __init__(self):
        self.trades = []
        self.impacts = []
        
    def generate_empirical_data(self, num_samples=100):
        """Generate realistic empirical market impact data"""
        order_sizes = np.logspace(3, 7, num_samples)  # 1K to 10M shares
        true_lambda = 0.0003  # True impact parameter
        alpha = 0.55  # Power-law exponent (between linear and sqrt)
        
        daily_volumes = np.random.uniform(1e7, 1e8, num_samples)
        
        # Generate true impacts with power-law + noise
        true_impacts = []
        for q, v in zip(order_sizes, daily_volumes):
            # Power-law model: impact ∝ (Q/V)^α
            impact = true_lambda * (q / v) ** alpha
            # Add noise
            noise = np.random.normal(0, impact * 0.1)  # 10% noise
            true_impacts.append(impact + noise)
        
        return order_sizes, true_impacts, daily_volumes
    
    def fit_models(self, order_sizes, impacts, daily_volumes):
        """Fit different impact models to data"""
        
        # Linear model: impact = λ × Q
        def linear(q, lamb):
            return lamb * q
        
        # Square-root model: impact ∝ √Q
        def sqrt_model(q, lamb, v_daily):
            return lamb * np.sqrt(q / v_daily)
        
        # Power-law model: impact ∝ Q^α
        def power_law(q, lamb, alpha, v_daily):
            return lamb * (q / v_daily) ** alpha
        
        # Fit linear
        try:
            popt_linear, _ = curve_fit(linear, order_sizes, impacts)
            impacts_linear = linear(order_sizes, *popt_linear)
            rmse_linear = np.sqrt(np.mean((impacts - impacts_linear)**2))
        except:
            popt_linear = [0]
            rmse_linear = np.inf
        
        # Fit power-law (more complex)
        try:
            def power_law_fit(q, lamb, alpha):
                return lamb * (q / np.mean(daily_volumes)) ** alpha
            
            popt_power, _ = curve_fit(power_law_fit, order_sizes, impacts, p0=[0.0003, 0.5])
            impacts_power = power_law_fit(order_sizes, *popt_power)
            rmse_power = np.sqrt(np.mean((impacts - impacts_power)**2))
        except:
            popt_power = [0, 0.5]
            rmse_power = np.inf
        
        # Fit square-root (assume daily vol = mean)
        try:
            def sqrt_fit(q, lamb):
                return lamb * np.sqrt(q / np.mean(daily_volumes))
            
            popt_sqrt, _ = curve_fit(sqrt_fit, order_sizes, impacts)
            impacts_sqrt = sqrt_fit(order_sizes, *popt_sqrt)
            rmse_sqrt = np.sqrt(np.mean((impacts - impacts_sqrt)**2))
        except:
            popt_sqrt = [0]
            rmse_sqrt = np.inf
        
        return {
            'linear': {'params': popt_linear, 'predictions': impacts_linear, 'rmse': rmse_linear},
            'sqrt': {'params': popt_sqrt, 'predictions': impacts_sqrt, 'rmse': rmse_sqrt},
            'power': {'params': popt_power, 'predictions': impacts_power, 'rmse': rmse_power}
        }

# Scenario 1: Generate and fit models
print("Scenario 1: Market Impact Model Comparison")
print("=" * 80)

modeler = PriceImpactModeler()
order_sizes, impacts, daily_volumes = modeler.generate_empirical_data(num_samples=150)

models = modeler.fit_models(order_sizes, impacts, daily_volumes)

print("Model Fit Quality (RMSE):")
for model_name, model_data in models.items():
    print(f"  {model_name:>10}: RMSE = {model_data['rmse']:.6f}")

print(f"\nModel Parameters:")
print(f"  Linear:    λ = {models['linear']['params'][0]:.6f}")
print(f"  Sqrt:      λ = {models['sqrt']['params'][0]:.6f}")
print(f"  Power-law: λ = {models['power']['params'][0]:.6f}, α = {models['power']['params'][1]:.3f}")

# Scenario 2: Impact prediction comparison
print(f"\n\nScenario 2: Impact Predictions for Different Order Sizes")
print("=" * 80)

test_sizes = [10000, 50000, 100000, 500000, 1000000]
avg_daily_volume = np.mean(daily_volumes)

for size in test_sizes:
    impact_linear = models['linear']['params'][0] * size
    impact_sqrt = models['sqrt']['params'][0] * np.sqrt(size / avg_daily_volume)
    impact_power = models['power']['params'][0] * (size / avg_daily_volume) ** models['power']['params'][1]
    
    print(f"Order Size: {size:>10,} shares")
    print(f"  Linear model: {impact_linear*10000:>8.2f} cents")
    print(f"  Sqrt model:   {impact_sqrt*10000:>8.2f} cents")
    print(f"  Power model:  {impact_power*10000:>8.2f} cents")
    print()

# Scenario 3: Kyle Model Equilibrium
print(f"\n\nScenario 3: Kyle Model Impact Calculation")
print("=" * 80)

# Kyle parameters
sigma_v = 0.01  # Fundamental value volatility
sigma_u = 50000  # Uninformed volume std dev
lambda_kyle = 2 * sigma_v / sigma_u

print(f"Kyle Model Setup:")
print(f"  Value volatility: {sigma_v:.4f} (1%)")
print(f"  Uninformed volume σ: {sigma_u:,} shares")
print(f"  Implied λ: {lambda_kyle:.8f}")

kyle_test_sizes = [10000, 100000, 500000]
print(f"\nKyle Model Impact (for informed trader order size):")
for size in kyle_test_sizes:
    impact = lambda_kyle * size
    print(f"  {size:>10,} shares: {impact*10000:>8.2f} cents impact")

# Scenario 4: Almgren-Chriss execution schedule
print(f"\n\nScenario 4: Almgren-Chriss Optimal Execution")
print("=" * 80)

def almgren_chriss_cost(execution_sizes, lambdas_permanent, lambdas_temporary, volatility, time_periods):
    """Calculate AC cost function"""
    total_cost = 0
    unexecuted = sum(execution_sizes)
    
    for i, size in enumerate(execution_sizes):
        # Permanent impact cost
        permanent_cost = lambdas_permanent[i] * np.sqrt(size)
        
        # Temporary impact cost
        temporary_cost = lambdas_temporary[i] * size
        
        # Timing risk cost (from unexecuted quantity)
        timing_cost = volatility * np.sqrt(time_periods[i]) * unexecuted
        
        total_cost += permanent_cost + temporary_cost + timing_cost
        unexecuted -= size
    
    return total_cost

# Simple schedule comparison
total_order = 100000
periods = 10
per_period = total_order / periods

# Uniform execution
uniform_sizes = [per_period] * periods
uniform_lambdas_perm = [0.0001] * periods
uniform_lambdas_temp = [0.0001] * periods
uniform_times = np.arange(1, periods + 1)

uniform_cost = almgren_chriss_cost(uniform_sizes, uniform_lambdas_perm, uniform_lambdas_temp, 0.02, uniform_times)

# Front-loaded execution (more at start)
front_loaded = [per_period * 1.5 if i < 3 else per_period * 0.8 for i in range(periods)]
front_cost = almgren_chriss_cost(front_loaded, uniform_lambdas_perm, uniform_lambdas_temp, 0.02, uniform_times)

# Back-loaded execution (less at start)
back_loaded = [per_period * 0.8 if i < 3 else per_period * 1.2 for i in range(periods)]
back_cost = almgren_chriss_cost(back_loaded, uniform_lambdas_perm, uniform_lambdas_temp, 0.02, uniform_times)

print(f"Execution Schedule Comparison (100K order, 10 periods):")
print(f"  Uniform execution:    Cost = ${uniform_cost:>12,.0f}")
print(f"  Front-loaded:         Cost = ${front_cost:>12,.0f} ({(front_cost/uniform_cost - 1)*100:>+6.1f}%)")
print(f"  Back-loaded:          Cost = ${back_cost:>12,.0f} ({(back_cost/uniform_cost - 1)*100:>+6.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Model fits
sorted_idx = np.argsort(order_sizes)
axes[0, 0].scatter(order_sizes[sorted_idx], np.array(impacts)[sorted_idx], alpha=0.5, s=20, label='Data')
axes[0, 0].plot(order_sizes[sorted_idx], models['linear']['predictions'][sorted_idx], linewidth=2, label='Linear')
axes[0, 0].plot(order_sizes[sorted_idx], models['sqrt']['predictions'][sorted_idx], linewidth=2, label='Sqrt')
axes[0, 0].plot(order_sizes[sorted_idx], models['power']['predictions'][sorted_idx], linewidth=2, label='Power-law')
axes[0, 0].set_xlabel('Order Size (log scale)')
axes[0, 0].set_ylabel('Price Impact')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('Scenario 1: Model Fits to Data')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Model predictions
sizes_plot = np.logspace(3, 7, 50)
impacts_linear = models['linear']['params'][0] * sizes_plot
impacts_sqrt = models['sqrt']['params'][0] * np.sqrt(sizes_plot / avg_daily_volume)
impacts_power = models['power']['params'][0] * (sizes_plot / avg_daily_volume) ** models['power']['params'][1]

axes[0, 1].loglog(sizes_plot, impacts_linear * 10000, linewidth=2, label='Linear')
axes[0, 1].loglog(sizes_plot, impacts_sqrt * 10000, linewidth=2, label='Sqrt')
axes[0, 1].loglog(sizes_plot, impacts_power * 10000, linewidth=2, label='Power (α=0.55)')
axes[0, 1].set_xlabel('Order Size (log scale)')
axes[0, 1].set_ylabel('Impact (cents, log scale)')
axes[0, 1].set_title('Scenario 2: Scaling Behavior')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: RMSE comparison
model_names = ['Linear', 'Sqrt', 'Power-law']
rmses = [models['linear']['rmse'], models['sqrt']['rmse'], models['power']['rmse']]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))

bars = axes[1, 0].bar(model_names, rmses, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_title('Scenario 1: Model Fit Quality')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{rmse:.5f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Execution schedule costs
schedules = ['Uniform', 'Front-loaded', 'Back-loaded']
costs = [uniform_cost, front_cost, back_cost]
colors_sched = ['blue', 'red', 'green']

bars = axes[1, 1].bar(schedules, costs, color=colors_sched, alpha=0.7)
axes[1, 1].set_ylabel('Total Cost ($)')
axes[1, 1].set_title('Scenario 4: Execution Schedule Comparison')
axes[1, 1].grid(alpha=0.3, axis='y')

for bar, cost in zip(bars, costs):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Power-law exponent α ≈ {models['power']['params'][1]:.3f} (closer to √ than linear)")
print(f"Best-fit model: Power-law (lowest RMSE)")
print(f"Kyle model λ: {lambda_kyle:.8f} (impacts only informed trader orders)")
print(f"Almgren-Chriss shows front-loading {'reduces' if front_cost < uniform_cost else 'increases'} cost")
