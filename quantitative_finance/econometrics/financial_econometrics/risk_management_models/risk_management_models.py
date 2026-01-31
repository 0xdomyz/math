import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# Set seed
np.random.seed(42)

# Generate simulated returns (fat-tailed: Student-t distribution)
T_history = 1000  # Historical data
T_backtest = 250  # Backtesting period
df_true = 5  # True degrees of freedom (fat tails)

# Generate returns using Student-t (fat-tailed)
returns_history = stats.t.rvs(df=df_true, loc=0.0005, scale=0.015, size=T_history)
returns_backtest = stats.t.rvs(df=df_true, loc=0.0005, scale=0.015, size=T_backtest)

# Portfolio value
portfolio_value = 10_000_000  # $10M

# VaR parameters
alpha = 0.99  # 99% confidence

print("="*70)
print("Risk Management: VaR & Expected Shortfall Analysis")
print("="*70)
print(f"Portfolio value: ${portfolio_value:,}")
print(f"Historical data: {T_history} days")
print(f"Backtesting period: {T_backtest} days")
print(f"Confidence level: {alpha*100}%")
print("")

# Method 1: Parametric VaR (Normal assumption)
mu_param = returns_history.mean()
sigma_param = returns_history.std()
z_alpha = stats.norm.ppf(alpha)

VaR_parametric_pct = -(mu_param - z_alpha * sigma_param)
VaR_parametric = VaR_parametric_pct * portfolio_value

ES_parametric_pct = -(mu_param - sigma_param * stats.norm.pdf(z_alpha) / (1 - alpha))
ES_parametric = ES_parametric_pct * portfolio_value

print("Method 1: Parametric VaR (Normal Assumption)")
print("-"*70)
print(f"  Mean: {mu_param*100:.4f}%")
print(f"  Std Dev: {sigma_param*100:.4f}%")
print(f"  VaR (99%): ${VaR_parametric:,.0f} ({VaR_parametric_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_parametric:,.0f} ({ES_parametric_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_parametric/VaR_parametric:.3f}")
print("")

# Method 2: Historical Simulation VaR
returns_sorted = np.sort(returns_history)
var_index = int((1 - alpha) * T_history)

VaR_historical_pct = -returns_sorted[var_index]
VaR_historical = VaR_historical_pct * portfolio_value

# ES: Average of returns beyond VaR
ES_historical_pct = -returns_sorted[:var_index].mean()
ES_historical = ES_historical_pct * portfolio_value

print("Method 2: Historical Simulation")
print("-"*70)
print(f"  VaR (99%): ${VaR_historical:,.0f} ({VaR_historical_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_historical:,.0f} ({ES_historical_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_historical/VaR_historical:.3f}")
print(f"  Worst historical loss: ${-returns_sorted[0]*portfolio_value:,.0f}")
print("")

# Method 3: Monte Carlo VaR (fit Student-t)
# Fit Student-t to historical data
params_t = stats.t.fit(returns_history)
df_fit, loc_fit, scale_fit = params_t

# Simulate
N_simulations = 10000
returns_mc = stats.t.rvs(df=df_fit, loc=loc_fit, scale=scale_fit, size=N_simulations)
returns_mc_sorted = np.sort(returns_mc)
var_index_mc = int((1 - alpha) * N_simulations)

VaR_mc_pct = -returns_mc_sorted[var_index_mc]
VaR_mc = VaR_mc_pct * portfolio_value

ES_mc_pct = -returns_mc_sorted[:var_index_mc].mean()
ES_mc = ES_mc_pct * portfolio_value

print("Method 3: Monte Carlo (Student-t Distribution)")
print("-"*70)
print(f"  Fitted df: {df_fit:.2f}")
print(f"  Fitted loc: {loc_fit*100:.4f}%")
print(f"  Fitted scale: {scale_fit*100:.4f}%")
print(f"  VaR (99%): ${VaR_mc:,.0f} ({VaR_mc_pct*100:.3f}%)")
print(f"  ES (99%): ${ES_mc:,.0f} ({ES_mc_pct*100:.3f}%)")
print(f"  ES/VaR ratio: {ES_mc/VaR_mc:.3f}")
print("")

# Backtesting
print("="*70)
print("Backtesting (Out-of-Sample)")
print("="*70)

methods = {
    'Parametric': VaR_parametric_pct,
    'Historical': VaR_historical_pct,
    'Monte Carlo': VaR_mc_pct
}

for method_name, var_pct in methods.items():
    # Count exceptions (losses exceeding VaR)
    losses = -returns_backtest  # Convert to losses
    exceptions = np.sum(losses > var_pct)
    exception_rate = exceptions / T_backtest
    expected_exceptions = (1 - alpha) * T_backtest
    
    # Kupiec Test
    p0 = 1 - alpha  # Theoretical exception rate
    p_hat = exception_rate  # Observed exception rate
    
    if exceptions > 0:
        LR = -2 * (np.log((1-p0)**(T_backtest - exceptions) * p0**exceptions) - 
                   np.log((1-p_hat)**(T_backtest - exceptions) * p_hat**exceptions))
    else:
        LR = -2 * np.log((1-p0)**(T_backtest))
    
    p_value = 1 - stats.chi2.cdf(LR, df=1)
    
    # Traffic light
    if exceptions <= 4:
        zone = "GREEN"
    elif exceptions <= 9:
        zone = "YELLOW"
    else:
        zone = "RED"
    
    print(f"\n{method_name} VaR:")
    print(f"  Expected exceptions: {expected_exceptions:.1f}")
    print(f"  Observed exceptions: {exceptions}")
    print(f"  Exception rate: {exception_rate*100:.2f}%")
    print(f"  Kupiec LR statistic: {LR:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'REJECT model' if p_value < 0.05 else 'Cannot reject model'}")
    print(f"  Basel Traffic Light: {zone}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Return distribution with VaR levels
axes[0, 0].hist(returns_history * 100, bins=50, color='blue', alpha=0.6, density=True, label='Historical')

# Overlay fitted distributions
x_range = np.linspace(returns_history.min(), returns_history.max(), 200)
axes[0, 0].plot(x_range * 100, stats.norm.pdf(x_range, mu_param, sigma_param) * 100, 
               linewidth=2, color='green', label='Normal fit', linestyle='--')
axes[0, 0].plot(x_range * 100, stats.t.pdf(x_range, df_fit, loc_fit, scale_fit) * 100,
               linewidth=2, color='red', label='Student-t fit')

# Mark VaR levels
axes[0, 0].axvline(-VaR_parametric_pct * 100, color='green', linestyle=':', linewidth=2, label='VaR Parametric')
axes[0, 0].axvline(-VaR_historical_pct * 100, color='orange', linestyle=':', linewidth=2, label='VaR Historical')
axes[0, 0].axvline(-VaR_mc_pct * 100, color='red', linestyle=':', linewidth=2, label='VaR MC')

axes[0, 0].set_xlabel('Daily Return (%)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Return Distribution & VaR Levels (99%)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Backtesting - Exceptions over time
losses_backtest = -returns_backtest * 100
days = np.arange(T_backtest)

axes[0, 1].plot(days, losses_backtest, linewidth=0.8, color='blue', alpha=0.7)
axes[0, 1].axhline(VaR_parametric_pct * 100, color='green', linestyle='--', linewidth=2, label='VaR Parametric')
axes[0, 1].axhline(VaR_historical_pct * 100, color='orange', linestyle='--', linewidth=2, label='VaR Historical')
axes[0, 1].axhline(VaR_mc_pct * 100, color='red', linestyle='--', linewidth=2, label='VaR MC')

# Mark exceptions
for method_name, var_pct in methods.items():
    exceptions_idx = np.where(losses_backtest > var_pct * 100)[0]
    if method_name == 'Monte Carlo':  # Only show MC exceptions for clarity
        axes[0, 1].scatter(exceptions_idx, losses_backtest[exceptions_idx], 
                          s=80, marker='x', color='red', linewidth=2, label='Exceptions (MC)')

axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Loss (%)')
axes[0, 1].set_title('Backtesting: Losses vs VaR Thresholds')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: QQ plot (check normality assumption)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns_history)))
sample_quantiles = np.sort(returns_history)

axes[1, 0].scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10, color='blue')
axes[1, 0].plot(theoretical_quantiles, theoretical_quantiles, 'r--', linewidth=2, label='Perfect fit')
axes[1, 0].set_xlabel('Theoretical Quantiles (Normal)')
axes[1, 0].set_ylabel('Sample Quantiles')
axes[1, 0].set_title('QQ-Plot: Testing Normality')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: VaR & ES comparison
methods_labels = ['Parametric\n(Normal)', 'Historical', 'Monte Carlo\n(Student-t)']
var_values = [VaR_parametric, VaR_historical, VaR_mc]
es_values = [ES_parametric, ES_historical, ES_mc]

x_pos = np.arange(len(methods_labels))
width = 0.35

axes[1, 1].bar(x_pos - width/2, np.array(var_values)/1e6, width, label='VaR (99%)', color='blue', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, np.array(es_values)/1e6, width, label='ES (99%)', color='red', alpha=0.7)

axes[1, 1].set_ylabel('Risk Measure ($M)')
axes[1, 1].set_title('VaR vs Expected Shortfall Comparison')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods_labels)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

# Add ES/VaR ratios as text
for i, (v, e) in enumerate(zip(var_values, es_values)):
    ratio = e/v
    axes[1, 1].text(i, e/1e6 + 0.05, f'{ratio:.2f}Ã—', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('risk_management_var_es.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("Key Insights:")
print("="*70)
print("1. Parametric VaR underestimates risk (assumes Normal; ignores fat tails)")
print("   â†’ Backtesting shows too many exceptions")
print("")
print("2. Monte Carlo with Student-t captures fat tails better")
print("   â†’ Fewer backtest failures; higher VaR (more conservative)")
print("")
print("3. Expected Shortfall (ES) 15-40% higher than VaR")
print("   â†’ Accounts for tail severity (not just probability)")
print("")
print("4. QQ-plot shows deviation in tails from Normal")
print("   â†’ Justifies Student-t or other fat-tailed distributions")
