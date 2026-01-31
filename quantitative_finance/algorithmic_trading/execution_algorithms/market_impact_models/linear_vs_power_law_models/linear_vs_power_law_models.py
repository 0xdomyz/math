import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# Generate synthetic trade data
np.random.seed(42)
n_trades = 200

# Order sizes (% of daily volume)
order_pct = np.random.uniform(0.01, 2.0, n_trades)

# True underlying power-law: I = 0.15 * V^0.5 + noise
true_alpha = 0.15
true_lambda = 0.50
impact_true = true_alpha * (order_pct ** true_lambda)

# Add noise (realistic; orders don't follow model perfectly)
noise = np.random.normal(0, 0.3, n_trades)  # ~0.3 bps std error
impact_observed = impact_true + noise

# Ensure no negative impacts
impact_observed = np.maximum(impact_observed, 0.01)

# Create DataFrame
df = pd.DataFrame({
    'order_pct': order_pct,
    'impact': impact_observed,
})

print(f"Trade data: {len(df)} trades, size range {df['order_pct'].min():.2f}–{df['order_pct'].max():.2f}% of volume\n")

# Model 1: Linear regression
# I = α + β × V
X_linear = np.column_stack([np.ones(n_trades), order_pct])
coeffs_linear = np.linalg.lstsq(X_linear, impact_observed, rcond=None)[0]
alpha_linear, beta_linear = coeffs_linear
impact_pred_linear = alpha_linear + beta_linear * order_pct

# Model 2: Power-law (via log-log)
# log(I) = log(α) + λ × log(V)
log_order = np.log(order_pct)
log_impact = np.log(impact_observed)
X_powerlaw = np.column_stack([np.ones(n_trades), log_order])
coeffs_log = np.linalg.lstsq(X_powerlaw, log_impact, rcond=None)[0]
log_alpha, lambda_fit = coeffs_log
alpha_powerlaw = np.exp(log_alpha)
impact_pred_powerlaw = alpha_powerlaw * (order_pct ** lambda_fit)

# Model 3: Kinked model
# Low tier: V < 0.5% → β₁
# High tier: V ≥ 0.5% → β₂
kink_threshold = 0.5
low_tier_mask = order_pct < kink_threshold
high_tier_mask = order_pct >= kink_threshold

alpha_low, beta_low = np.polyfit(order_pct[low_tier_mask], impact_observed[low_tier_mask], 1)
alpha_high, beta_high = np.polyfit(order_pct[high_tier_mask], impact_observed[high_tier_mask], 1)

impact_pred_kinked = np.where(
    low_tier_mask,
    alpha_low + beta_low * order_pct,
    alpha_high + beta_high * order_pct
)

# Metrics
models = {
    'Linear': impact_pred_linear,
    'Power-Law': impact_pred_powerlaw,
    'Kinked': impact_pred_kinked,
}

metrics = []
for name, pred in models.items():
    r2 = r2_score(impact_observed, pred)
    rmse = np.sqrt(mean_squared_error(impact_observed, pred))
    mae = np.mean(np.abs(impact_observed - pred))
    
    metrics.append({
        'Model': name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
    })

metrics_df = pd.DataFrame(metrics)
print("MODEL COMPARISON:\n")
print(metrics_df.to_string(index=False))

# Parameters
print(f"\n\nMODEL PARAMETERS:")
print(f"Linear:     α = {alpha_linear:.4f}, β = {beta_linear:.4f}")
print(f"Power-Law:  α = {alpha_powerlaw:.4f}, λ = {lambda_fit:.4f}")
print(f"Kinked:     Low (V<{kink_threshold}%): β₁={beta_low:.4f}")
print(f"            High (V≥{kink_threshold}%): β₂={beta_high:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Data and model fits (linear scale)
ax = axes[0, 0]
ax.scatter(order_pct, impact_observed, alpha=0.6, s=40, label='Observed', color='black')
ax.plot(order_pct, impact_pred_linear, linewidth=2, label=f'Linear (R²={r2_score(impact_observed, impact_pred_linear):.3f})', color='blue')
ax.plot(order_pct, impact_pred_powerlaw, linewidth=2, label=f'Power-Law (R²={r2_score(impact_observed, impact_pred_powerlaw):.3f})', color='red')
ax.plot(order_pct, impact_pred_kinked, linewidth=2, label=f'Kinked (R²={r2_score(impact_observed, impact_pred_kinked):.3f})', color='green', linestyle='--')
ax.axvline(kink_threshold, color='gray', linestyle=':', alpha=0.5, label=f'Kink at {kink_threshold}%')
ax.set_xlabel('Order Size (% of daily volume)')
ax.set_ylabel('Market Impact (bps)')
ax.set_title('Model Fits: Linear vs Power-Law vs Kinked')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Log-log plot (power-law visibility)
ax = axes[0, 1]
ax.loglog(order_pct, impact_observed, 'o', alpha=0.6, markersize=6, label='Observed', color='black')
ax.loglog(order_pct, impact_pred_powerlaw, linewidth=2, label=f'Power-Law: I={alpha_powerlaw:.3f}×V^{lambda_fit:.2f}', color='red')
# Linear on log-log appears as curved line
order_range = np.logspace(np.log10(order_pct.min()), np.log10(order_pct.max()), 100)
ax.loglog(order_range, alpha_linear + beta_linear * order_range, linewidth=2, label='Linear', color='blue', linestyle='--')
ax.set_xlabel('Order Size (% of daily volume) [log scale]')
ax.set_ylabel('Market Impact (bps) [log scale]')
ax.set_title('Log-Log Plot: Power-Law Appears Linear')
ax.legend()
ax.grid(alpha=0.3, which='both')

# Plot 3: Residuals vs fitted values
ax = axes[1, 0]
for name, pred in models.items():
    residuals = impact_observed - pred
    ax.scatter(pred, residuals, alpha=0.6, s=40, label=name)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Fitted Impact (bps)')
ax.set_ylabel('Residuals (bps)')
ax.set_title('Residual Plot (check for patterns)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Quantile-quantile (normality of residuals)
from scipy.stats import probplot
ax = axes[1, 1]
residuals_powerlaw = impact_observed - impact_pred_powerlaw
probplot(residuals_powerlaw, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Residual Normality Check (Power-Law)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity: Extrapolation to extreme sizes
print("\n\nEXTRAPOLATION TO EXTREME SIZES:")
test_sizes = [0.01, 0.1, 1.0, 5.0, 10.0]  # % of volume
print("\nOrder Size (% vol) | Linear (bps) | Power-Law (bps) | Ratio")
print("-" * 60)
for size in test_sizes:
    linear_pred = alpha_linear + beta_linear * size
    powerlaw_pred = alpha_powerlaw * (size ** lambda_fit)
    ratio = linear_pred / powerlaw_pred if powerlaw_pred > 0 else np.inf
    print(f"{size:16.2f} | {linear_pred:12.3f} | {powerlaw_pred:14.3f} | {ratio:6.2f}x")

print("\n→ Key insight: At large sizes, linear significantly overstates impact")
print(f"  (power-law more conservative at {test_sizes[-1]}% volume)")