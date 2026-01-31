# Linear vs Power-Law Market Impact Models

## 1. Concept Skeleton
**Definition:** Mathematical models quantifying how trade size affects execution price; linear assumes constant impact per unit, power-law captures acceleration of costs at larger volumes  
**Purpose:** Predict price concessions for various order sizes; optimize execution pace; estimate total transaction costs  
**Prerequisites:** Market microstructure, order book dynamics, liquidity measures, dimensionless analysis

## 2. Comparative Framing
| Model | **Linear** | **Power-Law** | **Kinked** | **Empirical** |
|-------|-----------|---------------|-----------|--------------|
| **Formula** | I = α + β×V | I = α×V^λ | Piecewise linear | Non-parametric |
| **Behavior** | Constant slope | Accelerating | Flat then steep | Data-driven |
| **Realism** | Low (oversimplified) | High for large trades | Medium | Highest |
| **Parameters** | 2 (α, β) | 2 (α, λ) | 3+ (kink points) | Many (varies) |
| **Computation** | Trivial | Simple | Moderate | Data-intensive |

## 3. Examples + Counterexamples

**Linear Model Example:**  
Selling 10,000 shares at market impact β = 0.50 bps/$1M volume. Mid price $100, daily volume $10M.  
Impact = 0.50 × (10k×100 / 10M) = 0.50 × 0.1 = 0.05 bps. Execution price: $99.9995 (5 basis points impact)

**Power-Law Example:**  
Same trade; empirical λ = 0.5 (square-root law). If impact doubles at 4× volume:  
I₁ = impact at V₁; I₂ = impact at 4V₁. Then I₂/I₁ = (4V₁/V₁)^0.5 = 2. Doubling confirmed.

**Counter-Example (Linear Fails):**  
Selling 100,000 shares (1% of daily volume) with linear model predicts 5 bps total impact. Actual: 15–20 bps observed. Linear underestimates; power-law (λ=0.6) predicts 12–18 bps (much closer)

**Kinked Model Example:**  
Small trades (<0.1% volume): β₁ = 0.3 bps/$1M (tight). Large trades (0.1–1% volume): β₂ = 1.0 bps/$1M (steep). Kink at 0.1%. Reflects market maker behavior shift (competitive below, defensive above)

**Edge Case (Microstructure Limit):**  
Order size: 1 share at $100, daily volume 10M shares, bid-ask spread 1 penny (1 bps).  
Power-law predicts impact: α × (1/10M)^λ ≈ near-zero. Reality: Bid-ask spread floor (1 penny minimum). Models break at tiny volumes.

## 4. Layer Breakdown
```
Market Impact Model Framework:
├─ Linear Model:
│   ├─ Specification: Impact = α + β × (Order_Size / Market_Volume)
│   │   ├─ α: Baseline fixed cost (market microstructure floor; bid-ask spread)
│   │   ├─ β: Slope (additional cost per unit of relative order size)
│   │   ├─ Order_Size: Trade quantity (shares)
│   │   ├─ Market_Volume: Benchmark volume (daily, hourly, or participation)
│   │   └─ Result: Impact in basis points or % of price
│   │
│   ├─ Interpretation:
│   │   ├─ Fixed component (α): Spread + dealer costs
│   │   ├─ Variable component (β × V): Adverse selection + inventory risk
│   │   ├─ Linearity assumption: Impact scales uniformly with order size
│   │   └─ Limitation: Ignores diminishing or accelerating effects at extremes
│   │
│   ├─ Estimation (from data):
│   │   ├─ Regress observed execution prices vs order sizes
│   │   ├─ Cross-sectional: Many orders of various sizes
│   │   ├─ Time-series: Same security over time with varying order sizes
│   │   ├─ OLS regression: I_i = α + β × V_i + ε_i
│   │   └─ Standard errors: Confidence intervals on α, β
│   │
│   ├─ Example Calibration (Large-Cap Equity):
│   │   ├─ Observed data: 200 trades, sizes 0.01–1% of daily volume
│   │   ├─ Regression output:
│   │   │   ├─ α (intercept): 0.4 bps (fixed bid-ask floor)
│   │   │   ├─ β (slope): 0.8 bps per 1% of volume
│   │   │   ├─ R²: 0.65 (moderate fit)
│   │   │   └─ Residual std: 2.5 bps (unexplained variation)
│   │   └─ Model: I = 0.4 + 0.8 × V (V as % of daily volume)
│   │
│   ├─ Application:
│   │   ├─ 0.05% order: I = 0.4 + 0.8×0.05 = 0.44 bps
│   │   ├─ 0.50% order: I = 0.4 + 0.8×0.50 = 0.80 bps
│   │   ├─ 2.00% order: I = 0.4 + 0.8×2.00 = 2.00 bps
│   │   └─ Scaling: Doubles when order size doubles (linear property)
│   │
│   └─ Advantages & Drawbacks:
│       ├─ Pros: Simplicity, interpretability, fast computation
│       ├─ Cons: Inaccurate for large/small orders; misses nonlinearity
│       └─ Best for: Small-to-mid size trades on liquid securities
│
├─ Power-Law Model:
│   ├─ Specification: Impact = α × (Order_Size / Market_Volume)^λ
│   │   ├─ α: Multiplier (sets scale of impact)
│   │   ├─ λ: Exponent (elasticity; how impact scales with volume)
│   │   │   ├─ λ = 0.5: Square-root law (empirically common)
│   │   │   ├─ λ = 1.0: Linear (special case of power-law)
│   │   │   ├─ λ = 1.5: Super-linear (very aggressive scaling)
│   │   │   └─ λ ∈ [0.4, 0.7]: Typical empirical range
│   │   └─ Behavior: Impact nonlinear; accelerates with order size
│   │
│   ├─ Interpretation:
│   │   ├─ Square-root law (λ=0.5):
│   │   │   ├─ 4× order size → 2× impact (proportional to √V)
│   │   │   ├─ Intuition: Market liquidity depth increases gradually; not linearly
│   │   │   ├─ Empirical support: Almgren, Chriss, Bouchaud et al.
│   │   │   └─ Mechanism: Limited inventory at each price level
│   │   │
│   │   ├─ Sub-linear vs Super-linear:
│   │   │   ├─ λ < 1: Concave (improved terms at larger size; rare)
│   │   │   ├─ λ = 1: Linear (uniform scaling)
│   │   │   ├─ λ > 1: Convex (worse terms at larger size; common)
│   │   │   └─ Typical: λ ≈ 0.6 balances between √(V) and V
│   │
│   ├─ Estimation (Nonlinear Regression):
│   │   ├─ Log-log specification: log(I) = log(α) + λ × log(V)
│   │   ├─ OLS on logs: Simplifies nonlinear relationship
│   │   ├─ Example calibration:
│   │   │   ├─ 200 trades (0.01–1% of daily volume)
│   │   │   ├─ Log-linear regression:
│   │   │   │   ├─ Intercept (log-α): -2.0 (α = exp(-2) ≈ 0.135)
│   │   │   │   ├─ Slope (λ): 0.55 (square-root-like behavior)
│   │   │   │   ├─ R²: 0.78 (better fit than linear)
│   │   │   │   └─ Model: I = 0.135 × V^0.55
│   │   └─ Interpretation: 10× volume → 3.6× impact (vs 10× linear)
│   │
│   ├─ Application Comparison (Linear vs Power-Law):
│   │   ├─ 0.05% trade:
│   │   │   ├─ Linear (α=0.4, β=0.8): I = 0.44 bps
│   │   │   ├─ Power-law (α=0.135, λ=0.55): I = 0.135 × 0.0005^0.55 = 0.032 bps
│   │   │   └─ Power-law lower (liquid small trades)
│   │   │
│   │   ├─ 0.50% trade:
│   │   │   ├─ Linear: I = 0.80 bps
│   │   │   ├─ Power-law: I = 0.135 × 0.005^0.55 = 0.169 bps
│   │   │   └─ Close agreement (mid-range)
│   │   │
│   │   └─ 2.00% trade:
│   │       ├─ Linear: I = 2.00 bps
│   │       ├─ Power-law: I = 0.135 × 0.02^0.55 = 0.377 bps
│   │       └─ Linear overstates (large trades less costly than linear predicts)
│   │
│   └─ Advantages & Drawbacks:
│       ├─ Pros: Captures nonlinearity, empirically accurate, better for wide size range
│       ├─ Cons: Slightly more complex estimation, λ sensitive to data quality
│       └─ Best for: All trade sizes; when precision matters
│
├─ Kinked (Piecewise Linear) Model:
│   ├─ Specification: Combines linear segments at different volume levels
│   │   ├─ Tier 1: V < V_kink → Impact = α₁ + β₁ × V (competitive)
│   │   ├─ Tier 2: V_kink ≤ V < V_kink2 → Impact = α₂ + β₂ × V (intermediate)
│   │   ├─ Tier 3: V ≥ V_kink2 → Impact = α₃ + β₃ × V (defensive)
│   │   └─ Example kinks: 0.1%, 0.5% of daily volume
│   │
│   ├─ Interpretation:
│   │   ├─ Market maker strategy:
│   │   │   ├─ Small orders: Tight spreads (compete for order flow)
│   │   │   ├─ Medium orders: Standard spreads (normal business)
│   │   │   ├─ Large orders: Wide spreads (inventory risk, adverse selection)
│   │   │   └─ Kinks reflect behavioral thresholds
│   │
│   ├─ Estimation:
│   │   ├─ Choose kink points: Economically motivated or data-driven
│   │   ├─ Fit linear segment in each tier
│   │   ├─ Ensure continuity at kinks (minimize discontinuity)
│   │   └─ Compare to alternatives (linear, power-law) via info criterion (AIC, BIC)
│   │
│   ├─ Application:
│   │   ├─ Tier structure:
│   │   │   ├─ 0–0.1% volume: β = 0.3 bps (tight)
│   │   │   ├─ 0.1–0.5% volume: β = 0.7 bps (normal)
│   │   │   ├─ 0.5%+ volume: β = 1.5 bps (defensive)
│   │   │   └─ Captures qualitative behavior of market depth
│   │
│   └─ Use Case:
│       ├─ When market behavior changes at identifiable thresholds
│       ├─ Example: Equity options with multiple dealer tiers
│       └─ Advantage: Interpretable; reflects business reality
│
├─ Empirical Considerations:
│   ├─ Data Quality:
│   │   ├─ Trade prices: Must be actual fills (not bids/offers)
│   │   ├─ Order sizes: Exactly as executed (not requested)
│   │   ├─ Volume benchmarks: Consistent definition (daily, hourly, by participant)
│   │   ├─ Survivorship bias: Exclude failed/partial fills
│   │   └─ Cleaning: Remove outliers (data errors, corporate actions)
│   │
│   ├─ Sample Selection:
│   │   ├─ Homogeneous securities: Separate models per security (or market cap bucket)
│   │   ├─ Time period: Market regime matters (low vol vs high vol)
│   │   ├─ Volume ranges: Consider model fit across size spectrum
│   │   ├─ Participation rate: Define relative to available liquidity
│   │   └─ Time-of-day effects: Morning vs afternoon may differ
│   │
│   ├─ Model Validation:
│   │   ├─ In-sample R²: Goodness of fit to historical data
│   │   ├─ Out-of-sample error: Test on held-out data (time series split)
│   │   ├─ Cross-validation: K-fold or rolling window
│   │   ├─ Residual analysis: Plot residuals vs predicted; check normality
│   │   └─ Sensitivity: Parameter stability across subsamples
│   │
│   └─ Stability Over Time:
│       ├─ Reestimate: Monthly or quarterly (market changes)
│       ├─ Structural breaks: Market microstructure evolution (decimalization, MiFID II)
│       ├─ Stress periods: Recalibrate during volatility spikes
│       └─ Forward-testing: Track prediction errors; flag degradation
```

## 5. Mini-Project: Comparing Linear vs Power-Law Impact Models

**Goal:** Estimate both models on synthetic trade data and compare accuracy.

```python
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
```

**Key Insights:**
- Power-law (λ=0.5) typically provides better out-of-sample fit than linear
- Linear adequate for small/mid-size trades; overstates large order costs
- Kinked model captures behavioral thresholds; useful for multi-tier market making
- Square-root law empirically supported across equities, FX, futures

## 6. Relationships & Dependencies
- **To Optimal Execution:** Impact model is input to Almgren-Chriss optimization
- **To Algorithm Design:** Determines participation rate strategy (POV, IS, VWAP)
- **To Risk Measurement:** Impact models quantify execution risk in portfolio optimization
- **To Cost Benchmarking:** VWAP/TWAP implementation shortfall depends on accurate impact

## References
- [Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=208282)
- [Bouchaud et al. (2009) "Market Microstructure of Bitcoin"](https://arxiv.org/abs/1011.3725)
- [Gatheral (2010) "No-Dynamic-Arbitrage and Market Impact"](https://www.math.nyu.edu/faculty/gatheral/)

