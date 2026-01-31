# Square-Root Law of Market Impact

## 1. Concept Skeleton
**Definition:** Empirical regularity where market impact scales with square root of order size (I ∝ √Q) rather than linearly, reflecting capacity constraints and liquidity provider risk in order book markets  
**Purpose:** Quantify execution costs for institutional orders, optimize trade scheduling, estimate price impact for large positions, benchmark transaction costs  
**Prerequisites:** Order book mechanics, market microstructure, statistical estimation, concave functions, capacity theory

## 2. Comparative Framing
| Impact Model | Linear | Square-Root | Logarithmic | Power-Law General |
|--------------|--------|-------------|-------------|-------------------|
| **Form** | I ∝ Q | I ∝ √Q | I ∝ log(Q) | I ∝ Q^δ |
| **Exponent** | δ = 1.0 | δ = 0.5 | δ → 0 | δ ∈ [0.3, 0.7] |
| **Large Orders** | Overestimates | Accurate | Underestimates | Flexible |
| **Theoretical Basis** | Kyle model | Farmer-Lillo | Information theory | Empirical |

## 3. Examples + Counterexamples

**Simple Example:**  
Impact for 100 shares: $0.10. Square-root law predicts 10,000 shares (100x): $0.10 × √100 = $1.00, not $10 (linear would vastly overestimate)

**Failure Case:**  
Flash crash: All liquidity vanishes, 1000 shares moves price 10%, square-root law breaks down completely in extreme illiquidity regime

**Edge Case:**  
Very small orders (<1% ADV): Square-root approximates linear, distinction irrelevant below certain threshold

## 4. Layer Breakdown
```
Square-Root Law Framework:
├─ Mathematical Formulation:
│   ├─ Basic Form:
│   │   ├─ I = Y × σ × √(Q / V)
│   │   ├─ I: Market impact ($ or bps)
│   │   ├─ Y: Scaling constant (dimensionless)
│   │   ├─ σ: Daily volatility (%)
│   │   ├─ Q: Order size (shares)
│   │   ├─ V: Daily volume (shares)
│   │   └─ Q/V: Participation rate
│   ├─ Almgren et al. (2005) Formulation:
│   │   ├─ I = ε × σ × (Q / ADV)^δ
│   │   ├─ ε: Constant (~0.3-1.0 for δ=0.5)
│   │   ├─ ADV: Average daily volume
│   │   ├─ δ: Power-law exponent (~0.5 empirically)
│   │   └─ Calibrated to institutional data
│   ├─ Temporary Impact:
│   │   ├─ I_temp = γ × σ × √(Q/V) × (v/V)^δ_v
│   │   ├─ v: Instantaneous trading rate
│   │   ├─ δ_v: Intraday exponent (~0.6)
│   │   └─ Depends on urgency
│   └─ Permanent Impact:
│       ├─ I_perm = η × σ × (Q/V)^δ_p
│       ├─ δ_p: Often closer to linear (0.7-0.9)
│       ├─ Less concave than temporary
│       └─ Information component stronger
├─ Theoretical Foundations:
│   ├─ Farmer-Lillo Mechanical Model:
│   │   ├─ Order book as queueing system
│   │   ├─ Liquidity arrives stochastically
│   │   ├─ Large orders deplete multiple levels
│   │   ├─ Square-root from Poisson statistics
│   │   └─ Capacity constraint interpretation
│   ├─ Kyle Model (Linear):
│   │   ├─ Strategic informed trader
│   │   ├─ Linear equilibrium: I ∝ Q
│   │   ├─ Single trader assumption
│   │   └─ Square-root emerges with multiple traders
│   ├─ Bouchaud et al. (2004):
│   │   ├─ Latent liquidity model
│   │   ├─ Hidden orders replenish gradually
│   │   ├─ Square-root from renewal process
│   │   └─ Explains concavity empirically
│   └─ Information Capacity:
│       ├─ Shannon information theory
│       ├─ Impact ~ √(information content)
│       ├─ Large orders dilute per-share info
│       └─ Diminishing adverse selection per share
├─ Empirical Evidence:
│   ├─ Almgren et al. (2005):
│   │   ├─ Sample: 10,000+ institutional orders
│   │   ├─ Estimated δ ≈ 0.5-0.6
│   │   ├─ Robust across stock characteristics
│   │   ├─ Separate temporary vs permanent
│   │   └─ Industry standard calibration
│   ├─ Lillo, Farmer, Mantegna (2003):
│   │   ├─ LSE order book data
│   │   ├─ Confirmed square-root for large orders
│   │   ├─ Linear for very small orders
│   │   ├─ Crossover at ~1% ADV
│   │   └─ Universal across stocks
│   ├─ Bershova, Rakhlin (2013):
│   │   ├─ US equity markets
│   │   ├─ δ varies: 0.3 (liquid) to 0.7 (illiquid)
│   │   ├─ Time-varying estimates
│   │   ├─ Regime-dependent
│   │   └─ Crisis periods: higher δ (less concave)
│   └─ Cross-Market Evidence:
│       ├─ Futures: δ ≈ 0.4-0.5 (Cont et al.)
│       ├─ FX: δ ≈ 0.5-0.6
│       ├─ Bonds: δ ≈ 0.6-0.8 (less liquid)
│       └─ Universal across asset classes
├─ Practical Applications:
│   ├─ Pre-Trade Cost Estimation:
│   │   ├─ Estimate impact before execution
│   │   ├─ Input to Almgren-Chriss optimizer
│   │   ├─ Compare broker quotes
│   │   ├─ Size optimization (trade vs don't trade)
│   │   └─ Risk budgeting for strategies
│   ├─ Execution Algorithm Design:
│   │   ├─ VWAP slice sizing: minimize √Q impact
│   │   ├─ Optimal urgency: balance time risk vs impact
│   │   ├─ Child order scheduling
│   │   ├─ Dark pool routing decisions
│   │   └─ Adaptive learning from realized impact
│   ├─ Transaction Cost Analysis (TCA):
│   │   ├─ Benchmark: Expected impact vs actual
│   │   ├─ Skill attribution (outperformance)
│   │   ├─ Broker evaluation
│   │   ├─ Regulatory reporting (MiFID II)
│   │   └─ Client reporting (fiduciary duty)
│   ├─ Portfolio Construction:
│   │   ├─ Capacity constraints for strategies
│   │   ├─ Maximum AUM before impact erodes alpha
│   │   ├─ Rebalancing frequency optimization
│   │   ├─ Liquidity-adjusted returns
│   │   └─ Risk model adjustments
│   └─ Market Making:
│       ├─ Inventory management sizing
│       ├─ Quote sizing to limit adverse selection
│       ├─ Risk limits based on expected impact
│       └─ Hedging cost estimation
├─ Deviations & Limitations:
│   ├─ Very Small Orders:
│   │   ├─ Below ~0.1% ADV: nearly linear
│   │   ├─ Spread cost dominates
│   │   ├─ Tick size effects
│   │   └─ Square-root not applicable
│   ├─ Very Large Orders:
│   │   ├─ Above ~25% ADV: steeper than √Q
│   │   ├─ Market capacity exhausted
│   │   ├─ δ → 1 (linear or worse)
│   │   └─ Requires multi-day execution
│   ├─ Illiquid Stocks:
│   │   ├─ Higher exponent (δ ≈ 0.7-0.9)
│   │   ├─ Less concavity
│   │   ├─ Sparse order book
│   │   └─ Square-root less applicable
│   ├─ High-Frequency Trades:
│   │   ├─ Sub-second execution
│   │   ├─ Order book replenishment matters
│   │   ├─ Microstructure noise dominates
│   │   └─ Different scaling (toxic flow)
│   └─ Crisis Periods:
│       ├─ 2008/2020: δ increased to ~0.8
│       ├─ Liquidity evaporation
│       ├─ Square-root underestimates
│       └─ Regime-switching models needed
├─ Estimation Methods:
│   ├─ Regression Approach:
│   │   ├─ Log(I) = α + δ×Log(Q/V) + ε
│   │   ├─ OLS on meta-order data
│   │   ├─ Control for volatility, time-of-day
│   │   ├─ Robust standard errors (clustering)
│   │   └─ Test δ = 0.5 hypothesis
│   ├─ MLE Estimation:
│   │   ├─ Specify distributional assumptions
│   │   ├─ Likelihood: P(I | Q, σ, θ)
│   │   ├─ Estimate (Y, δ) jointly
│   │   ├─ More efficient than OLS
│   │   └─ Handles heteroskedasticity
│   ├─ Nonparametric:
│   │   ├─ Kernel regression: E[I | Q]
│   │   ├─ No functional form assumption
│   │   ├─ Flexible but requires large sample
│   │   └─ Visualize concavity directly
│   └─ Bayesian Methods:
│       ├─ Prior on (Y, δ) from literature
│       ├─ Update with proprietary data
│       ├─ Posterior distributions
│       └─ Uncertainty quantification
├─ Extensions & Refinements:
│   ├─ Stock-Specific Calibration:
│   │   ├─ Cluster stocks by liquidity profile
│   │   ├─ Large-cap: δ ≈ 0.4-0.5
│   │   ├─ Small-cap: δ ≈ 0.6-0.8
│   │   ├─ Estimate per stock if data sufficient
│   │   └─ Shrinkage estimators (James-Stein)
│   ├─ Time-of-Day Effects:
│   │   ├─ Open/close: higher Y (wider spreads)
│   │   ├─ Midday: lower Y (peak liquidity)
│   │   ├─ Time-varying parameter model
│   │   └─ Intraday seasonality adjustment
│   ├─ Volume Conditioning:
│   │   ├─ High volume days: lower impact
│   │   ├─ Low volume days: higher impact
│   │   ├─ Forecast daily volume first
│   │   └─ Adaptive participation rate
│   ├─ Order Book State:
│   │   ├─ Spread: wider → higher Y
│   │   ├─ Depth: thinner → higher Y
│   │   ├─ Imbalance: adverse → higher Y
│   │   └─ Real-time adjustment
│   └─ Multi-Day Impact:
│       ├─ Very large orders (>1 day ADV)
│       ├─ Impact accumulates across days
│       ├─ Coordination risk
│       └─ Frazzini et al. (2018) model
└─ Controversies & Debates:
    ├─ Universality:
    │   ├─ Is δ = 0.5 truly universal?
    │   ├─ Or artifact of sample/period?
    │   ├─ Conflicting evidence across studies
    │   └─ Consensus: 0.4-0.7 with variation
    ├─ Causality:
    │   ├─ Does trade size cause impact?
    │   ├─ Or both driven by information?
    │   ├─ Endogeneity bias in estimates
    │   └─ Instrumental variable approaches
    ├─ Temporary vs Permanent:
    │   ├─ Different exponents for each?
    │   ├─ Temporary: ~0.5
    │   ├─ Permanent: ~0.7-0.9
    │   └─ Decomposition challenging
    ├─ Regime Dependence:
    │   ├─ Single δ oversimplifies
    │   ├─ Need regime-switching models
    │   ├─ Normal vs stress periods
    │   └─ Computational burden
    └─ Microstructure Evolution:
        ├─ HFT changed liquidity provision
        ├─ Dark pools alter impact dynamics
        ├─ Historical estimates outdated?
        └─ Continuous recalibration needed
```

**Interaction:** Order size determination → Square-root law estimate → Expected impact calculation → Execution strategy optimization → Actual impact measurement → Model refinement

## 5. Mini-Project
Estimate and validate square-root law with simulation:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score

class SquareRootImpactModel:
    """Square-root market impact model"""
    
    def __init__(self, Y=1.0, delta=0.5):
        self.Y = Y          # Scaling constant
        self.delta = delta  # Power-law exponent
    
    def predict_impact(self, Q, V, sigma):
        """
        Predict market impact
        
        Q: Order size (shares)
        V: Daily volume (shares)
        sigma: Daily volatility (%)
        
        Returns: Impact in bps
        """
        participation = Q / V
        impact_pct = self.Y * sigma * (participation ** self.delta)
        impact_bps = impact_pct * 10000
        
        return impact_bps
    
    def fit(self, Q_array, V_array, sigma_array, I_array):
        """
        Fit model to data
        
        I_array: Observed impacts (bps)
        """
        # Transform to log-log for linear regression
        # log(I) = log(Y×σ) + δ×log(Q/V)
        
        participation = Q_array / V_array
        log_participation = np.log(participation)
        log_impact = np.log(I_array)
        log_vol_scaled_impact = np.log(I_array / (sigma_array * 10000))
        
        # Regression: log(I/σ) = log(Y) + δ×log(Q/V)
        slope, intercept, r_value, p_value, std_err = linregress(
            log_participation, log_vol_scaled_impact
        )
        
        self.delta = slope
        self.Y = np.exp(intercept)
        
        return {
            'delta': self.delta,
            'Y': self.Y,
            'R2': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

def simulate_market_impact_data(n_samples=1000, true_delta=0.5, noise_level=0.3):
    """
    Generate synthetic market impact data
    
    Includes:
    - Square-root law with noise
    - Stock characteristics (volatility, volume)
    - Realistic parameter ranges
    """
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Stock characteristics
        ADV = np.random.lognormal(14, 1.5)  # Mean ~1M shares, varies widely
        sigma = np.random.uniform(0.01, 0.04)  # 1-4% daily vol
        
        # Order size (0.1% to 50% of ADV)
        participation_rate = np.random.uniform(0.001, 0.5)
        Q = participation_rate * ADV
        
        # True impact (square-root law)
        Y_true = 1.0
        impact_true = Y_true * sigma * (participation_rate ** true_delta)
        
        # Add noise (measurement error, other factors)
        noise = np.random.normal(1.0, noise_level)
        impact_observed = impact_true * noise
        
        # Convert to bps
        impact_bps = impact_observed * 10000
        
        data.append({
            'Q': Q,
            'ADV': ADV,
            'sigma': sigma,
            'participation': participation_rate,
            'impact_bps': impact_bps,
            'impact_true_bps': impact_true * 10000
        })
    
    return pd.DataFrame(data)

def compare_models(df):
    """Compare linear vs square-root vs power-law models"""
    
    Q = df['Q'].values
    V = df['ADV'].values
    sigma = df['sigma'].values
    I_obs = df['impact_bps'].values
    
    models = {}
    
    # Model 1: Linear (δ = 1.0)
    model_linear = SquareRootImpactModel(delta=1.0)
    
    # Fit Y only (delta fixed)
    participation = Q / V
    I_normalized = I_obs / (sigma * 10000 * participation)
    model_linear.Y = np.median(I_normalized)
    
    I_pred_linear = model_linear.predict_impact(Q, V, sigma)
    r2_linear = r2_score(I_obs, I_pred_linear)
    
    models['Linear (δ=1.0)'] = {
        'model': model_linear,
        'predictions': I_pred_linear,
        'R2': r2_linear
    }
    
    # Model 2: Square-root (δ = 0.5)
    model_sqrt = SquareRootImpactModel(delta=0.5)
    
    # Fit Y only
    I_normalized = I_obs / (sigma * 10000 * (participation ** 0.5))
    model_sqrt.Y = np.median(I_normalized)
    
    I_pred_sqrt = model_sqrt.predict_impact(Q, V, sigma)
    r2_sqrt = r2_score(I_obs, I_pred_sqrt)
    
    models['Square-root (δ=0.5)'] = {
        'model': model_sqrt,
        'predictions': I_pred_sqrt,
        'R2': r2_sqrt
    }
    
    # Model 3: Power-law (estimate δ)
    model_power = SquareRootImpactModel()
    fit_results = model_power.fit(Q, V, sigma, I_obs)
    
    I_pred_power = model_power.predict_impact(Q, V, sigma)
    r2_power = r2_score(I_obs, I_pred_power)
    
    models['Power-law (δ estimated)'] = {
        'model': model_power,
        'predictions': I_pred_power,
        'R2': r2_power,
        'fit_results': fit_results
    }
    
    return models

# Generate data
print("="*80)
print("SQUARE-ROOT LAW OF MARKET IMPACT")
print("="*80)

print("\nGenerating synthetic market impact data...")
df_data = simulate_market_impact_data(n_samples=2000, true_delta=0.5, noise_level=0.3)

print(f"\nData Summary (N={len(df_data)}):")
print(f"  Participation rate: {df_data['participation'].min():.2%} to {df_data['participation'].max():.2%}")
print(f"  Order size: {df_data['Q'].min():.0f} to {df_data['Q'].max():.0f} shares")
print(f"  ADV: {df_data['ADV'].min():.0f} to {df_data['ADV'].max():.0f} shares")
print(f"  Volatility: {df_data['sigma'].min():.1%} to {df_data['sigma'].max():.1%}")
print(f"  Impact: {df_data['impact_bps'].min():.1f} to {df_data['impact_bps'].max():.1f} bps")

# Fit models
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

models = compare_models(df_data)

for name, result in models.items():
    print(f"\n{name}:")
    print(f"  Y = {result['model'].Y:.4f}")
    print(f"  δ = {result['model'].delta:.4f}")
    print(f"  R² = {result['R2']:.4f}")
    
    if 'fit_results' in result:
        print(f"  p-value: {result['fit_results']['p_value']:.6f}")
        print(f"  Std error: {result['fit_results']['std_err']:.6f}")

# Find best model
best_model_name = max(models.items(), key=lambda x: x[1]['R2'])[0]
print(f"\nBest Model: {best_model_name} (highest R²)")

# Subsample analysis by order size
print("\n" + "="*80)
print("IMPACT BY ORDER SIZE BUCKET")
print("="*80)

df_data['participation_bucket'] = pd.cut(df_data['participation'], 
                                          bins=[0, 0.01, 0.05, 0.1, 0.5],
                                          labels=['<1%', '1-5%', '5-10%', '>10%'])

bucket_stats = df_data.groupby('participation_bucket').agg({
    'impact_bps': ['mean', 'std', 'count'],
    'participation': 'mean'
}).round(2)

print("\nAverage Impact by Participation Rate:")
print(bucket_stats)

# Test square-root scaling
print("\n" + "="*80)
print("SQUARE-ROOT SCALING TEST")
print("="*80)

# Compare impact at different sizes (theoretical)
base_Q = 1000
base_V = 1000000
base_sigma = 0.02

model_sqrt = models['Square-root (δ=0.5)']['model']

print(f"\nTheoretical Impact Scaling (σ={base_sigma:.1%}, V={base_V:,}):")
print(f"  Order Size | Linear (δ=1) | Square-root (δ=0.5) | Ratio")
print(f"  -----------|--------------|---------------------|-------")

for multiplier in [1, 4, 16, 100]:
    Q = base_Q * multiplier
    
    impact_linear = model_sqrt.Y * base_sigma * (Q / base_V) * 10000
    impact_sqrt = model_sqrt.predict_impact(Q, base_V, base_sigma)
    
    ratio = impact_linear / impact_sqrt if impact_sqrt > 0 else 0
    
    print(f"  {Q:>9,} | {impact_linear:>11.2f} | {impact_sqrt:>18.2f} | {ratio:>6.2f}x")

print(f"\nKey Insight: Square-root law saves {(1 - 1/np.sqrt(100)):.0%} vs linear for 100x size")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Impact vs participation (log-log)
axes[0, 0].scatter(df_data['participation'], df_data['impact_bps'], 
                   alpha=0.3, s=10, label='Observed')

# Overlay model predictions
part_range = np.logspace(np.log10(df_data['participation'].min()),
                         np.log10(df_data['participation'].max()), 100)
sigma_ref = df_data['sigma'].median()

for name, result in models.items():
    model = result['model']
    impact_pred = model.Y * sigma_ref * (part_range ** model.delta) * 10000
    axes[0, 0].plot(part_range, impact_pred, linewidth=2, label=name)

axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('Impact vs Participation Rate (Log-Log)')
axes[0, 0].set_xlabel('Participation Rate (Q/ADV)')
axes[0, 0].set_ylabel('Impact (bps)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuals by model
best_model = models[best_model_name]
residuals = df_data['impact_bps'] - best_model['predictions']

axes[0, 1].scatter(best_model['predictions'], residuals, alpha=0.3, s=10)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title(f'Residuals: {best_model_name}')
axes[0, 1].set_xlabel('Predicted Impact (bps)')
axes[0, 1].set_ylabel('Residual (bps)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: R² comparison
model_names = list(models.keys())
r2_values = [models[name]['R2'] for name in model_names]

axes[0, 2].bar(range(len(model_names)), r2_values, alpha=0.7)
axes[0, 2].set_xticks(range(len(model_names)))
axes[0, 2].set_xticklabels(model_names, rotation=15, ha='right')
axes[0, 2].set_title('Model Comparison (R²)')
axes[0, 2].set_ylabel('R²')
axes[0, 2].set_ylim([0, 1])
axes[0, 2].grid(axis='y', alpha=0.3)

# Plot 4: Actual vs predicted
axes[1, 0].scatter(df_data['impact_bps'], best_model['predictions'], 
                   alpha=0.3, s=10)
axes[1, 0].plot([df_data['impact_bps'].min(), df_data['impact_bps'].max()],
                [df_data['impact_bps'].min(), df_data['impact_bps'].max()],
                'r--', linewidth=2, label='Perfect fit')
axes[1, 0].set_title(f'Actual vs Predicted: {best_model_name}')
axes[1, 0].set_xlabel('Actual Impact (bps)')
axes[1, 0].set_ylabel('Predicted Impact (bps)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 5: Impact by bucket
bucket_means = df_data.groupby('participation_bucket')['impact_bps'].mean()
bucket_stds = df_data.groupby('participation_bucket')['impact_bps'].std()

axes[1, 1].bar(range(len(bucket_means)), bucket_means, 
               yerr=bucket_stds, alpha=0.7, capsize=5)
axes[1, 1].set_xticks(range(len(bucket_means)))
axes[1, 1].set_xticklabels(bucket_means.index, rotation=15)
axes[1, 1].set_title('Average Impact by Participation Bucket')
axes[1, 1].set_ylabel('Impact (bps)')
axes[1, 1].grid(axis='y', alpha=0.3)

# Plot 6: Distribution of estimated delta (bootstrap)
print("\n" + "="*80)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("="*80)

n_bootstrap = 100
delta_estimates = []

for i in range(n_bootstrap):
    # Resample
    df_boot = df_data.sample(frac=1.0, replace=True)
    
    # Fit model
    model_boot = SquareRootImpactModel()
    fit_boot = model_boot.fit(df_boot['Q'].values, df_boot['ADV'].values,
                               df_boot['sigma'].values, df_boot['impact_bps'].values)
    
    delta_estimates.append(fit_boot['delta'])

delta_estimates = np.array(delta_estimates)

axes[1, 2].hist(delta_estimates, bins=30, alpha=0.7, edgecolor='black')
axes[1, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='True δ=0.5')
axes[1, 2].axvline(delta_estimates.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean={delta_estimates.mean():.3f}')
axes[1, 2].set_title('Bootstrap Distribution of δ')
axes[1, 2].set_xlabel('Estimated δ')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nBootstrap Results:")
print(f"  Mean δ: {delta_estimates.mean():.4f}")
print(f"  Std δ: {delta_estimates.std():.4f}")
print(f"  95% CI: [{np.percentile(delta_estimates, 2.5):.4f}, {np.percentile(delta_estimates, 97.5):.4f}]")

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Square-root law (δ≈0.5) consistently outperforms linear model")
print(f"2. Concavity reduces large order costs dramatically vs linear extrapolation")
print(f"3. Power-law with estimated δ provides best fit, validates theory")
print(f"4. Bootstrap CI confirms δ≈0.5 robust estimate")
print(f"5. Model essential for institutional execution cost forecasting")
print(f"6. Deviations occur for very small (<0.1% ADV) and large (>25% ADV) orders")
```

## 6. Challenge Round
Why square-root specifically, not other concave functions?
- **Theoretical**: Farmer-Lillo queueing + Poisson arrivals → √Q naturally
- **Empirical**: Fits data better than log(Q) or Q^0.3 alternatives
- **Simplicity**: Easier to compute and communicate than complex nonlinear forms
- **Universal**: Holds across markets, asset classes, time periods (robustness)
- **Debate**: Some argue δ=0.6 better for equities, 0.4 for futures, not strictly 0.5

When does square-root law fail?
- **Extreme illiquidity**: Small-cap stocks, δ → 1 (linear) as depth vanishes
- **Flash crashes**: All bets off, impact non-continuous, regime shift
- **Very large orders**: Above 50% ADV, capacity exhausted, steeper than √Q
- **Sub-second trading**: HFT regime different, adverse selection dominant
- **Crisis periods**: 2008/2020 saw temporary breakdown, higher exponents

## 7. Key References
- [Almgren et al. (2005): Direct Estimation of Equity Market Impact](https://www.risk.net/derivatives/equity-derivatives/1507748/direct-estimation-equity-market-impact)
- [Farmer, Gerig, Lillo, Waelbroeck (2013): How Efficiency Shapes Market Impact](https://qfinance.pm-research.com/content/4/2/111.short)
- [Bouchaud, Farmer, Lillo (2009): How Markets Slowly Digest Changes in Supply and Demand](https://arxiv.org/abs/0809.0822)

---
**Status:** Empirical law central to institutional trading | **Complements:** Market Impact Models, Optimal Execution, Capacity Analysis
