# Price Impact Models

## 1. Concept Skeleton
**Definition:** Mathematical frameworks quantifying price movement from order size; scaling laws; parametric approximations of market impact  
**Purpose:** Predict execution costs; optimize order scheduling; benchmark execution quality; risk management  
**Prerequisites:** Market impact concept, execution algorithms, statistical estimation, optimization

## 2. Comparative Framing
| Model | Formula | Permanent | Temporary | Use Case |
|-------|---------|-----------|-----------|----------|
| **Linear** | Impact ∝ Q | Simple | Wrong | Baseline |
| **Square-root** | Impact ∝ √Q | Standard | Partial | Optimal execution |
| **Almgren-Chriss** | Linear + decay | Yes | Yes | Algorithm design |
| **Kyle** | Game-theoretic | Informed only | MM inventory | Theory |
| **Power-law** | Impact ∝ Q^α | Flexible | Nonlinear | Empirical fitting |

## 3. Examples + Counterexamples

**Square-Root Works:**  
10K order → 2 cents impact; 40K order → 4 cents (not 8) → √4 = 2x → square-root scales correctly → most empirical studies confirm

**Linear Fails Badly:**  
Assume linear ($0.002 per 1000 shares); 10K → $0.02 (correct); 100K → $0.20 (huge overestimate; actually $0.063) → linear dangerously underprices small orders, overprices large

**Kyle Model Too Theoretical:**  
Works perfectly in 1-asset, 2-trader model → real markets have 1000s of assets, millions of traders → structure fails → useful for intuition but not practical

**Nonlinear (Power-law) Capture Reality:**  
Empirical data: 10K → 2c, 100K → 7c (not √100 = 10c) → power-law α=0.6 fits better → more realistic but harder to calibrate

## 4. Layer Breakdown
```
Price Impact Models Framework:
├─ Foundational Models:
│   ├─ Linear Model:
│   │   - Formula: Impact = λ × Q (Q = order size in shares)
│   │   - Assumption: Impact proportional to volume
│   │   - Advantage: Simple, interpretable
│   │   - Disadvantage: Empirically wrong (underestimates small, overestimates large)
│   │   - Application: Quick estimates, benchmarking
│   │   - Reality: Fails because market has more depth in small vs large trades
│   ├─ Square-Root Model:
│   │   - Formula: Impact = σ × √(Q / V_daily)
│   │   - Components: σ = volatility, Q = order size, V_daily = daily volume
│   │   - Intuition: Large orders relative to daily volume hit deeper into book
│   │   - Empirical: 70-80% of studies confirm
│   │   - Advantage: Better captures law of one price (market depth)
│   │   - Application: Optimal execution, execution algorithms
│   ├─ Power-Law Model:
│   │   - Formula: Impact = λ × (Q / V_daily)^α
│   │   - α ≈ 0.5-0.6 typically (between linear and square-root)
│   │   - Advantage: Flexible, fits empirical data better
│   │   - Disadvantage: Harder to estimate, less interpretable
│   │   - Recent: Gaining acceptance in academic literature
│   └─ Temporary Component (Bid-Ask):
│       - Formula: Temporary = s/2 (half the spread)
│       - Reverts quickly (minutes)
│       - Permanent = Total - Temporary
│       - Decomposition: Total impact = Permanent + Temporary
│
├─ Almgren-Chriss Optimal Execution Framework:
│   ├─ Model Setup:
│   │   - Objective: Minimize cost = market impact + timing risk
│   │   - Constraints: Must execute full order in T seconds
│   │   - Decision: How much to execute per second?
│   │   - Tradeoff: Execute faster → less price risk but more impact
│   │   - Execute slower → less impact but more price risk
│   ├─ Components:
│   │   - Permanent impact: I_P = λ × √(n_i) (from trades)
│   │   - Temporary impact: I_T = σ_s × n_i (from spread and MM inventory)
│   │   - Volatility cost: σ × √(t) × (X - Σn_i) (unexecuted portion)
│   │   - Total cost: Σ(I_P + I_T) + σ × √(t) × position_risk
│   ├─ Optimal Solution:
│   │   - Derived: Follows hyperbolic path
│   │   - Meaning: Execute more initially, less later
│   │   - Intuition: Reduces position risk from unexecuted quantity
│   │   - Parameter: Risk aversion λ determines trajectory
│   │   - High λ: Execute fast (ignore impact), accept market risk
│   │   - Low λ: Execute slow (minimize impact), accept position risk
│   ├─ Empirical Application:
│   │   - Input: Order size, daily volume, volatility, time horizon
│   │   - Output: Optimal execution schedule (how much per minute)
│   │   - Advantage: Mathematically optimal trade-off
│   │   - Limitation: Assumes impact is known/constant
│   │   - Practice: Widely used by fund managers
│   └─ Extensions:
│       - Stochastic: Add random walk to prices
│       - Multi-period: Discrete vs continuous trading
│       - Slippage: Model imperfect execution
│       - Learning: Update estimate as execution proceeds
│
├─ Kyle Model (Game-Theoretic):
│   ├─ Setup:
│   │   - Informed trader: Knows true value
│   │   - MM (market maker): Sets prices
│   │   - Uninformed: Random demand/supply
│   │   - Duration: Single period (one trade)
│   ├─ Equilibrium:
│   │   - Price: p = v + λ × (order flow)
│   │   - λ (impact) depends on:
│   │     - Uncertainty about fundamental value
│   │     - Uninformed demand volatility
│   │     - MM risk aversion
│   │   - Result: More uninformed noise → lower impact (harder to detect informed)
│   ├─ Predictions:
│   │   - Spread: Wide if more uncertainty
│   │   - Price move: Permanent (all from informed trading)
│   │   - No temporary: Temporary component not modeled (not in model)
│   │   - Liquidity: More uninformed volume → deeper market
│   ├─ Empirical Fit:
│   │   - Partial: Explains why spreads vary
│   │   - Missing: Doesn't explain inventory effects (temporary)
│   │   - Theoretical: Useful for qualitative intuition
│   │   - Applied: Hard to estimate λ parameter
│   └─ Limitations:
│       - One period: Real markets continuous
│       - Two traders: Real markets many
│       - Symmetric info: Real has many info asymmetries
│       - Static: Doesn't model learning/update
│
├─ Empirical Findings (Market Impact Laws):
│   ├─ Scaling with Order Size:
│   │   - Doubled volume: ~√2 ≈ 1.4x impact (not 2x)
│   │   - 4x volume: ~2x impact
│   │   - Supports square-root law well
│   │   - Typical: 10K→2c, 100K→6-7c, 1M→20c
│   ├─ Asset Class Differences:
│   │   - Large cap: √ law holds strongly
│   │   - Small cap: Steeper than √ (closer to linear)
│   │   - ETFs: Similar to large cap
│   │   - Futures: Steeper (less liquidity)
│   │   - Currencies: Moderate to √
│   ├─ Liquidity Effects:
│   │   - High volume periods: Smaller impact
│   │   - Low volume (illiquid): Larger impact
│   │   - Spread environment: Wider spreads → larger impact
│   │   - Order type: Market orders > limit orders
│   ├─ Temporal Decay:
│   │   - Immediate: Full impact present
│   │   - Minutes: 70-80% remains
│   │   - Hours: 30-50% remains
│   │   - Days: 10-20% remains (mostly permanent)
│   │   - Interpretation: Permanent component settles over hours
│   ├─ Asymmetric Buy/Sell:
│   │   - Buy vs sell same size: Different impacts
│   │   - Dealer preference: Often can sell cheaper than buy (short-biased)
│   │   - Effect: 10-30% asymmetry typical
│   │   - Implication: Directional demand matters
│   └─ Seasonal Patterns:
│       - Month-end: Rebalancing increases impact
│       - Open/close: Auctions reduce impact temporarily
│       - Earnings: Volatility spikes increase impact
│       - Market stress: Impact dramatically higher
│
├─ Practical Calibration:
│   ├─ Historical Regression:
│   │   - Method: Regress price changes on order flow
│   │   - Coefficient: Estimate of λ
│   │   - Data: Need order-level data + prices
│   │   - Sample: Typically 100+ days of history
│   │   - Output: λ = 0.0001-0.0005 typical (asset dependent)
│   ├─ TAQ Database Method:
│   │   - Source: TAQ (Trade and Quote) data from exchanges
│   │   - Processing: Match trades to order book snapshots
│   │   - Advantage: Detailed microstructure data
│   │   - Challenge: High-frequency noise
│   │   - Practice: Filter, aggregate, run regression
│   ├─ Cross-Sectional Estimation:
│   │   - Method: Compare different stocks at same time
│   │   - Regression: Impact = f(size, volatility, spread, volume)
│   │   - Advantage: Robust across time
│   │   - Challenge: Omitted variable bias likely
│   ├─ Information-Based Estimation:
│   │   - Use PIN measure (probability of informed trading)
│   │   - High PIN: Higher expected impact
│   │   - Advantage: Theory-grounded
│   │   - Challenge: PIN complex to estimate
│   └─ Real-Time Monitoring:
│       - Execution: Track actual vs predicted impact
│       - Adjustment: Update estimate as new data arrives
│       - Feedback: Improve estimate in-the-moment
│       - Benefit: Adaptive algorithm learning
│
├─ Market Impact During Stress:
│   ├─ Flash Crash (May 6, 2010):
│   │   - Normal impact: ~5-10 cents per order
│   │   - Flash crash: $3-4 moves (100x normal)
│   │   - Duration: 5 minutes
│   │   - Cause: Liquidity evaporation + algorithmic cascade
│   │   - Lesson: Normal models break down in stress
│   ├─ Liquidity Drought:
│   │   - Mechanism: MM pulls bids → price gaps widen
│   │   - Effect: Order cannot fill at reasonable price
│   │   - Recovery: Takes hours/days for depth restoration
│   │   - Prediction: Models can't predict this (regime change)
│   ├─ Contagion Effects:
│   │   - One asset stress: Spreads widen across portfolio
│   │   - Correlation: Usually uncorrelated pairs move together
│   │   - Cause: Systemic liquidity demand (everyone sells)
│   │   - Impact: Much larger than single-stock models
│   └─ Fat Tail Behavior:
│       - Normal distribution: Assume impact ~N(μ, σ)
│       - Reality: Distribution has fat tails
│       - Implication: 2-3σ events happen 10x more often
│       - Risk: Standard models underestimate tail risk
│
├─ Advanced Model Extensions:
│   ├─ Nonlinear Impact:
│   │   - Assumption: Impact linear in order size (may not be)
│   │   - Reality: Convex (larger orders hit deeper, exponential cost)
│   │   - Model: Impact ∝ Q^α, α > 1 at extremes
│   │   - Implication: Very large orders far more expensive
│   ├─ Information Dynamics:
│   │   - Assumption: Impact constant (it's not)
│   │   - Reality: Depends on public information
│   │   - Timing: Before earnings → high impact
│   │   - After earnings → normal impact
│   │   - Model: λ = f(information_uncertainty)
│   ├─ Cross-Asset Impacts:
│   │   - Correlated assets: Buy one → affects others
│   │   - Mechanism: Market makers hedge across assets
│   │   - Effect: Systematic impact extension
│   │   - Example: Large S&P 500 buy → all components rise
│   ├─ Machine Learning Approaches:
│   │   - Model: Neural networks to learn λ from data
│   │   - Advantage: Captures nonlinearities automatically
│   │   - Data: Needs large historical dataset
│   │   - Challenge: Overfitting, non-stationary environments
│   └─ Regime-Switching Models:
│       - Normal regime: √ law holds
│       - Stress regime: Higher β (steeper impact)
│       - Detection: Monitor liquidity indicators
│       - Benefit: Switch models based on regime
│
└─ Implementation Challenges:
    ├─ Parameter Estimation:
    │   - λ changes over time (non-stationary)
    │   - Depends on market conditions
    │   - Historical data may not predict future
    │   - Solution: Rolling window, adaptive updates
    ├─ Order Size Boundary:
    │   - What counts as "order size"?
    │   - One large order? or split across venues?
    │   - Includes/excludes dark pool volume?
    │   - Definition affects λ estimate materially
    ├─ Endogeneity Problem:
    │   - Informed traders trade when want to
    │   - Creates correlation: Order flow + price
    │   - Causality unclear: Does order cause price or vice versa?
    │   - Solution: Use instrumental variables
    ├─ Multiple Time Scales:
    │   - Impact decay at different rates
    │   - 10ms decay ≠ 10 second decay
    │   - Model must specify time horizon
    │   - Risk: Applying 1-minute model to 1-second execution
    └─ Regime Dependence:
        - Model calibrated in normal times
        - Breaks in stress
        - Robustness testing: Use crisis data
        - Challenge: Crisis data sparse (hard to learn from)
```

**Interaction:** Trader submits large market order → exchange consults impact model → predicts √(size) movement → distributes execution across time/venues accordingly → outcome matches prediction (model validated)

## 5. Mini-Project
Estimate and compare multiple price impact models:
```python
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
```

## 6. Challenge Round
Square-root impact law is empirically validated, yet traders still get surprised by execution costs—why doesn't everyone use this model?

- **Parameter uncertainty**: λ varies with market conditions; estimated λ from old data doesn't match current reality → need adaptive updates → manual recalibration misses changes
- **Nonlinearities at extremes**: √ law breaks for very large/very small orders → model assumes regime that doesn't hold at boundaries → prediction fails exactly when needed
- **Omitted variables**: Doesn't account for information content of order → informed buyers get worse prices than uninformed → model can't distinguish → residual shock
- **Liquidity regime changes**: Model calibrated in calm → crisis arrives → liquidity evaporates → impact becomes 10x model predictions → model assumes stationarity that doesn't hold
- **Endogeneity**: Smart traders trade when want to (selection effect) → order timing correlated with price moves → causality backwards → model overstates impact at selected times

## 7. Key References
- [Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf)
- [Kyle (1985) - Continuous Auctions and Insider Trading](https://www.jstor.org/stable/1913210)
- [Bouchaud et al (2004) - Empirical Market Microstructure Properties](https://arxiv.org/abs/cond-mat/0406224)
- [Gatheral et al (2012) - Price Impact Models and Market Microstructure Noise](https://arxiv.org/abs/1202.6283)

---
**Status:** Mathematical quantification of market impact | **Complements:** Execution Algorithms, Optimal Execution, Permanent/Temporary Impact
