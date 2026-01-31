# Information Ratio

## 1. Concept Skeleton
**Definition:** Active return (alpha) per unit of active risk (tracking error)  
**Purpose:** Measure active management skill relative to benchmark, evaluate value of deviating from index  
**Prerequisites:** Portfolio tracking, benchmark selection, alpha/beta decomposition

## 2. Comparative Framing
| Metric | Information Ratio | Sharpe Ratio | Appraisal Ratio |
|--------|-------------------|--------------|-----------------|
| **Numerator** | Active return (Rp - Rb) | Excess return (Rp - rf) | Jensen's alpha |
| **Denominator** | Tracking error (TE) | Total volatility (σp) | Idiosyncratic risk |
| **Measures** | Active management skill | Total risk-adjusted return | Pure stock selection |
| **Benchmark** | Portfolio-specific | Risk-free rate | CAPM expected return |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio return: 12%, Benchmark return: 10%, Tracking error: 4%  
IR = (12-10)/4 = 0.50 (50 basis points alpha per 1% tracking error)

**Failure Case:**  
Closet indexer: 10.2% return, benchmark 10%, tracking error 0.5%  
IR = 0.2/0.5 = 0.40 (looks good, but essentially passive with high fees)

**Edge Case:**  
Tracking error = 0: Perfect index replication, IR undefined (division by zero)

## 4. Layer Breakdown
```
Information Ratio Calculation:
├─ Inputs:
│   ├─ Portfolio Returns: Rp,t (time series)
│   ├─ Benchmark Returns: Rb,t (must be relevant investable index)
│   └─ Time Period: Minimum 3 years, preferably 5+ years
├─ Active Return Computation:
│   ├─ Excess Returns: αt = Rp,t - Rb,t
│   ├─ Mean Active Return: α̅ = Mean(αt)
│   └─ Annualization: Multiply by periods per year
├─ Tracking Error Computation:
│   ├─ Standard Deviation: TE = StdDev(αt)
│   ├─ Annualization: TE × √(periods per year)
│   └─ Interpretation: Consistency of active deviations
├─ Information Ratio:
│   ├─ Formula: IR = α̅ / TE
│   ├─ Interpretation:
│   │   ├─ IR > 0.5: Good active management
│   │   ├─ IR > 1.0: Excellent (top quartile)
│   │   └─ IR < 0: Destroying value vs benchmark
│   └─ Consistency: IR measures repeatability, not just magnitude
├─ Relationship to Sharpe:
│   ├─ Sharpe² = IR² + Sharpe²benchmark (fundamental law)
│   ├─ IR captures value-add from active management
│   └─ Can decompose total Sharpe into passive + active components
└─ Fundamental Law of Active Management:
    ├─ IR = IC × √BR (Grinold & Kahn)
    ├─ IC: Information Coefficient (skill)
    ├─ BR: Breadth (number of independent bets)
    └─ Practical: Even modest skill + many bets → high IR
```

**Interaction:** Higher alpha or lower tracking error (more consistent alpha) → Higher IR

## 5. Mini-Project
Evaluate active management strategies using Information Ratio:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Parameters
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

# Benchmark
benchmark = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
benchmark_returns = benchmark.pct_change().dropna()

# Active fund strategies (simulated)
def create_active_strategy(benchmark_returns, alpha_annual, te_annual, 
                          name, skill_level):
    """
    Simulate active fund with target alpha and tracking error
    skill_level: consistency of alpha generation (0-1)
    """
    n = len(benchmark_returns)
    np.random.seed(hash(name) % 2**32)
    
    # Daily alpha and TE
    alpha_daily = alpha_annual / 252
    te_daily = te_annual / np.sqrt(252)
    
    # Generate active returns with varying skill
    noise = np.random.normal(0, te_daily, n)
    trend = alpha_daily * (1 + skill_level * np.sin(np.linspace(0, 4*np.pi, n)))
    
    active_returns = trend + noise
    portfolio_returns = benchmark_returns + active_returns
    
    return portfolio_returns, active_returns

# Define strategies
strategies = {
    'Skilled Active': {'alpha': 0.03, 'te': 0.05, 'skill': 0.7},
    'Lucky Active': {'alpha': 0.02, 'te': 0.08, 'skill': 0.2},
    'Closet Indexer': {'alpha': 0.005, 'te': 0.015, 'skill': 0.5},
    'High Conviction': {'alpha': 0.04, 'te': 0.12, 'skill': 0.6},
    'Unlucky Active': {'alpha': -0.01, 'te': 0.06, 'skill': 0.3},
}

# Generate returns
results = {}
for name, params in strategies.items():
    port_rets, active_rets = create_active_strategy(
        benchmark_returns, params['alpha'], params['te'], name, params['skill']
    )
    
    # Calculate metrics
    annual_alpha = active_rets.mean() * 252
    annual_te = active_rets.std() * np.sqrt(252)
    ir = annual_alpha / annual_te if annual_te > 0 else 0
    
    # Portfolio metrics
    port_annual_return = port_rets.mean() * 252
    port_annual_vol = port_rets.std() * np.sqrt(252)
    sharpe = (port_annual_return - 0.03) / port_annual_vol
    
    # Benchmark metrics
    bench_annual_return = benchmark_returns.mean() * 252
    bench_annual_vol = benchmark_returns.std() * np.sqrt(252)
    bench_sharpe = (bench_annual_return - 0.03) / bench_annual_vol
    
    # Statistical significance (t-statistic)
    t_stat = annual_alpha / (annual_te / np.sqrt(len(active_rets)/252))
    
    results[name] = {
        'Alpha (Annual)': annual_alpha,
        'Tracking Error': annual_te,
        'Information Ratio': ir,
        'Portfolio Return': port_annual_return,
        'Portfolio Volatility': port_annual_vol,
        'Sharpe Ratio': sharpe,
        't-statistic': t_stat,
        'Active Returns': active_rets,
        'Portfolio Returns': port_rets
    }

# Add benchmark
results['Benchmark (SPY)'] = {
    'Alpha (Annual)': 0.0,
    'Tracking Error': 0.0,
    'Information Ratio': np.nan,
    'Portfolio Return': bench_annual_return,
    'Portfolio Volatility': bench_annual_vol,
    'Sharpe Ratio': bench_sharpe,
    't-statistic': np.nan,
    'Active Returns': pd.Series([0]*len(benchmark_returns), index=benchmark_returns.index),
    'Portfolio Returns': benchmark_returns
}

# Summary table
summary_cols = ['Alpha (Annual)', 'Tracking Error', 'Information Ratio', 
                'Portfolio Return', 'Sharpe Ratio', 't-statistic']
summary = pd.DataFrame({k: {col: v[col] for col in summary_cols} 
                       for k, v in results.items()}).T
summary = summary.sort_values('Information Ratio', ascending=False)

print("Active Management Performance Analysis")
print("=" * 90)
print(summary.round(4))
print("\n" + "=" * 90)
print("Interpretation:")
print("  IR > 0.5: Good active management")
print("  IR > 1.0: Excellent (top quartile)")
print("  |t-stat| > 2: Statistically significant alpha (95% confidence)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: IR comparison
ir_values = summary['Information Ratio'].dropna().sort_values(ascending=True)
colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' 
          for x in ir_values]
axes[0, 0].barh(range(len(ir_values)), ir_values, color=colors)
axes[0, 0].set_yticks(range(len(ir_values)))
axes[0, 0].set_yticklabels(ir_values.index)
axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=1,
                   label='IR = 0.5 (Good)')
axes[0, 0].axvline(1.0, color='darkgreen', linestyle='--', linewidth=1,
                   label='IR = 1.0 (Excellent)')
axes[0, 0].set_xlabel('Information Ratio')
axes[0, 0].set_title('Information Ratio Comparison')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Alpha vs Tracking Error scatter
scatter_data = summary.dropna()
scatter = axes[0, 1].scatter(scatter_data['Tracking Error'],
                             scatter_data['Alpha (Annual)'],
                             s=200, alpha=0.6,
                             c=scatter_data['Information Ratio'],
                             cmap='RdYlGn', vmin=-0.5, vmax=1.5)

for idx in scatter_data.index:
    axes[0, 1].annotate(idx, 
                       (scatter_data.loc[idx, 'Tracking Error'],
                        scatter_data.loc[idx, 'Alpha (Annual)']),
                       fontsize=8, ha='center')

# Draw IR isolines
te_range = np.linspace(0.01, 0.15, 100)
for ir_level in [-0.5, 0, 0.5, 1.0]:
    alpha_line = ir_level * te_range
    axes[0, 1].plot(te_range, alpha_line, 'k--', alpha=0.3, linewidth=1)
    axes[0, 1].text(te_range[-1], alpha_line[-1], f'IR={ir_level}',
                   fontsize=8, ha='left')

axes[0, 1].axhline(0, color='black', linewidth=1)
axes[0, 1].set_xlabel('Tracking Error (Annual)')
axes[0, 1].set_ylabel('Alpha (Annual)')
axes[0, 1].set_title('Active Risk-Return Profile')
plt.colorbar(scatter, ax=axes[0, 1], label='Information Ratio')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Cumulative active returns
for name in ['Skilled Active', 'Lucky Active', 'Closet Indexer', 'Unlucky Active']:
    cumulative = (1 + results[name]['Active Returns']).cumprod()
    axes[1, 0].plot(cumulative.index, cumulative.values, 
                   label=name, linewidth=2)

axes[1, 0].axhline(1.0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cumulative Active Return')
axes[1, 0].set_title('Active Return Trajectories')
axes[1, 0].legend(loc='best', fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Rolling Information Ratio (1-year window)
window = 252  # 1 year
for name in ['Skilled Active', 'High Conviction', 'Closet Indexer']:
    active_rets = results[name]['Active Returns']
    
    rolling_alpha = active_rets.rolling(window).mean() * 252
    rolling_te = active_rets.rolling(window).std() * np.sqrt(252)
    rolling_ir = rolling_alpha / rolling_te
    
    axes[1, 1].plot(rolling_ir.index, rolling_ir.values,
                   label=name, linewidth=2)

axes[1, 1].axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Rolling 1-Year Information Ratio')
axes[1, 1].set_title('Information Ratio Stability Over Time')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(-2, 3)

plt.tight_layout()
plt.show()

# Fundamental Law of Active Management illustration
print("\n" + "=" * 90)
print("Fundamental Law of Active Management: IR = IC × √BR")
print("=" * 90)
print("\nAssuming IC (Information Coefficient) = 0.05 (realistic skill level):")
print(f"{'Breadth (Bets/Year)':<25} {'Expected IR':<15} {'Alpha (2% TE)':<20}")
print("-" * 60)
for breadth in [10, 50, 100, 250, 500]:
    ic = 0.05
    expected_ir = ic * np.sqrt(breadth)
    expected_alpha = expected_ir * 0.02  # Assuming 2% tracking error
    print(f"{breadth:<25} {expected_ir:<15.3f} {expected_alpha:<20.2%}")

print("\nInsight: Many small bets (high breadth) with modest skill beats")
print("         few large bets (low breadth) even with higher skill")
```

## 6. Challenge Round
When is Information Ratio the right metric?
- Active fund evaluation: Direct measure of manager value-add
- Benchmark-relative mandates: When goal is to beat specific index
- Risk budgeting: Allocate tracking error to highest IR strategies
- Portable alpha: Evaluate alpha generation independent of beta exposure

IR limitations:
- Benchmark selection critical: Wrong benchmark makes IR meaningless
- Backward-looking: Past IR doesn't guarantee future skill
- Closet indexing: High IR with tiny alpha may not justify fees
- Market-neutral strategies: IR undefined if no clear benchmark
- Correlation with benchmark: Assumes similar risk profile

## 7. Key References
- [Grinold & Kahn (1999) "Active Portfolio Management"](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826)
- [Investopedia - Information Ratio](https://www.investopedia.com/terms/i/informationratio.asp)
- [CFA Institute - Fundamental Law of Active Management](https://www.cfainstitute.org/en/research/foundation/2000/the-fundamental-law-of-active-management)

---
**Status:** Active management evaluation standard | **Complements:** Sharpe Ratio, Tracking Error, Jensen's Alpha
