# Alpha (α)

## 1. Concept Skeleton
**Definition:** Excess return above CAPM prediction; risk-adjusted performance measure indicating skill or mispricing after accounting for systematic risk  
**Purpose:** Evaluate active management skill, identify market inefficiencies, separate luck from skill  
**Prerequisites:** CAPM, beta, regression analysis, Jensen's alpha, risk-adjusted returns

## 2. Comparative Framing
| Alpha Type | Jensen's Alpha | Portable Alpha | Smart Beta | Statistical Alpha |
|------------|----------------|----------------|------------|-------------------|
| **Definition** | Intercept from CAPM regression | Excess return transferable to other strategies | Factor-based systematic returns | Regression residual |
| **Source** | Stock selection skill | Leverage + derivatives | Exploiting known anomalies | Idiosyncratic outperformance |
| **Persistence** | Debated (often luck) | Strategy-dependent | Decays as crowded | Unlikely to persist |
| **CAPM View** | True skill | Arbitrage opportunity | Risk premium (not alpha) | Measurement error |

## 3. Examples + Counterexamples

**Simple Example:**  
Fund returns 15%, β=1.2, market returns 10%, rf=3%  
Expected (CAPM): 3% + 1.2×(10%-3%) = 11.4%  
Alpha: 15% - 11.4% = +3.6% (positive alpha, outperformance)

**Failure Case:**  
Madoff fund: Consistent +10% returns with β≈0, massive "alpha" → too good to be true, was fraud

**Edge Case:**  
Bear market alpha: Fund returns -5%, market -20%, β=0.8, rf=3%  
Expected: 3% + 0.8×(-20%-3%) = -15.4%  
Alpha: -5% - (-15.4%) = +10.4% (positive alpha despite negative absolute return)

## 4. Layer Breakdown
```
Alpha Framework and Interpretation:
├─ Mathematical Definitions:
│   ├─ Jensen's Alpha: αi = Ri - [rf + βi(Rm - rf)]
│   ├─ Regression Form: Ri - rf = αi + βi(Rm - rf) + εi
│   ├─ Annualized: α_annual = α_period × periods_per_year
│   └─ Information Ratio: IR = α / Tracking_Error (consistency)
├─ Sources of Alpha:
│   ├─ Genuine Skill:
│   │   ├─ Superior information processing
│   │   ├─ Better valuation models
│   │   ├─ Unique insights (industry expertise)
│   │   └─ Execution advantages (speed, technology)
│   ├─ Market Inefficiencies:
│   │   ├─ Behavioral biases (overreaction, underreaction)
│   │   ├─ Structural constraints (index rebalancing)
│   │   ├─ Information asymmetries
│   │   └─ Liquidity provision
│   ├─ Measurement Issues:
│   │   ├─ Incorrect beta estimation
│   │   ├─ Wrong benchmark (should be multi-factor)
│   │   ├─ Survivorship bias in data
│   │   └─ Data mining / overfitting
│   └─ Statistical Noise:
│       ├─ Random variation (luck)
│       ├─ Short measurement periods
│       └─ Non-normal return distributions
├─ Testing Alpha Significance:
│   ├─ Null Hypothesis: H₀: α = 0
│   ├─ t-statistic: t = α / SE(α)
│   ├─ Critical Value: |t| > 1.96 for 95% confidence
│   ├─ p-value: Probability of observing α if H₀ true
│   └─ Multiple Testing: Bonferroni correction for many funds
├─ Alpha Decay:
│   ├─ Discovery: Academic paper identifies anomaly
│   ├─ Exploitation: Hedge funds implement strategy
│   ├─ Crowding: More capital chases same alpha
│   ├─ Arbitrage: Prices adjust, spreads narrow
│   └─ Disappearance: Alpha → 0 as markets efficient
├─ Gross vs Net Alpha:
│   ├─ Gross Alpha: Before fees, costs
│   ├─ Net Alpha: After management fees, expenses
│   ├─ Transaction Costs: Bid-ask, market impact
│   ├─ Typical Fees: 2% management + 20% performance
│   └─ Reality: Gross alpha often exists, net alpha rare
├─ CAPM Perspective:
│   ├─ Efficient Market: α = 0 for all assets (equilibrium)
│   ├─ Positive Alpha: Undervalued, buy signal
│   ├─ Negative Alpha: Overvalued, sell signal
│   ├─ Zero Alpha: Fairly priced given risk
│   └─ Market Portfolio: α = 0 by definition
└─ Multi-Factor Context:
    ├─ CAPM Alpha: May capture omitted factors (size, value)
    ├─ Fama-French Alpha: More stringent (3/5 factors)
    ├─ True Alpha: Unexplained by all known factors
    └─ Smart Beta: Systematic factor returns mislabeled as alpha
```

**Interaction:** Alpha measures return unexplained by beta exposure; statistical significance crucial

## 5. Mini-Project
Calculate, test, and decompose alpha for mutual funds and strategies:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# Download fund and market data
funds = {
    'Berkshire Hathaway': 'BRK-B',
    'ARK Innovation': 'ARKK',
    'Vanguard 500': 'VOO',
    'Active Growth': 'VUG',
    'Value Fund': 'VTV'
}

market_ticker = 'SPY'
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading fund data...")
all_tickers = list(funds.values()) + [market_ticker]
data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Risk-free rate
rf_annual = 0.03
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Excess returns
excess_returns = returns.subtract(rf_daily, axis=0)
market_excess = excess_returns[market_ticker]

def calculate_alpha(fund_returns, market_returns, rf_rate=0):
    """
    Calculate Jensen's alpha via CAPM regression
    Returns detailed statistics including significance tests
    """
    # Align data
    aligned = pd.concat([fund_returns, market_returns], axis=1).dropna()
    aligned.columns = ['fund', 'market']
    
    # Regression: fund = alpha + beta * market + error
    X = sm.add_constant(aligned['market'])
    model = sm.OLS(aligned['fund'], X).fit()
    
    alpha_daily = model.params[0]
    beta = model.params[1]
    alpha_se = model.bse[0]
    t_stat = model.tvalues[0]
    p_value = model.pvalues[0]
    r_squared = model.rsquared
    
    # Annualize
    alpha_annual = alpha_daily * 252
    alpha_se_annual = alpha_se * np.sqrt(252)
    
    # Calculate actual vs expected returns
    actual_return = aligned['fund'].mean() * 252
    expected_return = beta * aligned['market'].mean() * 252
    
    # Information ratio
    residuals = model.resid
    tracking_error = residuals.std() * np.sqrt(252)
    information_ratio = alpha_annual / tracking_error if tracking_error > 0 else np.nan
    
    return {
        'alpha_annual': alpha_annual,
        'alpha_se_annual': alpha_se_annual,
        't_statistic': t_stat,
        'p_value': p_value,
        'beta': beta,
        'r_squared': r_squared,
        'actual_return': actual_return,
        'expected_return': expected_return,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'n_obs': len(aligned),
        'residuals': residuals
    }

# Calculate alpha for all funds
alpha_results = {}
for name, ticker in funds.items():
    if ticker in excess_returns.columns:
        results = calculate_alpha(excess_returns[ticker], market_excess)
        alpha_results[name] = results

# Convert to DataFrame
alpha_df = pd.DataFrame(alpha_results).T
alpha_df = alpha_df.sort_values('alpha_annual', ascending=False)

print("\n" + "=" * 110)
print("JENSEN'S ALPHA ANALYSIS")
print("=" * 110)
print(f"{'Fund':<25} {'Alpha':>10} {'Std Err':>10} {'t-stat':>8} {'p-value':>10} {'Beta':>8} {'IR':>8}")
print("-" * 110)

for name in alpha_df.index:
    row = alpha_df.loc[name]
    sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
    print(f"{name:<25} {row['alpha_annual']:>9.2%} {row['alpha_se_annual']:>9.2%} "
          f"{row['t_statistic']:>8.2f} {row['p_value']:>10.4f} {row['beta']:>8.3f} "
          f"{row['information_ratio']:>8.3f} {sig}")

print("\n*** p<0.01, ** p<0.05, * p<0.1")

# Rolling alpha analysis
def rolling_alpha(fund_returns, market_returns, window=252):
    """Calculate rolling alpha over time"""
    alphas = []
    betas = []
    dates = []
    
    for i in range(window, len(fund_returns)):
        window_fund = fund_returns.iloc[i-window:i]
        window_market = market_returns.iloc[i-window:i]
        
        aligned = pd.concat([window_fund, window_market], axis=1).dropna()
        if len(aligned) > 50:
            X = sm.add_constant(aligned.iloc[:, 1])
            model = sm.OLS(aligned.iloc[:, 0], X).fit()
            
            alpha = model.params[0] * 252  # Annualize
            beta = model.params[1]
            
            alphas.append(alpha)
            betas.append(beta)
            dates.append(fund_returns.index[i])
    
    return pd.DataFrame({'alpha': alphas, 'beta': betas}, index=dates)

# Calculate rolling alpha for selected funds
rolling_results = {}
example_funds = ['Berkshire Hathaway', 'ARK Innovation', 'Vanguard 500']
for name in example_funds:
    if name in alpha_results:
        ticker = funds[name]
        rolling_results[name] = rolling_alpha(
            excess_returns[ticker], market_excess, window=252
        )

# Simulate luck vs skill
def simulate_managers(n_managers=1000, n_periods=60, market_return=0.08/12, market_vol=0.15/np.sqrt(12)):
    """
    Simulate random managers with no skill (α=0)
    Show how many appear to have alpha by chance
    """
    np.random.seed(42)
    
    # Generate market returns
    market_returns = np.random.normal(market_return, market_vol, n_periods)
    
    manager_alphas = []
    manager_t_stats = []
    
    for _ in range(n_managers):
        # True alpha = 0, but random beta
        beta = np.random.uniform(0.8, 1.2)
        
        # Generate manager returns = beta * market + noise (no true alpha)
        idiosyncratic_vol = 0.05 / np.sqrt(12)
        noise = np.random.normal(0, idiosyncratic_vol, n_periods)
        manager_returns = beta * market_returns + noise
        
        # Estimate alpha
        X = sm.add_constant(market_returns)
        model = sm.OLS(manager_returns, X).fit()
        
        alpha = model.params[0] * 12  # Annualize
        t_stat = model.tvalues[0]
        
        manager_alphas.append(alpha)
        manager_t_stats.append(t_stat)
    
    return np.array(manager_alphas), np.array(manager_t_stats)

# Run simulation
simulated_alphas, simulated_tstats = simulate_managers(n_managers=1000, n_periods=60)

# Count false positives
false_positives_5pct = np.sum(np.abs(simulated_tstats) > 1.96) / len(simulated_tstats)
apparent_skill = np.sum(simulated_alphas > 0.02) / len(simulated_alphas)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha comparison with confidence intervals
fund_names = alpha_df.index
alphas = alpha_df['alpha_annual'].values
alpha_ses = alpha_df['alpha_se_annual'].values
colors = ['green' if p < 0.05 and a > 0 else 'red' if p < 0.05 and a < 0 
          else 'gray' for a, p in zip(alphas, alpha_df['p_value'])]

y_pos = np.arange(len(fund_names))
bars = axes[0, 0].barh(y_pos, alphas, color=colors, alpha=0.6)
axes[0, 0].errorbar(alphas, y_pos, xerr=1.96*alpha_ses,
                    fmt='none', ecolor='black', capsize=5)

axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(fund_names)
axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Jensen\'s Alpha (Annual %)')
axes[0, 0].set_title('Alpha with 95% Confidence Intervals')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Rolling alpha over time
for name in rolling_results.keys():
    if name in rolling_results:
        axes[0, 1].plot(rolling_results[name].index, 
                       rolling_results[name]['alpha'],
                       linewidth=2, label=name, alpha=0.8)

axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Rolling 1-Year Alpha')
axes[0, 1].set_title('Time-Varying Alpha')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Luck vs Skill simulation
axes[1, 0].hist(simulated_alphas * 100, bins=50, alpha=0.7, 
               edgecolor='black', color='steelblue')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, 
                  label='True Alpha = 0')
axes[1, 0].axvline(2, color='orange', linestyle='--', linewidth=2,
                  label=f'{apparent_skill:.1%} appear > 2%')

# Mark significant alphas
sig_threshold = 1.96 * np.std(simulated_alphas)
axes[1, 0].axvline(sig_threshold * 100, color='green', linestyle=':', linewidth=2, alpha=0.5)
axes[1, 0].axvline(-sig_threshold * 100, color='green', linestyle=':', linewidth=2, alpha=0.5)

axes[1, 0].set_xlabel('Estimated Alpha (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Luck vs Skill: 1000 Managers with True α=0\n{false_positives_5pct:.1%} false positives at 5% level')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Information Ratio vs Alpha
ir_values = alpha_df['information_ratio'].values
alpha_values = alpha_df['alpha_annual'].values

scatter = axes[1, 1].scatter(ir_values, alpha_values, s=300, alpha=0.6,
                            c=alpha_df['p_value'], cmap='RdYlGn_r',
                            vmin=0, vmax=0.1)

for i, name in enumerate(fund_names):
    axes[1, 1].annotate(name.split()[0], 
                       (ir_values[i], alpha_values[i]),
                       fontsize=8, ha='center', va='bottom')

axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].axvline(0.5, color='orange', linestyle='--', linewidth=1, 
                  alpha=0.5, label='IR = 0.5 (Good)')
axes[1, 1].axvline(1.0, color='green', linestyle='--', linewidth=1,
                  alpha=0.5, label='IR = 1.0 (Excellent)')

axes[1, 1].set_xlabel('Information Ratio')
axes[1, 1].set_ylabel('Alpha (Annual %)')
axes[1, 1].set_title('Alpha Magnitude vs Consistency (color = p-value)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='p-value')

plt.tight_layout()
plt.show()

# Detailed performance attribution
print("\n" + "=" * 110)
print("PERFORMANCE ATTRIBUTION")
print("=" * 110)
print(f"{'Fund':<25} {'Actual Return':>15} {'Expected (CAPM)':>18} {'Alpha':>10} {'Attribution':>15}")
print("-" * 110)

for name in alpha_df.index:
    row = alpha_df.loc[name]
    actual = row['actual_return']
    expected = expected_return = rf_annual + row['beta'] * (market_excess.mean() * 252)
    alpha = row['alpha_annual']
    
    # Attribution
    rf_contrib = rf_annual
    beta_contrib = row['beta'] * (market_excess.mean() * 252)
    alpha_contrib = alpha
    
    print(f"{name:<25} {actual:>14.2%} {expected:>17.2%} {alpha:>9.2%}")
    print(f"{'  Breakdown:':<25} rf={rf_contrib:>5.2%} + β×MRP={beta_contrib:>5.2%} + α={alpha_contrib:>5.2%}")

# Alpha persistence test
print("\n" + "=" * 110)
print("ALPHA PERSISTENCE TEST (First half vs Second half)")
print("=" * 110)

for name, ticker in list(funds.items())[:3]:  # Sample funds
    if ticker in excess_returns.columns:
        fund_rets = excess_returns[ticker]
        n = len(fund_rets)
        midpoint = n // 2
        
        # First half
        first_half = calculate_alpha(fund_rets.iloc[:midpoint], 
                                     market_excess.iloc[:midpoint])
        # Second half
        second_half = calculate_alpha(fund_rets.iloc[midpoint:],
                                      market_excess.iloc[midpoint:])
        
        correlation = "Positive" if (first_half['alpha_annual'] > 0 and 
                                     second_half['alpha_annual'] > 0) else "Negative"
        
        print(f"\n{name}:")
        print(f"  First half:  α = {first_half['alpha_annual']:>6.2%} (p={first_half['p_value']:.4f})")
        print(f"  Second half: α = {second_half['alpha_annual']:>6.2%} (p={second_half['p_value']:.4f})")
        print(f"  Persistence: {correlation}")

print("\n" + "=" * 110)
print("KEY INSIGHTS")
print("=" * 110)
print(f"1. Statistically significant alpha is rare (requires |t| > 2, typically)")
print(f"2. In simulation, {false_positives_5pct:.1%} of zero-alpha managers appear significant")
print(f"3. Information Ratio > 0.5 suggests consistent skill, not just luck")
print(f"4. Alpha persistence is key test: does it continue in out-of-sample periods?")
print(f"5. Gross alpha may exist, but net alpha (after fees) is harder to achieve")
```

## 6. Challenge Round
When is observed alpha not true skill?
- Data mining: Testing many strategies until one "works" (p-hacking)
- Survivorship bias: Failed funds disappear, only winners remain in dataset
- Omitted factors: CAPM alpha may be Fama-French beta (size, value)
- Luck: Random variation masquerading as skill (see simulation)
- Market inefficiency closing: Historical alpha disappears as arbitraged away

Alpha persistence vs mean reversion:
- Skill hypothesis: Alpha persists over time (correlation across periods)
- Luck hypothesis: Alpha mean-reverts to zero (no correlation)
- Evidence: Mixed; top quartile has slight persistence, but most is transitory
- Fees matter: Gross alpha persists more than net alpha

Common alpha misconceptions:
- "Positive alpha = buy": Must be statistically significant and persistent
- "CAPM alpha = true alpha": Could be capturing other systematic factors
- "High IR means high alpha": Could have low alpha but very low tracking error
- "Past alpha predicts future": Often doesn't; luck vs skill hard to distinguish
- "Alpha is free money": Requires capital, risk-taking, often disappears quickly

Practical considerations:
- Fees: 2% + 20% structure means need >3% gross alpha for positive net
- Capacity: Alpha strategies have limited capacity before impact
- Crowding: As more capital chases alpha, spreads narrow
- Transaction costs: High turnover erodes alpha
- Tax efficiency: Realized gains reduce after-tax alpha

## 7. Key References
- [Jensen, M. (1968) "The Performance of Mutual Funds in 1945-1964"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1968.tb00815.x)
- [Fama & French (2010) "Luck versus Skill in Mutual Fund Returns"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X10001315)
- [Carhart (1997) "On Persistence in Mutual Fund Performance"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03808.x)
- [Investopedia - Alpha](https://www.investopedia.com/terms/a/alpha.asp)

---
**Status:** Active management holy grail | **Complements:** Beta, Jensen's Alpha, Information Ratio, CAPM
