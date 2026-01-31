import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import yfinance as yf

# Compute certainty equivalents for portfolios; compare to alternatives

def get_portfolio_returns(start_date, end_date):
    """
    Fetch portfolio returns for analysis.
    """
    tickers = {
        'Aggressive (80% SPY, 20% BND)': ['SPY', 'BND'],
        'Moderate (60% SPY, 40% BND)': ['SPY', 'BND'],
        'Conservative (40% SPY, 60% BND)': ['SPY', 'BND'],
        'Bonds Only (100% BND)': ['BND'],
        'Stocks Only (100% SPY)': ['SPY'],
    }
    
    # Download data
    all_tickers = list(set([t for ts in tickers.values() for t in ts]))
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    portfolio_returns = {}
    for portfolio_name, weights_dict in [
        ('Aggressive (80% SPY, 20% BND)', {'SPY': 0.8, 'BND': 0.2}),
        ('Moderate (60% SPY, 40% BND)', {'SPY': 0.6, 'BND': 0.4}),
        ('Conservative (40% SPY, 60% BND)', {'SPY': 0.4, 'BND': 0.6}),
        ('Bonds Only (100% BND)', {'BND': 1.0}),
        ('Stocks Only (100% SPY)', {'SPY': 1.0}),
    ]:
        port_ret = sum(weights_dict.get(t, 0) * returns[t] for t in all_tickers if t in returns.columns)
        portfolio_returns[portfolio_name] = port_ret
    
    return pd.DataFrame(portfolio_returns), returns[['SPY', 'BND']]


def compute_certainty_equivalents(returns_dict, lambda_values):
    """
    Compute CE for each portfolio and λ value.
    """
    results = []
    
    for portfolio_name, port_returns in returns_dict.items():
        annual_mean = port_returns.mean() * 252
        annual_std = port_returns.std() * np.sqrt(252)
        annual_var = annual_std ** 2
        
        for lambda_coeff in lambda_values:
            ce = annual_mean - (lambda_coeff / 2) * annual_var
            risk_premium = (lambda_coeff / 2) * annual_var
            
            results.append({
                'Portfolio': portfolio_name,
                'Lambda': lambda_coeff,
                'Expected Return': annual_mean,
                'Volatility': annual_std,
                'Variance': annual_var,
                'CE': ce,
                'Risk Premium': risk_premium,
                'Sharpe Ratio': (annual_mean - 0.02) / annual_std if annual_std > 0 else 0
            })
    
    return pd.DataFrame(results)


def simulate_wealth_paths(returns, initial_wealth, n_paths, n_steps):
    """
    Monte Carlo simulation of portfolio wealth paths.
    """
    mean_return = returns.mean() * 252
    std_return = returns.std() * np.sqrt(252)
    
    # Geometric random walk
    random_returns = np.random.normal(mean_return, std_return, (n_paths, n_steps))
    
    wealth_paths = np.zeros((n_paths, n_steps + 1))
    wealth_paths[:, 0] = initial_wealth
    
    for t in range(n_steps):
        wealth_paths[:, t + 1] = wealth_paths[:, t] * (1 + random_returns[:, t])
    
    return wealth_paths


def interpret_ce_as_insurance_value(ce, expected_return, risk_aversion):
    """
    Interpret CE as implied insurance value.
    """
    return expected_return - ce  # Amount investor willing to pay to eliminate uncertainty


# Main Analysis
print("=" * 100)
print("CERTAINTY EQUIVALENT & EXPECTED UTILITY THEORY")
print("=" * 100)

# 1. Data
print("\n1. PORTFOLIO DATA & RETURNS")
print("-" * 100)

portfolio_returns, asset_returns = get_portfolio_returns('2015-01-01', '2024-01-01')

print("\nHistorical Returns (2015-2024):")
print(portfolio_returns.describe().loc[['mean', 'std']].T)

# Annualize
annual_summary = pd.DataFrame({
    'Annual Return': portfolio_returns.mean() * 252,
    'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
    'Sharpe Ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252))
})
print("\n" + annual_summary.to_string())

# 2. Certainty equivalents for different λ
print("\n2. CERTAINTY EQUIVALENTS BY RISK AVERSION COEFFICIENT")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
ce_results = compute_certainty_equivalents(portfolio_returns, lambda_values)

# Pivot for readability
ce_pivot = ce_results.pivot_table(values='CE', index='Portfolio', columns='Lambda')
print("\nCertainty Equivalent (%):  [Expected Return - Risk Premium]")
print((ce_pivot * 100).to_string())

rp_pivot = ce_results.pivot_table(values='Risk Premium', index='Portfolio', columns='Lambda')
print("\n\nRisk Premium (%) paid for variance:")
print((rp_pivot * 100).to_string())

# 3. Insurance value interpretation
print("\n3. IMPLIED INSURANCE VALUE (What investor would pay to eliminate uncertainty)")
print("-" * 100)

for lambda_coeff in [2.0, 4.0]:
    subset = ce_results[ce_results['Lambda'] == lambda_coeff]
    print(f"\nFor Risk Aversion λ = {lambda_coeff}:")
    for _, row in subset.iterrows():
        insurance_value = row['Expected Return'] - row['CE']
        print(f"  {row['Portfolio']:30s}: Would pay {insurance_value*100:5.2f}% of return to eliminate risk")

# 4. Comparison: Which portfolio best for different investors?
print("\n4. OPTIMAL PORTFOLIO CHOICE BY RISK AVERSION")
print("-" * 100)

for lambda_coeff in [1.0, 2.0, 4.0, 8.0]:
    subset = ce_results[ce_results['Lambda'] == lambda_coeff].sort_values('CE', ascending=False)
    best = subset.iloc[0]
    print(f"\nλ = {lambda_coeff} (Risk aversion):")
    print(f"  Best choice: {best['Portfolio']}")
    print(f"    Expected return: {best['Expected Return']*100:.2f}%")
    print(f"    Volatility: {best['Volatility']*100:.2f}%")
    print(f"    Certainty equivalent: {best['CE']*100:.2f}%")
    print(f"    Risk premium paid: {best['Risk Premium']*100:.2f}%")

# 5. Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: CE by portfolio and λ
ax = axes[0, 0]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio]
    ax.plot(subset['Lambda'], subset['CE'] * 100, 'o-', label=portfolio, linewidth=2, markersize=6)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=11)
ax.set_ylabel('Certainty Equivalent (%)', fontsize=11)
ax.set_title('Certainty Equivalent by Portfolio & λ', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

# Plot 2: Risk premium by portfolio and λ
ax = axes[0, 1]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio]
    ax.plot(subset['Lambda'], subset['Risk Premium'] * 100, 's--', label=portfolio, linewidth=2, markersize=6)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=11)
ax.set_ylabel('Risk Premium Paid (%)', fontsize=11)
ax.set_title('Risk Premium by Portfolio & λ', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

# Plot 3: E[R] vs CE scatter
ax = axes[0, 2]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio].drop_duplicates('Portfolio', keep='last')
    ax.scatter(subset['Expected Return'] * 100, subset['CE'] * 100, s=150, label=portfolio, alpha=0.7)
    
    # Add 45-degree line (E[R] = CE, risk-neutral)
    ax.plot([0, 15], [0, 15], 'k--', alpha=0.3, linewidth=1)

ax.set_xlabel('Expected Return (%)', fontsize=11)
ax.set_ylabel('Certainty Equivalent (%)', fontsize=11)
ax.set_title('Expected Return vs Certainty Equivalent', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 4: Utility comparison for moderate investor (λ=2)
ax = axes[1, 0]
lambda_coeff = 2.0
subset = ce_results[ce_results['Lambda'] == lambda_coeff].sort_values('Expected Return')

ax.barh(subset['Portfolio'], subset['CE'] * 100, label='Certainty Equivalent', alpha=0.7, color='#2ecc71')
ax.barh(subset['Portfolio'], (subset['Expected Return'] - subset['CE']) * 100, 
        left=subset['CE'] * 100, label='Risk Premium Paid', alpha=0.7, color='#e74c3c')

ax.set_xlabel('Return (%)', fontsize=11)
ax.set_title(f'Return Composition (λ={lambda_coeff})', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='x')

# Plot 5: Wealth distribution (MC simulation)
ax = axes[1, 1]

moderate_returns = portfolio_returns['Moderate (60% SPY, 40% BND)']
wealth_paths = simulate_wealth_paths(moderate_returns, 100000, 1000, 50)

percentiles = [10, 25, 50, 75, 90]
colors_dist = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#e74c3c']

for percentile, color in zip(percentiles, colors_dist):
    ax.plot(np.arange(51), np.percentile(wealth_paths, percentile, axis=0), 
           label=f'{percentile}th percentile', linewidth=2, color=color)

final_wealth = wealth_paths[:, -1]
final_mean = final_wealth.mean()
final_ce = final_mean - (lambda_coeff / 2) * final_wealth.std()**2  # Approximate CE

ax.axhline(y=final_mean, color='blue', linestyle='--', linewidth=2, label='Expected wealth')
ax.axhline(y=final_ce, color='green', linestyle='--', linewidth=2, label=f'CE (λ={lambda_coeff})')

ax.set_xlabel('Time Steps (5-year horizon)', fontsize=11)
ax.set_ylabel('Wealth ($)', fontsize=11)
ax.set_title('Wealth Paths & Certainty Equivalent', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 6: Allais Paradox illustration
ax = axes[1, 2]

# Outcomes (in $000s)
outcomes = ['Gamble 1', 'Gamble 2', 'Gamble 3', 'Gamble 4']
ev = [500, 510, 110, 100]  # Expected values
ce_illustrative = [400, 410, 50, 90]  # Certainty equivalents (with λ=2, λ=4 for last two)

x = np.arange(len(outcomes))
width = 0.35

ax.bar(x - width/2, ev, width, label='Expected Value', alpha=0.7, color='#3498db')
ax.bar(x + width/2, ce_illustrative, width, label='Certainty Equivalent (λ=2-4)', alpha=0.7, color='#e74c3c')

ax.set_ylabel('Value ($000s)', fontsize=11)
ax.set_title('Allais Paradox: EU vs EV Preference Reversals', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(outcomes, fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('certainty_equivalent_expected_utility.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: certainty_equivalent_expected_utility.png")
plt.show()

# 6. Key insights
print("\n5. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
CERTAINTY EQUIVALENT DEFINITION:
├─ CE is risk-free return investor indifferent between having & uncertain portfolio
├─ Formula (mean-variance): CE = E[Rp] - (λ/2)σp²
├─ Interpretation: E[Rp] - CE = risk premium (return sacrificed for uncertainty)
└─ Key insight: Higher λ → lower CE (more averse; demands return for risk)

EXPECTED UTILITY THEORY (von Neumann-Morgenstern):
├─ Axiom 1 (Completeness): Can rank all options
├─ Axiom 2 (Transitivity): Preferences consistent (A>B, B>C → A>C)
├─ Axiom 3 (Continuity): Numerical representation exists
├─ Axiom 4 (Independence): C irrelevant if choosing between A vs B
├─ Theorem: If satisfy all 4, rational to maximize E[U(R)]
└─ Practical: Explains insurance, diversification, risk management

RISK AVERSION COEFFICIENTS:
├─ Absolute (ARA = -u''/u'): How much $ pay to eliminate risk
├─ Relative (RRA = W × ARA): % of wealth to allocate to risky
├─ Empirical: λ ≈ 1-2 (stocks), λ ≈ 5-10 (insurance), λ ≈ 0.5-1 (lab)
└─ Calibration: Match observed allocation choices to infer λ

EQUITY PREMIUM PUZZLE:
├─ Observed: Stocks 8% premium, variance implies λ ≈ 0.7
├─ But people act with λ ≈ 2-3, which implies 2-3% premium (not 8%)
├─ Resolutions: Rare disasters, myopic loss aversion, information disagreement, frictions
└─ Current thinking: Combination; no single explanation

PRACTICAL DECISION RULE:
├─ Step 1: Estimate portfolio E[R] and σ
├─ Step 2: Determine your λ (questionnaire or revealed)
├─ Step 3: Compute CE = E[R] - (λ/2)σ²
├─ Step 4: Compare CE across options; choose highest
├─ Step 5: Sensitivity analysis (if λ ±1, does choice change?)
└─ Step 6: Rebalance if belief or λ changes (life events, market shocks)

VIOLATIONS & EXTENSIONS:
├─ Allais Paradox: Preference reversal violates Independence axiom
├─ Framing Effects: Same choice, different wording → different decision
├─ Reference Dependence: CE depends on starting wealth, not absolute level
├─ Loss Aversion: Losing $X worse than gaining $X is good (λ asymmetric)
└─ Modern: Prospect theory, behavioral portfolio theory address these

YOUR ANALYSIS:
├─ Best portfolio for λ=2 (moderate): {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Portfolio']}
├─  Expected return: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Expected Return']*100:.2f}%
├─ Certainty equivalent: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['CE']*100:.2f}%
├─ Risk premium paid: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Risk Premium']*100:.2f}%
└─ Interpretation: Willing to give up {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Risk Premium']*100:.2f}% return for volatility reduction
""")

print("=" * 100)