import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# Download data
tickers = ['AAPL', 'MSFT', 'TSLA', 'JNJ', 'XOM', 'JPM', 'WMT', 'DIS']
market_ticker = 'SPY'

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers + [market_ticker], start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Risk-free rate (approximate)
rf_annual = 0.03
rf_daily = (1 + rf_annual) ** (1/252) - 1

# Calculate excess returns
excess_returns = returns.subtract(rf_daily, axis=0)
market_excess = excess_returns[market_ticker]

def decompose_risk(stock_returns, market_returns):
    """
    Decompose stock risk into systematic and idiosyncratic components
    Returns: beta, alpha, R², systematic_vol, idiosyncratic_vol, total_vol
    """
    # Align data
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['stock', 'market']
    
    # Regression
    X = sm.add_constant(aligned['market'])
    model = sm.OLS(aligned['stock'], X).fit()
    
    alpha = model.params[0]
    beta = model.params[1]
    r_squared = model.rsquared
    residuals = model.resid
    
    # Annualized statistics
    total_vol = aligned['stock'].std() * np.sqrt(252)
    market_vol = aligned['market'].std() * np.sqrt(252)
    residual_vol = residuals.std() * np.sqrt(252)
    
    # Systematic and idiosyncratic volatility
    systematic_vol = beta * market_vol
    idiosyncratic_vol = residual_vol
    
    # Verification: total_vol² ≈ systematic_vol² + idiosyncratic_vol²
    implied_total = np.sqrt(systematic_vol**2 + idiosyncratic_vol**2)
    
    return {
        'beta': beta,
        'alpha_annual': alpha * 252,
        'r_squared': r_squared,
        'total_vol': total_vol,
        'systematic_vol': systematic_vol,
        'idiosyncratic_vol': idiosyncratic_vol,
        'systematic_pct': (systematic_vol**2 / total_vol**2) * 100,
        'idiosyncratic_pct': (idiosyncratic_vol**2 / total_vol**2) * 100,
        'residuals': residuals
    }

# Analyze each stock
risk_components = {}
for ticker in tickers:
    components = decompose_risk(excess_returns[ticker], market_excess)
    risk_components[ticker] = components

# Convert to DataFrame for display
summary = pd.DataFrame({
    ticker: {
        'Beta': comp['beta'],
        'R²': comp['r_squared'],
        'Total Vol': comp['total_vol'],
        'Systematic Vol': comp['systematic_vol'],
        'Idiosyncratic Vol': comp['idiosyncratic_vol'],
        'Systematic %': comp['systematic_pct'],
        'Idiosyncratic %': comp['idiosyncratic_pct']
    }
    for ticker, comp in risk_components.items()
}).T

print("\n" + "=" * 100)
print("RISK DECOMPOSITION: SYSTEMATIC VS IDIOSYNCRATIC")
print("=" * 100)
print(summary.round(4))

# Simulate portfolio diversification
def simulate_portfolio_risk(returns, market_returns, n_stocks, n_simulations=100):
    """Simulate portfolios with n stocks to show diversification"""
    np.random.seed(42)
    
    portfolio_total_vols = []
    portfolio_systematic_vols = []
    portfolio_idiosyncratic_vols = []
    portfolio_betas = []
    
    for _ in range(n_simulations):
        # Randomly select n stocks
        selected = np.random.choice(tickers, min(n_stocks, len(tickers)), replace=False)
        
        # Equal weight portfolio
        weights = np.array([1/len(selected)] * len(selected))
        port_returns = returns[selected].dot(weights)
        
        # Decompose portfolio risk
        components = decompose_risk(port_returns, market_returns)
        
        portfolio_total_vols.append(components['total_vol'])
        portfolio_systematic_vols.append(components['systematic_vol'])
        portfolio_idiosyncratic_vols.append(components['idiosyncratic_vol'])
        portfolio_betas.append(components['beta'])
    
    return {
        'total': np.array(portfolio_total_vols),
        'systematic': np.array(portfolio_systematic_vols),
        'idiosyncratic': np.array(portfolio_idiosyncratic_vols),
        'betas': np.array(portfolio_betas)
    }

# Simulate for different portfolio sizes
portfolio_sizes = [1, 2, 3, 5, 8]
diversification_results = {}

for n in portfolio_sizes:
    diversification_results[n] = simulate_portfolio_risk(
        excess_returns, market_excess, n, n_simulations=200
    )

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Risk decomposition by stock
systematic_vols = [risk_components[t]['systematic_vol'] for t in tickers]
idiosyncratic_vols = [risk_components[t]['idiosyncratic_vol'] for t in tickers]

x_pos = np.arange(len(tickers))
width = 0.8

bars1 = axes[0, 0].bar(x_pos, systematic_vols, width, label='Systematic Risk', 
                       alpha=0.8, color='red')
bars2 = axes[0, 0].bar(x_pos, idiosyncratic_vols, width, bottom=systematic_vols,
                       label='Idiosyncratic Risk', alpha=0.8, color='blue')

axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(tickers, rotation=45, ha='right')
axes[0, 0].set_ylabel('Volatility (Annual)')
axes[0, 0].set_title('Risk Decomposition by Stock')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Add total volatility markers
for i, ticker in enumerate(tickers):
    total = risk_components[ticker]['total_vol']
    axes[0, 0].plot([i-width/2, i+width/2], [total, total], 
                   'k-', linewidth=2, marker='_', markersize=10)

# Plot 2: Diversification effect on risk components
n_sizes = sorted(diversification_results.keys())
avg_total = [diversification_results[n]['total'].mean() for n in n_sizes]
avg_systematic = [diversification_results[n]['systematic'].mean() for n in n_sizes]
avg_idiosyncratic = [diversification_results[n]['idiosyncratic'].mean() for n in n_sizes]

axes[0, 1].plot(n_sizes, avg_total, 'ko-', linewidth=3, markersize=10, 
               label='Total Risk')
axes[0, 1].plot(n_sizes, avg_systematic, 'ro-', linewidth=3, markersize=10,
               label='Systematic Risk')
axes[0, 1].plot(n_sizes, avg_idiosyncratic, 'bo-', linewidth=3, markersize=10,
               label='Idiosyncratic Risk')

axes[0, 1].set_xlabel('Number of Stocks in Portfolio')
axes[0, 1].set_ylabel('Portfolio Volatility (Annual)')
axes[0, 1].set_title('Diversification: Idiosyncratic Risk Elimination')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Add theoretical 1/sqrt(n) curve for idiosyncratic
single_stock_idio = avg_idiosyncratic[0]
theoretical_idio = [single_stock_idio / np.sqrt(n) for n in n_sizes]
axes[0, 1].plot(n_sizes, theoretical_idio, 'b--', linewidth=2, alpha=0.5,
               label='1/√n (theoretical)')

# Plot 3: Beta vs R² scatter
betas = summary['Beta'].values
r_squareds = summary['R²'].values
colors_beta = ['green' if b < 1 else 'red' for b in betas]

axes[1, 0].scatter(betas, r_squareds, s=300, alpha=0.6, c=colors_beta)

for i, ticker in enumerate(tickers):
    axes[1, 0].annotate(ticker, (betas[i], r_squareds[i]),
                       fontsize=10, ha='center', va='bottom')

axes[1, 0].axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
                  label='Market Beta')
axes[1, 0].set_xlabel('Beta (Systematic Risk Sensitivity)')
axes[1, 0].set_ylabel('R² (Proportion of Variance Explained by Market)')
axes[1, 0].set_title('Systematic Exposure: Beta vs Explanatory Power')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Time series of systematic vs total risk (AAPL example)
example_ticker = 'AAPL'
stock_excess = excess_returns[example_ticker]

# Rolling window regression
window = 252  # 1 year
rolling_beta = []
rolling_r2 = []
rolling_dates = []

for i in range(window, len(stock_excess)):
    window_stock = stock_excess.iloc[i-window:i]
    window_market = market_excess.iloc[i-window:i]
    
    aligned = pd.concat([window_stock, window_market], axis=1).dropna()
    if len(aligned) > 50:  # Minimum observations
        X = sm.add_constant(aligned.iloc[:, 1])
        model = sm.OLS(aligned.iloc[:, 0], X).fit()
        
        rolling_beta.append(model.params[1])
        rolling_r2.append(model.rsquared)
        rolling_dates.append(stock_excess.index[i])

axes[1, 1].plot(rolling_dates, rolling_beta, 'r-', linewidth=2, label='Beta')
axes[1, 1].axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_ylabel('Beta', color='r')
axes[1, 1].tick_params(axis='y', labelcolor='r')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_title(f'{example_ticker}: Time-Varying Systematic Risk (1-year window)')
axes[1, 1].grid(alpha=0.3)

# Secondary y-axis for R²
ax2 = axes[1, 1].twinx()
ax2.plot(rolling_dates, rolling_r2, 'b-', linewidth=2, alpha=0.7, label='R²')
ax2.set_ylabel('R² (Systematic %)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.set_ylim(0, 1)

# Combine legends
lines1, labels1 = axes[1, 1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print("\n1. INDIVIDUAL STOCK RISK DECOMPOSITION:")
for ticker in tickers:
    sys_pct = risk_components[ticker]['systematic_pct']
    idio_pct = risk_components[ticker]['idiosyncratic_pct']
    beta = risk_components[ticker]['beta']
    print(f"   {ticker:6s}: {sys_pct:>5.1f}% systematic (β={beta:.2f}), "
          f"{idio_pct:>5.1f}% idiosyncratic")

print("\n2. DIVERSIFICATION EFFECTIVENESS:")
for n in n_sizes:
    total_avg = diversification_results[n]['total'].mean()
    idio_avg = diversification_results[n]['idiosyncratic'].mean()
    reduction = (1 - idio_avg / diversification_results[1]['idiosyncratic'].mean()) * 100
    print(f"   {n:2d} stocks: Avg idiosyncratic risk = {idio_avg:.2%}, "
          f"Reduction = {reduction:.1f}%")

print("\n3. THEORETICAL IMPLICATIONS:")
print(f"   • Systematic risk cannot be diversified away (remains ~{avg_systematic[-1]:.2%})")
print(f"   • Idiosyncratic risk approaches zero (from {avg_idiosyncratic[0]:.2%} to {avg_idiosyncratic[-1]:.2%})")
print(f"   • CAPM: Only systematic risk (β) earns risk premium")
print(f"   • Well-diversified: R² > 0.7, idiosyncratic < 30% of total variance")

# Calculate minimum stocks for 80% idiosyncratic elimination
target_reduction = 0.80
initial_idio = diversification_results[1]['idiosyncratic'].mean()
print(f"\n4. To eliminate 80% of idiosyncratic risk:")
theoretical_n = int((1 / (1 - target_reduction)) ** 2)
print(f"   Theoretical (1/√n formula): ~{theoretical_n} stocks")
print(f"   Practical: 20-30 stocks typically sufficient")