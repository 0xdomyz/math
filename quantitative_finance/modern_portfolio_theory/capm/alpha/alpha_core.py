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