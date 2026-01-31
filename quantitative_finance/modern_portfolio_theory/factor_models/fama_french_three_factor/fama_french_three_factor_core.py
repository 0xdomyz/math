import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats

# Download test portfolio data (various ETFs with different styles)
tickers = {
    'SPY': 'Large-Cap Blend',
    'IWM': 'Small-Cap Blend', 
    'IWD': 'Large-Cap Value',
    'IWF': 'Large-Cap Growth',
    'IWN': 'Small-Cap Value',
    'IWO': 'Small-Cap Growth',
    'VTV': 'Value ETF',
    'VUG': 'Growth ETF'
}

end_date = datetime.now()
start_date = datetime(2015, 1, 1)  # Need longer history for robust estimates

print("Downloading portfolio data...")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Download Fama-French factors from Ken French's data library
# For this example, we'll construct proxy factors using ETFs
print("\nConstructing factor proxies from ETF data...")

# Risk-free rate proxy (using constant 2% annual = 0.167% monthly for simplicity)
# In practice, download actual T-bill rates
rf_monthly = 0.02 / 12

# Market factor: SPY (S&P 500)
market_returns = returns['SPY']
mkt_rf = market_returns - rf_monthly

# SMB proxy: Small-cap (IWM) - Large-cap (SPY)
smb = returns['IWM'] - returns['SPY']

# HML proxy: Value (VTV) - Growth (VUG)
hml = returns['VTV'] - returns['VUG']

# Create factor DataFrame
factors = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml
})

# Remove any NaN values
factors = factors.dropna()

def fama_french_regression(asset_returns, factors, rf=None):
    """
    Run Fama-French 3-factor regression
    
    Returns: Dictionary with regression results
    """
    if rf is None:
        rf = 0
    
    # Excess returns
    excess_returns = asset_returns - rf
    
    # Align dates
    common_dates = excess_returns.index.intersection(factors.index)
    y = excess_returns.loc[common_dates]
    X = factors.loc[common_dates]
    
    # Add constant for alpha
    X_with_const = sm.add_constant(X)
    
    # OLS regression
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Newey-West standard errors (corrects for autocorrelation/heteroskedasticity)
    results_robust = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params['Mkt-RF'],
        'smb': results.params['SMB'],
        'hml': results.params['HML'],
        'alpha_tstat': results_robust.tvalues['const'],
        'beta_tstat': results_robust.tvalues['Mkt-RF'],
        'smb_tstat': results_robust.tvalues['SMB'],
        'hml_tstat': results_robust.tvalues['HML'],
        'alpha_pval': results_robust.pvalues['const'],
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'residuals': results.resid,
        'fitted': results.fittedvalues,
        'results': results_robust
    }

def calculate_capm_alpha(asset_returns, market_returns, rf=0):
    """
    Calculate CAPM alpha for comparison
    """
    excess_asset = asset_returns - rf
    excess_market = market_returns - rf
    
    common_dates = excess_asset.index.intersection(excess_market.index)
    y = excess_asset.loc[common_dates]
    X = sm.add_constant(excess_market.loc[common_dates])
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params[excess_market.name] if hasattr(excess_market, 'name') else results.params.iloc[1],
        'rsquared': results.rsquared,
        'alpha_tstat': results.tvalues['const']
    }

# Run regressions for all assets
ff3_results = {}
capm_results = {}

for ticker in tickers.keys():
    asset_ret = returns[ticker]
    
    # Fama-French 3-factor
    ff3 = fama_french_regression(asset_ret, factors, rf_monthly)
    ff3_results[ticker] = ff3
    
    # CAPM for comparison
    capm = calculate_capm_alpha(asset_ret, market_returns, rf_monthly)
    capm_results[ticker] = capm

# Create comparison DataFrame
comparison_data = []
for ticker in tickers.keys():
    comparison_data.append({
        'Ticker': ticker,
        'Style': tickers[ticker],
        'CAPM Alpha': capm_results[ticker]['alpha'] * 12 * 100,  # Annualized %
        'CAPM Beta': capm_results[ticker]['beta'],
        'CAPM R²': capm_results[ticker]['rsquared'],
        'FF3 Alpha': ff3_results[ticker]['alpha'] * 12 * 100,  # Annualized %
        'FF3 Beta': ff3_results[ticker]['beta'],
        'FF3 SMB': ff3_results[ticker]['smb'],
        'FF3 HML': ff3_results[ticker]['hml'],
        'FF3 R²': ff3_results[ticker]['rsquared'],
        'Alpha t-stat': ff3_results[ticker]['alpha_tstat'],
        'Alpha Sig': 'Yes' if abs(ff3_results[ticker]['alpha_tstat']) > 1.96 else 'No'
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*120)
print("FAMA-FRENCH 3-FACTOR MODEL ANALYSIS")
print("="*120)
print(comparison_df.to_string(index=False))

# Detailed output for one example (Small-Cap Value)
print("\n" + "="*120)
print("DETAILED REGRESSION OUTPUT: IWN (Small-Cap Value ETF)")
print("="*120)
print(ff3_results['IWN']['results'].summary())

# Factor premiums over the period
print("\n" + "="*120)
print("FACTOR PREMIUMS (Sample Period)")
print("="*120)
factor_stats = pd.DataFrame({
    'Mean (Monthly %)': factors.mean() * 100,
    'Mean (Annual %)': factors.mean() * 12 * 100,
    'Std Dev (Annual %)': factors.std() * np.sqrt(12) * 100,
    'Sharpe Ratio': (factors.mean() / factors.std()) * np.sqrt(12),
    't-statistic': factors.mean() / (factors.std() / np.sqrt(len(factors)))
})
print(factor_stats.round(3))

# Attribution analysis