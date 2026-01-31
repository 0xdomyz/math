import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Note: You'll need FRED API key for macro data
# Sign up free at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = 'your_api_key_here'  # Replace with your key

# Download asset data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI']
end_date = datetime.now()
start_date = datetime(2010, 1, 1)

print("Downloading asset data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# For demonstration, we'll create synthetic macro factors
# In practice, download from FRED API
print("\nCreating synthetic macro factors (replace with actual data in production)...")

# Generate synthetic macro factor surprises (mean 0)
np.random.seed(42)
n_periods = len(returns)

macro_factors = pd.DataFrame({
    'GDP_surprise': np.random.normal(0, 0.01, n_periods),
    'Inflation_surprise': np.random.normal(0, 0.005, n_periods),
    'IntRate_surprise': np.random.normal(0, 0.003, n_periods),
    'Credit_spread': np.random.normal(0, 0.002, n_periods),
    'VIX_change': np.random.normal(0, 0.05, n_periods)
}, index=returns.index)

# Add some realistic correlation with returns
# Market correlation with macro factors
market_ret = returns['SPY']
macro_factors['GDP_surprise'] = 0.3 * market_ret + 0.7 * macro_factors['GDP_surprise']
macro_factors['Inflation_surprise'] = -0.2 * market_ret + 0.8 * macro_factors['Inflation_surprise']
macro_factors['IntRate_surprise'] = -0.25 * market_ret + 0.75 * macro_factors['IntRate_surprise']

def apt_regression_macro(asset_returns, factor_data):
    """
    Run APT regression using macroeconomic factors
    """
    # Align dates
    common_dates = asset_returns.index.intersection(factor_data.index)
    y = asset_returns.loc[common_dates]
    X = factor_data.loc[common_dates]
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # OLS regression
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Robust standard errors
    results_robust = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    factor_loadings = results.params[1:]  # Exclude constant
    
    return {
        'alpha': results.params['const'],
        'loadings': factor_loadings,
        'tvalues': results_robust.tvalues[1:],
        'pvalues': results_robust.pvalues[1:],
        'rsquared': results.rsquared,
        'residuals': results.resid,
        'results': results_robust
    }

def extract_statistical_factors(returns_data, n_factors=5):
    """
    Extract statistical factors using PCA
    """
    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_data)
    
    # PCA
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns_scaled)
    
    # Create DataFrame
    factor_df = pd.DataFrame(
        factors,
        index=returns_data.index,
        columns=[f'PC{i+1}' for i in range(n_factors)]
    )
    
    return factor_df, pca

def apt_regression_statistical(asset_returns, statistical_factors):
    """
    Run APT regression using statistical (PCA) factors
    """
    common_dates = asset_returns.index.intersection(statistical_factors.index)
    y = asset_returns.loc[common_dates]
    X = statistical_factors.loc[common_dates]
    
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'loadings': results.params[1:],
        'tvalues': results.tvalues[1:],
        'rsquared': results.rsquared,
        'results': results
    }

# Extract statistical factors (PCA)
print("\nExtracting statistical factors using PCA...")
stat_factors, pca_model = extract_statistical_factors(returns, n_factors=5)

# Explained variance by each principal component
print("\n" + "="*100)
print("PRINCIPAL COMPONENT ANALYSIS")
print("="*100)
variance_explained = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(5)],
    'Variance Explained (%)': pca_model.explained_variance_ratio_ * 100,
    'Cumulative (%)': np.cumsum(pca_model.explained_variance_ratio_) * 100
})
print(variance_explained.round(2).to_string(index=False))

# Run APT regressions - Macro factors
print("\nRunning APT regressions with macro factors...")
apt_macro_results = {}
for ticker in tickers:
    apt_macro_results[ticker] = apt_regression_macro(returns[ticker], macro_factors)

# Run APT regressions - Statistical factors
print("Running APT regressions with statistical factors...")
apt_stat_results = {}
for ticker in tickers:
    apt_stat_results[ticker] = apt_regression_statistical(returns[ticker], stat_factors)

# CAPM for comparison