import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# Download asset data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'XOM', 'PG', 'JNJ', 'MA', 'V']
end_date = datetime.now()
start_date = datetime(2019, 1, 1)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Split into estimation and validation periods
split_date = len(returns) // 2
estimation_returns = returns.iloc[:split_date]
validation_returns = returns.iloc[split_date:]

print(f"\nData periods:")
print(f"  Estimation: {estimation_returns.index[0].date()} to {estimation_returns.index[-1].date()}")
print(f"  Validation: {validation_returns.index[0].date()} to {validation_returns.index[-1].date()}")

# Calculate historical statistics
mu_est = estimation_returns.mean() * 252  # Annualized
sigma_est = estimation_returns.cov() * 252
rf = 0.02  # Risk-free rate

def portfolio_stats(weights, mu, sigma, rf=0):
    """Calculate portfolio return, volatility, Sharpe"""
    p_return = weights @ mu
    p_vol = np.sqrt(weights @ sigma @ weights)
    sharpe = (p_return - rf) / p_vol if p_vol > 0 else 0
    return p_return, p_vol, sharpe

def optimize_portfolio(mu, sigma, rf=0, constraint_type='unconstrained'):
    """
    Optimize portfolio weights
    
    constraint_type: 'unconstrained', 'no_short', 'concentration'
    """
    n = len(mu)
    
    if constraint_type == 'unconstrained':
        # Unconstrained (allow short selling)
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            # Singular matrix: use Moore-Penrose pseudoinverse
            sigma_inv = np.linalg.pinv(sigma)
        
        w = sigma_inv @ (mu - rf)
        w = w / w.sum()  # Normalize
        
    elif constraint_type == 'no_short':
        # No short selling constraint
        def objective(w):
            ret, vol, _ = portfolio_stats(w, mu, sigma, rf)
            return -ret / vol if vol > 0 else 0  # Negative for minimization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        
        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        w = result.x if result.success else np.ones(n) / n
        
    elif constraint_type == 'concentration':
        # Concentration limit: max 10% per asset
        def objective(w):
            ret, vol, _ = portfolio_stats(w, mu, sigma, rf)
            return -ret / vol if vol > 0 else 0
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 0.1) for _ in range(n))  # Max 10% per asset
        
        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        w = result.x if result.success else np.ones(n) / n
    
    return w / w.sum()  # Ensure normalized

# Optimization scenarios
print("\n" + "="*100)
print("OPTIMIZATION RESULTS (Estimation Period Data)")
print("="*100)

w_unconstrained = optimize_portfolio(mu_est, sigma_est, rf, 'unconstrained')
w_no_short = optimize_portfolio(mu_est, sigma_est, rf, 'no_short')
w_concentration = optimize_portfolio(mu_est, sigma_est, rf, 'concentration')
w_equal = np.ones(len(tickers)) / len(tickers)

weights_df = pd.DataFrame({
    'Equal-Weight': w_equal,
    'Unconstrained': w_unconstrained,
    'No Short': w_no_short,
    'Concentration': w_concentration
}, index=tickers)

print("\nPortfolio Weights:")
print(weights_df.round(4))

# In-sample performance (estimation period)
print("\n" + "="*100)
print("IN-SAMPLE PERFORMANCE (Estimation Period)")
print("="*100)

portfolios = {
    'Equal-Weight': w_equal,
    'Unconstrained': w_unconstrained,
    'No Short': w_no_short,
    'Concentration': w_concentration
}

insample_stats = {}
for name, w in portfolios.items():
    ret, vol, sharpe = portfolio_stats(w, mu_est, sigma_est, rf)
    insample_stats[name] = {'Return': ret, 'Volatility': vol, 'Sharpe': sharpe}

insample_df = pd.DataFrame(insample_stats).T
print(insample_df.round(4))

# Out-of-sample performance (validation period)
mu_val = validation_returns.mean() * 252
sigma_val = validation_returns.cov() * 252

print("\n" + "="*100)
print("OUT-OF-SAMPLE PERFORMANCE (Validation Period)")
print("="*100)

oosample_stats = {}
for name, w in portfolios.items():
    ret, vol, sharpe = portfolio_stats(w, mu_val, sigma_val, rf)
    oosample_stats[name] = {'Return': ret, 'Volatility': vol, 'Sharpe': sharpe}

oosample_df = pd.DataFrame(oosample_stats).T
print(oosample_df.round(4))

# Performance degradation
print("\n" + "="*100)
print("IN-SAMPLE vs OUT-OF-SAMPLE DEGRADATION")
print("="*100)

degradation = pd.DataFrame({
    'In-Sample Sharpe': insample_df['Sharpe'],
    'Out-of-Sample Sharpe': oosample_df['Sharpe'],
    'Degradation': insample_df['Sharpe'] - oosample_df['Sharpe'],
    'Degradation %': ((insample_df['Sharpe'] - oosample_df['Sharpe']) / insample_df['Sharpe'] * 100)
})

print(degradation.round(4))
print("\nNote: Large degradation indicates overfitting / estimation error")

# Weight stability analysis (sensitivity to estimation period)
print("\n" + "="*100)
print("WEIGHT STABILITY ANALYSIS")
print("="*100)

# Reestimate with slightly different periods
windows = [20, 40, 60, 80]  # Different training window sizes
sensitivity_results = {ticker: [] for ticker in tickers}

for window in windows:
    ret_window = estimation_returns.iloc[-window:]  # Most recent window months
    mu_window = ret_window.mean() * 252
    sigma_window = ret_window.cov() * 252
    
    w_window = optimize_portfolio(mu_window, sigma_window, rf, 'no_short')
    
    for i, ticker in enumerate(tickers):
        sensitivity_results[ticker].append(w_window[i])

sensitivity_df = pd.DataFrame(sensitivity_results, index=[f'{w}-month' for w in windows])

print("\nPortfolio Weight Sensitivity (No-Short Portfolio, Different Windows):")
print(sensitivity_df.round(4))

print("\nWeight Range by Asset (Min - Max):")
for ticker in tickers:
    w_range = sensitivity_df[ticker].max() - sensitivity_df[ticker].min()
    print(f"  {ticker}: {w_range:.4f} ({sensitivity_df[ticker].min():.4f} to {sensitivity_df[ticker].max():.4f})")

# Estimation error empirical study
print("\n" + "="*100)
print("ESTIMATION ERROR SIMULATION")
print("="*100)

# Bootstrap: Resample returns, reoptimize
n_simulations = 100
bootstrap_weights = {name: np.zeros((n_simulations, len(tickers))) 
                     for name in portfolios.keys()}
bootstrap_returns = []

np.random.seed(42)
for i in range(n_simulations):
    # Resample returns with replacement
    indices = np.random.choice(len(estimation_returns), len(estimation_returns), replace=True)
    ret_boot = estimation_returns.iloc[indices]
    
    mu_boot = ret_boot.mean() * 252
    sigma_boot = ret_boot.cov() * 252
    
    # Reoptimize
    for name, w in portfolios.items():
        w_boot = optimize_portfolio(mu_boot, sigma_boot, rf, 'no_short' if 'no' in name or 'Conc' in name else 'unconstrained')
        bootstrap_weights[name][i] = w_boot

# Analyze weight stability across bootstrap samples
print("\nWeight Standard Deviation Across Bootstrap Samples:")
weight_stability = {}
for name in portfolios.keys():
    weight_std = bootstrap_weights[name].std(axis=0)
    weight_stability[name] = weight_std.mean()  # Average across assets
    print(f"  {name:>20}: {weight_std.mean():.4f}")

print("\nInterpretation: Higher std dev = more unstable weights = estimation error problem")

# Correlation matrix analysis (condition number)
print("\n" + "="*100)
print("COVARIANCE MATRIX CONDITIONING")
print("="*100)

eigenvalues = np.linalg.eigvals(sigma_est)
eigenvalues_sorted = np.sort(eigenvalues)[::-1]

condition_number = eigenvalues_sorted[0] / eigenvalues_sorted[-1]

print(f"\nEigenvalue Range: {eigenvalues_sorted[-1]:.4f} to {eigenvalues_sorted[0]:.4f}")
print(f"Condition Number: {condition_number:.4f}")
print(f"Ratio Max/Min: {condition_number:.2f}x")

if condition_number > 100:
    print(f"\n⚠️  HIGH CONDITION NUMBER: Covariance matrix ill-conditioned")
    print(f"    Weights are very sensitive to estimation error")
    print(f"    Recommendation: Use shrinkage, constraints, or factor models")
else:
    print(f"\n✓ Moderate condition number: Matrix well-conditioned")

# Shrinkage Estimator (Ledoit-Wolf)