import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Estimate investor risk aversion from preferences and compute optimal allocation

def estimate_lambda_from_questionnaire(responses):
    """
    Estimate risk aversion coefficient λ from investor questionnaire.
    Responses: dict with keys like 'age', 'loss_comfort', 'portfolio_risk', 'gambling'
    """
    lambda_estimate = 1.0  # Start with neutral
    
    # Age-based adjustment
    age = responses.get('age', 50)
    lambda_estimate *= (age / 40)  # Older → higher λ
    
    # Loss comfort ("How would you feel losing 20% in one year?")
    # 1=devastating, 5=manageable
    loss_comfort = responses.get('loss_comfort', 3)
    lambda_estimate *= (6 - loss_comfort) / 3  # Lower comfort → higher λ
    
    # Portfolio risk tolerance ("What's ideal stock allocation?")
    # Measured as % stocks (higher = lower λ)
    stock_pct = responses.get('portfolio_risk', 60) / 100
    lambda_estimate *= (1 - stock_pct)  # Inverse relationship
    
    # Gambling preference ("Would you buy a 50-50 coin flip gamble?")
    # 1=never, 5=often
    gambling = responses.get('gambling', 2)
    lambda_estimate *= (3 - gambling) / 2  # More gambling → lower λ
    
    return lambda_estimate


def get_market_data(start_date, end_date):
    """
    Fetch asset returns for optimization.
    """
    tickers = {
        'Stocks': 'SPY',
        'Bonds': 'AGG',
        'Real Estate': 'VNQ',
        'Commodities': 'GSG',
    }
    
    data = yf.download(list(tickers.values()), start=start_date, end=end_date, 
                      progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns, tickers


def optimize_portfolio_mean_variance(returns, lambda_coeff, rf_rate=0.02):
    """
    Solve for optimal portfolio weights given risk aversion coefficient.
    Maximize: E[Rp] - (λ/2) σp²
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Objective function: -Utility (minimize negative)
    def neg_utility(w):
        portfolio_return = np.sum(mean_returns * w)
        portfolio_var = w @ cov_matrix @ w
        utility = portfolio_return - (lambda_coeff / 2) * portfolio_var
        return -utility
    
    # Constraints: weights sum to 1, no short selling
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weight
    x0 = np.array([1 / n_assets] * n_assets)
    
    result = minimize(neg_utility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, -result.fun


def compute_efficient_frontier(returns, lambda_range, rf_rate=0.02):
    """
    Compute efficient frontier for different risk aversion coefficients.
    """
    frontier = []
    
    for lambda_coeff in lambda_range:
        weights, utility = optimize_portfolio_mean_variance(returns, lambda_coeff, rf_rate)
        
        portfolio_return = returns.mean() * 252
        portfolio_var = ((weights * returns).std() * np.sqrt(252)) ** 2
        
        expected_return = np.sum(weights * returns.mean() * 252)
        portfolio_vol = np.sqrt(weights @ (returns.cov() * 252) @ weights)
        
        frontier.append({
            'lambda': lambda_coeff,
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'utility': utility,
            'weights': weights
        })
    
    return pd.DataFrame(frontier)

