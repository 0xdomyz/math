import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Compare rebalancing strategies: buy-and-hold, calendar, threshold, tactical

def simulate_market_returns(n_periods, n_assets, expected_returns, volatilities, 
                           correlation_matrix, seed=42):
    """
    Simulate correlated asset returns.
    """
    np.random.seed(seed)
    
    # Cholesky decomposition for correlated returns
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate uncorrelated random returns
    Z = np.random.normal(0, 1, (n_periods, n_assets))
    
    # Transform to correlated
    returns = expected_returns + Z @ L.T
    
    return returns


def rebalance_portfolio(weights, target_weights, method='none', threshold=0.05, 
                       transaction_cost=0.001):
    """
    Rebalance portfolio based on method.
    Returns new weights and cost incurred.
    """
    if method == 'none':
        # Buy-and-hold: No rebalancing
        return weights, 0.0
    
    elif method == 'calendar':
        # Calendar: Always rebalance to target
        turnover = np.sum(np.abs(weights - target_weights))
        cost = turnover * transaction_cost
        return target_weights.copy(), cost
    
    elif method == 'threshold':
        # Threshold: Rebalance if any asset drifts >threshold
        max_drift = np.max(np.abs(weights - target_weights))
        
        if max_drift > threshold:
            turnover = np.sum(np.abs(weights - target_weights))
            cost = turnover * transaction_cost
            return target_weights.copy(), cost
        else:
            return weights, 0.0
    
    elif method == 'tactical':
        # Tactical: Simplified momentum overlay (±10% from target)
        # Increase weight of assets with positive recent momentum
        momentum_signal = np.random.normal(0, 0.05, len(weights))  # Simplified
        tactical_weights = target_weights + 0.1 * momentum_signal
        tactical_weights = np.clip(tactical_weights, 0.1, 0.9)
        tactical_weights = tactical_weights / np.sum(tactical_weights)
        
        turnover = np.sum(np.abs(weights - tactical_weights))
        cost = turnover * transaction_cost
        return tactical_weights, cost
    
    return weights, 0.0


def backtest_rebalancing(returns, initial_weights, target_weights, method='none',
                        rebalance_frequency=1, threshold=0.05, transaction_cost=0.001):
    """
    Backtest rebalancing strategy.
    """
    n_periods, n_assets = returns.shape
    
    # Initialize
    weights = initial_weights.copy()
    portfolio_values = np.zeros(n_periods + 1)
    portfolio_values[0] = 1.0
    
    total_costs = 0.0
    rebalance_count = 0
    
    all_weights = np.zeros((n_periods + 1, n_assets))
    all_weights[0] = weights
    
    for t in range(n_periods):
        # Calculate period return
        period_return = np.dot(weights, returns[t])
        portfolio_values[t + 1] = portfolio_values[t] * (1 + period_return)
        
        # Update weights based on asset performance
        weights = weights * (1 + returns[t])
        weights = weights / np.sum(weights)  # Normalize
        
        # Check if rebalancing due
        should_rebalance = False
        
        if method == 'calendar':
            if (t + 1) % rebalance_frequency == 0:
                should_rebalance = True
        elif method == 'threshold':
            max_drift = np.max(np.abs(weights - target_weights))
            if max_drift > threshold:
                should_rebalance = True
        elif method == 'tactical':
            if (t + 1) % rebalance_frequency == 0:
                should_rebalance = True
        
        # Rebalance if needed
        if should_rebalance:
            weights, cost = rebalance_portfolio(weights, target_weights, method,
                                               threshold, transaction_cost)
            total_costs += cost
            rebalance_count += 1
            
            # Reduce portfolio value by cost
            portfolio_values[t + 1] *= (1 - cost)
        
        all_weights[t + 1] = weights
    
    return {
        'portfolio_values': portfolio_values,
        'final_value': portfolio_values[-1],
        'total_return': portfolio_values[-1] - 1.0,
        'cagr': (portfolio_values[-1] ** (1 / (n_periods / 12)) - 1) * 100,
        'total_costs': total_costs,
        'rebalance_count': rebalance_count,
        'all_weights': all_weights
    }

