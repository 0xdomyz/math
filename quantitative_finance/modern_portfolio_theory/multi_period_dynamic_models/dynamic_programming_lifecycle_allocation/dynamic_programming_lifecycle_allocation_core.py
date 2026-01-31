import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate lifecycle allocation strategies and compare outcomes

def simulate_lifecycle_path(start_age, retirement_age, death_age, initial_wealth, 
                            annual_contribution, stock_return, bond_return, 
                            stock_vol, bond_vol, correlation, glide_path_type='linear'):
    """
    Simulate wealth accumulation and decumulation over lifetime.
    """
    
    ages = np.arange(start_age, death_age + 1)
    n_periods = len(ages)
    
    # Initialize
    wealth = np.zeros(n_periods)
    wealth[0] = initial_wealth
    stock_allocation = np.zeros(n_periods)
    
    # Glide path rule
    for i, age in enumerate(ages):
        if glide_path_type == 'linear':
            # Linear: 110 - age
            stock_allocation[i] = max(0.2, min(0.95, (110 - age) / 100))
        elif glide_path_type == 'aggressive':
            # Aggressive: 120 - age
            stock_allocation[i] = max(0.3, min(1.0, (120 - age) / 100))
        elif glide_path_type == 'conservative':
            # Conservative: 100 - age
            stock_allocation[i] = max(0.1, min(0.8, (100 - age) / 100))
        else:  # 'static'
            stock_allocation[i] = 0.6  # Fixed 60/40
    
    # Simulate returns
    np.random.seed(42)
    
    for i in range(1, n_periods):
        age = ages[i]
        
        # Portfolio return (with correlation)
        z1 = np.random.normal(0, 1)
        z2 = correlation * z1 + np.sqrt(1 - correlation**2) * np.random.normal(0, 1)
        
        stock_ret = stock_return + stock_vol * z1
        bond_ret = bond_return + bond_vol * z2
        
        portfolio_return = (stock_allocation[i-1] * stock_ret + 
                          (1 - stock_allocation[i-1]) * bond_ret)
        
        # Update wealth
        if age < retirement_age:
            # Accumulation: Add contributions
            wealth[i] = wealth[i-1] * (1 + portfolio_return) + annual_contribution
        else:
            # Decumulation: Withdraw 4% of initial retirement wealth
            if age == retirement_age:
                retirement_wealth = wealth[i-1]
                annual_withdrawal = 0.04 * retirement_wealth
            
            wealth[i] = wealth[i-1] * (1 + portfolio_return) - annual_withdrawal
            wealth[i] = max(0, wealth[i])  # Can't go negative
    
    return ages, wealth, stock_allocation


def monte_carlo_lifecycle(n_simulations, start_age, retirement_age, death_age,
                          initial_wealth, annual_contribution, 
                          stock_return, bond_return, stock_vol, bond_vol, correlation,
                          glide_path_type):
    """
    Run Monte Carlo simulation for lifecycle wealth paths.
    """
    
    ages = np.arange(start_age, death_age + 1)
    n_periods = len(ages)
    
    all_wealth = np.zeros((n_simulations, n_periods))
    
    for sim in range(n_simulations):
        np.random.seed(sim)
        _, wealth, _ = simulate_lifecycle_path(
            start_age, retirement_age, death_age, initial_wealth,
            annual_contribution, stock_return, bond_return,
            stock_vol, bond_vol, correlation, glide_path_type
        )
        all_wealth[sim, :] = wealth
    
    # Statistics
    median_wealth = np.median(all_wealth, axis=0)
    percentile_10 = np.percentile(all_wealth, 10, axis=0)
    percentile_90 = np.percentile(all_wealth, 90, axis=0)
    
    # Terminal wealth at death
    terminal_wealth = all_wealth[:, -1]
    
    return {
        'ages': ages,
        'median_wealth': median_wealth,
        'p10': percentile_10,
        'p90': percentile_90,
        'terminal_wealth': terminal_wealth,
        'all_paths': all_wealth
    }

