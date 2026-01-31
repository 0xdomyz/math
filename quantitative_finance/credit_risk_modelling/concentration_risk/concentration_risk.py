"""
Concentration Risk Analysis
Extracted from concentration_risk.md

Implements Herfindahl-Hirschman Index (HHI), Gini coefficient, and portfolio
concentration metrics for credit risk assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

def main_concentration_risk():
    print("=== Portfolio Concentration Risk Analysis ===")
    
    # Create three portfolios with different concentration levels
    print("\n=== Portfolio Configurations ===")
    
    # Portfolio 1: Highly concentrated (10 large exposures)
    portfolio_1_size = 10
    portfolio_1 = pd.DataFrame({
        'Portfolio': 'Concentrated',
        'Exposure': np.random.lognormal(16, 0.5, portfolio_1_size),
        'Borrower': [f'Large_{i}' for i in range(portfolio_1_size)]
    })
    
    # Portfolio 2: Moderate concentration (100 medium exposures)
    portfolio_2_size = 100
    portfolio_2 = pd.DataFrame({
        'Portfolio': 'Moderate',
        'Exposure': np.random.lognormal(12, 1.0, portfolio_2_size),
        'Borrower': [f'Medium_{i}' for i in range(portfolio_2_size)]
    })
    
    # Portfolio 3: Well-diversified (1000 small exposures)
    portfolio_3_size = 1000
    portfolio_3 = pd.DataFrame({
        'Portfolio': 'Diversified',
        'Exposure': np.random.lognormal(8, 1.5, portfolio_3_size),
        'Borrower': [f'Small_{i}' for i in range(portfolio_3_size)]
    })
    
    # Combine and normalize
    portfolios = pd.concat([portfolio_1, portfolio_2, portfolio_3], ignore_index=True)
    
    # Scale so total exposure = $1B for each
    for portfolio_name in portfolios['Portfolio'].unique():
        mask = portfolios['Portfolio'] == portfolio_name
        total = portfolios.loc[mask, 'Exposure'].sum()
        portfolios.loc[mask, 'Exposure'] = portfolios.loc[mask, 'Exposure'] / total * 1e9
    
    print("Portfolio Summary:")
    for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
        subset = portfolios[portfolios['Portfolio'] == portfolio_name]
        print(f"\n{portfolio_name}:")
        print(f"  Number of exposures: {len(subset)}")
        print(f"  Total exposure: ${subset['Exposure'].sum()/1e9:.1f}B")
        print(f"  Average exposure: ${subset['Exposure'].mean()/1e6:.1f}M")
        print(f"  Largest exposure: ${subset['Exposure'].max()/1e6:.1f}M")
        print(f"  Smallest exposure: ${subset['Exposure'].min()/1e6:.1f}M")
    
    # Calculate concentration metrics
    print("\n=== Concentration Metrics ===")
    
    metrics_list = []
    for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
        subset = portfolios[portfolios['Portfolio'] == portfolio_name]
        
        # Weights
        weights = subset['Exposure'].values / subset['Exposure'].sum()
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights**2)
        
        # Numbers Equivalent
        n_eq = 1 / hhi if hhi > 0 else np.inf
        
        # Gini Coefficient
        sorted_exposures = np.sort(subset['Exposure'].values)
        cumsum = np.cumsum(sorted_exposures)
        n = len(sorted_exposures)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_exposures)) / (n * cumsum[-1]) - (n + 1) / n
        
        # Largest exposure as % of total
        top1_pct = weights.max() * 100
        top10_pct = weights[np.argsort(weights)[-10:] if len(weights) >= 10 else :].sum() * 100
        
        metrics_list.append({
            'Portfolio': portfolio_name,
            'HHI': hhi,
            'N_Equivalent': n_eq,
            'Gini': gini,
            'Top 1 %': top1_pct,
            'Top 10 %': top10_pct
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))
    
    # Loss scenarios by concentration
    print("\n=== Loss Scenarios: Concentration Impact ===")
    
    loss_scenarios = []
    
    for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
        subset = portfolios[portfolios['Portfolio'] == portfolio_name]
        
        if portfolio_name == 'Concentrated':
            pd_vals = np.full(len(subset), 0.02)
        elif portfolio_name == 'Moderate':
            pd_vals = np.full(len(subset), 0.03)
        else:
            pd_vals = np.full(len(subset), 0.05)
        
        lgd = 0.40  # Uniform LGD
        
        # Single default scenario: largest exposure defaults
        largest_idx = subset['Exposure'].idxmax()
        loss_largest = subset.loc[largest_idx, 'Exposure'] * lgd
        loss_largest_pct = loss_largest / 1e9 * 100
        
        # Top 5 defaults scenario
        top_5_idx = subset.nlargest(5, 'Exposure').index
        loss_top5 = (subset.loc[top_5_idx, 'Exposure'] * lgd).sum()
        loss_top5_pct = loss_top5 / 1e9 * 100
        
        loss_scenarios.append({
            'Portfolio': portfolio_name,
            'Largest Default': f'${loss_largest_pct:.2f}%',
            'Top 5 Default': f'${loss_top5_pct:.2f}%',
        })
    
    loss_df = pd.DataFrame(loss_scenarios)
    print(loss_df.to_string(index=False))
    
    print("\n=== Concentration Risk Summary ===")
    print(f"HHI, Gini, and N_eq provide complementary perspectives on diversification")
    print(f"Regulatory limits and granularity adjustments address concentration in capital rules")

if __name__ == "__main__":
    main_concentration_risk()
