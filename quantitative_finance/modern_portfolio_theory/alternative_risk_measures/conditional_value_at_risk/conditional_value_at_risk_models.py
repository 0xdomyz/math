import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def component_cvar(returns, weights, alpha=0.95, epsilon=0.0001):
    """
    Calculate marginal and component CVaR
    """
    base_cvar = historical_cvar(returns, weights, alpha)
    
    marginal_cvars = []
    for i in range(len(weights)):
        # Perturb weight
        perturbed_weights = weights.copy()
        perturbed_weights[i] += epsilon
        perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
        
        perturbed_cvar = historical_cvar(returns, perturbed_weights, alpha)
        marginal_cvar = (perturbed_cvar - base_cvar) / epsilon
        marginal_cvars.append(marginal_cvar)
    
    # Component CVaR
    component_cvars = weights * np.array(marginal_cvars)
    
    return {
        'portfolio_cvar': base_cvar,
        'marginal_cvar': marginal_cvars,
        'component_cvar': component_cvars
    }

cvar_comp = component_cvar(returns, cvar_port['weights'], 0.95)

print("\n" + "="*100)
print("CVaR CONTRIBUTION ANALYSIS (CVaR-Optimized Portfolio)")
print("="*100)
contrib_df = pd.DataFrame({
    'Weight': cvar_port['weights'],
    'Component CVaR': cvar_comp['component_cvar'],
    '% of Total': cvar_comp['component_cvar'] / cvar_comp['portfolio_cvar'] * 100
}, index=tickers)
print(contrib_df.round(4))
print(f"\nTotal Portfolio CVaR: {cvar_comp['portfolio_cvar']*np.sqrt(252):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficient Frontier (Return vs Volatility)
axes[0, 0].scatter(asset_vols, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 0].annotate(ticker, (asset_vols[i], asset_returns[i]), 
                       fontsize=8, ha='right')

axes[0, 0].plot(mv_df['volatility'], mv_df['return'], 'b-o', 
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 0].plot(cvar_df['volatility'], cvar_df['return'], 'r-s',
               label='CVaR Optimized', alpha=0.7, markersize=4)

axes[0, 0].set_xlabel('Volatility (Annual)')
axes[0, 0].set_ylabel('Expected Return (Annual)')
axes[0, 0].set_title('Efficient Frontiers: Mean-Variance vs CVaR')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Return vs CVaR
axes[0, 1].scatter(asset_cvars, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 1].annotate(ticker, (asset_cvars[i], asset_returns[i]),
                       fontsize=8, ha='right')

axes[0, 1].plot(mv_df['cvar'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 1].plot(cvar_df['cvar'], cvar_df['return'], 'r-s',
               label='CVaR Optimized', alpha=0.7, markersize=4)

axes[0, 1].set_xlabel('95% CVaR (Annual)')
axes[0, 1].set_ylabel('Expected Return (Annual)')
axes[0, 1].set_title('Mean-CVaR Frontier')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: VaR vs CVaR comparison
portfolio_returns_mv = returns @ mv_port['weights']
portfolio_returns_cvar = returns @ cvar_port['weights']

axes[1, 0].hist(portfolio_returns_mv * 100, bins=50, alpha=0.5, 
               label='Mean-Variance Portfolio', density=True, color='blue')
axes[1, 0].hist(portfolio_returns_cvar * 100, bins=50, alpha=0.5,
               label='CVaR Portfolio', density=True, color='red')

# Mark VaR and CVaR
mv_var_line = mv_port['var'] / np.sqrt(252) * 100
mv_cvar_line = mv_port['cvar'] / np.sqrt(252) * 100
cvar_var_line = cvar_port['var'] / np.sqrt(252) * 100
cvar_cvar_line = cvar_port['cvar'] / np.sqrt(252) * 100

axes[1, 0].axvline(-mv_var_line, color='blue', linestyle='--', alpha=0.7,
                  label=f'MV VaR: {mv_var_line:.2f}%')
axes[1, 0].axvline(-mv_cvar_line, color='blue', linestyle=':', alpha=0.7,
                  label=f'MV CVaR: {mv_cvar_line:.2f}%')
axes[1, 0].axvline(-cvar_cvar_line, color='red', linestyle=':', alpha=0.7,
                  label=f'CVaR CVaR: {cvar_cvar_line:.2f}%')

axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Return Distributions: MV vs CVaR Portfolios')
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Weight comparison
x = np.arange(len(tickers))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, mv_port['weights'], width, 
                       label='Mean-Variance', alpha=0.8)
bars2 = axes[1, 1].bar(x + width/2, cvar_port['weights'], width,
                       label='CVaR Optimized', alpha=0.8)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(tickers)
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_title('Portfolio Weight Comparison')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: CVaR vs VaR")
print("="*100)
print("1. CVaR always ≥ VaR (captures tail severity)")
print("2. CVaR optimization produces more diversified portfolios")
print("3. CVaR is coherent (sub-additive), VaR is not")
print("4. CVaR optimization is convex, enables global optimum")
print("5. Mean-CVaR frontier similar to mean-variance but better tail control")
print("6. CVaR portfolio typically has lower extreme losses")
print("7. Basel III moving from VaR to Expected Shortfall for market risk")
print("8. CVaR estimation requires more data (tail focus)")

# Coherence demonstration
print("\n" + "="*100)
print("COHERENCE PROPERTY DEMONSTRATION")
print("="*100)

# Create two simple portfolios
weights_A = np.array([0.8, 0.2, 0, 0, 0, 0, 0])
weights_B = np.array([0, 0, 0.6, 0.4, 0, 0, 0])
weights_combined = (weights_A + weights_B) / 2

var_A, cvar_A = calculate_var_cvar(returns, weights_A, 0.95)
var_B, cvar_B = calculate_var_cvar(returns, weights_B, 0.95)
var_combined, cvar_combined = calculate_var_cvar(returns, weights_combined, 0.95)

print(f"Portfolio A - VaR: {var_A*np.sqrt(252):.4f}, CVaR: {cvar_A*np.sqrt(252):.4f}")
print(f"Portfolio B - VaR: {var_B*np.sqrt(252):.4f}, CVaR: {cvar_B*np.sqrt(252):.4f}")
print(f"Combined    - VaR: {var_combined*np.sqrt(252):.4f}, CVaR: {cvar_combined*np.sqrt(252):.4f}")
print(f"\nVaR Sub-additivity: VaR(A+B) ≤ VaR(A) + VaR(B)?")
print(f"  {var_combined*np.sqrt(252):.4f} ≤ {(var_A + var_B)*np.sqrt(252):.4f}? {var_combined <= var_A + var_B}")
print(f"\nCVaR Sub-additivity: CVaR(A+B) ≤ CVaR(A) + CVaR(B)?")
print(f"  {cvar_combined*np.sqrt(252):.4f} ≤ {(cvar_A + cvar_B)*np.sqrt(252):.4f}? {cvar_combined <= cvar_A + cvar_B}")
print(f"\nCVaR is coherent (always sub-additive), VaR can fail!")