import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================================
# DATA GENERATION (Simulated Stock Universe)
# ============================================================================

n_stocks = 100
n_days = 252 * 2  # 2 years of daily data

# Generate stock returns with factor structure
# True factors: Value, Momentum, Quality
factor_returns = pd.DataFrame({
    'market': np.random.normal(0.0004, 0.01, n_days),  # Market factor
    'value': np.random.normal(0.0002, 0.005, n_days),   # Value premium
    'momentum': np.random.normal(0.0003, 0.006, n_days),  # Momentum
    'quality': np.random.normal(0.0001, 0.004, n_days)   # Quality
})

# Generate factor loadings (betas) for each stock
factor_loadings = pd.DataFrame({
    'market': np.ones(n_stocks),  # All stocks have market exposure
    'value': np.random.uniform(-1, 1, n_stocks),
    'momentum': np.random.uniform(-1, 1, n_stocks),
    'quality': np.random.uniform(-1, 1, n_stocks)
})

# Generate stock returns: R = Beta × Factor + Idiosyncratic
stock_returns = np.zeros((n_days, n_stocks))
for i in range(n_stocks):
    factor_contribution = (factor_returns.values * factor_loadings.iloc[i].values).sum(axis=1)
    idiosyncratic = np.random.normal(0, 0.015, n_days)  # Stock-specific noise
    stock_returns[:, i] = factor_contribution + idiosyncratic

# Generate factor characteristics (observable signals)
# These would normally come from fundamentals, but we simulate
characteristics = pd.DataFrame({
    f'stock_{i}': {
        'value_score': factor_loadings.loc[i, 'value'] + np.random.normal(0, 0.3),
        'momentum_score': factor_loadings.loc[i, 'momentum'] + np.random.normal(0, 0.3),
        'quality_score': factor_loadings.loc[i, 'quality'] + np.random.normal(0, 0.3)
    }
    for i in range(n_stocks)
}).T

print("="*70)
print("ALPHA GENERATION: MULTI-FACTOR MODEL")
print("="*70)
print(f"\nStock Universe: {n_stocks} stocks")
print(f"Time Period: {n_days} days ({n_days/252:.1f} years)")
print(f"Factors: Market, Value, Momentum, Quality")

# ============================================================================
# ALPHA SIGNAL GENERATION
# ============================================================================

def generate_composite_score(characteristics, weights={'value': 0.3, 'momentum': 0.4, 'quality': 0.3}):
    """
    Generate composite alpha score from multiple factors.
    Higher score = better expected return.
    """
    scores = (
        weights['value'] * characteristics['value_score'] +
        weights['momentum'] * characteristics['momentum_score'] +
        weights['quality'] * characteristics['quality_score']
    )
    
    # Standardize (z-score)
    scores = (scores - scores.mean()) / scores.std()
    
    return scores

composite_scores = generate_composite_score(characteristics)

print("\n" + "="*70)
print("SIGNAL GENERATION")
print("="*70)
print(f"\nComposite Score Statistics:")
print(f"   Mean: {composite_scores.mean():.3f}")
print(f"   Std:  {composite_scores.std():.3f}")
print(f"   Min:  {composite_scores.min():.3f}")
print(f"   Max:  {composite_scores.max():.3f}")

# ============================================================================
# PORTFOLIO CONSTRUCTION
# ============================================================================

def construct_long_short_portfolio(scores, n_long=20, n_short=20):
    """
    Long top-ranked stocks, short bottom-ranked.
    Market-neutral strategy.
    """
    # Rank stocks by score
    ranked = scores.sort_values(ascending=False)
    
    # Select long and short portfolios
    long_stocks = ranked.head(n_long).index.tolist()
    short_stocks = ranked.tail(n_short).index.tolist()
    
    # Equal weight within each side
    weights = pd.Series(0.0, index=scores.index)
    weights[long_stocks] = 1.0 / n_long
    weights[short_stocks] = -1.0 / n_short
    
    return weights

portfolio_weights = construct_long_short_portfolio(composite_scores, n_long=20, n_short=20)

print("\n" + "="*70)
print("PORTFOLIO CONSTRUCTION")
print("="*70)
print(f"\nLong/Short Portfolio:")
print(f"   Long positions:  {(portfolio_weights > 0).sum()}")
print(f"   Short positions: {(portfolio_weights < 0).sum()}")
print(f"   Gross exposure:  {portfolio_weights.abs().sum():.1f}x")
print(f"   Net exposure:    {portfolio_weights.sum():.1f}x (market neutral)")

# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_strategy(stock_returns, portfolio_weights, rebalance_freq=21):
    """
    Backtest alpha strategy with periodic rebalancing.
    rebalance_freq: Days between rebalancing (21 = monthly)
    """
    n_days, n_stocks = stock_returns.shape
    
    # Convert portfolio weights to array
    weight_array = portfolio_weights.values
    
    # Strategy returns (buy and hold until rebalance)
    strategy_returns = []
    holdings = []
    
    for day in range(n_days):
        # Rebalance periodically
        if day % rebalance_freq == 0:
            current_weights = weight_array.copy()
        
        # Calculate daily return
        daily_stock_returns = stock_returns[day, :]
        portfolio_return = np.dot(current_weights, daily_stock_returns)
        strategy_returns.append(portfolio_return)
        
        # Update weights due to price changes (no rebalancing)
        # Simplified: assume weights stay constant between rebalances
        # In reality, would drift with relative performance
    
    strategy_returns = np.array(strategy_returns)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Performance metrics
    total_return = cumulative_returns[-1]
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_vol = np.std(strategy_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Maximum drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'daily_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# Run backtest
backtest_results = backtest_strategy(stock_returns, portfolio_weights, rebalance_freq=21)

print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)
print(f"\nPerformance Metrics:")
print(f"   Total Return:      {backtest_results['total_return']*100:+.2f}%")
print(f"   Annualized Return: {backtest_results['annual_return']*100:+.2f}%")
print(f"   Annualized Vol:    {backtest_results['annual_vol']*100:.2f}%")
print(f"   Sharpe Ratio:      {backtest_results['sharpe_ratio']:.2f}")
print(f"   Maximum Drawdown:  {backtest_results['max_drawdown']*100:.2f}%")

# Benchmark (market factor only)
market_returns = factor_returns['market'].values
market_cumulative = np.cumprod(1 + market_returns) - 1
market_annual_return = (1 + market_cumulative[-1]) ** (252 / n_days) - 1
market_annual_vol = np.std(market_returns) * np.sqrt(252)
market_sharpe = market_annual_return / market_annual_vol

print(f"\nBenchmark (Market Factor):")
print(f"   Annualized Return: {market_annual_return*100:+.2f}%")
print(f"   Annualized Vol:    {market_annual_vol*100:.2f}%")
print(f"   Sharpe Ratio:      {market_sharpe:.2f}")

alpha = backtest_results['annual_return'] - market_annual_return
print(f"\n** ALPHA GENERATED: {alpha*100:+.2f}% per year **")

# ============================================================================
# FACTOR ATTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("ALPHA ATTRIBUTION (Which Factors Contributed?)")
print("="*70)

# Calculate correlation between portfolio weights and factor exposures
for factor in ['value', 'momentum', 'quality']:
    exposure = factor_loadings[factor].values
    correlation = np.corrcoef(portfolio_weights.values, exposure)[0, 1]
    print(f"\n{factor.capitalize()} Factor:")
    print(f"   Correlation with weights: {correlation:+.3f}")
    print(f"   Factor return (annualized): {factor_returns[factor].mean() * 252 * 100:+.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative returns
ax1 = axes[0, 0]
days_axis = np.arange(n_days)
ax1.plot(days_axis, backtest_results['cumulative_returns'] * 100, 
        linewidth=2, label='Alpha Strategy', color='blue')
ax1.plot(days_axis, market_cumulative * 100, 
        linewidth=2, label='Market Benchmark', color='gray', alpha=0.7)
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Cumulative Return (%)')
ax1.set_title('Alpha Strategy vs Market Benchmark')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling Sharpe ratio
ax2 = axes[0, 1]
window = 63  # ~3 months
rolling_returns = pd.Series(backtest_results['daily_returns'])
rolling_sharpe = (rolling_returns.rolling(window).mean() / 
                 rolling_returns.rolling(window).std()) * np.sqrt(252)
ax2.plot(days_axis[window:], rolling_sharpe[window:], linewidth=2, color='green')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Rolling Sharpe Ratio (63-day)')
ax2.set_title('Strategy Stability: Rolling Sharpe')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Return distribution
ax3 = axes[1, 0]
ax3.hist(backtest_results['daily_returns'] * 100, bins=50, 
        alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(x=backtest_results['daily_returns'].mean() * 100, 
           color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {backtest_results["daily_returns"].mean()*100:.3f}%')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Daily Return Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Drawdown chart
ax4 = axes[1, 1]
cumulative_wealth = np.cumprod(1 + backtest_results['daily_returns'])
running_max = np.maximum.accumulate(cumulative_wealth)
drawdown = (cumulative_wealth - running_max) / running_max * 100
ax4.fill_between(days_axis, drawdown, 0, alpha=0.3, color='red')
ax4.plot(days_axis, drawdown, linewidth=2, color='darkred')
ax4.set_xlabel('Trading Days')
ax4.set_ylabel('Drawdown (%)')
ax4.set_title(f'Underwater Chart (Max DD: {backtest_results["max_drawdown"]*100:.1f}%)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('alpha_generation_backtest.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: alpha_generation_backtest.png")
plt.show()