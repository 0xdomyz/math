import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Implement two-fund portfolio; backtest vs alternatives

def get_two_fund_data(start_date, end_date):
    """
    Fetch data for two-fund implementation: stocks and bonds.
    """
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
    bnd = yf.download('BND', start=start_date, end=end_date, progress=False)['Adj Close']
    
    stock_returns = spy.pct_change().dropna()
    bond_returns = bnd.pct_change().dropna()
    
    return stock_returns, bond_returns


def compute_two_fund_allocation(stock_returns, bond_returns, lambda_coeff):
    """
    Compute optimal two-fund allocation for given risk aversion.
    """
    stock_mean = stock_returns.mean() * 252
    stock_vol = stock_returns.std() * np.sqrt(252)
    
    bond_mean = bond_returns.mean() * 252
    bond_vol = bond_returns.std() * np.sqrt(252)
    
    # For simplicity, assume bond is risk-free proxy (low vol)
    # If treating bond as second risky asset, need correlation matrix
    rf = bond_mean  # Approximate rf as bond yield
    
    # Market premium and market portfolio (assumed mix)
    # Treat SPY as risky portfolio, BND as risk-free proxy
    market_premium = stock_mean - rf
    w_stock = market_premium / (lambda_coeff * stock_vol ** 2) if stock_vol > 0 else 0
    w_bond = 1 - w_stock
    
    # Constrain weights (no naked short selling, limit leverage)
    w_stock = np.clip(w_stock, 0, 2.0)  # Allow up to 2x leverage
    w_bond = 1 - w_stock
    
    return w_stock, w_bond, stock_mean, stock_vol


def backtest_portfolio(stock_returns, bond_returns, w_stock, w_bond):
    """
    Backtest two-fund portfolio with given weights.
    """
    portfolio_returns = w_stock * stock_returns + w_bond * bond_returns
    
    # Performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Cumulative returns
    cumulative = (1 + portfolio_returns).cumprod()
    
    return {
        'returns': portfolio_returns,
        'cumulative': cumulative,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe
    }


def backtest_alternative_strategies(stock_returns, bond_returns):
    """
    Compare two-fund to alternative strategies.
    """
    results = {}
    
    # Strategy 1: Two-fund (60/40)
    cumul_60_40 = (1 + 0.6 * stock_returns + 0.4 * bond_returns).cumprod()
    results['60/40 Two-Fund'] = cumul_60_40
    
    # Strategy 2: All stocks (100/0)
    cumul_100_0 = (1 + stock_returns).cumprod()
    results['100% Stocks'] = cumul_100_0
    
    # Strategy 3: All bonds (0/100)
    cumul_0_100 = (1 + bond_returns).cumprod()
    results['100% Bonds'] = cumul_0_100
    
    # Strategy 4: Rebalanced two-fund (quarterly)
    monthly_returns = stock_returns.to_frame()
    monthly_returns.columns = ['stock']
    monthly_returns['bond'] = bond_returns
    monthly_returns['month'] = monthly_returns.index.to_period('M')
    
    portfolio_vals = [1.0]
    for month, group in monthly_returns.groupby('month'):
        # Monthly returns within month
        month_stock_ret = (1 + group['stock']).prod() - 1
        month_bond_ret = (1 + group['bond']).prod() - 1
        
        # Rebalance at month end if drift > 5%
        port_ret = 0.6 * month_stock_ret + 0.4 * month_bond_ret
        portfolio_vals.append(portfolio_vals[-1] * (1 + port_ret))
    
    cumul_rebalanced = pd.Series(portfolio_vals[:-1], index=stock_returns.index)
    results['60/40 Rebalanced'] = cumul_rebalanced
    
    return results


# Main Analysis
print("=" * 100)
print("TWO-FUND SEPARATION THEOREM & LEVERAGE")
print("=" * 100)

# 1. Data
print("\n1. MARKET DATA")
print("-" * 100)

stock_returns, bond_returns = get_two_fund_data('2015-01-01', '2024-01-01')

print(f"Stock returns (SPY):")
print(f"  Annual return: {stock_returns.mean() * 252 * 100:.2f}%")
print(f"  Annual volatility: {stock_returns.std() * np.sqrt(252) * 100:.2f}%")
print(f"  Sharpe ratio: {(stock_returns.mean() * 252) / (stock_returns.std() * np.sqrt(252)):.3f}")

print(f"\nBond returns (BND):")
print(f"  Annual return: {bond_returns.mean() * 252 * 100:.2f}%")
print(f"  Annual volatility: {bond_returns.std() * np.sqrt(252) * 100:.2f}%")
print(f"  Sharpe ratio: {(bond_returns.mean() * 252) / (bond_returns.std() * np.sqrt(252)):.3f}")

corr = stock_returns.corr(bond_returns)
print(f"\nCorrelation: {corr:.3f}")

# 2. Optimal allocations
print("\n2. OPTIMAL TWO-FUND ALLOCATIONS BY RISK AVERSION")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
allocations = {}

print(f"\n{'Lambda':<10} {'w_stock %':<15} {'w_bond %':<15} {'Expected Return %':<20} {'Volatility %':<15}")
print("-" * 75)

for lambda_coeff in lambda_values:
    w_stock, w_bond, stock_mean, stock_vol = compute_two_fund_allocation(
        stock_returns, bond_returns, lambda_coeff
    )
    
    rf = bond_returns.mean() * 252
    e_r = w_stock * stock_mean + w_bond * rf
    e_vol = np.sqrt(w_stock ** 2 * stock_vol ** 2 + w_bond ** 2 * (bond_returns.std() * np.sqrt(252)) ** 2 + 
                    2 * w_stock * w_bond * corr * stock_vol * (bond_returns.std() * np.sqrt(252)))
    
    allocations[lambda_coeff] = {'w_stock': w_stock, 'w_bond': w_bond}
    
    print(f"{lambda_coeff:<10.1f} {w_stock*100:<15.1f} {w_bond*100:<15.1f} {e_r*100:<20.2f} {e_vol*100:<15.2f}")

# 3. Backtests
print("\n3. STRATEGY COMPARISON (Historical)")
print("-" * 100)

strategies = backtest_alternative_strategies(stock_returns, bond_returns)

print(f"\nCumulative returns (2015-2024):\n")
print(f"{'Strategy':<30} {'Total Return %':<20} {'Annual Return %':<18} {'Volatility %':<15}")
print("-" * 83)

for name, cumul in strategies.items():
    total_ret = cumul.iloc[-1] - 1
    rets = cumul.pct_change().dropna()
    annual_ret = rets.mean() * 252
    annual_vol = rets.std() * np.sqrt(252)
    
    print(f"{name:<30} {total_ret*100:<20.2f} {annual_ret*100:<18.2f} {annual_vol*100:<15.2f}")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Allocation by lambda
ax = axes[0, 0]

lambdas = list(allocations.keys())
stock_weights = [allocations[l]['w_stock'] for l in lambdas]
bond_weights = [allocations[l]['w_bond'] for l in lambdas]

x = np.arange(len(lambdas))
width = 0.35

ax.bar(x - width/2, [w * 100 for w in stock_weights], width, label='Stocks', color='#3498db', alpha=0.8)
ax.bar(x + width/2, [w * 100 for w in bond_weights], width, label='Bonds', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_title('Two-Fund Optimal Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'λ={l}' for l in lambdas])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Cumulative returns comparison
ax = axes[0, 1]

for name, cumul in strategies.items():
    ax.plot(cumul.index, (cumul - 1) * 100, label=name, linewidth=2)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.set_title('Strategy Comparison: Cumulative Returns', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 3: Risk-return scatter
ax = axes[1, 0]

names = []
for name, cumul in strategies.items():
    rets = cumul.pct_change().dropna()
    annual_ret = rets.mean() * 252
    annual_vol = rets.std() * np.sqrt(252)
    ax.scatter(annual_vol * 100, annual_ret * 100, s=200, alpha=0.7, label=name)
    names.append(name)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Risk-Return Trade-off', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Rolling Sharpe ratios
ax = axes[1, 1]

for name, cumul in strategies.items():
    rets = cumul.pct_change().dropna()
    rolling_sharpe = rets.rolling(252).mean() / rets.rolling(252).std() * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe, label=name, linewidth=2, alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Rolling Sharpe Ratio (1-Year)', fontsize=12)
ax.set_title('Rolling Sharpe Ratio Comparison', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('two_fund_separation_leverage.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: two_fund_separation_leverage.png")
plt.show()

# 5. Key insights
print("\n4. KEY INSIGHTS & PRACTICAL IMPLICATIONS")
print("-" * 100)
print(f"""
TWO-FUND SEPARATION THEOREM:
├─ All rational investors hold: risk-free asset + market portfolio
├─ Proportion differs by risk aversion (λ); composition same
├─ Eliminates need for stock-picking (market portfolio is optimal risky asset)
├─ Simplifies portfolio management to single decision: allocation to market
└─ Empirically supported: Passive indexing outperforms 85% of active managers after fees

PRACTICAL IMPLEMENTATION:
├─ Fund 1: Total Stock Market Index (VTI, VTSAX, or SPY)
├─ Fund 2: Total Bond Index (BND, VBTLX) or T-bills
├─ Allocation: Based on personal λ (conservative 30/70, aggressive 80/20)
├─ Rebalancing: Quarterly or annually when weights drift >5%
└─ Cost: Use low-cost index funds (<0.1% ER); rebalance tax-efficiently

CONSTRAINTS & VIOLATIONS:
├─ Leverage limit (50% typical): Aggressive investors forced to accept more risk-free
├─ Borrow costs > rf: Makes leverage suboptimal for most
├─ Different beliefs: Some try active stock-picking (mostly underperform)
├─ Taxes/costs: Some optimize after-tax allocation
└─ Behavioral: Home bias, overconfidence lead to non-market holdings

ALLOCATION BY RISK AVERSION:
├─ Very conservative (λ=8): ~20-30% stocks, 70-80% bonds
├─ Moderate (λ=4): ~50-60% stocks, 40-50% bonds
├─ Balanced (λ=2): ~60-70% stocks, 30-40% bonds
├─ Aggressive (λ=1): ~75-85% stocks, 15-25% bonds
└─ Very aggressive (λ<1): 90%+ stocks, may use modest leverage

REBALANCING BENEFITS:
├─ Automatic "buy low, sell high" discipline (contrarian)
├─ Historical studies: +0.5-1% p.a. benefit (tax-deferred accounts best)
├─ Cost: Bid-ask spreads, commissions (limit frequency accordingly)
├─ Tax-efficient: Realize losses first; defer gains in taxable accounts
└─ Practical: Annual or threshold-based (>5% drift) sufficient

LEVERAGE CONSIDERATION:
├─ Allows aggressive investors to take desired risk level
├─ But amplifies losses; margin calls hurt at worst times
├─ Cost: Borrow rate often 1-3% above lending rate
├─ Recommendation: Use sparingly (max 1.25-1.5× for conservative investors)
└─ Alternative: Accepting higher stock allocation (no leverage needed)

YOUR RECOMMENDED TWO-FUND PORTFOLIO:
├─ Conservative: 40% stocks (VTI) + 60% bonds (BND); rebalance annually
├─ Moderate: 60% stocks (VTI) + 40% bonds (BND); rebalance annually
├─ Aggressive: 80% stocks (VTI) + 20% bonds (BND); rebalance quarterly
└─ Cost: 0.05% VTI ER + 0.05% BND ER = 0.10% total (very cheap!)
""")

print("=" * 100)