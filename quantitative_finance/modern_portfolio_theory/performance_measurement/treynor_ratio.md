# Treynor Ratio

## 1. Concept Skeleton
**Definition:** Risk-adjusted return measure dividing excess return by systematic risk (beta)  
**Purpose:** Evaluate portfolio performance per unit of market risk for well-diversified portfolios  
**Prerequisites:** CAPM, beta calculation, systematic vs idiosyncratic risk distinction

## 2. Comparative Framing
| Aspect | Treynor Ratio | Sharpe Ratio | Jensen's Alpha |
|--------|---------------|--------------|----------------|
| **Risk Metric** | Beta (systematic) | Volatility (total) | Beta (systematic) |
| **Output** | Ratio (dimensionless) | Ratio (dimensionless) | Absolute return |
| **Best For** | Diversified portfolios | All portfolios | CAPM performance attribution |
| **Assumption** | Diversified away idiosyncratic risk | All risk matters | CAPM holds |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio A: 14% return, β=1.3, rf=3% → Treynor = (14-3)/1.3 = 8.46%  
Portfolio B: 12% return, β=0.9, rf=3% → Treynor = (12-3)/0.9 = 10.0%  
Portfolio B generates more return per unit market risk despite lower absolute return

**Failure Case:**  
Concentrated portfolio (3 stocks): β=1.2 but high idiosyncratic risk; Treynor looks good but Sharpe reveals total risk problem

**Edge Case:**  
Beta = 0 (perfect hedge): Treynor ratio undefined, division by zero

## 4. Layer Breakdown
```
Treynor Ratio Calculation:
├─ Inputs:
│   ├─ Portfolio Return (Rp): Annualized excess return over period
│   ├─ Portfolio Beta (βp): Cov(Rp, Rm) / Var(Rm)
│   └─ Risk-Free Rate (rf): Treasury bill rate
├─ Beta Estimation:
│   ├─ Regression: Rp,t - rf = α + β(Rm,t - rf) + εt
│   ├─ Minimum Period: 2-5 years monthly data (24-60 observations)
│   └─ Market Proxy: S&P 500, MSCI World, relevant benchmark
├─ Computation:
│   ├─ Excess Return: Rp - rf
│   ├─ Risk Adjustment: Divide by βp
│   └─ Treynor Ratio: (Rp - rf) / βp (% return per unit beta)
├─ Interpretation:
│   ├─ Higher is better (more return per market risk unit)
│   ├─ Only valid for diversified portfolios
│   ├─ Compare portfolios with same market benchmark
│   └─ Units are percent return (not dimensionless like Sharpe)
└─ Relationship to CAPM:
    ├─ Treynor = Excess Return / β
    ├─ Market Portfolio: Treynor = (Rm - rf) / 1.0 = Market Premium
    └─ Superior Performance: Treynor > Market Risk Premium
```

**Interaction:** Assumes idiosyncratic risk diversified away; beta captures only systematic exposure

## 5. Mini-Project
Compare Sharpe vs Treynor for diversified and concentrated portfolios:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# Parameters
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
rf_annual = 0.03

# Download market data
market = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
market_returns = market.pct_change().dropna()

# Portfolio definitions
portfolio_tickers = {
    'Diversified Large Cap': ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 'PG', 'V', 'HD'],
    'Concentrated Tech': ['NVDA', 'TSLA', 'AMD'],
    'Defensive': ['KO', 'PEP', 'WMT', 'PG', 'JNJ'],
    'High Beta': ['TSLA', 'NVDA', 'COIN', 'GME'],
}

def calculate_beta(portfolio_returns, market_returns):
    """Calculate portfolio beta via regression"""
    # Align dates
    aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    aligned.columns = ['portfolio', 'market']
    
    # Run regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        aligned['market'], aligned['portfolio']
    )
    
    return slope, r_value**2, intercept

def analyze_portfolio(tickers, market_returns, name):
    """Download data and calculate performance metrics"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        returns = data.pct_change().dropna()
        
        # Equal weight portfolio
        portfolio_returns = returns.mean(axis=1)
        
        # Align with market
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        aligned.columns = ['portfolio', 'market']
        
        # Calculate metrics
        rf_daily = (1 + rf_annual) ** (1/252) - 1
        
        # Annualized return and volatility
        annual_return = aligned['portfolio'].mean() * 252
        annual_vol = aligned['portfolio'].std() * np.sqrt(252)
        
        # Beta
        beta, r_squared, alpha_daily = calculate_beta(
            aligned['portfolio'], aligned['market']
        )
        alpha_annual = alpha_daily * 252
        
        # Sharpe ratio
        excess_return = annual_return - rf_annual
        sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Treynor ratio
        treynor_ratio = excess_return / beta if beta != 0 else np.nan
        
        # Market return for comparison
        market_annual_return = aligned['market'].mean() * 252
        market_annual_vol = aligned['market'].std() * np.sqrt(252)
        
        return {
            'Name': name,
            'Annual Return': annual_return,
            'Volatility': annual_vol,
            'Beta': beta,
            'R-squared': r_squared,
            'Alpha (annual)': alpha_annual,
            'Sharpe Ratio': sharpe_ratio,
            'Treynor Ratio': treynor_ratio,
            'N Assets': len(tickers),
            'Returns': aligned['portfolio']
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return None

# Analyze all portfolios
results = []
for name, tickers in portfolio_tickers.items():
    result = analyze_portfolio(tickers, market_returns, name)
    if result:
        results.append(result)

# Market portfolio metrics
market_annual_return = market_returns.mean() * 252
market_annual_vol = market_returns.std() * np.sqrt(252)
market_sharpe = (market_annual_return - rf_annual) / market_annual_vol
market_treynor = (market_annual_return - rf_annual) / 1.0  # Beta = 1

results.append({
    'Name': 'Market (SPY)',
    'Annual Return': market_annual_return,
    'Volatility': market_annual_vol,
    'Beta': 1.0,
    'R-squared': 1.0,
    'Alpha (annual)': 0.0,
    'Sharpe Ratio': market_sharpe,
    'Treynor Ratio': market_treynor,
    'N Assets': 500,
    'Returns': market_returns
})

# Create summary DataFrame
summary_cols = ['Name', 'N Assets', 'Annual Return', 'Volatility', 
                'Beta', 'R-squared', 'Alpha (annual)', 
                'Sharpe Ratio', 'Treynor Ratio']
summary = pd.DataFrame([{k: r[k] for k in summary_cols} for r in results])
summary = summary.sort_values('Treynor Ratio', ascending=False)

print("Portfolio Performance Analysis: Sharpe vs Treynor")
print("=" * 100)
print(summary.to_string(index=False))
print("\n" + "=" * 100)
print(f"Risk-Free Rate: {rf_annual:.2%}")
print(f"Market Risk Premium: {(market_annual_return - rf_annual):.2%}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sharpe vs Treynor comparison
portfolios = summary[summary['Name'] != 'Market (SPY)']
x_pos = np.arange(len(portfolios))
width = 0.35

bars1 = axes[0, 0].bar(x_pos - width/2, portfolios['Sharpe Ratio'], 
                       width, label='Sharpe Ratio', alpha=0.8)
bars2 = axes[0, 0].bar(x_pos + width/2, portfolios['Treynor Ratio'], 
                       width, label='Treynor Ratio', alpha=0.8)

axes[0, 0].set_xlabel('Portfolio')
axes[0, 0].set_ylabel('Risk-Adjusted Return')
axes[0, 0].set_title('Sharpe vs Treynor Ratio Comparison')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(portfolios['Name'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].axhline(market_sharpe, color='blue', linestyle='--', alpha=0.5)
axes[0, 0].axhline(market_treynor, color='orange', linestyle='--', alpha=0.5)

# Plot 2: Beta vs R-squared scatter
scatter = axes[0, 1].scatter(summary['Beta'], summary['R-squared'],
                            s=summary['N Assets']*2, alpha=0.6,
                            c=summary['Sharpe Ratio'], cmap='viridis')
for idx, row in summary.iterrows():
    axes[0, 1].annotate(row['Name'].split()[0], 
                       (row['Beta'], row['R-squared']),
                       fontsize=8, ha='center')

axes[0, 1].set_xlabel('Portfolio Beta')
axes[0, 1].set_ylabel('R-squared (diversification)')
axes[0, 1].set_title('Systematic Risk vs Diversification')
axes[0, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Sharpe Ratio')

# Plot 3: When Sharpe and Treynor disagree
sharpe_rank = summary.sort_values('Sharpe Ratio', ascending=False)['Name'].tolist()
treynor_rank = summary.sort_values('Treynor Ratio', ascending=False)['Name'].tolist()

y_pos = np.arange(len(sharpe_rank))
axes[1, 0].barh(y_pos, range(len(sharpe_rank), 0, -1), alpha=0.5, label='Sharpe Ranking')
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(sharpe_rank)
axes[1, 0].set_xlabel('Rank (Higher is Better)')
axes[1, 0].set_title('Sharpe Ratio Rankings')
axes[1, 0].invert_xaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

axes2 = axes[1, 0].twinx()
y_pos2 = [sharpe_rank.index(name) for name in treynor_rank]
axes2.barh(y_pos2, range(1, len(treynor_rank)+1), alpha=0.5, 
          color='orange', label='Treynor Ranking')
axes2.set_yticks(y_pos)
axes2.set_yticklabels([''] * len(y_pos))
axes2.set_xlabel('Rank (Higher is Better)')

# Plot 4: Idiosyncratic vs Systematic risk
total_risk = summary['Volatility']
systematic_risk = summary['Beta'] * market_annual_vol
idiosyncratic_risk = np.sqrt(np.maximum(0, total_risk**2 - systematic_risk**2))

x_pos = np.arange(len(summary))
axes[1, 1].bar(x_pos, systematic_risk, label='Systematic Risk (β×σm)', alpha=0.8)
axes[1, 1].bar(x_pos, idiosyncratic_risk, bottom=systematic_risk,
              label='Idiosyncratic Risk', alpha=0.8)

axes[1, 1].set_xlabel('Portfolio')
axes[1, 1].set_ylabel('Volatility (Annualized)')
axes[1, 1].set_title('Risk Decomposition: Systematic vs Idiosyncratic')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(summary['Name'], rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print("1. Treynor Ratio only meaningful for well-diversified portfolios (high R²)")
print("2. Sharpe considers all risk; Treynor only systematic risk")
print("3. For concentrated portfolios, Sharpe < Treynor×market beta (idiosyncratic penalty)")
print("\nInterpretation Guide:")
print("- High Treynor, Low Sharpe → Undiversified, high idiosyncratic risk")
print("- High R² (>0.8) → Treynor valid, mostly systematic risk")
print("- Low R² (<0.6) → Use Sharpe, significant diversifiable risk remains")
```

## 6. Challenge Round
When should you prefer Treynor over Sharpe?
- Well-diversified institutional portfolios: Idiosyncratic risk eliminated
- Part of larger portfolio: When evaluating contribution to multi-manager fund
- Active manager evaluation: Isolate systematic exposure skill from stock selection
- CAPM framework: Consistency with equilibrium pricing theory

When Treynor fails:
- Concentrated positions: Ignores diversifiable risk that matters
- Alternative assets: Beta to equity market irrelevant for commodities, real estate
- Non-linear exposures: Options, structured products have unstable betas
- Market-neutral strategies: Beta ≈ 0 makes Treynor undefined

## 7. Key References
- [Treynor, J. (1965) "How to Rate Management of Investment Funds"](https://www.cfapubs.org/doi/abs/10.2469/faj.v21.n1.63)
- [Investopedia - Treynor Ratio](https://www.investopedia.com/terms/t/treynorratio.asp)
- [Sharpe vs Treynor Comparison (CFA Institute)](https://www.cfainstitute.org/en/research/cfa-digest/2002/06/performance-measurement)

---
**Status:** Systematic risk-adjusted performance | **Complements:** Sharpe Ratio, Jensen's Alpha, Information Ratio
