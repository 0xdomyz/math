# Fama-French Three-Factor Model

## 1. Concept Skeleton
**Definition:** Asset pricing model extending CAPM with size (SMB) and value (HML) factors; explains cross-sectional return variation  
**Purpose:** Capture systematic risk beyond market beta; explain size and value premiums; improve performance attribution  
**Prerequisites:** CAPM, regression analysis, market capitalization, book-to-market ratio

## 2. Comparative Framing
| Model | CAPM | Fama-French 3-Factor | Carhart 4-Factor | Fama-French 5-Factor |
|-------|------|---------------------|------------------|---------------------|
| **Factors** | Market only | Market + Size + Value | + Momentum | + Profitability + Investment |
| **R² (avg)** | ~70% | ~90% | ~92% | ~93% |
| **Anomalies Explained** | None | Size, Value | + Momentum | + Profitability, Investment |
| **Data Start** | 1926 | 1926 | 1926 | 1963 |
| **Complexity** | Simple (1 factor) | Moderate (3 factors) | Moderate (4 factors) | Higher (5 factors) |

## 3. Examples + Counterexamples

**Value Premium Example:**  
High B/M stocks (value) outperform low B/M (growth) by ~5% annually (1926-2020)  
HML factor captures this premium; positive HML loading → value tilt

**Size Premium Example:**  
Small-cap stocks historically outperform large-cap by ~3% annually  
SMB factor captures small-cap premium; positive SMB loading → small-cap exposure

**Failure Case:**  
Tech bubble (1998-2000): Growth stocks (negative HML) outperformed dramatically  
Value premium reversed temporarily; factor models struggle with regime shifts

## 4. Layer Breakdown
```
Fama-French Three-Factor Model:
├─ Model Specification:
│   ├─ Ri - rf = αi + βi(Rm - rf) + si·SMB + hi·HML + εi
│   ├─ Ri: Asset/portfolio return
│   ├─ rf: Risk-free rate (T-bill)
│   ├─ Rm - rf: Market excess return (Mkt-RF)
│   ├─ SMB: Small Minus Big (size factor)
│   ├─ HML: High Minus Low (value factor)
│   ├─ αi: Intercept (alpha, unexplained return)
│   └─ βi, si, hi: Factor loadings (sensitivities)
├─ Factor Construction (Fama-French methodology):
│   ├─ SMB (Size Factor):
│   │   ├─ Sort NYSE/AMEX/NASDAQ stocks by market cap
│   │   ├─ Median split: Small (below) vs Big (above)
│   │   ├─ Within each size group, split by B/M into 3 groups
│   │   ├─ SMB = (Small Value + Small Neutral + Small Growth)/3 
│   │   │        - (Big Value + Big Neutral + Big Growth)/3
│   │   └─ Returns difference: Small - Big portfolios
│   ├─ HML (Value Factor):
│   │   ├─ Book-to-Market ratio = Book equity / Market equity
│   │   ├─ Sort into 3 groups: High (top 30%), Medium (40%), Low (bottom 30%)
│   │   ├─ Within each B/M group, split by size
│   │   ├─ HML = (Small Value + Big Value)/2 
│   │   │        - (Small Growth + Big Growth)/2
│   │   └─ Returns difference: High B/M - Low B/M portfolios
│   └─ Mkt-RF (Market Factor):
│       ├─ Market return: Value-weighted return of all stocks
│       ├─ Minus risk-free rate (1-month T-bill)
│       └─ CRSP value-weighted index typically used
├─ Economic Rationale:
│   ├─ Size Premium:
│   │   ├─ Small firms riskier: Higher failure rates, less liquidity
│   │   ├─ Information asymmetry: Less analyst coverage
│   │   ├─ Transaction costs: Higher bid-ask spreads
│   │   └─ Compensation for these risks → higher expected returns
│   ├─ Value Premium:
│   │   ├─ Financial distress: High B/M = low market price (troubles)
│   │   ├─ Mean reversion: Mispricing corrects over time
│   │   ├─ Risk-based: Value firms more sensitive to economic distress
│   │   └─ Behavioral: Overreaction to bad news creates opportunity
│   └─ vs CAPM: Market beta insufficient; size and value are priced risks
├─ Interpretation of Factor Loadings:
│   ├─ βi > 1: High market sensitivity (aggressive)
│   ├─ si > 0: Small-cap tilt; si < 0: Large-cap tilt
│   ├─ hi > 0: Value tilt; hi < 0: Growth tilt
│   ├─ α ≈ 0: Returns fully explained by factors (no skill)
│   └─ α > 0: Outperformance after adjusting for factor exposures
├─ Estimation Process:
│   ├─ Data: Monthly returns for asset and 3 factors
│   ├─ Time Series Regression: OLS of excess returns on factors
│   ├─ Standard Errors: Newey-West for autocorrelation/heteroskedasticity
│   ├─ R²: Proportion of variance explained (typically 0.85-0.95)
│   └─ t-statistics: Test significance of α and factor loadings
├─ Applications:
│   ├─ Performance Attribution: Decompose returns into factors + alpha
│   ├─ Risk Management: Understand factor exposures (style drift detection)
│   ├─ Expected Return Estimation: E[Ri] = rf + βi·E[Rm-rf] + si·E[SMB] + hi·E[HML]
│   ├─ Portfolio Construction: Target specific factor tilts
│   └─ Benchmarking: Compare alpha after factor adjustment (not just vs S&P 500)
├─ Empirical Evidence (1926-2020):
│   ├─ Market Premium (Rm - rf): ~8% annually
│   ├─ Size Premium (SMB): ~3% annually (weaker post-1980)
│   ├─ Value Premium (HML): ~5% annually (persistent across countries)
│   ├─ R² Improvement: 70% (CAPM) → 90% (FF3)
│   └─ Alpha Reduction: Many active managers have α ≈ 0 after FF3 adjustment
└─ Limitations & Critiques:
    ├─ Data Mining: Factors chosen retrospectively (hindsight bias)
    ├─ Size Premium Diminishment: Weaker since discovery (1981+)
    ├─ Factor Timing: Premiums time-varying (value struggled 2007-2020)
    ├─ Missing Factors: Doesn't capture momentum, profitability
    ├─ Risk vs Mispricing: Debate whether premiums are risk-based or behavioral
    ├─ International: Factors work globally but magnitudes differ
    └─ Factor Crowding: As strategies become popular, premiums may compress
```

**Interaction:** Time series regression decomposes returns into market, size, value exposures plus alpha

## 5. Mini-Project
Implement Fama-French three-factor model for portfolio analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats

# Download test portfolio data (various ETFs with different styles)
tickers = {
    'SPY': 'Large-Cap Blend',
    'IWM': 'Small-Cap Blend', 
    'IWD': 'Large-Cap Value',
    'IWF': 'Large-Cap Growth',
    'IWN': 'Small-Cap Value',
    'IWO': 'Small-Cap Growth',
    'VTV': 'Value ETF',
    'VUG': 'Growth ETF'
}

end_date = datetime.now()
start_date = datetime(2015, 1, 1)  # Need longer history for robust estimates

print("Downloading portfolio data...")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Download Fama-French factors from Ken French's data library
# For this example, we'll construct proxy factors using ETFs
print("\nConstructing factor proxies from ETF data...")

# Risk-free rate proxy (using constant 2% annual = 0.167% monthly for simplicity)
# In practice, download actual T-bill rates
rf_monthly = 0.02 / 12

# Market factor: SPY (S&P 500)
market_returns = returns['SPY']
mkt_rf = market_returns - rf_monthly

# SMB proxy: Small-cap (IWM) - Large-cap (SPY)
smb = returns['IWM'] - returns['SPY']

# HML proxy: Value (VTV) - Growth (VUG)
hml = returns['VTV'] - returns['VUG']

# Create factor DataFrame
factors = pd.DataFrame({
    'Mkt-RF': mkt_rf,
    'SMB': smb,
    'HML': hml
})

# Remove any NaN values
factors = factors.dropna()

def fama_french_regression(asset_returns, factors, rf=None):
    """
    Run Fama-French 3-factor regression
    
    Returns: Dictionary with regression results
    """
    if rf is None:
        rf = 0
    
    # Excess returns
    excess_returns = asset_returns - rf
    
    # Align dates
    common_dates = excess_returns.index.intersection(factors.index)
    y = excess_returns.loc[common_dates]
    X = factors.loc[common_dates]
    
    # Add constant for alpha
    X_with_const = sm.add_constant(X)
    
    # OLS regression
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Newey-West standard errors (corrects for autocorrelation/heteroskedasticity)
    results_robust = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params['Mkt-RF'],
        'smb': results.params['SMB'],
        'hml': results.params['HML'],
        'alpha_tstat': results_robust.tvalues['const'],
        'beta_tstat': results_robust.tvalues['Mkt-RF'],
        'smb_tstat': results_robust.tvalues['SMB'],
        'hml_tstat': results_robust.tvalues['HML'],
        'alpha_pval': results_robust.pvalues['const'],
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'residuals': results.resid,
        'fitted': results.fittedvalues,
        'results': results_robust
    }

def calculate_capm_alpha(asset_returns, market_returns, rf=0):
    """
    Calculate CAPM alpha for comparison
    """
    excess_asset = asset_returns - rf
    excess_market = market_returns - rf
    
    common_dates = excess_asset.index.intersection(excess_market.index)
    y = excess_asset.loc[common_dates]
    X = sm.add_constant(excess_market.loc[common_dates])
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params[excess_market.name] if hasattr(excess_market, 'name') else results.params.iloc[1],
        'rsquared': results.rsquared,
        'alpha_tstat': results.tvalues['const']
    }

# Run regressions for all assets
ff3_results = {}
capm_results = {}

for ticker in tickers.keys():
    asset_ret = returns[ticker]
    
    # Fama-French 3-factor
    ff3 = fama_french_regression(asset_ret, factors, rf_monthly)
    ff3_results[ticker] = ff3
    
    # CAPM for comparison
    capm = calculate_capm_alpha(asset_ret, market_returns, rf_monthly)
    capm_results[ticker] = capm

# Create comparison DataFrame
comparison_data = []
for ticker in tickers.keys():
    comparison_data.append({
        'Ticker': ticker,
        'Style': tickers[ticker],
        'CAPM Alpha': capm_results[ticker]['alpha'] * 12 * 100,  # Annualized %
        'CAPM Beta': capm_results[ticker]['beta'],
        'CAPM R²': capm_results[ticker]['rsquared'],
        'FF3 Alpha': ff3_results[ticker]['alpha'] * 12 * 100,  # Annualized %
        'FF3 Beta': ff3_results[ticker]['beta'],
        'FF3 SMB': ff3_results[ticker]['smb'],
        'FF3 HML': ff3_results[ticker]['hml'],
        'FF3 R²': ff3_results[ticker]['rsquared'],
        'Alpha t-stat': ff3_results[ticker]['alpha_tstat'],
        'Alpha Sig': 'Yes' if abs(ff3_results[ticker]['alpha_tstat']) > 1.96 else 'No'
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*120)
print("FAMA-FRENCH 3-FACTOR MODEL ANALYSIS")
print("="*120)
print(comparison_df.to_string(index=False))

# Detailed output for one example (Small-Cap Value)
print("\n" + "="*120)
print("DETAILED REGRESSION OUTPUT: IWN (Small-Cap Value ETF)")
print("="*120)
print(ff3_results['IWN']['results'].summary())

# Factor premiums over the period
print("\n" + "="*120)
print("FACTOR PREMIUMS (Sample Period)")
print("="*120)
factor_stats = pd.DataFrame({
    'Mean (Monthly %)': factors.mean() * 100,
    'Mean (Annual %)': factors.mean() * 12 * 100,
    'Std Dev (Annual %)': factors.std() * np.sqrt(12) * 100,
    'Sharpe Ratio': (factors.mean() / factors.std()) * np.sqrt(12),
    't-statistic': factors.mean() / (factors.std() / np.sqrt(len(factors)))
})
print(factor_stats.round(3))

# Attribution analysis
def attribution_analysis(ticker, ff3_result, factors_df, period_returns):
    """
    Decompose total return into factor contributions
    """
    total_return = (1 + period_returns).prod() - 1
    
    # Factor contributions (geometric approximation using arithmetic for simplicity)
    factor_contrib = {
        'Market': ff3_result['beta'] * factors_df['Mkt-RF'].sum(),
        'Size (SMB)': ff3_result['smb'] * factors_df['SMB'].sum(),
        'Value (HML)': ff3_result['hml'] * factors_df['HML'].sum(),
        'Alpha': ff3_result['alpha'] * len(factors_df),
        'Residual': ff3_result['residuals'].sum()
    }
    
    return factor_contrib, total_return

# Attribution for Small-Cap Value
iwn_contrib, iwn_total = attribution_analysis('IWN', ff3_results['IWN'], factors, returns['IWN'])

print("\n" + "="*120)
print(f"RETURN ATTRIBUTION: IWN (Total Return: {iwn_total*100:.2f}%)")
print("="*120)
for source, contrib in iwn_contrib.items():
    pct_of_total = (contrib / iwn_total * 100) if iwn_total != 0 else 0
    print(f"{source:>15}: {contrib*100:>8.2f}% ({pct_of_total:>6.1f}% of total)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha comparison (CAPM vs FF3)
x = np.arange(len(tickers))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, comparison_df['CAPM Alpha'], width, 
                       label='CAPM Alpha', alpha=0.8)
bars2 = axes[0, 0].bar(x + width/2, comparison_df['FF3 Alpha'], width,
                       label='FF3 Alpha', alpha=0.8)

axes[0, 0].set_xlabel('ETF')
axes[0, 0].set_ylabel('Alpha (% per year)')
axes[0, 0].set_title('Alpha Comparison: CAPM vs Fama-French 3-Factor')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(comparison_df['Ticker'], rotation=45, ha='right')
axes[0, 0].axhline(0, color='black', linewidth=0.5)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Factor loadings heatmap
factor_loadings = comparison_df[['Ticker', 'FF3 Beta', 'FF3 SMB', 'FF3 HML']].set_index('Ticker')
im = axes[0, 1].imshow(factor_loadings.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

axes[0, 1].set_yticks(range(len(factor_loadings.columns)))
axes[0, 1].set_yticklabels(factor_loadings.columns)
axes[0, 1].set_xticks(range(len(factor_loadings)))
axes[0, 1].set_xticklabels(factor_loadings.index, rotation=45, ha='right')
axes[0, 1].set_title('Factor Loadings Heatmap')

# Add text annotations
for i in range(len(factor_loadings.columns)):
    for j in range(len(factor_loadings)):
        text = axes[0, 1].text(j, i, f'{factor_loadings.iloc[j, i]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=axes[0, 1])

# Plot 3: R² comparison
axes[1, 0].scatter(comparison_df['CAPM R²'], comparison_df['FF3 R²'], s=100, alpha=0.7)
for idx, row in comparison_df.iterrows():
    axes[1, 0].annotate(row['Ticker'], 
                       (row['CAPM R²'], row['FF3 R²']),
                       fontsize=8, ha='right')

# Add diagonal line
axes[1, 0].plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='Equal R²')
axes[1, 0].set_xlabel('CAPM R²')
axes[1, 0].set_ylabel('Fama-French 3-Factor R²')
axes[1, 0].set_title('Explanatory Power: CAPM vs FF3')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim([0.5, 1.0])
axes[1, 0].set_ylim([0.5, 1.0])

# Plot 4: Factor premiums over time (rolling 12-month)
rolling_window = 12
rolling_premiums = factors.rolling(window=rolling_window).mean() * 12 * 100

axes[1, 1].plot(rolling_premiums.index, rolling_premiums['Mkt-RF'], 
               label='Market Premium', linewidth=2)
axes[1, 1].plot(rolling_premiums.index, rolling_premiums['SMB'],
               label='Size Premium (SMB)', linewidth=2)
axes[1, 1].plot(rolling_premiums.index, rolling_premiums['HML'],
               label='Value Premium (HML)', linewidth=2)

axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Rolling 12-Month Premium (%)')
axes[1, 1].set_title('Factor Premiums Over Time (12-Month Rolling)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical tests
print("\n" + "="*120)
print("STATISTICAL SIGNIFICANCE OF FACTOR LOADINGS")
print("="*120)
print(f"{'Ticker':>8} {'Beta':>8} {'t-stat':>8} {'SMB':>8} {'t-stat':>8} {'HML':>8} {'t-stat':>8}")
print("-"*120)

for ticker in tickers.keys():
    ff3 = ff3_results[ticker]
    print(f"{ticker:>8} {ff3['beta']:>8.3f} {ff3['beta_tstat']:>8.2f} "
          f"{ff3['smb']:>8.3f} {ff3['smb_tstat']:>8.2f} "
          f"{ff3['hml']:>8.3f} {ff3['hml_tstat']:>8.2f}")

print("\nInterpretation: |t-stat| > 1.96 indicates significance at 5% level")

# Key insights
print("\n" + "="*120)
print("KEY INSIGHTS: FAMA-FRENCH THREE-FACTOR MODEL")
print("="*120)
print("1. FF3 substantially increases R² vs CAPM (typically +15-20 percentage points)")
print("2. Small-cap stocks have positive SMB loadings (IWM, IWN, IWO)")
print("3. Value stocks have positive HML loadings (IWD, IWN, VTV)")
print("4. Growth stocks have negative HML loadings (IWF, IWO, VUG)")
print("5. Many alphas become insignificant after factor adjustment (skill vs style)")
print("6. Factor premiums time-varying: Value struggled 2015-2020, rebounded 2021+")
print("7. SMB factor weaker in recent decades (size premium diminished)")
print("8. Factor model useful for style analysis and performance attribution")

# Factor correlation analysis
print("\n" + "="*120)
print("FACTOR CORRELATION MATRIX")
print("="*120)
factor_corr = factors.corr()
print(factor_corr.round(3))
print("\nNote: Low correlations indicate factors capture independent sources of risk/return")
```

## 6. Challenge Round
Why does Fama-French outperform CAPM empirically?
- Size premium: Small firms historically earn ~3% more (higher risk, less liquidity)
- Value premium: High B/M firms earn ~5% more (distress risk or mispricing correction)
- CAPM anomalies: Market beta alone doesn't explain cross-sectional returns
- Additional risk factors: Size and value capture systematic risks beyond market
- R² improvement: 70% (CAPM) → 90% (FF3) for typical equity portfolios

Limitations and ongoing debates:
- Data mining criticism: Factors selected because they worked historically (hindsight)
- Size premium diminished: Weaker since Banz (1981) published finding (crowding?)
- Risk vs behavioral: Is value premium compensation for distress or investor overreaction?
- Factor timing: Value struggled 2007-2020; momentum (excluded from FF3) persisted
- International evidence: Factors work globally but premiums vary by market
- Missing factors: Momentum, profitability, investment (led to FF5)

When should you use FF3 vs CAPM?
- Performance attribution: FF3 separates skill (alpha) from style (factor exposure)
- Risk management: Understand true factor exposures (not just beta)
- Expected returns: Better estimates incorporating size/value premiums
- Portfolio construction: Target specific factor tilts (smart beta strategies)
- CAPM sufficient: For broad market index funds with no style tilt

## 7. Key References
- [Fama, E.F. & French, K.R. (1993) "Common Risk Factors in Returns on Stocks and Bonds"](https://www.sciencedirect.com/science/article/abs/pii/0304405X93900235) - Original paper
- [Fama, E.F. & French, K.R. (2015) "A Five-Factor Asset Pricing Model"](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002323) - Extension to 5 factors
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) - Official factor data
- [Investopedia - Fama French Three-Factor Model](https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp)

---
**Status:** Standard asset pricing model (Nobel Prize 2013) | **Complements:** CAPM, Carhart 4-Factor, FF5
