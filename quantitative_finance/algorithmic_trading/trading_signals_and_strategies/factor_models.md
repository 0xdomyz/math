# Factor Models in Algorithmic Trading

## 1. Concept Skeleton
**Definition:** Systematic approach decomposing asset returns into factor exposures (beta × factor return) plus alpha; factors explain return variations beyond market beta; classic examples: Fama-French 3/5-factor model (market, value, size, profitability, investment)  
**Purpose:** Identify systematic sources of return; reduce idiosyncratic risk via factor diversification; decompose strategy returns into explainable components; construct factor-efficient portfolios  
**Prerequisites:** Regression analysis, correlation/covariance matrices, risk decomposition, asset pricing theory, factor estimation

## 2. Comparative Framing
| Aspect | CAPM (1-Factor) | Fama-French 3-Factor | Fama-French 5-Factor | Alternative Factors |
|--------|-----------------|----------------------|----------------------|-------------------|
| **Factors** | Market (Rm - Rf) | Market + Value (HML) + Size (SMB) | FF3 + Profitability (RMW) + Investment (CMA) | Momentum, Quality, Liquidity, Volatility |
| **Return Equation** | Ri = α + β(Rm) + ε | Ri = α + β(Rm) + βᵥ(HML) + βₛ(SMB) + ε | Ri = α + β(Rm) + βᵥ(HML) + βₛ(SMB) + βₚ(RMW) + βᵢ(CMA) + ε | Custom portfolio of 6+ factors |
| **Explanatory Power** | ~70% of returns (R²) | ~90% of returns | ~93-95% of returns | 85-98% (varies) |
| **Alpha Significance** | High (exploitable anomalies) | Reduced (residuals smaller) | Further reduced | Depends on factors chosen |
| **Computation** | Simple (1 regression) | Moderate (3 regressions) | Moderate (5 regressions) | Complex (factor construction) |
| **Data Availability** | Public (market returns) | Public (FF portal) | Public (FF portal) | Varies (proprietary sources) |
| **Implementation Lag** | Real-time | Monthly updates | Monthly updates | Varies |
| **Academic Consensus** | Universal (since 1964) | High (since 1993) | Moderate (since 2015) | Emerging |
| **Profitability** | Declining (overcrowded) | Declining (crowded) | Still viable | Varies widely |

## 3. Examples + Counterexamples

**CAPM Failure (2008 Crisis):**  
CAPM predicts: Portfolio β = 1.0 → moves with market, σ = market volatility (15%).  
Reality: 2008 crash → all betas spike to 1.5-2.0, vol jumps to 40%+. Returns uncorrelated to CAPM prediction.  
Lesson: Single-factor model misses tail risk, doesn't capture factor beta changes in crises.

**Fama-French 3-Factor Success (2000-2010):**  
Value factor (HML): High book-to-market (cheap) stocks beat low book-to-market (expensive) stocks.  
Period: Tech crash (2000-2002) devastated growth (low HML), value thrived. +100%+ returns in 3 years.  
FF3 correctly predicted value outperformance via HML factor exposure.

**Size Factor Reversal (Small-Cap Trap):**  
1980s-2000s: Small-cap effect documented; small stocks beat large (SMB > 0). Widely exploited.  
2000s onwards: Small-cap factor reversal; large-cap tech (mega-cap) dominates. SMB becomes negative.  
Factor crowding: Once "anomaly" discovered, capital floods in, effect diminishes.

**Profitability Factor Discovery (2015):**  
FF5 adds profitability (RMW): Profitable firms outperform unprofitable. Seems obvious.  
Surprising finding: Market underprices profitability (market is irrational). Exploitable alpha.  
Post-discovery: Profitability factor return compresses as more capital targets it. 2020+ less profitable.

**Alternative Factor Disaster (Volatility Collapse 2020):**  
Low-volatility factor: Stocks with low historical vol beat market (risk premium inversion).  
During March 2020 crash: Vol explodes, low-vol stocks underperform (-40%+). Factor breaks.  
Lesson: Low-vol factor works in "normal" regimes; fails in tail events (when vol truly high).

**Momentum Factor Reversal 2008 & 2020:**  
Momentum factor (WML: winner - loser): Winning stocks continue. Built-in crowding risk.  
2008/2020: Risk-off environment → all longs liquidated simultaneously. Momentum crashes -50%+.  
Factor crowding creates momentum crash risk (everyone exits same time).

## 4. Layer Breakdown
```
Factor Model Architecture:

├─ Single-Factor Model (CAPM)
│  ├─ Formula:
│  │   Ri(t) = αi + βi × Rm(t) + εi(t)
│  │   - Ri: Asset return
│  │   - βi: Beta (market sensitivity)
│  │   - Rm: Market return
│  │   - εi: Idiosyncratic risk
│  ├─ Estimation:
│  │   - Regress asset returns on market returns
│  │   - β = Cov(Ri, Rm) / Var(Rm)
│  │   - α = Mean(Ri) - β × Mean(Rm)
│  ├─ Interpretation:
│  │   - β > 1: Aggressive (moves more than market)
│  │   - β = 1: In sync with market
│  │   - β < 1: Defensive (moves less than market)
│  │   - 0 < β < 1: Lower risk than market
│  │   - β < 0: Uncorrelated/hedging (rare)
│  └─ Limitations:
│      - Ignores value/size/profitability effects
│      - High R² but systematic over/underpricing of specific types
│      - Beta unstable across regimes (jumps in crises)
│
├─ Fama-French 3-Factor Model
│  ├─ Formula:
│  │   Ri(t) = αi + βm × Rm(t) + βv × HML(t) + βs × SMB(t) + εi(t)
│  │   - HML: Value factor (High - Low book-to-market)
│  │   - SMB: Size factor (Small - Big market cap)
│  ├─ Factor Construction:
│  │   ├─ Market (Rm): Value-weighted market portfolio return
│  │   ├─ HML: Long high book-to-market (cheap), short low (expensive)
│  │   │   - Longing distressed value plays, shorting overpriced growth
│  │   │   - Historical excess return: +3-5% annual (value premium)
│  │   └─ SMB: Long small-cap, short large-cap
│  │       - Small-cap premium: +2-3% historical (disputed in recent years)
│  ├─ Factor Returns (Historical):
│  │   - Market premium: ~5-7% annual above risk-free
│  │   - Value premium (HML): +3-5% annual
│  │   - Size premium (SMB): +2-3% annual (mostly 1980s-1990s)
│  ├─ Empirical Results:
│  │   - R² typically 90%+ (explains most variance)
│  │   - Residual α smaller (harder to find alpha after controlling for factors)
│  │   - Cross-sectional: Systematically explains 50%+ of stock return cross-section
│  └─ Trading Application:
│      - Portfolio targeting high HML + low SMB exposure = value strategy
│      - Portfolio targeting low HML + low SMB exposure = growth/small-cap
│      - Factor-neutral construction: βv ≈ 0, βs ≈ 0 (market-only exposure)
│
├─ Fama-French 5-Factor Model (2015)
│  ├─ Addition to FF3:
│  │   - RMW: Profitability (Robust - Weak operating profitability)
│  │   - CMA: Investment (Conservative - Aggressive capital investment)
│  ├─ Profitability Factor (RMW):
│  │   ├─ Long: High operating profitability (ROA, ROE, profit margins)
│  │   ├─ Short: Low operating profitability
│  │   ├─ Rationale: Market underprices sustainable profitable businesses
│  │   ├─ Excess return: +2-3% annual
│  │   └─ Finding: Distinct from value; profitable cheap stocks best
│  ├─ Investment Factor (CMA):
│  │   ├─ Long: Conservative investment (low capex-to-assets)
│  │   ├─ Short: Aggressive investment (high capex, R&D)
│  │   ├─ Rationale: Market overprices growth stories (overinvestment destroys value)
│  │   ├─ Excess return: +3-5% annual
│  │   └─ Finding: High-capex firms (Tesla, pharma) underperform expected
│  ├─ Returns Explained:
│  │   - FF3: ~90% of variance
│  │   - FF5: ~93-95% of variance (incremental gains modest)
│  ├─ Trading Strategy:
│  │   - Targeting: High HML + high RMW + high CMA = "quality value"
│  │   - Characteristics: Cheap, profitable, conservative (best performers)
│  │   - Opposite: Growth, unprofitable, aggressive (worst)
│  └─ Evolution:
│      - FF3 dominance: 1993-2015 (22 years)
│      - FF5 adoption: 2015-present (crowding of profitability/investment effects)
│
├─ Multi-Factor Strategy Construction
│  ├─ Factor Exposure Matrix:
│  │   ├─ For each asset: Calculate exposures [βm, βv, βs, βp, βi]
│  │   ├─ Linear factor model: Expected return = Σ(β × Factor premium)
│  │   ├─ Example: Stock A: βm=1.1, βv=0.8, βs=0.3, βp=0.9, βi=-0.2
│  │   ├─ Factor premiums (annual): Rm=5%, HML=3%, SMB=2%, RMW=2.5%, CMA=3%
│  │   ├─ Expected return = 1.1×5% + 0.8×3% + 0.3×2% + 0.9×2.5% + (-0.2)×3%
│  │   └─ Expected return = 5.5% + 2.4% + 0.6% + 2.25% - 0.6% = 10.15%
│  │
│  ├─ Portfolio Construction Approaches:
│  │   ├─ Long-only (traditional):
│  │   │   - Weights optimized for maximum expected return per risk
│  │   │   - Factor exposure: βv ≈ market long-only benchmark (0.3-0.6)
│  │   │   - Returns: Benchmark + factor tilts
│  │   │   - Typical alpha: +0.5-1.5% annual
│  │   ├─ Long-short (factor arbitrage):
│  │   │   - Isolate specific factor: βm ≈ 0 (market-neutral), βv ≈ 1 (pure value)
│  │   │   - Long value factor, short growth factor
│  │   │   - Returns: Factor premium alone (3-5% for HML)
│  │   │   - Volatility: 8-12% (less correlated to market)
│  │   ├─ Multi-factor systematic:
│  │   │   - Combine 6+ factors (classic + momentum + quality + liquidity)
│  │   │   - Equal risk contribution across factors
│  │   │   - Diversification: Factors decorrelated (different return drivers)
│  │   │   - Returns: +4-8% annual, volatility 6-10%
│  │   └─ Dynamic factor weighting:
│  │       - Adjust factor exposures based on regime/valuation
│  │       - Value overweight in late cycle, growth in early cycle
│  │       - Tactical: +1-3% annual from timing (if correct)
│  │
│  ├─ Constraints & Practical Issues:
│  │   ├─ Short-selling: Regulatory, cost, borrow availability
│  │   ├─ Transaction costs: Rebalancing 4-12 times/year (1-3% costs)
│  │   ├─ Survivorship bias: Data includes dead companies (overstates returns)
│  │   ├─ Factor crowding: Popular factors (value, momentum) decline as more capital targets
│  │   ├─ Regime shifts: Factor premia can be negative (value underperforms 2010-2019)
│  │   ├─ Small-cap premium: Disappears post-discovery (cost exceeds premium)
│  │   └─ Implementation: Smart beta ETFs (easy access but higher fees)
│  │
│  └─ Risk Management:
│      ├─ Factor concentration risk: Portfolio 100% value exposure → crashes when value crashes
│      ├─ Crowding risk: All factor investors exit simultaneously (flash crash)
│      ├─ Correlation breakdown: Factors uncorrelated in normal times, correlate 1.0 in crisis
│      ├─ Leverage risk: Leveraged factor strategies amplify tail losses
│      ├─ Monitoring: Track factor exposures daily, rebalance when drift >10%
│      └─ Hedging: Can hedge against factor reversals (expensive)
│
├─ Factor Decay & Anomaly Life Cycle
│  ├─ Discovery Phase (Pre-publication):
│  │   - Researcher finds anomaly (value outperforms, size premium exists)
│  │   - Papers submitted, rejection, revision, acceptance (1-3 years)
│  │   - Factor trades profitably in this period (unknown to market)
│  │   - Excess return: +5-10% annual
│  │
│  ├─ Publication Phase (1-3 years post):
│  │   - Paper published in top journal (e.g., JF, JFE)
│  │   - Academics discuss; some practitioners aware
│  │   - Moderate capital flows to factor
│  │   - Excess return: +3-6% annual (still attractive)
│  │
│  ├─ Popularization Phase (3-5 years):
│  │   - Smart beta ETFs launch (easy access)
│  │   - Billions in AUM flow to factor
│  │   - Crowding: Factor becomes saturated
│  │   - Excess return: +1-3% annual (compressed, barely beats costs)
│  │
│  ├─ Saturation Phase (5+ years):
│  │   - Factor becomes mainstream (academic canon)
│  │   - Trillions potentially exposed (if FF factors)
│  │   - Excess return: 0-2% annual (may disappear)
│  │   - Reversals possible: Value underperforms growth (2010-2019)
│  │   - Anomaly life span: ~15-20 years (from discovery to saturation)
│  │
│  └─ Historical Examples:
│      - Value anomaly: Discovered 1980s, mainstream 2010s, underperforming 2010-2020
│      - Momentum anomaly: Discovered 1993, crowded 2010s, crashes 2008, 2020
│      - Low-volatility anomaly: Discovered 2000s, crowded 2010s, crashes 2020
│
├─ Alternative & Emerging Factors
│  ├─ Momentum (WML: Winner - Loser):
│  │   ├─ Strategy: Long recent winners, short recent losers
│  │   ├─ Excess return: +5-8% historical (high)
│  │   ├─ Drawback: Crashes 2-3 times per decade; -50% drawdowns
│  │   ├─ Crowding: Highly popular, return declining
│  │   └─ Application: Momentum crash indicator for risk management
│  │
│  ├─ Quality:
│  │   ├─ Definition: High profitability, low debt, stable earnings
│  │   ├─ Excess return: +2-4% annual
│  │   ├─ Advantage: Outperforms in downturns (defensive)
│  │   ├─ Disadvantage: Expensive (already priced in by smart money)
│  │   └─ Combination: Quality + value = sweet spot
│  │
│  ├─ Liquidity:
│  │   ├─ Strategy: Long liquid assets, short illiquid (liquidity premium)
│  │   ├─ Excess return: +2-5% (varies by market stress)
│  │   ├─ Highly regime-dependent: Disappears/reverses in crises
│  │   └─ Data issues: Hard to measure liquidity consistently
│  │
│  ├─ Minimum Volatility / Low Beta:
│  │   ├─ Strategy: Low historical volatility outperforms
│  │   ├─ Excess return: +1-3% annual (post-cost)
│  │   ├─ Drawback: Crashes when vol spikes (2020, 2022)
│  │   ├─ Crowding: Billions in low-vol ETFs (crash risk)
│  │   └─ Mechanism: Leverage-constrained investors use low-vol as equity substitute
│  │
│  └─ Newly Proposed:
│      - ESG (Environmental/Social/Governance): +1-2%, mostly sentiment-driven
│      - Earnings quality: +1-3% but hard to measure
│      - Dividend policy: +1-2%, mostly reflected in payout ratio
│      - Patent/innovation: +1-2%, data limited
│
└─ Practical Factor Investing Workflow
   ├─ Data Acquisition:
   │   ├─ Raw returns: Bloomberg, FactSet, Refinitiv
   │   ├─ Factor data: Kenneth French data library (free), AQR, Morningstar
   │   ├─ Fundamental data: S&P Capital IQ, Bloomberg, SEC Edgar
   │   └─ Frequency: Daily returns, monthly factor updates
   │
   ├─ Factor Construction:
   │   ├─ Time series factors: Portfolio of stocks sorted by characteristic
   │   ├─ Cross-sectional factors: Current holdings ranked
   │   ├─ Rolling window: Update monthly (trades incur costs)
   │   ├─ Weighting: Market-cap, equal-weight, or risk-parity
   │   └─ Validation: Ensure factors correlated with historical premia
   │
   ├─ Strategy Design:
   │   ├─ Factor selection: Choose 3-6 factors based on philosophy
   │   ├─ Risk budgeting: Allocate risk contribution equally across factors
   │   ├─ Leverage: Typically 1-2x (to stabilize volatility)
   │   ├─ Constraints: Position limits (max 5% in any stock), sector limits (max 30%)
   │   └─ Rebalancing: Monthly to quarterly (trade-off: costs vs factor drift)
   │
   ├─ Backtesting:
   │   ├─ Period: Minimum 10 years (preferably 30+ years)
   │   ├─ Out-of-sample: Walk-forward validation (not just historical fit)
   │   ├─ Include costs: Transaction costs (0.5-1%), slippage (0.1-0.5%), fees (0.5-1%)
   │   ├─ Stress test: Worst periods (2008, 2020, 1987, etc.)
   │   └─ Robustness: Parameter sensitivity, factor definition changes
   │
   ├─ Implementation:
   │   ├─ Index replication: Replicate factor using top holdings (~30-50 stocks)
   │   ├─ Smart beta ETF: 0.2-0.5% annual fee (easier)
   │   ├─ Separate account: Custom optimization (complex, higher minimum)
   │   └─ Hybrid: Core index + factor tilt overlay
   │
   ├─ Monitoring:
   │   ├─ Factor exposure: Track each factor beta monthly
   │   ├─ Performance attribution: How much return from each factor?
   │   ├─ Regime detection: When do factors underperform?
   │   ├─ Crowding metrics: Estimate how much capital in factor
   │   └─ Adjust: Reduce crowded factors, tilt to uncrowded
   │
   └─ Ongoing Research:
       ├─ Track new factor discoveries (academic journals, blogs)
       ├─ Test new factors on historical data
       ├─ Monitor factor crowding (sentiment, positioning)
       ├─ Identify regime changes (factors switchover)
       └─ Adapt strategy (dynamic factor weights)
```

**Interaction:** Factor identification → Data collection → Portfolio construction → Risk decomposition → Backtesting → Implementation → Monitoring → Rebalancing → Adaptation.

## 5. Mini-Project
Implement and backtest 5-factor model on equity portfolio:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Simulate factor returns (historical approximations)
np.random.seed(42)
n_months = 240  # 20 years

# Generate realistic factor returns
market_ret = np.random.normal(0.006, 0.04, n_months)  # 6% annual, 4% vol
hml_ret = np.random.normal(0.002, 0.08, n_months)    # 2% value premium, 8% vol
smb_ret = np.random.normal(0.001, 0.06, n_months)    # 1% size premium, 6% vol
rmw_ret = np.random.normal(0.002, 0.05, n_months)    # 2% profitability, 5% vol
cma_ret = np.random.normal(0.0025, 0.04, n_months)   # 2.5% investment, 4% vol

# Hypothetical stock returns (mixture of factors + noise)
stock_alpha = 0.0005  # 0.5% monthly alpha (~6% annual)
stock_ret = (stock_alpha + 
             1.0 * market_ret +    # β_m = 1.0
             0.5 * hml_ret +       # β_v = 0.5 (value tilt)
             -0.3 * smb_ret +      # β_s = -0.3 (large-cap tilt)
             0.4 * rmw_ret +       # β_p = 0.4 (quality tilt)
             -0.2 * cma_ret +      # β_i = -0.2 (growth-oriented capex)
             np.random.normal(0, 0.02, n_months))  # idiosyncratic noise

dates = pd.date_range('2004-01-31', periods=n_months, freq='M')

# Combine into DataFrame
ff5_data = pd.DataFrame({
    'market': market_ret,
    'hml': hml_ret,
    'smb': smb_ret,
    'rmw': rmw_ret,
    'cma': cma_ret,
    'stock': stock_ret
}, index=dates)

print("="*100)
print("FAMA-FRENCH 5-FACTOR MODEL ANALYSIS")
print("="*100)

# Step 1: Descriptive statistics
print(f"\nStep 1: Factor & Stock Return Statistics (20 years)")
print(f"-" * 50)

stats_df = pd.DataFrame({
    'Mean (%)': ff5_data.mean() * 100 * 12,  # Annualized
    'Std (%)': ff5_data.std() * 100 * np.sqrt(12),  # Annualized
    'Min (%)': ff5_data.min() * 100,
    'Max (%)': ff5_data.max() * 100,
})

print(stats_df.round(2))

# Step 2: Factor correlation
print(f"\nStep 2: Factor Correlation Matrix")
print(f"-" * 50)

corr_matrix = ff5_data.corr()
print(corr_matrix.round(3))

# Step 3: Regression analysis (FF5 model)
print(f"\nStep 3: Fama-French 5-Factor Regression")
print(f"-" * 50)

X = ff5_data[['market', 'hml', 'smb', 'rmw', 'cma']]
y = ff5_data['stock']

# Add constant
X_with_const = np.column_stack([np.ones(len(X)), X])

# OLS regression
betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
alpha_coef = betas[0]
beta_market, beta_hml, beta_smb, beta_rmw, beta_cma = betas[1:]

# Predictions and residuals
y_pred = X_with_const @ betas
residuals = y - y_pred

# R-squared
ss_total = np.sum((y - y.mean())**2)
ss_residual = np.sum(residuals**2)
r_squared = 1 - (ss_residual / ss_total)

# Adjusted R-squared
n = len(y)
k = 5  # number of factors
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

# Standard errors
residual_std_err = np.sqrt(ss_residual / (n - k - 1))
var_covar_matrix = residual_std_err**2 * np.linalg.inv(X_with_const.T @ X_with_const)
std_errors = np.sqrt(np.diag(var_covar_matrix))

# T-statistics and p-values
from scipy import stats as sp_stats
t_stats = betas / std_errors
p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), n - k - 1))

# Print regression results
results_df = pd.DataFrame({
    'Factor': ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA'],
    'Coefficient': betas,
    'Std Error': std_errors,
    'T-stat': t_stats,
    'P-value': p_values,
})

print(results_df.to_string(index=False))

print(f"\nModel Fit:")
print(f"  R-squared: {r_squared:.4f}")
print(f"  Adjusted R-squared: {adj_r_squared:.4f}")
print(f"  Residual Std Error: {residual_std_err * 100:.3f}%/month ({residual_std_err * np.sqrt(12) * 100:.2f}%/year)")

# Annualized alpha
alpha_annual = alpha_coef * 12 * 100
print(f"  Annualized Alpha: {alpha_annual:.2f}%")

# Step 4: Performance decomposition
print(f"\nStep 4: Return Decomposition (Annualized)")
print(f"-" * 50)

# Average factor returns and contributions
factor_means = X.mean()
factor_contributions = (betas[1:] * factor_means).values * 12 * 100

decomp_df = pd.DataFrame({
    'Factor': ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA'],
    'Beta': [alpha_coef, beta_market, beta_hml, beta_smb, beta_rmw, beta_cma],
    'Factor Return': [0, factor_means['market']*12*100, factor_means['hml']*12*100, 
                      factor_means['smb']*12*100, factor_means['rmw']*12*100, factor_means['cma']*12*100],
    'Contribution (%)': [alpha_annual] + list(factor_contributions),
})

print(decomp_df.to_string(index=False))

total_return = y.mean() * 12 * 100
decomp_total = decomp_df['Contribution (%)'].sum()
print(f"\nTotal Stock Return: {total_return:.2f}%")
print(f"Explained (decomposition): {decomp_total:.2f}%")

# Step 5: Rolling window analysis
print(f"\nStep 5: Rolling Factor Exposures (24-month rolling window)")
print(f"-" * 50)

window = 24
rolling_betas = {}

for factor in ['market', 'hml', 'smb', 'rmw', 'cma']:
    betas_rolling = []
    for i in range(len(ff5_data) - window):
        X_window = ff5_data[['market', 'hml', 'smb', 'rmw', 'cma']].iloc[i:i+window]
        y_window = ff5_data['stock'].iloc[i:i+window]
        X_window_const = np.column_stack([np.ones(window), X_window])
        betas_window = np.linalg.lstsq(X_window_const, y_window, rcond=None)[0]
        
        if factor == 'market':
            betas_rolling.append(betas_window[1])
        elif factor == 'hml':
            betas_rolling.append(betas_window[2])
        elif factor == 'smb':
            betas_rolling.append(betas_window[3])
        elif factor == 'rmw':
            betas_rolling.append(betas_window[4])
        elif factor == 'cma':
            betas_rolling.append(betas_window[5])
    
    rolling_betas[factor] = betas_rolling

rolling_dates = dates[window:]

print(f"Average β (across rolling windows):")
for factor, betas_r in rolling_betas.items():
    print(f"  {factor.upper()}: {np.mean(betas_r):.3f} (±{np.std(betas_r):.3f})")

# VISUALIZATION
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Cumulative returns
ax = axes[0, 0]
ax.plot(dates, np.exp(np.log(1 + ff5_data['stock']).cumsum()), label='Stock', linewidth=2)
ax.plot(dates, np.exp(np.log(1 + ff5_data['market']).cumsum()), label='Market Factor', alpha=0.7)
ax.set_title('Cumulative Returns: Stock vs Market')
ax.set_ylabel('Cumulative Return (log scale)')
ax.legend()
ax.set_yscale('log')
ax.grid(alpha=0.3)

# Plot 2: Factor correlation heatmap
ax = axes[0, 1]
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(['Market', 'HML', 'SMB', 'RMW', 'CMA', 'Stock'], rotation=45)
ax.set_yticklabels(['Market', 'HML', 'SMB', 'RMW', 'CMA', 'Stock'])
ax.set_title('Factor Correlation Matrix')
plt.colorbar(im, ax=ax)

# Plot 3: Regression fit (actual vs predicted)
ax = axes[1, 0]
ax.scatter(y_pred * 100, y * 100, alpha=0.5, s=20)
ax.plot([-5, 15], [-5, 15], 'r--', linewidth=2)
ax.set_xlabel('Predicted Return (%)')
ax.set_ylabel('Actual Return (%)')
ax.set_title(f'FF5 Fit (R² = {r_squared:.4f})')
ax.grid(alpha=0.3)

# Plot 4: Residuals
ax = axes[1, 1]
ax.hist(residuals * 100, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Residual Distribution (σ = {residual_std_err*100:.2f}%/month)')
ax.grid(alpha=0.3, axis='y')

# Plot 5: Rolling betas
ax = axes[2, 0]
ax.plot(rolling_dates, rolling_betas['market'], label='Market β', linewidth=1.5)
ax.plot(rolling_dates, rolling_betas['hml'], label='HML β', linewidth=1.5)
ax.plot(rolling_dates, rolling_betas['smb'], label='SMB β', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_title('Rolling Factor Exposures (24-month window)')
ax.set_ylabel('Beta')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Return contribution stacked
ax = axes[2, 1]
contributions = [alpha_annual] + list(factor_contributions)
labels = ['Alpha', 'Market', 'HML', 'SMB', 'RMW', 'CMA']
colors = ['gray', 'blue', 'green', 'red', 'orange', 'purple']
bars = ax.bar(labels, contributions, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Contribution to Return (%/year)')
ax.set_title('Return Decomposition by Factor')
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, contributions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top')

plt.tight_layout()
plt.show()

print(f"\n" + "="*100)
print("INSIGHTS")
print(f"="*100)
print(f"- FF5 explains {r_squared*100:.1f}% of stock return variance")
print(f"- Key exposures: Market β={beta_market:.2f}, Value β={beta_hml:.2f}, Quality β={beta_rmw:.2f}")
print(f"- Annualized alpha: {alpha_annual:.2f}% (may be exploitable if >2%)")
print(f"- Factor betas vary over time (rolling window shows instability)")
print(f"- Residuals approximately normal (model reasonably specified)")
```

## 6. Challenge Round
- Run FF5 regression on 50-stock portfolio; identify dominant factors
- Backtest value factor strategy (HML long-short) vs market; measure Sharpe ratio
- Test mean-reversion of factors (do factor returns revert?); correlation stability
- Design 3-factor portfolio: Market-neutral long HML, short SMB, long RMW
- Monitor factor crowding: Estimate total AUM in each Fama-French factor globally

## 7. Key References
- [Fama & French, "The Three-Factor Model" (1993), JFE](https://www.sciencedirect.com/science/article/pii/0304405X93900235) — Foundational factor framework
- [Fama & French, "A Five-Factor Asset Pricing Model" (2015), JFE](https://www.sciencedirect.com/science/article/pii/S0304405X15000033) — Extension to profitability/investment
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) — Benchmark factor returns (free, public)
- [AQR, "Factor Investing" (2020)](https://www.aqr.com/Insights/Research/Paper/The-Case-for-Factor-Investing) — Comprehensive framework and empirics

---
**Status:** Established factor framework (1993-present, increasingly crowded) | **Complements:** Mean Reversion, Momentum, Risk Decomposition, Portfolio Construction
