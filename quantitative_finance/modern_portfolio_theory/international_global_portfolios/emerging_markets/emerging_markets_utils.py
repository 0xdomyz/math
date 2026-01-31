import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_performance_by_regime(returns, crisis_scores):
    """
    Analyze returns during different market regimes (normal vs crisis).
    """
    # Define crisis regime: Crisis score > 1 std above mean
    crisis_threshold = crisis_scores.mean() + crisis_scores.std()
    
    regimes = {}
    for col in crisis_scores.columns:
        col_base = col.replace('_CrisisScore', '')
        if col_base in returns.columns:
            
            # Mark crisis periods
            is_crisis = crisis_scores[col] > crisis_threshold[col]
            
            # Calculate returns in each regime
            normal_return = returns.loc[~is_crisis, col_base].mean() * 252
            crisis_return = returns.loc[is_crisis, col_base].mean() * 252
            normal_vol = returns.loc[~is_crisis, col_base].std() * np.sqrt(252)
            crisis_vol = returns.loc[is_crisis, col_base].std() * np.sqrt(252)
            
            regimes[col_base] = {
                'Normal Return': normal_return,
                'Normal Vol': normal_vol,
                'Crisis Return': crisis_return,
                'Crisis Vol': crisis_vol,
                'Crisis Freq': is_crisis.sum() / len(is_crisis)
            }
    
    return pd.DataFrame(regimes).T


# Main Analysis
print("=" * 100)
print("EMERGING MARKETS VALUATION & CRISIS DETECTION")
print("=" * 100)

# Get 10 years of data
prices, indices = get_em_data('2014-01-01', '2024-01-01')

# 1. Volatility analysis
print("\n1. VOLATILITY PATTERNS: EM vs US")
print("-" * 100)

rolling_vol, rolling_corr_dict = calculate_valuation_metrics(prices, '2014-01-01', '2024-01-01')

latest_vol = rolling_vol.iloc[-1]
print(f"Current Volatility (Latest):")
for idx, vol in latest_vol.items():
    print(f"  {idx}: {vol:.2%}")

print(f"\nHistorical Volatility (Mean/Min/Max):")
for col in rolling_vol.columns:
    mean_v = rolling_vol[col].mean()
    min_v = rolling_vol[col].min()
    max_v = rolling_vol[col].max()
    print(f"  {col}: {mean_v:.2%} (range {min_v:.2%} - {max_v:.2%})")

# 2. Correlation analysis
print("\n2. CORRELATION WITH US STOCKS: Contagion Indicator")
print("-" * 100)

rolling_corr_df = pd.DataFrame(rolling_corr_dict)
latest_corr = rolling_corr_df.iloc[-1]
print(f"Current Correlations (Latest):")
for idx, corr in latest_corr.items():
    print(f"  {idx}: {corr:.3f}")

print(f"\nHistorical Correlations (Mean/Min/Max):")
for col in rolling_corr_df.columns:
    mean_c = rolling_corr_df[col].mean()
    min_c = rolling_corr_df[col].min()
    max_c = rolling_corr_df[col].max()
    status = "HIGH" if mean_c > 0.80 else "MODERATE" if mean_c > 0.65 else "LOW"
    print(f"  {col}: {mean_c:.3f} (range {min_c:.3f} - {max_c:.3f}) [{status}]")

# 3. Performance by regime
print("\n3. PERFORMANCE IN NORMAL vs CRISIS REGIMES")
print("-" * 100)

crisis_scores = EM_crisis_score(rolling_vol, rolling_corr_df)
returns = prices.pct_change().dropna()

perf_by_regime = calculate_performance_by_regime(returns, crisis_scores)
print(perf_by_regime.round(4))

# 4. Buy/sell signals
print("\n4. VALUATION SIGNALS (Based on Price vs 200-day MA)")
print("-" * 100)

signals_df = valuation_buy_sell_signals(prices)
latest_signals = signals_df.iloc[-1]

print(f"Current Signals:")
for idx, sig in latest_signals.items():
    idx_base = idx.replace('_Signal', '')
    if sig > 0.5:
        action = "STRONG BUY"
    elif sig > 0:
        action = "BUY"
    elif sig < -0.5:
        action = "STRONG SELL"
    elif sig < 0:
        action = "SELL"
    else:
        action = "NEUTRAL"
    
    print(f"  {idx_base}: {action} (score: {sig:.2f})")

# 5. Crisis periods historical
print("\n5. HISTORICAL CRISIS PERIODS (EEM Index)")
print("-" * 100)

crisis_threshold_eem = crisis_scores['EEM_CrisisScore'].mean() + crisis_scores['EEM_CrisisScore'].std()
crisis_periods = crisis_scores[crisis_scores['EEM_CrisisScore'] > crisis_threshold_eem]

print(f"Crisis Score Threshold: {crisis_threshold_eem:.2f}")
print(f"Days in Crisis (>threshold): {len(crisis_periods)} days ({len(crisis_periods)/len(crisis_scores)*100:.1f}%)")
print(f"\nMajor Crisis Dates:")

# Find crisis clusters
if len(crisis_periods) > 0:
    crisis_dates = crisis_periods.index
    for crisis_date in crisis_dates[-20:]:  # Last 20 crisis signals
        return_day = returns.loc[crisis_date, 'EEM'] if crisis_date in returns.index else np.nan
        print(f"  {crisis_date.strftime('%Y-%m-%d')}: Score={crisis_scores.loc[crisis_date, 'EEM_CrisisScore']:.2f}, Return={return_day:.2%}")

# 6. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Rolling volatility
ax = axes[0, 0]
for col in ['EEM', 'ASHR', 'INDY']:
    if col in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=2, alpha=0.8)
ax.axhline(0.25, color='red', linestyle='--', label='High Vol (25%)')
ax.axhline(0.15, color='green', linestyle='--', label='Low Vol (15%)')
ax.set_title('Rolling 252-Day Volatility', fontweight='bold')
ax.set_ylabel('Volatility')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Rolling correlations
ax = axes[0, 1]
for col in rolling_corr_df.columns:
    ax.plot(rolling_corr_df.index, rolling_corr_df[col], label=col.replace('_Corr_SPY', ''), linewidth=2, alpha=0.8)
ax.axhline(0.85, color='red', linestyle='--', label='Crisis (0.85+)')
ax.axhline(0.70, color='orange', linestyle='--', label='Normal (0.70)')
ax.set_title('Correlation with US Stocks (Crisis Indicator)', fontweight='bold')
ax.set_ylabel('Correlation')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# Plot 3: Crisis score
ax = axes[0, 2]
crisis_score_eem = crisis_scores['EEM_CrisisScore']
ax.plot(crisis_score_eem.index, crisis_score_eem, linewidth=2, color='steelblue')
ax.fill_between(crisis_score_eem.index, 0, crisis_score_eem, 
                 where=(crisis_score_eem > crisis_threshold_eem), 
                 alpha=0.3, color='red', label='Crisis Zone')
ax.axhline(crisis_threshold_eem, color='red', linestyle='--', label='Threshold')
ax.set_title('EM Crisis Score (EEM)', fontweight='bold')
ax.set_ylabel('Crisis Score (z-score)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Price vs 200-day MA
ax = axes[1, 0]
ax.plot(prices.index, prices['EEM'], label='EEM Price', linewidth=2)
ma_200 = prices['EEM'].rolling(200).mean()
ax.plot(ma_200.index, ma_200, label='200-day MA', linewidth=2, linestyle='--')
ax.fill_between(prices.index, 0.9*ma_200, 1.1*ma_200, alpha=0.2, color='gray', label='Buy/Sell Zone')
ax.set_title('EEM Price vs 200-Day Moving Average', fontweight='bold')
ax.set_ylabel('Price')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 5: Performance by regime
ax = axes[1, 1]
regimes_to_plot = perf_by_regime[['Normal Return', 'Crisis Return']].sort_values('Normal Return')
x = np.arange(len(regimes_to_plot))
width = 0.35
ax.bar(x - width/2, regimes_to_plot['Normal Return'] * 100, width, label='Normal Regime', alpha=0.8)
ax.bar(x + width/2, regimes_to_plot['Crisis Return'] * 100, width, label='Crisis Regime', alpha=0.8)
ax.set_ylabel('Annual Return (%)')
ax.set_title('Returns by Market Regime', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regimes_to_plot.index, fontsize=8)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Volatility comparison
ax = axes[1, 2]
volatility_comparison = perf_by_regime[['Normal Vol', 'Crisis Vol']]
x = np.arange(len(volatility_comparison))
width = 0.35
ax.bar(x - width/2, volatility_comparison['Normal Vol'] * 100, width, label='Normal Vol', alpha=0.8)
ax.bar(x + width/2, volatility_comparison['Crisis Vol'] * 100, width, label='Crisis Vol', alpha=0.8)
ax.set_ylabel('Volatility (%)')
ax.set_title('Volatility in Different Regimes', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(volatility_comparison.index, fontsize=8)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('em_valuation_crisis_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: em_valuation_crisis_model.png")
plt.show()

# 7. EM allocation recommendation
print("\n6. EM ALLOCATION FRAMEWORK")
print("-" * 100)

current_vol_eem = rolling_vol['EEM'].iloc[-1]
current_corr_eem = rolling_corr_df['EEM_Corr_SPY'].iloc[-1]
current_price = prices['EEM'].iloc[-1]
ma_200_current = prices['EEM'].rolling(200).mean().iloc[-1]

print(f"""
CURRENT CONDITIONS (as of {prices.index[-1].strftime('%Y-%m-%d')}):
├─ EEM Volatility: {current_vol_eem:.2%}
├─ EEM-US Correlation: {current_corr_eem:.3f}
├─ Price vs 200-MA: {current_price / ma_200_current:.2%}
└─ Crisis Score: {crisis_scores['EEM_CrisisScore'].iloc[-1]:.2f}

ALLOCATION RECOMMENDATION:
├─ Volatility-adjusted:
│   ├─ If vol < 15%: Increase EM to 25-30% (low risk environment)
│   ├─ If vol 15-25%: Target 20% (normal)
│   ├─ If vol > 25%: Reduce to 10-15% (crisis risk)
│
├─ Correlation-adjusted:
│   ├─ If corr < 0.70: Increase EM (great diversification benefit)
│   ├─ If corr 0.70-0.80: Normal 20% allocation
│   ├─ If corr > 0.85: Reduce EM (diversification lost; contagion risk)
│
├─ Valuation-adjusted:
│   ├─ If price < 0.90 × MA200: Increase EM (oversold opportunity)
│   ├─ If price 0.90-1.10 × MA200: Maintain allocation
│   ├─ If price > 1.10 × MA200: Reduce EM (overextended)
│
└─ COMBINED RECOMMENDATION:
    ├─ Bullish case (low vol, low corr, oversold): 30% EM
    ├─ Normal case (moderate metrics): 20% EM
    └─ Bearish case (high vol, high corr, extended): 10% EM
""")

print("=" * 100)