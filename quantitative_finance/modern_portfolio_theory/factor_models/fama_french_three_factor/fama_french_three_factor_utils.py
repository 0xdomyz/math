import numpy as np
import matplotlib.pyplot as plt

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