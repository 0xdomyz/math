import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json

# Download and analyze risk-free rates; compare proxies

def get_treasury_yields():
    """
    Fetch current U.S. Treasury yields from FRED (Federal Reserve Economic Data).
    Using yfinance as proxy; actual FRED API would be better but requires key.
    """
    # Approximate yields based on Treasury price data
    treasury_maturities = {
        '3M': '^IRX',      # 3-month T-bill
        '6M': '^IRX',      # 6-month T-bill (approximated)
        '1Y': 'IEF',       # 7-10 year Treasury ETF (proxy)
        '2Y': '^IEY',      # 2-year yield (estimated)
        '5Y': '^FVX',      # 5-year yield
        '10Y': '^TNX',     # 10-year yield
        '30Y': '^TYX',     # 30-year yield
    }
    
    yields = {}
    for name, ticker in treasury_maturities.items():
        try:
            data = yf.download(ticker, start='2020-01-01', end=datetime.now(), 
                             progress=False, interval='1d')
            if 'Adj Close' in data.columns:
                current = data['Adj Close'].iloc[-1]
                yields[name] = current / 100  # Convert to decimal
            else:
                yields[name] = np.nan
        except:
            yields[name] = np.nan
    
    return yields


def simulate_treasuries_pf(years_data=5):
    """
    Create historical Treasury yield data; simulate via synthetic returns.
    """
    dates = pd.date_range(end=datetime.now(), periods=252*years_data, freq='D')
    
    # Synthetic Treasury yields over time (realistic patterns)
    t_bill_yields = np.linspace(0.02, 0.045, len(dates)) + np.random.normal(0, 0.002, len(dates))
    t_2y_yields = np.linspace(0.025, 0.048, len(dates)) + np.random.normal(0, 0.003, len(dates))
    t_5y_yields = np.linspace(0.03, 0.050, len(dates)) + np.random.normal(0, 0.004, len(dates))
    t_10y_yields = np.linspace(0.035, 0.049, len(dates)) + np.random.normal(0, 0.005, len(dates))
    t_30y_yields = np.linspace(0.04, 0.048, len(dates)) + np.random.normal(0, 0.005, len(dates))
    
    yields_df = pd.DataFrame({
        'Date': dates,
        '3M': np.maximum(t_bill_yields, 0.001),
        '2Y': np.maximum(t_2y_yields, 0.001),
        '5Y': np.maximum(t_5y_yields, 0.001),
        '10Y': np.maximum(t_10y_yields, 0.001),
        '30Y': np.maximum(t_30y_yields, 0.001),
    })
    yields_df.set_index('Date', inplace=True)
    
    return yields_df


def compute_duration_price_sensitivity(yield_change, duration, current_price=100):
    """
    Compute Treasury price change from yield move using duration formula.
    % price change ≈ -duration × Δyield
    """
    price_change_pct = -duration * yield_change
    new_price = current_price * (1 + price_change_pct)
    return price_change_pct, new_price


def extract_real_rate_from_tips(nominal_yield, tips_yield):
    """
    Extract real rate from nominal vs TIPS yields.
    Real rate ≈ nominal yield - TIPS yield (simplified)
    More precisely: (1 + nominal) = (1 + real) × (1 + inflation)
    """
    real_rate = nominal_yield - tips_yield
    return real_rate


def term_premium_decomposition(yields_dict):
    """
    Estimate term premium from yield curve shape.
    Term premium ≈ Longer yield - Shorter yield
    """
    term_premium_2y = yields_dict.get('2Y', 0) - yields_dict.get('3M', 0)
    term_premium_5y = yields_dict.get('5Y', 0) - yields_dict.get('2Y', 0)
    term_premium_10y = yields_dict.get('10Y', 0) - yields_dict.get('5Y', 0)
    term_premium_total = yields_dict.get('30Y', 0) - yields_dict.get('3M', 0)
    
    return {
        '2Y TP': term_premium_2y,
        '5Y TP': term_premium_5y,
        '10Y TP': term_premium_10y,
        'Total TP (30Y-3M)': term_premium_total
    }


# Main Analysis
print("=" * 100)
print("RISK-FREE RATE: NATURE, MEASUREMENT & APPLICATION")
print("=" * 100)

# 1. Historical Treasury yields
print("\n1. HISTORICAL TREASURY YIELD CURVE (Synthetic Data)")
print("-" * 100)

yields_hist = simulate_treasuries_pf(years_data=5)

print("\nCurrent Yields (end of period):")
print(yields_hist.iloc[-1] * 100)

print("\nHistorical Statistics (%):")
print((yields_hist * 100).describe().T)

# 2. Yield curve shape
print("\n2. YIELD CURVE ANALYSIS")
print("-" * 100)

current_yields = yields_hist.iloc[-1]
term_premiums = term_premium_decomposition(current_yields)

print("\nCurrent Yield Curve (%):")
for maturity, yield_val in current_yields.items():
    print(f"  {maturity:5s}: {yield_val*100:5.2f}%")

print("\nTerm Premiums (Longer - Shorter Maturity):")
for tp_name, tp_val in term_premiums.items():
    print(f"  {tp_name:20s}: {tp_val*100:5.2f}%")

# 3. Duration & Interest Rate Risk
print("\n3. TREASURY DURATION & PRICE SENSITIVITY")
print("-" * 100)

durations = {'3M': 0.25, '2Y': 1.9, '5Y': 4.5, '10Y': 8.2, '30Y': 20.0}
rate_move = 0.01  # 1% rate increase

print(f"\nIf yields rise by {rate_move*100:.1f}% (100 basis points):\n")
print(f"{'Maturity':<10} {'Duration':<12} {'Price Change %':<18} {'New Price':<15}")
print("-" * 55)

for maturity in ['3M', '2Y', '5Y', '10Y', '30Y']:
    duration = durations[maturity]
    price_change_pct, new_price = compute_duration_price_sensitivity(rate_move, duration)
    print(f"{maturity:<10} {duration:<12.2f} {price_change_pct*100:<18.2f} {new_price:<15.2f}")

print("\nKey insight: Longer-duration bonds suffer larger price declines (interest rate risk)")

# 4. Real vs Nominal RF rates
print("\n4. REAL VS NOMINAL RISK-FREE RATE")
print("-" * 100)

nominal_10y = yields_hist['10Y'].iloc[-1]
expected_inflation = 0.025  # 2.5% assumption

real_rf = nominal_10y - expected_inflation

print(f"\nNominal 10Y Treasury Yield: {nominal_10y*100:.2f}%")
print(f"Expected Inflation (assumption): {expected_inflation*100:.2f}%")
print(f"Implied Real RF: {real_rf*100:.2f}%")

print(f"\nFisher Equation: r_nominal = r_real + π^e")
print(f"Verification: {real_rf*100:.2f}% + {expected_inflation*100:.2f}% = {nominal_10y*100:.2f}%")

# 5. CAPM sensitivity to rf choice
print("\n5. CAPM EXPECTED RETURN SENSITIVITY TO RF CHOICE")
print("-" * 100)

market_premium = 0.06  # 6% equity risk premium assumption
betas = {'Low-beta defensive': 0.6, 'Market (SPY)': 1.0, 'High-beta cyclical': 1.4}

print(f"\nAssuming Market Risk Premium = {market_premium*100:.1f}%\n")
print(f"Expected Returns with Different RF Choices:\n")
print(f"{'Asset':<25} {'Low RF (2%)':<18} {'Mid RF (4%)':<18} {'High RF (5%)':<15}")
print("-" * 75)

for asset, beta in betas.items():
    rf_low = 0.02
    rf_mid = 0.04
    rf_high = 0.05
    
    e_r_low = rf_low + beta * market_premium
    e_r_mid = rf_mid + beta * market_premium
    e_r_high = rf_high + beta * market_premium
    
    print(f"{asset:<25} {e_r_low*100:<18.2f} {e_r_mid*100:<18.2f} {e_r_high*100:<15.2f}")

print("\nKey insight: Higher RF → Higher expected returns across all assets (proportional to beta)")

# 6. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Yield curve evolution
ax = axes[0, 0]
for col in ['3M', '2Y', '5Y', '10Y', '30Y']:
    ax.plot(yields_hist.index, yields_hist[col] * 100, label=col, linewidth=2)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Yield (%)', fontsize=11)
ax.set_title('U.S. Treasury Yield Curve Evolution', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Current yield curve shape
ax = axes[0, 1]
maturities = [0.25, 2, 5, 10, 30]  # Years
current_yields_list = [yields_hist[m].iloc[-1] * 100 for m in ['3M', '2Y', '5Y', '10Y', '30Y']]

ax.plot(maturities, current_yields_list, 'o-', linewidth=2.5, markersize=8, color='#3498db')
ax.fill_between(maturities, current_yields_list, alpha=0.3, color='#3498db')

ax.set_xlabel('Maturity (Years)', fontsize=11)
ax.set_ylabel('Yield (%)', fontsize=11)
ax.set_title('Current Yield Curve Shape', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)

# Plot 3: Price sensitivity to yield moves
ax = axes[1, 0]
yield_moves = np.linspace(-0.02, 0.02, 50)
for maturity in ['3M', '5Y', '10Y', '30Y']:
    duration = durations[maturity]
    price_changes = [-duration * move * 100 for move in yield_moves]
    ax.plot(yield_moves * 100, price_changes, label=f'{maturity} (D≈{duration:.1f})', linewidth=2)

ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Yield Change (Basis Points)', fontsize=11)
ax.set_ylabel('Price Change (%)', fontsize=11)
ax.set_title('Treasury Price Sensitivity (Duration Risk)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: CAPM return expectations by rf
ax = axes[1, 1]
rf_range = np.linspace(0.01, 0.06, 20)
market_premium = 0.06

for asset, beta in betas.items():
    expected_returns = [rf + beta * market_premium for rf in rf_range]
    ax.plot(rf_range * 100, np.array(expected_returns) * 100, 'o-', label=asset, linewidth=2, markersize=5)

ax.set_xlabel('Risk-Free Rate (%)', fontsize=11)
ax.set_ylabel('Expected Return (%)', fontsize=11)
ax.set_title('CAPM Expected Returns vs RF Choice', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('risk_free_rate_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: risk_free_rate_analysis.png")
plt.show()

# 7. Key insights
print("\n6. KEY INSIGHTS & PRACTICAL RECOMMENDATIONS")
print("-" * 100)
print(f"""
RISK-FREE RATE FUNDAMENTALS:
├─ Definition: Theoretical asset with zero default/reinvestment risk
├─ Proxy: U.S. Treasury securities (best approximation)
├─ Horizon-matched selection:
│   ├─ Short-term (< 5 years): Use 2Y or 5Y Treasury
│   ├─ Long-term (> 10 years): Use 10Y Treasury
│   └─ Mismatch hurts portfolio (reinvestment or duration risk)
│
├─ Components of Treasury yield:
│   ├─ Real interest rate (1-3%): Time preference, productivity
│   ├─ Expected inflation (1-3%): Purchasing power
│   ├─ Term premium (0-1.5%): Duration risk compensation
│   └─ Total: 2-7% depending on economic regime
│
├─ Key risks NOT eliminated (despite "risk-free" name):
│   ├─ Inflation risk: If realized inflation > expected, real return falls
│   ├─ Interest rate risk: If rates rise, bond prices fall
│   ├─ Reinvestment risk: Coupons may reinvest at lower rates
│   └─ Opportunity cost: Better returns available elsewhere
│
├─ CAPM sensitivity:
│   ├─ 1% increase in RF → Expected returns increase ~0.5-1% (beta-weighted)
│   ├─ High-beta stocks more leveraged to RF changes
│   ├─ Update RF quarterly as rate environment changes
│   └─ Small RF errors → large valuation errors (sensitivity high)
│
├─ Historical context:
│   ├─ Pre-2008: RF ≈ 4-5% (normal)
│   ├─ 2008-2022: RF ≈ 0-2% (emergency low)
│   ├─ 2022-present: RF ≈ 4-5% (normalized)
│   └─ Implication: Static RF assumptions become stale; adapt to rate regime
│
├─ Recommended best practices:
│   ├─ Use current (spot) Treasury yield, not historical average
│   ├─ Match Treasury maturity to investment horizon
│   ├─ Update RF monthly or after Fed announcements
│   ├─ Sensitivity analysis: Model range (RF ± 1%) not point estimate
│   ├─ Real RF analysis: Compare to TIPS yield for inflation-adjusted returns
│   └─ Consider term premium: 10Y yield includes ~0.5-1% for duration risk
│
├─ Portfolio implications:
│   ├─ Rising RF → Rebalance away from equities (better yield on bonds)
│   ├─ Falling RF → Rebalance into equities (bonds less attractive)
│   ├─ Valuation inversely sensitive: 1% RF increase → 10-20% equity valuation decline
│   └─ Horizon mismatch: Conservative investors may overpay for long-term certainty
│
└─ Calculation example (for CAPM):
    ├─ Current 10Y Treasury: {nominal_10y*100:.2f}%
    ├─ Market Risk Premium: 6%
    ├─ Low-beta stock (β=0.6): E[R] = {nominal_10y*100:.2f}% + 0.6 × 6% = {(nominal_10y + 0.6*0.06)*100:.2f}%
    ├─ High-beta stock (β=1.4): E[R] = {nominal_10y*100:.2f}% + 1.4 × 6% = {(nominal_10y + 1.4*0.06)*100:.2f}%
    └─ Implication: {(nominal_10y + 1.4*0.06)*100:.2f}% - {(nominal_10y + 0.6*0.06)*100:.2f}% = {0.8*0.06*100:.2f}% expected return difference from beta alone
""")

print("=" * 100)