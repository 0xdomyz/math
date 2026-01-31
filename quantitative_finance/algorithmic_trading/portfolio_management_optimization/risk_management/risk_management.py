import numpy as np
import pandas as pd
from scipy.stats import norm

# Portfolio setup: 3 assets
assets = ['Tech Stock', 'Bond', 'Commodity']
positions = np.array([1_000_000, 500_000, 500_000])  # Dollar holdings
volatilities = np.array([0.25, 0.08, 0.30])  # Annual volatilities
correlation = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 1.0, 0.1],
    [0.2, 0.1, 1.0]
])

# Covariance matrix (annualized)
cov_matrix = np.outer(volatilities, volatilities) * correlation

# Portfolio volatility (annualized)
weights = positions / positions.sum()
portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
portfolio_vol = np.sqrt(portfolio_variance)
portfolio_value = positions.sum()

# 1-day 95% VaR (parametric)
confidence_level = 0.95
z_score = norm.ppf(confidence_level)
time_horizon_days = 1
daily_vol = portfolio_vol / np.sqrt(252)
var_95 = portfolio_value * daily_vol * z_score

# 1-day 99% VaR
z_score_99 = norm.ppf(0.99)
var_99 = portfolio_value * daily_vol * z_score_99

print("=" * 60)
print("PORTFOLIO RISK METRICS")
print("=" * 60)
print(f"Portfolio Value:              ${portfolio_value:>12,.0f}")
print(f"Portfolio Volatility (Annual): {portfolio_vol:>11.2%}")
print(f"Portfolio Volatility (Daily):  {daily_vol:>11.2%}")
print()
print(f"1-Day 95% Value-at-Risk:      ${var_95:>12,.0f}")
print(f"1-Day 99% Value-at-Risk:      ${var_99:>12,.0f}")
print()
print("Interpretation: 95% confident daily loss will not exceed VaR_95.")
print("=" * 60)

# Position sizing example: Kelly criterion
# Assume strategy with 60% win rate, avg win $1000, avg loss $600
win_prob = 0.60
avg_win = 1000
avg_loss = 600

kelly_fraction = (win_prob / avg_loss) - ((1 - win_prob) / avg_win)
kelly_percent = kelly_fraction * 100

# Conservative: Use half-Kelly
half_kelly_percent = kelly_percent * 0.5

print("\nPOSITION SIZING (Kelly Criterion)")
print("=" * 60)
print(f"Win Probability:              {win_prob:>12.0%}")
print(f"Average Win:                  ${avg_win:>12,.0f}")
print(f"Average Loss:                 ${avg_loss:>12,.0f}")
print()
print(f"Full Kelly Fraction:          {kelly_percent:>11.2f}%")
print(f"Half-Kelly (Conservative):    {half_kelly_percent:>11.2f}%")
print()
print(f"Recommended Position Size:    ${portfolio_value * half_kelly_percent/100:>12,.0f}")
print("=" * 60)

# Stress test: Simulate 2008-style crisis
print("\nSTRESS TEST: Financial Crisis Scenario")
print("=" * 60)
stress_return = -0.40  # 40% market drop
stress_vol_multiplier = 3  # Volatility triples
stress_correlation = 0.95  # Correlations converge to 1

stress_positions = positions * (1 + stress_return)
stress_loss = positions.sum() - stress_positions.sum()
stress_loss_pct = stress_loss / portfolio_value

print(f"Scenario: Market drops 40%, vol triples, correlation → 0.95")
print(f"Portfolio Loss:               ${stress_loss:>12,.0f}")
print(f"Portfolio Loss %:             {stress_loss_pct:>11.2%}")
print()

if stress_loss_pct > 0.50:
    print("⚠️  WARNING: Portfolio at risk of >50% drawdown in crisis!")
    print("   Recommendation: Reduce leverage or add tail hedges.")
else:
    print("✓ Portfolio survives stress test with manageable drawdown.")
print("=" * 60)