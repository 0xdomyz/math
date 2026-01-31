"""
Structural Models (Merton Model)
Extracted from structural_models.md

Links firm value to default risk using option pricing theory.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(42)

print("=== Merton Structural Model ===")

# Firm data
equity_price = 50
shares_outstanding = 10
equity_market_value = equity_price * shares_outstanding
book_debt = 150
market_debt = 140
equity_volatility = 0.35
risk_free_rate = 0.03
time_horizon = 1.0

print(f"Firm Equity Market Value: ${equity_market_value}M")
print(f"Firm Debt (Book): ${book_debt}M")
print(f"Equity Volatility: {equity_volatility:.1%}")


def merton_equations(params, E, D, sigma_E, r, T):
    """
    Solve for asset value A and volatility sigma_A.
    Equations:
    1. E = A*N(d1) - D*exp(-rT)*N(d2)
    2. sigma_E * E = sigma_A * A * N(d1)
    """
    A, sigma_A = params

    if A <= D or sigma_A <= 0:
        return [1e10, 1e10]

    d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)

    call_value = A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    nd1_value = norm.cdf(d1) * A

    eq1 = call_value - E
    eq2 = sigma_E * E - sigma_A * nd1_value

    return eq1**2 + eq2**2


# Initial guess
initial_guess = [equity_market_value + book_debt * 0.5, equity_volatility / 0.8]

# Solve for asset value and volatility
result = minimize(
    lambda params: merton_equations(
        params,
        equity_market_value,
        book_debt,
        equity_volatility,
        risk_free_rate,
        time_horizon,
    ),
    initial_guess,
    method="Nelder-Mead",
)

A_est, sigma_A_est = result.x

print(f"\n=== Calibration Results ===")
print(f"Implied Asset Value: ${A_est:.1f}M")
print(f"Implied Asset Volatility: {sigma_A_est:.1%}")
print(f"Assets / Debt ratio: {A_est / book_debt:.2f}x")

# Calculate Merton distance to default
dd_merton = (
    np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * time_horizon
) / (sigma_A_est * np.sqrt(time_horizon))
pd_merton = norm.cdf(-dd_merton)

print(f"\nDistance to Default (DD): {dd_merton:.2f}")
print(f"1-Year PD (Merton): {pd_merton:.2%}")

# Risk-neutral adjustment
market_risk_premium = 0.06
mu_risk_neutral = risk_free_rate

dd_rn = (
    np.log(A_est / book_debt) + (mu_risk_neutral - 0.5 * sigma_A_est**2) * time_horizon
) / (sigma_A_est * np.sqrt(time_horizon))
pd_rn = norm.cdf(-dd_rn)

print(f"PD (Risk-Neutral): {pd_rn:.2%}")

lgd_assumption = 0.40
implied_cds_spread = (pd_rn * lgd_assumption / time_horizon) * 10000

print(f"Implied CDS Spread: {implied_cds_spread:.0f} bps")

# Scenario analysis
print(f"\n=== Sensitivity Analysis ===")
print("Asset Value (% of current) | Distance to Default | PD")
print("-" * 55)

asset_scenarios = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
for scenario in asset_scenarios:
    A_scenario = A_est * scenario
    dd_scenario = (
        np.log(A_scenario / book_debt)
        + (risk_free_rate - 0.5 * sigma_A_est**2) * time_horizon
    ) / (sigma_A_est * np.sqrt(time_horizon))
    pd_scenario = norm.cdf(-dd_scenario)
    print(
        f"{scenario*100:6.0f}% ({A_scenario:6.1f}M)      | {dd_scenario:18.2f} | {pd_scenario:6.2%}"
    )

# Monte Carlo simulation
print(f"\n=== Monte Carlo Default Simulation ===")
n_paths = 10000
dt = 1 / 252
n_steps = int(time_horizon / dt)

asset_paths = np.zeros((n_paths, n_steps + 1))
asset_paths[:, 0] = A_est

np.random.seed(42)
for step in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    asset_paths[:, step + 1] = asset_paths[:, step] * np.exp(
        (risk_free_rate - 0.5 * sigma_A_est**2) * dt + sigma_A_est * dW
    )

min_asset_value = asset_paths.min(axis=1)
defaults_mc = (min_asset_value < book_debt).astype(int)
pd_mc = defaults_mc.mean()

print(f"Monte Carlo PD (n_paths={n_paths}): {pd_mc:.2%}")
print(f"Merton PD: {pd_merton:.2%}")
print(f"Difference: {(pd_mc - pd_merton)*100:.2f} percentage points")

# Multi-period PD
print(f"\n=== Multi-Year PD ===")
for T_year in [1, 3, 5]:
    dd_year = (
        np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * T_year
    ) / (sigma_A_est * np.sqrt(T_year))
    pd_year = norm.cdf(-dd_year)
    print(f"{T_year}-Year PD: {pd_year:.2%}")

print("\n=== Merton Model Summary ===")
print(f"Approach: Structural model linking firm value to default risk")
print(f"Key insight: Default is rational economic decision when assets < debt")
print(f"Limitation: Assumes continuous asset paths (ignores jump risk)")

if __name__ == "__main__":
    print("\nMerton structural model execution complete.")
