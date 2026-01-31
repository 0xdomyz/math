"""
Reduced-Form Models
Extracted from reduced_form_models.md

Market-based PD extraction using credit spreads and intensity models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize

np.random.seed(42)

print("=== Reduced-Form PD from Market Spreads ===")

# Market data: Credit spreads
spreads_market = pd.DataFrame(
    {
        "Maturity": [0.5, 1, 2, 3, 5, 7, 10],
        "CDS_Spread_bps": [80, 120, 160, 200, 250, 280, 300],
        "Bond_Spread_bps": [100, 140, 180, 220, 280, 310, 330],
    }
)

print("Market CDS and Bond Spreads:")
print(spreads_market)

# Assumptions
coupon = 0.04
face_value = 100
recovery_rate = 0.40
risk_free_rate = 0.03


def extract_hazard_rate_from_cds(cds_bps, recovery):
    """Simplified extraction: λ = CDS_spread / LGD"""
    lgd = 1 - recovery
    hazard_rate = (cds_bps / 10000) / lgd
    return hazard_rate


spreads_market["LGD"] = 1 - recovery_rate
spreads_market["Hazard_Rate"] = spreads_market["CDS_Spread_bps"].apply(
    lambda x: extract_hazard_rate_from_cds(x, recovery_rate)
)

spreads_market["Survival_Prob"] = np.exp(
    -spreads_market["Hazard_Rate"] * spreads_market["Maturity"]
)
spreads_market["PD_Cumulative"] = 1 - spreads_market["Survival_Prob"]

print("\n=== PD Extraction ===")
print(spreads_market[["Maturity", "CDS_Spread_bps", "Hazard_Rate", "PD_Cumulative"]])

spreads_market["Forward_PD"] = spreads_market["Hazard_Rate"]

print("\n=== Forward Default Rates ===")
for idx, row in spreads_market.iterrows():
    print(
        f"Year {row['Maturity']:.1f}: Forward PD = {row['Forward_PD']:.2%}, Cumulative PD = {row['PD_Cumulative']:.2%}"
    )

# Stochastic intensity model
print("\n=== Stochastic Intensity Model ===")

maturity_range = np.linspace(0.5, 10, 50)

from numpy.polynomial import Polynomial

p = Polynomial.fit(spreads_market["Maturity"], spreads_market["Hazard_Rate"], 2)
hazard_fitted = p(maturity_range)

hazard_interp = interp1d(
    spreads_market["Maturity"],
    spreads_market["Hazard_Rate"],
    kind="cubic",
    fill_value="extrapolate",
)

# Simulate hazard rate paths (Vasicek-type)
dt = 0.01
n_steps = int(10 / dt)
n_paths = 1000

kappa = 0.3
theta = 0.15
sigma = 0.05

lambda_paths = np.zeros((n_paths, n_steps + 1))
lambda_paths[:, 0] = spreads_market["Hazard_Rate"].iloc[0]

np.random.seed(42)
for step in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    lambda_paths[:, step + 1] = (
        lambda_paths[:, step]
        + kappa * (theta - lambda_paths[:, step]) * dt
        + sigma * dW
    )
    lambda_paths[:, step + 1] = np.maximum(lambda_paths[:, step + 1], 0)

time_axis = np.arange(n_steps + 1) * dt
survival_paths = np.exp(-np.cumsum(lambda_paths * dt, axis=1))

default_prob_simulated = 1 - survival_paths.mean(axis=0)

print("\nStochastic Intensity Simulation:")
print(f"Initial hazard rate: {lambda_paths[0, 0]:.2%}")
print(f"Mean long-run hazard: {lambda_paths[:, -1].mean():.2%}")
print(f"Std of final hazard rate: {lambda_paths[:, -1].std():.2%}")

# Term structure
print("\n=== Term Structure of Credit Risk ===")
print("Maturity | CDS Spread | Bond Spread | Spread Difference")
print("-" * 55)
for idx, row in spreads_market.iterrows():
    diff = row["Bond_Spread_bps"] - row["CDS_Spread_bps"]
    print(
        f"{row['Maturity']:6.1f}Y  | {row['CDS_Spread_bps']:9d} | {row['Bond_Spread_bps']:10d} | {diff:17d}"
    )

print("\nAsset swap spread (Bond - CDS) reflects liquidity/basis risk")

# CDS index modeling
print("\n=== CDS Index Modeling ===")
n_names = 125
individual_spreads = np.random.normal(150, 80, n_names)
individual_spreads = np.maximum(individual_spreads, 10)

index_spread = np.mean(individual_spreads)
index_std = np.std(individual_spreads)

print(f"Number of names: {n_names}")
print(f"Index spread: {index_spread:.0f} bps")
print(f"Std of individual spreads: {index_std:.0f} bps")
print(f"Implied correlation: {(index_spread / np.mean(individual_spreads)):.2f}")

print("\n=== Reduced-Form Summary ===")
print(f"Approach: Market prices → Hazard rate → PD")
print(f"Advantage: Forward-looking, based on real transactions")
print(f"Limitation: Sensitive to market liquidity, bid-ask spreads")

if __name__ == "__main__":
    print("\nReduced-form models execution complete.")
