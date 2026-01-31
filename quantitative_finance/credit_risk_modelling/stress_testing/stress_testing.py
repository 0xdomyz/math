"""
Stress Testing
Extracted from stress_testing.md

Portfolio stress testing under various economic scenarios.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

print("=== Portfolio Stress Testing ===")

# Base portfolio
print("\n=== Base Portfolio ===")

portfolio_base = pd.DataFrame(
    {
        "Exposure_Name": [
            "US Corps AAA",
            "US Corps A",
            "US Corps BBB",
            "US Corps BB",
            "Real Estate",
            "Consumer Finance",
            "International",
            "Treasury",
        ],
        "Amount": [200, 250, 300, 150, 200, 100, 150, 50],
        "Rating": ["AAA", "A", "BBB", "BB", "BBB", "BB", "A", "AAA"],
        "Sector": [
            "Corporate",
            "Corporate",
            "Corporate",
            "Corporate",
            "Real Estate",
            "Consumer",
            "Corporate",
            "Sovereign",
        ],
        "Base_PD_%": [0.01, 0.10, 0.50, 2.00, 0.80, 3.00, 0.15, 0.00],
        "LGD_%": [30, 35, 40, 45, 50, 60, 40, 0],
        "Duration_years": [5, 5, 5, 5, 5, 3, 5, 2],
        "Spread_bps": [50, 100, 150, 300, 200, 400, 120, 0],
    }
)

total_exposure = portfolio_base["Amount"].sum()
print(f"Total Portfolio: ${total_exposure}M")
print(portfolio_base.to_string(index=False))

portfolio_base["Base_EL"] = (
    portfolio_base["Base_PD_%"]
    / 100
    * portfolio_base["LGD_%"]
    / 100
    * portfolio_base["Amount"]
)
base_el = portfolio_base["Base_EL"].sum()
print(f"\nBase Case Expected Loss: ${base_el:.1f}M ({base_el/total_exposure*100:.2f}%)")

# Define stress scenarios
print("\n=== Stress Scenarios ===")

scenarios = {
    "Recession_Mild": {
        "Description": "Mild Recession: +2pp unemployment, GDP -1%",
        "PD_Multiplier": {"AAA": 2.0, "A": 2.0, "BBB": 3.0, "BB": 5.0},
        "LGD_Multiplier": {
            "Corporate": 1.05,
            "Real Estate": 1.15,
            "Consumer": 1.10,
            "Sovereign": 1.0,
        },
        "Spread_Change_bps": {
            "Corporate": 100,
            "Real Estate": 200,
            "Consumer": 150,
            "Sovereign": 0,
        },
        "Correlation_Adjustment": 0.40,
    },
    "Recession_Severe": {
        "Description": "Severe Recession: +5pp unemployment, GDP -3%",
        "PD_Multiplier": {"AAA": 5.0, "A": 5.0, "BBB": 10.0, "BB": 25.0},
        "LGD_Multiplier": {
            "Corporate": 1.15,
            "Real Estate": 1.35,
            "Consumer": 1.30,
            "Sovereign": 1.0,
        },
        "Spread_Change_bps": {
            "Corporate": 300,
            "Real Estate": 600,
            "Consumer": 400,
            "Sovereign": 50,
        },
        "Correlation_Adjustment": 0.70,
    },
    "Financial_Crisis": {
        "Description": "Financial Crisis: Systemic breakdown",
        "PD_Multiplier": {"AAA": 10.0, "A": 15.0, "BBB": 30.0, "BB": 100.0},
        "LGD_Multiplier": {
            "Corporate": 1.30,
            "Real Estate": 1.50,
            "Consumer": 1.50,
            "Sovereign": 1.20,
        },
        "Spread_Change_bps": {
            "Corporate": 600,
            "Real Estate": 1000,
            "Consumer": 800,
            "Sovereign": 200,
        },
        "Correlation_Adjustment": 0.95,
    },
    "Sector_RealEstate": {
        "Description": "Real Estate Shock: RE prices -40%",
        "PD_Multiplier": {"AAA": 1.0, "A": 1.0, "BBB": 1.5, "BB": 3.0},
        "LGD_Multiplier": {
            "Corporate": 1.0,
            "Real Estate": 1.50,
            "Consumer": 1.05,
            "Sovereign": 1.0,
        },
        "Spread_Change_bps": {
            "Corporate": 50,
            "Real Estate": 800,
            "Consumer": 100,
            "Sovereign": 0,
        },
        "Correlation_Adjustment": 0.30,
    },
}

# Calculate losses under each scenario
stress_results = []

for scenario_name, scenario_params in scenarios.items():
    portfolio_stress = portfolio_base.copy()

    def get_pd_mult(rating):
        return scenario_params["PD_Multiplier"].get(rating, 1.0)

    portfolio_stress["Stress_PD_%"] = portfolio_stress.apply(
        lambda row: row["Base_PD_%"] * get_pd_mult(row["Rating"]), axis=1
    )

    def get_lgd_mult(sector):
        return scenario_params["LGD_Multiplier"].get(sector, 1.0)

    portfolio_stress["Stress_LGD_%"] = portfolio_stress.apply(
        lambda row: min(row["LGD_%"] * get_lgd_mult(row["Sector"]), 100), axis=1
    )

    portfolio_stress["Stress_EL"] = (
        portfolio_stress["Stress_PD_%"]
        / 100
        * portfolio_stress["Stress_LGD_%"]
        / 100
        * portfolio_stress["Amount"]
    )

    credit_loss = portfolio_stress["Stress_EL"].sum()

    def get_spread_change(sector):
        return scenario_params["Spread_Change_bps"].get(sector, 0)

    portfolio_stress["Spread_Delta_bps"] = portfolio_stress["Sector"].apply(
        get_spread_change
    )
    portfolio_stress["MVL_from_Spread"] = (
        portfolio_stress["Duration_years"]
        * portfolio_stress["Spread_Delta_bps"]
        / 10000
        * portfolio_stress["Amount"]
    )

    market_value_loss = portfolio_stress["MVL_from_Spread"].sum()

    total_loss = credit_loss + market_value_loss
    loss_pct = total_loss / total_exposure * 100

    stress_results.append(
        {
            "Scenario": scenario_name,
            "Description": scenario_params["Description"],
            "Credit_Loss_$M": credit_loss,
            "MVL_$M": market_value_loss,
            "Total_Loss_$M": total_loss,
            "Loss_%_Portfolio": loss_pct,
            "Correlation_Adj": scenario_params["Correlation_Adjustment"],
        }
    )

stress_df = pd.DataFrame(stress_results)

print("\nStress Test Results:")
print(
    stress_df[
        ["Scenario", "Credit_Loss_$M", "MVL_$M", "Total_Loss_$M", "Loss_%_Portfolio"]
    ].to_string(index=False)
)

# Capital impact
print("\n=== Capital Impact ===")

capital = 800
current_loss = base_el

for idx, row in stress_df.iterrows():
    remaining_capital = capital - row["Total_Loss_$M"]
    capital_ratio = remaining_capital / total_exposure * 100
    capital_buffer = "✓" if capital_ratio > 6.5 else "✗"

    print(f"\n{row['Scenario']}:")
    print(f"  Loss: ${row['Total_Loss_$M']:.1f}M")
    print(f"  Remaining capital: ${remaining_capital:.1f}M")
    print(f"  Capital ratio: {capital_ratio:.2f}% {capital_buffer}")

# Sensitivity analysis
print("\n=== Sensitivity Analysis: PD Impact ===")

pd_multipliers_range = np.linspace(1, 20, 10)
severe_recession_scenario = scenarios["Recession_Severe"]

print(f"PD Multiplier | Credit Loss ($M) | % of Portfolio")
print("-" * 50)

for mult in pd_multipliers_range:
    portfolio_sens = portfolio_base.copy()
    portfolio_sens["Stress_PD_%"] = portfolio_sens.apply(
        lambda row: row["Base_PD_%"]
        * mult
        * severe_recession_scenario["PD_Multiplier"].get(row["Rating"], 1.0),
        axis=1,
    )
    portfolio_sens["Stress_PD_%"] = portfolio_sens["Stress_PD_%"].clip(upper=100)

    portfolio_sens["Stress_LGD_%"] = portfolio_sens.apply(
        lambda row: min(
            row["LGD_%"]
            * severe_recession_scenario["LGD_Multiplier"].get(row["Sector"], 1.0),
            100,
        ),
        axis=1,
    )

    portfolio_sens["Stress_EL"] = (
        portfolio_sens["Stress_PD_%"]
        / 100
        * portfolio_sens["Stress_LGD_%"]
        / 100
        * portfolio_sens["Amount"]
    )

    credit_loss = portfolio_sens["Stress_EL"].sum()
    print(
        f"{mult:13.1f} | {credit_loss:15.1f} | {credit_loss/total_exposure*100:14.2f}%"
    )

# Reverse stress test
print("\n=== Reverse Stress Test: Capital Depletion Scenarios ===")

target_loss = capital * 0.5
portfolio_reverse = portfolio_base.copy()

for mult in np.linspace(1, 100, 1000):
    portfolio_reverse["Stress_PD_%"] = portfolio_reverse["Base_PD_%"] * mult
    portfolio_reverse["Stress_PD_%"] = portfolio_reverse["Stress_PD_%"].clip(upper=100)
    portfolio_reverse["Stress_EL"] = (
        portfolio_reverse["Stress_PD_%"]
        / 100
        * portfolio_reverse["LGD_%"]
        / 100
        * portfolio_reverse["Amount"]
    )

    loss = portfolio_reverse["Stress_EL"].sum()

    if loss >= target_loss:
        critical_mult = mult
        break

print(f"Critical PD multiplier for 50% capital loss: {critical_mult:.1f}x")
print(f"Implied scenario: Average PD increases {critical_mult:.1f}x from base")

print("\n=== Stress Testing Summary ===")
print(f"Portfolio stress tested under {len(scenarios)} scenarios")
print(f"Most severe: {stress_df.loc[stress_df['Total_Loss_$M'].idxmax(), 'Scenario']}")
print(
    f"Maximum loss: ${stress_df['Total_Loss_$M'].max():.1f}M ({stress_df['Loss_%_Portfolio'].max():.1f}% of portfolio)"
)

if __name__ == "__main__":
    print("\nStress testing execution complete.")
