"""
Loss Given Default (LGD) Analysis
Extracted from loss_given_default.md

Implements LGD simulation for mortgages with collateral valuation, economic
cycle effects, and corporate bond recovery analysis by seniority.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)


def main_loss_given_default():
    print("=== Mortgage LGD Analysis ===")

    n_mortgages = 1000
    mortgages = pd.DataFrame(
        {
            "original_balance": np.random.normal(300000, 100000, n_mortgages),
            "ltv_at_origination": np.random.uniform(0.70, 0.95, n_mortgages),
            "months_seasoned": np.random.uniform(0, 360, n_mortgages),
        }
    )

    mortgages["original_balance"] = mortgages["original_balance"].abs()

    # Economic condition impact
    economic_condition = np.random.choice(
        ["Boom", "Normal", "Crisis"], n_mortgages, p=[0.2, 0.5, 0.3]
    )
    value_change = np.where(
        economic_condition == "Boom",
        1.15,
        np.where(
            economic_condition == "Normal",
            1.02,
            np.random.uniform(0.7, 0.85, n_mortgages),
        ),
    )

    mortgages["current_property_value"] = (
        mortgages["original_balance"] / mortgages["ltv_at_origination"] * value_change
    )

    # Loan balance remaining
    amortization_factor = 1 - (mortgages["months_seasoned"] / 360) * 0.3
    mortgages["current_balance"] = mortgages["original_balance"] * amortization_factor

    # Calculate LGD
    sale_costs_pct = 0.08
    legal_costs_pct = 0.02

    mortgages["gross_recovery"] = mortgages["current_property_value"]
    mortgages["net_recovery"] = mortgages["gross_recovery"] * (
        1 - sale_costs_pct - legal_costs_pct
    )
    mortgages["lgd"] = np.maximum(
        (mortgages["current_balance"] - mortgages["net_recovery"])
        / mortgages["current_balance"],
        0,
    )

    print(f"Average LGD: {mortgages['lgd'].mean():.1%}")
    print(f"Median LGD: {mortgages['lgd'].median():.1%}")
    print(f"75th percentile LGD: {mortgages['lgd'].quantile(0.75):.1%}")
    print(f"Percentage with LGD=0: {(mortgages['lgd']==0).sum()/len(mortgages):.1%}")

    # LGD by economic condition
    print("\n=== LGD by Economic Condition ===")
    for condition in ["Boom", "Normal", "Crisis"]:
        mask = economic_condition == condition
        print(f"{condition:8s}: Mean LGD = {mortgages[mask]['lgd'].mean():.1%}")

    # Corporate bond recovery
    print("\n=== Corporate Bond Recovery Analysis ===")
    n_defaults = 100
    bonds = pd.DataFrame(
        {
            "debt_amount": np.random.lognormal(15, 2, n_defaults),
            "seniority": np.random.choice(
                ["Senior Secured", "Senior Unsecured", "Subordinated"],
                n_defaults,
                p=[0.3, 0.5, 0.2],
            ),
        }
    )

    recovery_rates = {
        "Senior Secured": 0.70,
        "Senior Unsecured": 0.40,
        "Subordinated": 0.15,
    }
    bonds["recovery_rate"] = bonds["seniority"].map(recovery_rates)
    bonds["lgd"] = 1 - bonds["recovery_rate"]

    print("\n| Seniority          | Recovery Rate | Avg LGD |")
    for seniority in ["Senior Secured", "Senior Unsecured", "Subordinated"]:
        subset = bonds[bonds["seniority"] == seniority]
        print(
            f"| {seniority:18s} | {subset['recovery_rate'].iloc[0]:12.1%} | {subset['lgd'].iloc[0]:6.1%} |"
        )


if __name__ == "__main__":
    main_loss_given_default()
