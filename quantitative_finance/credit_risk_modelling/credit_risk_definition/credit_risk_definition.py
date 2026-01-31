"""
Credit Risk Definition and Fundamentals
Extracted from credit_risk_definition.md

Implements basic credit risk metrics: PD, LGD, EAD, and Expected Loss calculations
across different loan types and borrower segments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main_credit_risk_definition():
    print("=== Credit Risk Across Loan Types ===")

    # Define typical credit risk profiles by loan type
    loan_types = {
        "Government Bond": {"PD": 0.001, "LGD": 0.40, "EAD": 1000000},
        "AAA Corporate": {"PD": 0.005, "LGD": 0.30, "EAD": 5000000},
        "BBB Corporate": {"PD": 0.020, "LGD": 0.40, "EAD": 5000000},
        "High-Yield Bond": {"PD": 0.080, "LGD": 0.50, "EAD": 3000000},
        "Prime Auto Loan": {"PD": 0.015, "LGD": 0.20, "EAD": 25000},
        "Subprime Auto Loan": {"PD": 0.080, "LGD": 0.35, "EAD": 20000},
        "Prime Mortgage": {"PD": 0.005, "LGD": 0.25, "EAD": 400000},
        "Subprime Mortgage": {"PD": 0.050, "LGD": 0.40, "EAD": 350000},
    }

    # Calculate expected loss
    results = []
    for loan_type, params in loan_types.items():
        pd_val = params["PD"]
        lgd_val = params["LGD"]
        ead_val = params["EAD"]
        el = pd_val * lgd_val * ead_val

        results.append(
            {
                "Loan Type": loan_type,
                "PD (%)": pd_val * 100,
                "LGD (%)": lgd_val * 100,
                "EAD": ead_val,
                "Expected Loss": el,
                "EL per $1K": el / ead_val * 1000,
            }
        )

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Time series of default rates (economic cycle)
    print("\n=== Default Rate Variation Over Economic Cycle ===")
    years = np.arange(2000, 2021)
    # Simulated default rate with business cycle
    cycle_default = (
        2
        + 1.5 * np.sin(2 * np.pi * years / 10)
        + 0.5 * np.random.normal(0, 1, len(years))
    )
    cycle_default = np.maximum(cycle_default, 0.5)  # Floor at 0.5%

    print("PIT rates spike in downturns; TTC rates average over full cycle")
    print(f"Average default rate: {cycle_default.mean():.2f}%")
    print(f"Peak default rate: {cycle_default.max():.2f}%")
    print(f"Minimum default rate: {cycle_default.min():.2f}%")


if __name__ == "__main__":
    main_credit_risk_definition()
