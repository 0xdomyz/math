"""
Exposure at Default (EAD) Modeling
Extracted from exposure_at_default.md

Implements EAD calculations for mortgages, credit cards, credit lines with CCF,
and derivatives with potential future exposure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)


def main_exposure_at_default():
    print("=== Mortgage EAD Over Time ===")

    # Mortgage amortization
    principal = 300000
    annual_rate = 0.04
    years = 30
    monthly_rate = annual_rate / 12
    n_months = years * 12

    # Calculate monthly payment
    monthly_payment = (
        principal
        * (monthly_rate * (1 + monthly_rate) ** n_months)
        / ((1 + monthly_rate) ** n_months - 1)
    )

    # Generate payment schedule
    months = np.arange(0, n_months + 1)
    balance = np.zeros(len(months))
    balance[0] = principal

    for month in range(1, len(months)):
        interest_paid = balance[month - 1] * monthly_rate
        principal_paid = monthly_payment - interest_paid
        balance[month] = max(0, balance[month - 1] - principal_paid)

    accrued_interest = np.zeros(len(months))
    ead_mortgage = balance + accrued_interest

    print(f"Mortgage: ${principal:,.0f}")
    print(f"Monthly payment: ${monthly_payment:,.0f}")
    print(f"\nYears | Principal | EAD")
    for year in [0, 5, 10, 15, 20, 25, 30]:
        month_idx = year * 12
        if month_idx < len(months):
            print(
                f"{year:5d} | ${balance[month_idx]:9,.0f} | ${ead_mortgage[month_idx]:10,.0f}"
            )

    # Credit card utilization
    print("\n=== Credit Card EAD ===")
    credit_limits = np.array([5000, 10000, 15000, 20000, 25000])
    utilization_rates = np.array([0.25, 0.50, 0.75, 0.90, 0.95])
    interest_rates = np.array([0.18, 0.20, 0.21, 0.22, 0.23])

    card_data = []
    for limit, util, apr in zip(credit_limits, utilization_rates, interest_rates):
        drawn = limit * util
        monthly_interest = drawn * (apr / 12)
        ead = drawn + monthly_interest
        card_data.append(
            {
                "Credit Limit": limit,
                "Utilization %": util * 100,
                "Drawn Amount": drawn,
                "Monthly Interest": monthly_interest,
                "EAD": ead,
            }
        )

    card_df = pd.DataFrame(card_data)
    print(card_df.to_string(index=False))

    # Committed credit line with CCF
    print("\n=== Credit Line EAD with Credit Conversion Factor ===")
    n_accounts = 1000
    committed_amount = np.random.lognormal(12, 1.5, n_accounts)
    drawn_pct = np.random.beta(2, 5, n_accounts)
    drawn_amount = committed_amount * drawn_pct
    accrued_interest_line = drawn_amount * 0.015

    # CCF scenarios
    ccf_normal = 0.75
    ccf_stressed = 0.95
    ccf_default = 1.00

    ead_normal = (
        drawn_amount
        + accrued_interest_line
        + (committed_amount - drawn_amount) * ccf_normal
    )
    ead_stressed = (
        drawn_amount
        + accrued_interest_line
        + (committed_amount - drawn_amount) * ccf_stressed
    )
    ead_at_default = (
        drawn_amount
        + accrued_interest_line
        + (committed_amount - drawn_amount) * ccf_default
    )

    print(f"Average Committed Amount: ${committed_amount.mean():,.0f}")
    print(f"Average % Drawn: {drawn_pct.mean()*100:.1f}%")
    print(f"Average EAD (Normal): ${ead_normal.mean():,.0f}")
    print(f"Average EAD (Stressed): ${ead_stressed.mean():,.0f}")
    print(f"Average EAD (At Default): ${ead_at_default.mean():,.0f}")
    print(f"EAD Increase: {(ead_at_default.mean() / ead_normal.mean() - 1) * 100:.1f}%")

    # Derivative exposure
    print("\n=== Derivative EAD (Simplified FX Swap) ===")
    n_swaps = 100
    notional_usd = np.random.lognormal(15, 1, n_swaps)
    maturity_years = np.random.uniform(1, 10, n_swaps)

    # Mark-to-market and PFE
    mtm = np.random.normal(0, notional_usd * 0.02, n_swaps)
    volatility = 0.10
    pfe = 2 * volatility * notional_usd * np.sqrt(maturity_years)
    ead_derivative = np.maximum(mtm, 0) + pfe

    print(f"Average Notional: ${notional_usd.mean():,.0f}")
    print(f"Average MTM: ${mtm.mean():,.0f}")
    print(f"Average PFE: ${pfe.mean():,.0f}")
    print(f"Average EAD: ${ead_derivative.mean():,.0f}")


if __name__ == "__main__":
    main_exposure_at_default()
