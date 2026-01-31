"""
Credit Value-at-Risk (Credit VaR)
Extracted from credit_var.md

Implements parametric, Monte Carlo, and historical simulation approaches
for calculating portfolio Credit VaR and Expected Shortfall.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)


def main_credit_var():
    print("=== Credit Value-at-Risk Analysis ===")

    # Portfolio of loans
    n_loans = 100
    portfolio = pd.DataFrame(
        {
            "Loan_ID": np.arange(n_loans),
            "Amount": np.random.lognormal(12, 1.5, n_loans),
            "Rating": np.random.choice(
                ["AAA", "AA", "A", "BBB", "BB", "B"],
                n_loans,
                p=[0.05, 0.10, 0.20, 0.35, 0.20, 0.10],
            ),
        }
    )

    # PD and LGD by rating
    rating_params = {
        "AAA": {"PD": 0.0001, "LGD": 0.30},
        "AA": {"PD": 0.0005, "LGD": 0.32},
        "A": {"PD": 0.002, "LGD": 0.35},
        "BBB": {"PD": 0.008, "LGD": 0.40},
        "BB": {"PD": 0.030, "LGD": 0.45},
        "B": {"PD": 0.100, "LGD": 0.50},
    }

    portfolio["PD"] = portfolio["Rating"].map(lambda r: rating_params[r]["PD"])
    portfolio["LGD"] = portfolio["Rating"].map(lambda r: rating_params[r]["LGD"])
    portfolio["EL"] = portfolio["PD"] * portfolio["LGD"] * portfolio["Amount"]

    print(f"Portfolio size: {len(portfolio)} loans")
    print(f"Total exposure: ${portfolio['Amount'].sum()/1e6:.1f}M")
    print(f"Average PD: {portfolio['PD'].mean():.2%}")
    print(f"Total Expected Loss: ${portfolio['EL'].sum()/1e3:.0f}K")

    # Method 1: Parametric (Delta-Normal) VaR
    print("\n=== Method 1: Parametric (Delta-Normal) VaR ===")

    total_el = portfolio["EL"].sum()
    variances = (portfolio["PD"] * (1 - portfolio["PD"])) * (
        portfolio["LGD"] * portfolio["Amount"]
    ) ** 2

    # Account for correlation
    pairwise_cov = 0
    for i in range(len(portfolio)):
        for j in range(i + 1, len(portfolio)):
            std_i = np.sqrt(variances.iloc[i])
            std_j = np.sqrt(variances.iloc[j])
            pairwise_cov += 2 * 0.30 * std_i * std_j

    total_var_correlated = variances.sum() + pairwise_cov
    total_std_correlated = np.sqrt(total_var_correlated)

    # VaR at different confidence levels
    confidence_levels = [0.95, 0.99, 0.999]
    var_parametric = {}

    for conf in confidence_levels:
        z_score = stats.norm.ppf(conf)
        var_amount = total_el + z_score * total_std_correlated
        var_parametric[conf] = var_amount

    print("Parametric VaR (assuming normal distribution):")
    print("Confidence | VaR Amount | UL = VaR - EL")
    print("-" * 45)
    for conf in confidence_levels:
        ul = var_parametric[conf] - total_el
        print(
            f"{conf*100:6.1f}%    | ${var_parametric[conf]/1e6:9.2f}M | ${ul/1e6:8.2f}M"
        )

    # Method 2: Monte Carlo Simulation
    print("\n=== Method 2: Monte Carlo Credit VaR ===")

    n_simulations = 10000
    default_matrix = np.random.rand(n_simulations, len(portfolio))
    losses_mc = np.zeros(n_simulations)

    for sim in range(n_simulations):
        for i, loan in enumerate(portfolio.itertuples()):
            if default_matrix[sim, i] < loan.PD:
                losses_mc[sim] += loan.LGD * loan.Amount

    var_mc = {}
    for conf in confidence_levels:
        var_mc[conf] = np.percentile(losses_mc, conf * 100)

    print(f"Simulations: {n_simulations}")
    print("Monte Carlo VaR:")
    print("Confidence | VaR Amount | UL = VaR - EL")
    print("-" * 45)
    for conf in confidence_levels:
        ul = var_mc[conf] - total_el
        print(f"{conf*100:6.1f}%    | ${var_mc[conf]/1e6:9.2f}M | ${ul/1e6:8.2f}M")

    # Expected Shortfall (ES) = Average of tail losses
    print("\n=== Expected Shortfall (CVaR) ===")
    print("Average loss in worst scenarios:")
    for conf in confidence_levels:
        threshold = np.percentile(losses_mc, conf * 100)
        tail_losses = losses_mc[losses_mc >= threshold]
        es = tail_losses.mean()
        print(f"{conf*100:6.1f}%: VaR = ${var_mc[conf]/1e6:.2f}M, ES = ${es/1e6:.2f}M")

    print("\n=== Credit VaR Summary ===")
    print(f"Portfolio 99% VaR (MC): ${var_mc[0.99]/1e6:.2f}M")
    print(f"Capital buffer for tail risk: ${(var_mc[0.99] - total_el)/1e6:.2f}M")
    print(f"Ratio VaR/EL: {var_mc[0.99]/total_el:.2f}x")


if __name__ == "__main__":
    main_credit_var()
