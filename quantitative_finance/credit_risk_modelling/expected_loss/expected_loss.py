"""
Expected Loss (EL) Portfolio Analysis
Extracted from expected_loss.md

Implements portfolio EL calculation with correlation adjustments, multi-year EL,
stress scenario analysis, and concentration metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)


def calculate_correlated_el(portfolio_data, within_corr, cross_corr):
    """Simplified Vasicek model for correlated portfolio EL"""
    total_el_ind = portfolio_data["EL_Individual"].sum()
    correlation_factor = 1 + within_corr * 0.5
    el_correlated = total_el_ind * correlation_factor
    return el_correlated


def main_expected_loss():
    print("=== Portfolio Expected Loss Analysis ===")

    n_loans = 500

    portfolio = pd.DataFrame(
        {
            "Loan_ID": np.arange(n_loans),
            "Segment": np.random.choice(
                ["Corporate", "SME", "Retail"], n_loans, p=[0.3, 0.4, 0.3]
            ),
            "Size": np.random.lognormal(12, 1.5, n_loans),
        }
    )

    # Assign risk parameters by segment
    segment_params = {
        "Corporate": {"PD": 0.015, "LGD": 0.35, "Volatility": 0.08},
        "SME": {"PD": 0.035, "LGD": 0.45, "Volatility": 0.12},
        "Retail": {"PD": 0.025, "LGD": 0.40, "Volatility": 0.10},
    }

    portfolio["PD"] = portfolio["Segment"].map(lambda x: segment_params[x]["PD"])
    portfolio["LGD"] = portfolio["Segment"].map(lambda x: segment_params[x]["LGD"])
    portfolio["EAD"] = portfolio["Size"]
    portfolio["EL_Individual"] = portfolio["PD"] * portfolio["LGD"] * portfolio["EAD"]

    print(f"\nPortfolio size: ${portfolio['Size'].sum()/1e6:.1f}M")
    print(f"Number of loans: {len(portfolio)}")

    # Summary by segment
    print("\n=== Expected Loss by Segment ===")
    for segment in ["Corporate", "SME", "Retail"]:
        subset = portfolio[portfolio["Segment"] == segment]
        print(f"\n{segment}:")
        print(f"  Count: {len(subset)}")
        print(f"  Portfolio: ${subset['Size'].sum()/1e6:.1f}M")
        print(f"  Total EL: ${subset['EL_Individual'].sum()/1e3:.0f}K")
        print(
            f"  EL% of Portfolio: {subset['EL_Individual'].sum() / subset['Size'].sum():.2%}"
        )

    # Portfolio-level EL
    portfolio_el_independent = portfolio["EL_Individual"].sum()
    correlation_within_segment = 0.30
    correlation_cross_segment = 0.10

    portfolio_el_correlated = calculate_correlated_el(
        portfolio, correlation_within_segment, correlation_cross_segment
    )
    el_increase_pct = (
        (portfolio_el_correlated - portfolio_el_independent)
        / portfolio_el_independent
        * 100
    )

    print(f"\n=== Expected Loss Aggregation ===")
    print(f"Sum of individual ELs: ${portfolio_el_independent/1e3:.0f}K")
    print(f"Correlated portfolio EL: ${portfolio_el_correlated/1e3:.0f}K")
    print(f"Correlation effect: +{el_increase_pct:.1f}%")

    # Multi-year expected loss
    print("\n=== Multi-Year Expected Loss ===")
    years = [1, 3, 5]
    for year in years:
        cumulative_pd = 1 - (1 - portfolio["PD"]) ** year
        annual_el = (cumulative_pd * portfolio["LGD"] * portfolio["EAD"]).sum()
        print(f"{year}-year cumulative EL: ${annual_el/1e3:.0f}K")

    # Stress scenario analysis
    print("\n=== Stress Scenario EL ===")
    scenarios = {
        "Base Case": {"pd_mult": 1.0, "lgd_mult": 1.0},
        "Mild Stress": {"pd_mult": 1.5, "lgd_mult": 1.1},
        "Severe Stress": {"pd_mult": 3.0, "lgd_mult": 1.3},
        "Crisis": {"pd_mult": 5.0, "lgd_mult": 1.5},
    }

    stress_results = []
    for scenario_name, multipliers in scenarios.items():
        stressed_pd = portfolio["PD"] * multipliers["pd_mult"]
        stressed_lgd = np.minimum(portfolio["LGD"] * multipliers["lgd_mult"], 1.0)
        stressed_el = (stressed_pd * stressed_lgd * portfolio["EAD"]).sum()
        stress_results.append(
            {
                "Scenario": scenario_name,
                "Total EL": stressed_el,
                "EL % of Portfolio": stressed_el / portfolio["Size"].sum() * 100,
            }
        )

    stress_df = pd.DataFrame(stress_results)
    print(stress_df.to_string(index=False))


if __name__ == "__main__":
    main_expected_loss()
