"""
Basel Accords for Credit Risk - Consolidated Python Implementation
Extracted from markdown files in basel_accords subfolder

This module implements Basel I, II, and III capital calculations including:
- Basel I crude risk weights
- Basel II Standardized and IRB approaches
- Basel III enhanced capital requirements and stress testing
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# Basel Capital Calculator Class
# ============================================================================


class BaselCapitalCalculator:
    """Calculate regulatory capital under Basel frameworks"""

    def __init__(self):
        self.minimum_cet1 = 0.045  # 4.5%
        self.ccb = 0.025  # Capital conservation buffer
        self.minimum_total = 0.08  # 8%

    def basel_i_rwa(self, exposure, asset_class="corporate"):
        """Basel I risk-weighted assets (crude buckets)"""
        weights = {
            "sovereign_oecd": 0.0,
            "bank_oecd": 0.2,
            "mortgage": 0.5,
            "corporate": 1.0,
            "other": 1.0,
        }

        rw = weights.get(asset_class, 1.0)
        return exposure * rw

    def basel_ii_standardized_rwa(
        self, exposure, rating="unrated", asset_class="corporate"
    ):
        """Basel II Standardized Approach"""
        # Corporate risk weights by external rating
        corporate_weights = {
            "AAA": 0.20,
            "AA": 0.20,
            "A": 0.50,
            "BBB": 1.00,
            "BB": 1.00,
            "B": 1.50,
            "CCC": 1.50,
            "unrated": 1.00,
        }

        # Asset class specific
        if asset_class == "corporate":
            rw = corporate_weights.get(rating, 1.0)
        elif asset_class == "retail":
            rw = 0.75
        elif asset_class == "mortgage":
            rw = 0.35
        elif asset_class == "sovereign":
            sovereign_weights = {
                "AAA": 0.0,
                "AA": 0.0,
                "A": 0.20,
                "BBB": 0.50,
                "BB": 1.00,
                "B": 1.00,
                "unrated": 1.00,
            }
            rw = sovereign_weights.get(rating, 1.0)
        else:
            rw = 1.0

        return exposure * rw

    def basel_ii_irb_correlation(self, pd, asset_class="corporate"):
        """Asset correlation R in Basel IRB formula"""
        if asset_class == "corporate":
            # Corporate correlation formula
            R = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + 0.24 * (
                1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
            )
        elif asset_class == "retail":
            # Retail (simplified)
            R = 0.15
        else:
            R = 0.12

        return R

    def basel_ii_irb_rw(self, pd, lgd, ead, maturity=2.5, asset_class="corporate"):
        """
        Basel II IRB risk weight formula

        Parameters:
        - pd: Probability of default (annual)
        - lgd: Loss given default (as fraction)
        - ead: Exposure at default
        - maturity: Effective maturity in years
        """
        # Floor for PD
        pd = max(pd, 0.0003)  # 0.03% minimum

        # Correlation
        R = self.basel_ii_irb_correlation(pd, asset_class)

        # Maturity adjustment
        b = (0.11852 - 0.05478 * np.log(pd)) ** 2
        maturity_factor = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

        # Risk weight calculation
        K = (
            lgd
            * norm.cdf((norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
            - lgd * pd
        ) * maturity_factor

        # 1.06 scaling factor (G-SIB buffer) and 12.5 multiplier (inverse of 8%)
        rw = K * 1.06 * 12.5

        # Floor and cap
        rw = max(rw, 0.0)
        rw = min(rw, 12.5)  # 100% risk weight cap

        return rw


# ============================================================================
# Basel I, II, III Capital Calculations
# ============================================================================


def main_basel_accords():
    print("=" * 70)
    print("BASEL ACCORDS: CAPITAL REQUIREMENTS CALCULATION")
    print("=" * 70)

    calculator = BaselCapitalCalculator()

    # Example portfolio
    portfolio = [
        {
            "name": "AAA Corporate",
            "exposure": 100,
            "rating": "AAA",
            "asset_class": "corporate",
            "pd": 0.0002,
            "lgd": 0.30,
        },
        {
            "name": "A Corporate",
            "exposure": 150,
            "rating": "A",
            "asset_class": "corporate",
            "pd": 0.002,
            "lgd": 0.40,
        },
        {
            "name": "BBB Corporate",
            "exposure": 200,
            "rating": "BBB",
            "asset_class": "corporate",
            "pd": 0.008,
            "lgd": 0.45,
        },
        {
            "name": "Prime Mortgage",
            "exposure": 300,
            "rating": "AAA",
            "asset_class": "mortgage",
            "pd": 0.001,
            "lgd": 0.25,
        },
        {
            "name": "Retail",
            "exposure": 250,
            "rating": "unrated",
            "asset_class": "retail",
            "pd": 0.015,
            "lgd": 0.50,
        },
    ]

    results = []

    for loan in portfolio:
        # Basel I
        rwa_basel_i = calculator.basel_i_rwa(loan["exposure"], loan["asset_class"])
        capital_basel_i = rwa_basel_i * 0.08

        # Basel II Standardized
        rwa_basel_ii_std = calculator.basel_ii_standardized_rwa(
            loan["exposure"], loan["rating"], loan["asset_class"]
        )
        capital_basel_ii_std = rwa_basel_ii_std * 0.08

        # Basel II IRB
        rw_irb = calculator.basel_ii_irb_rw(
            loan["pd"], loan["lgd"], loan["exposure"], asset_class=loan["asset_class"]
        )
        rwa_basel_ii_irb = loan["exposure"] * (rw_irb / 12.5)  # Convert back to RWA
        capital_basel_ii_irb = rwa_basel_ii_irb * 0.08

        results.append(
            {
                "Loan": loan["name"],
                "Exposure": loan["exposure"],
                "Basel_I_Capital": capital_basel_i,
                "Basel_II_Std_Capital": capital_basel_ii_std,
                "Basel_II_IRB_Capital": capital_basel_ii_irb,
            }
        )

    df = pd.DataFrame(results)

    print("\nPortfolio Capital Requirements (in millions):")
    print(df.to_string(index=False))

    print(f"\nTotal Exposure: ${df['Exposure'].sum():.0f}M")
    print(f"Total Basel I Capital: ${df['Basel_I_Capital'].sum():.1f}M")
    print(
        f"Total Basel II Standardized Capital: ${df['Basel_II_Std_Capital'].sum():.1f}M"
    )
    print(f"Total Basel II IRB Capital: ${df['Basel_II_IRB_Capital'].sum():.1f}M")

    # Basel III Stress Test
    print("\n" + "=" * 70)
    print("BASEL III STRESS TEST")
    print("=" * 70)

    # Bank starting position
    tier1_capital = 50  # $50B
    tier2_capital = 20  # $20B
    rwa = 400  # $400B risk-weighted assets
    total_assets = 1000  # $1000B total assets

    print(f"\nBank Starting Position:")
    print(f"  Tier 1 Capital: ${tier1_capital}B")
    print(f"  Tier 2 Capital: ${tier2_capital}B")
    print(f"  Risk-Weighted Assets: ${rwa}B")
    print(f"  Total Assets: ${total_assets}B")

    # Baseline ratios
    baseline = {
        "CET1": tier1_capital / rwa,
        "Tier1": tier1_capital / rwa,
        "Total": (tier1_capital + tier2_capital) / rwa,
        "Leverage": tier1_capital / total_assets,
    }

    print(f"\nBaseline Ratios:")
    for metric, value in baseline.items():
        print(f"  {metric}: {value*100:.2f}%")

    # Stress scenarios
    scenarios = {
        "Mild Downturn": {"tier1_loss": -5, "rwa_increase": 50, "tier2_loss": -2},
        "Moderate Recession": {
            "tier1_loss": -15,
            "rwa_increase": 150,
            "tier2_loss": -5,
        },
        "Severe Crisis": {"tier1_loss": -25, "rwa_increase": 250, "tier2_loss": -15},
    }

    minimums = {
        "CET1": 0.045,
        "Tier1": 0.065,
        "Total": 0.105,
        "Leverage": 0.03,
    }

    print(f"\nStress Test Results:")
    for scenario_name, shocks in scenarios.items():
        t1 = tier1_capital + shocks["tier1_loss"]
        t2 = tier2_capital + shocks["tier2_loss"]
        rwa_stressed = rwa + shocks["rwa_increase"]

        ratios = {
            "CET1": t1 / rwa_stressed,
            "Total": (t1 + t2) / rwa_stressed,
        }

        print(f"\n{scenario_name}:")
        print(
            f"  CET1: {ratios['CET1']*100:.2f}% (min: {minimums['CET1']*100:.2f}%) {'✓ PASS' if ratios['CET1'] >= minimums['CET1'] else '✗ FAIL'}"
        )
        print(
            f"  Total: {ratios['Total']*100:.2f}% (min: {minimums['Total']*100:.2f}%) {'✓ PASS' if ratios['Total'] >= minimums['Total'] else '✗ FAIL'}"
        )


if __name__ == "__main__":
    main_basel_accords()
