"""
IFRS 9 / CECL: Consolidated Python Code
Extracted from markdown files in ifrs_9_cecl subfolder

This module contains code for:
- Expected credit loss models
- Significant increase in credit risk detection
- Forward-looking information incorporation
- Lifetime vs 12-month ECL calculations
- Three-stage approach implementation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, roc_curve

# ============================================================================
# EXPECTED CREDIT LOSS MODELS
# ============================================================================

# Set seed
np.random.seed(42)

# Portfolio parameters
n_loans = 500
loan_amounts = np.random.uniform(50_000, 500_000, n_loans)
maturities = np.random.randint(1, 10, n_loans)  # 1-10 years remaining
lgd = 0.40  # 40% LGD (constant for simplicity)
eir = 0.05  # 5% effective interest rate (discount rate)


# PD term structure (annual marginal PD)
def generate_pd_term_structure(maturity, base_pd=0.01, scenario="base"):
    """Generate annual marginal PD for loan over maturity."""
    if scenario == "base":
        pds = [base_pd * (1 + 0.5 * t) for t in range(maturity)]
    elif scenario == "adverse":
        pds = [base_pd * 2 * (1 + 0.5 * t) for t in range(maturity)]
    elif scenario == "upside":
        pds = [base_pd * 0.5 * (1 + 0.5 * t) for t in range(maturity)]

    return np.array(pds)


def calc_12m_ecl(ead, pd_12m, lgd):
    """Stage 1: 12-month ECL (no discounting for simplicity)."""
    return ead * pd_12m * lgd


def calc_lifetime_ecl(ead, pds, lgd, eir):
    """Stage 2/3: Lifetime ECL with discounting."""
    ecl = 0
    survival_prob = 1.0

    for t, pd_t in enumerate(pds, start=1):
        marginal_loss = ead * pd_t * lgd * survival_prob
        discount_factor = 1 / (1 + eir) ** t
        ecl += marginal_loss * discount_factor
        survival_prob *= 1 - pd_t

    return ecl


# ============================================================================
# SIGNIFICANT INCREASE IN CREDIT RISK (SICR)
# ============================================================================


def detect_sicr(pd_orig, pd_current, dpd, rating_orig, rating_current, watchlist=False):
    """
    Detect significant increase in credit risk using multiple indicators.

    Returns dict with SICR flags for each indicator.
    """
    sicr_flags = {}

    # 30 DPD backstop
    sicr_flags["30dpd"] = dpd >= 30

    # Relative PD change > 2×
    sicr_flags["relative_pd"] = (pd_current / pd_orig) > 2.0 if pd_orig > 0 else False

    # Absolute PD > 5%
    sicr_flags["absolute_pd"] = pd_current > 0.05

    # Rating downgrade ≥ 2 notches
    sicr_flags["rating"] = (rating_current - rating_orig) >= 2

    # Watchlist
    sicr_flags["watchlist"] = watchlist

    # Combined (OR logic)
    sicr_flags["combined"] = any(
        [
            sicr_flags["30dpd"],
            sicr_flags["relative_pd"],
            sicr_flags["absolute_pd"],
            sicr_flags["rating"],
            sicr_flags["watchlist"],
        ]
    )

    return sicr_flags


# ============================================================================
# FORWARD-LOOKING INFORMATION
# ============================================================================

# Macroeconomic scenarios
scenarios = {
    "Base": {
        "weight": 0.50,
        "gdp_growth": [2.5, 2.8, 2.3],
        "unemployment": [5.0, 4.8, 5.0],
        "house_price_change": [2.0, 2.5, 2.0],
    },
    "Adverse": {
        "weight": 0.30,
        "gdp_growth": [-1.5, 0.0, 1.0],
        "unemployment": [7.5, 8.0, 7.0],
        "house_price_change": [-10.0, -5.0, 0.0],
    },
    "Upside": {
        "weight": 0.20,
        "gdp_growth": [4.0, 4.5, 3.5],
        "unemployment": [4.0, 3.5, 4.0],
        "house_price_change": [5.0, 6.0, 4.0],
    },
}


def calc_pd_macro(base_pd, gdp_growth, unemployment):
    """PD model: log(PD) = α + β₁·GDP + β₂·Unemp"""
    beta_gdp = -0.15
    beta_unemp = 0.08

    gdp_effect = beta_gdp * (gdp_growth - 2.5)
    unemp_effect = beta_unemp * (unemployment - 5.0)

    adjusted_pd = base_pd * np.exp(gdp_effect + unemp_effect)
    return adjusted_pd.clip(0.001, 0.50)


def calc_lgd_macro(base_lgd, house_price_change):
    """LGD model: LGD adjusts with collateral value"""
    lgd_adjustment = -0.10 * house_price_change
    adjusted_lgd = base_lgd + lgd_adjustment
    return adjusted_lgd.clip(0.10, 0.90)


# ============================================================================
# LIFETIME VS 12-MONTH ECL
# ============================================================================


def calculate_ecl_by_maturity(loan_amount, annual_pd, lgd, eir, maturities):
    """
    Calculate both 12-month and lifetime ECL for different maturities.

    Returns DataFrame with results.
    """
    results = []

    ecl_12m = loan_amount * annual_pd * lgd

    for maturity in maturities:
        pds = [annual_pd] * maturity

        lifetime_ecl = 0
        survival_prob = 1.0

        for year in range(1, maturity + 1):
            pd_year = pds[year - 1]
            marginal_loss = loan_amount * pd_year * lgd * survival_prob
            discount_factor = 1 / (1 + eir) ** year
            ecl_year = marginal_loss * discount_factor
            lifetime_ecl += ecl_year
            survival_prob *= 1 - pd_year

        cumulative_pd = 1 - survival_prob

        results.append(
            {
                "maturity": maturity,
                "ecl_12m": ecl_12m,
                "ecl_lifetime": lifetime_ecl,
                "ratio": lifetime_ecl / ecl_12m,
                "cumulative_pd": cumulative_pd,
                "coverage_ratio_12m": (ecl_12m / loan_amount) * 100,
                "coverage_ratio_lifetime": (lifetime_ecl / loan_amount) * 100,
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# THREE-STAGE APPROACH
# ============================================================================


def classify_stage(default_flag, sicr_flag):
    """
    Classify loan into Stage 1, 2, or 3.

    Stage 1: Performing, no SICR (12-month ECL)
    Stage 2: Performing, SICR (lifetime ECL)
    Stage 3: Default (lifetime ECL with 100% PD)
    """
    if default_flag == 1:
        return 3
    elif sicr_flag:
        return 2
    else:
        return 1


def calculate_stage_ecl(stage, amount, pd_12m, pd_lifetime, lgd):
    """Calculate ECL based on stage classification."""
    if stage == 1:
        return amount * pd_12m * lgd
    elif stage == 2:
        return amount * pd_lifetime * lgd
    else:  # Stage 3
        return amount * lgd  # PD = 100%


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IFRS 9 ECL Model: Consolidated Implementation")
    print("=" * 70)

    # Generate portfolio
    portfolio = []

    for i in range(n_loans):
        loan = {
            "loan_id": i,
            "amount": loan_amounts[i],
            "maturity": maturities[i],
            "base_pd_12m": np.random.uniform(0.005, 0.02),
        }

        loan["pds_base"] = generate_pd_term_structure(
            loan["maturity"], loan["base_pd_12m"], "base"
        )
        loan["pds_adverse"] = generate_pd_term_structure(
            loan["maturity"], loan["base_pd_12m"], "adverse"
        )
        loan["pds_upside"] = generate_pd_term_structure(
            loan["maturity"], loan["base_pd_12m"], "upside"
        )

        loan["ecl_12m"] = calc_12m_ecl(loan["amount"], loan["base_pd_12m"], lgd)
        loan["ecl_lifetime_base"] = calc_lifetime_ecl(
            loan["amount"], loan["pds_base"], lgd, eir
        )
        loan["ecl_lifetime_adverse"] = calc_lifetime_ecl(
            loan["amount"], loan["pds_adverse"], lgd, eir
        )
        loan["ecl_lifetime_upside"] = calc_lifetime_ecl(
            loan["amount"], loan["pds_upside"], lgd, eir
        )

        loan["ecl_lifetime_weighted"] = (
            0.5 * loan["ecl_lifetime_base"]
            + 0.3 * loan["ecl_lifetime_adverse"]
            + 0.2 * loan["ecl_lifetime_upside"]
        )

        portfolio.append(loan)

    df = pd.DataFrame(portfolio)

    print(f"Number of Loans: {n_loans}")
    print(f"Total Exposure: ${df['amount'].sum():,.0f}")
    print(f"Average Maturity: {df['maturity'].mean():.1f} years")
    print("")

    total_12m = df["ecl_12m"].sum()
    total_lifetime_weighted = df["ecl_lifetime_weighted"].sum()

    print(f"12-Month ECL (Stage 1):      ${total_12m:,.0f}")
    print(f"Lifetime ECL (Weighted):     ${total_lifetime_weighted:,.0f}")
    print(f"Lifetime ECL / 12m ECL:      {total_lifetime_weighted / total_12m:.1f}×")
    print("")

    coverage_12m = (total_12m / df["amount"].sum()) * 100
    coverage_lifetime = (total_lifetime_weighted / df["amount"].sum()) * 100

    print(f"Coverage Ratio (12m ECL):    {coverage_12m:.2f}%")
    print(f"Coverage Ratio (Lifetime):   {coverage_lifetime:.2f}%")
    print("")
    print("IFRS 9 ECL model execution complete.")
