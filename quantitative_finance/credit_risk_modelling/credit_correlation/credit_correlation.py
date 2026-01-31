"""
Credit Correlation Analysis
Extracted from credit_correlation.md

Implements pairwise default correlation, factor models, and copula-based
joint distribution modeling for credit portfolios.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)


def main_credit_correlation():
    print("=== Credit Correlation Analysis ===")

    # Simulate default data for 2 firms over simulations
    n_simulations = 1000

    # True underlying correlation in asset returns
    rho_true = 0.35

    print(f"True asset correlation: {rho_true:.2f}")

    # Generate correlated default times
    # Single factor model: V_i = β_i × M + √(1 - β_i²) × ε_i
    beta_1, beta_2 = 0.6, 0.5  # Factor loadings

    # Common factor and idiosyncratic shocks
    M = np.random.normal(0, 1, n_simulations)
    eps_1 = np.random.normal(0, 1, n_simulations)
    eps_2 = np.random.normal(0, 1, n_simulations)

    # Firm values
    V_1 = beta_1 * M + np.sqrt(1 - beta_1**2) * eps_1
    V_2 = beta_2 * M + np.sqrt(1 - beta_2**2) * eps_2

    # Default thresholds (based on PD)
    pd_1, pd_2 = 0.05, 0.05
    threshold_1 = stats.norm.ppf(pd_1)
    threshold_2 = stats.norm.ppf(pd_2)

    # Defaults
    default_1 = (V_1 < threshold_1).astype(int)
    default_2 = (V_2 < threshold_2).astype(int)

    # Empirical statistics
    emp_pd_1 = default_1.mean()
    emp_pd_2 = default_2.mean()
    joint_default = (default_1 * default_2).sum()
    joint_pd = joint_default / n_simulations

    print(f"\nEmpirical Results (n={n_simulations}):")
    print(f"Firm 1 default rate: {emp_pd_1:.2%}")
    print(f"Firm 2 default rate: {emp_pd_2:.2%}")
    print(f"Joint default rate: {joint_pd:.2%}")

    # Calculate correlations
    p_11 = joint_pd
    p_10 = (default_1 * (1 - default_2)).mean()
    p_01 = ((1 - default_1) * default_2).mean()
    p_00 = ((1 - default_1) * (1 - default_2)).mean()

    corr_default = (p_11 - emp_pd_1 * emp_pd_2) / np.sqrt(
        emp_pd_1 * (1 - emp_pd_1) * emp_pd_2 * (1 - emp_pd_2)
    )

    print(f"\nDefault Correlation (Tetrachoric): {corr_default:.3f}")

    # Asset value correlation
    corr_asset = np.corrcoef(V_1, V_2)[0, 1]
    print(f"Asset Correlation (Realized): {corr_asset:.3f}")
    print(
        f"Expected Asset Correlation: {np.corrcoef(beta_1 * M, beta_2 * M)[0, 1]:.3f}"
    )

    # Conditional default probability
    if default_1.sum() > 0:
        cond_prob_d2_given_d1 = (default_1 * default_2).sum() / default_1.sum()
        uncond_prob_d2 = emp_pd_2
        print(f"\nP(Firm 2 defaults | Firm 1 defaults): {cond_prob_d2_given_d1:.2%}")
        print(f"P(Firm 2 defaults | unconditional): {uncond_prob_d2:.2%}")
        print(f"Contagion effect: {cond_prob_d2_given_d1 / uncond_prob_d2:.2f}x")

    # 2x2 Contingency table
    print(f"\nContingency Table (n={n_simulations}):")
    print(f"              No Default D2  | Default D2")
    print(
        f"No Default D1      {p_00*n_simulations:6.0f}      |    {p_01*n_simulations:6.0f}"
    )
    print(
        f"Default D1         {p_10*n_simulations:6.0f}      |    {p_11*n_simulations:6.0f}"
    )

    # Correlation under different scenarios
    print("\n=== Scenario Analysis: Correlation Variation ===")

    # Scenario 1: Normal times (low common factor realization)
    normal_mask = M < stats.norm.ppf(0.5)
    default_1_normal = default_1[normal_mask]
    default_2_normal = default_2[normal_mask]
    corr_normal = np.corrcoef(default_1_normal, default_2_normal)[0, 1]

    # Scenario 2: Crisis times (high common factor realization)
    crisis_mask = M > stats.norm.ppf(0.9)
    default_1_crisis = default_1[crisis_mask]
    default_2_crisis = default_2[crisis_mask]
    corr_crisis = np.corrcoef(default_1_crisis, default_2_crisis)[0, 1]

    print(f"Normal times correlation: {np.nan_to_num(corr_normal):.3f}")
    print(f"Crisis times correlation: {np.nan_to_num(corr_crisis):.3f}")

    # Multi-firm portfolio with correlation matrix
    print("\n=== Multi-Firm Portfolio with Correlation Matrix ===")

    n_firms = 10
    corr_matrix = np.zeros((n_firms, n_firms))

    # Build correlation matrix: sector structure
    for i in range(n_firms):
        for j in range(n_firms):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i % 3 == j % 3:  # Same sector
                corr_matrix[i, j] = 0.50
            else:  # Different sector
                corr_matrix[i, j] = 0.20

    print("\nCorrelation Matrix Structure:")
    print(f"Within-sector correlation: 0.50")
    print(f"Cross-sector correlation: 0.20")

    print("\n=== Credit Correlation Summary ===")
    print(f"Pairwise correlation captures co-movement in defaults")
    print(f"Factor models explain correlation through systemic risk")
    print(f"Correlation is regime-dependent (increases in crisis)")


if __name__ == "__main__":
    main_credit_correlation()
