"""
Internal Ratings-Based (IRB) Approach
Extracted from internal_ratings_based.md

Basel III IRB approach for PD estimation and capital calculation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

print("=" * 70)
print("IRB APPROACH: PD ESTIMATION AND CAPITAL CALCULATION")
print("=" * 70)

np.random.seed(42)
n = 5000

# Generate synthetic credit data
financial_ratio = np.random.normal(0, 1, n)
leverage = np.random.normal(0, 1, n)
profitability = np.random.normal(0, 1, n)

z = -3.5 + 0.8 * financial_ratio + 0.6 * leverage - 0.5 * profitability
prob_default = 1 / (1 + np.exp(-z))
defaults = (np.random.uniform(0, 1, n) < prob_default).astype(int)

data = pd.DataFrame(
    {
        "FinancialRatio": financial_ratio,
        "Leverage": leverage,
        "Profitability": profitability,
        "Default": defaults,
    }
)

print(f"\nPortfolio Statistics:")
print(f"   Observations: {n}")
print(f"   Defaults: {defaults.sum()} ({defaults.mean():.2%})")

# Train PD model
X = data[["FinancialRatio", "Leverage", "Profitability"]]
y = data["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

pd_model = LogisticRegression(random_state=42, max_iter=1000)
pd_model.fit(X_train, y_train)

data["PD"] = pd_model.predict_proba(X)[:, 1]
data["PD_IRB"] = np.maximum(data["PD"], 0.0003)  # Basel floor

print(f"\nPD Model Performance:")
print(
    f"   Training AUC: {roc_auc_score(y_train, pd_model.predict_proba(X_train)[:, 1]):.4f}"
)
print(f"   Test AUC: {roc_auc_score(y_test, pd_model.predict_proba(X_test)[:, 1]):.4f}")
print(f"   Mean PD: {data['PD_IRB'].mean():.4%}")
print(f"   Min PD (after floor): {data['PD_IRB'].min():.4%}")
print(f"   Max PD: {data['PD_IRB'].max():.4%}")


# IRB RWA calculation functions
def calculate_correlation(pd, asset_class="corporate"):
    """Calculate asset value correlation per Basel IRB formula."""
    if asset_class == "corporate":
        rho = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + 0.24 * (
            1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
        )
    elif asset_class == "retail_mortgage":
        rho = 0.15
    elif asset_class == "retail_revolving":
        rho = 0.04
    else:
        rho = 0.03 * (1 - np.exp(-35 * pd)) / (1 - np.exp(-35)) + 0.16 * (
            1 - (1 - np.exp(-35 * pd)) / (1 - np.exp(-35))
        )
    return rho


def calculate_maturity_adjustment(pd, maturity=2.5):
    """Calculate maturity adjustment b(PD) per Basel formula."""
    b_pd = (0.11852 - 0.05478 * np.log(pd)) ** 2
    ma = (1 + (maturity - 2.5) * b_pd) / (1 - 1.5 * b_pd)
    return ma


def calculate_irb_capital(pd, lgd, ead, maturity=2.5, asset_class="corporate"):
    """Calculate IRB capital requirement K per Basel formula."""
    rho = calculate_correlation(pd, asset_class)

    g_pd = stats.norm.ppf(pd)
    g_999 = stats.norm.ppf(0.999)

    n_term = stats.norm.cdf(g_pd + np.sqrt(rho) * g_999)
    k_base = lgd * n_term - pd * lgd

    if asset_class == "corporate":
        ma = calculate_maturity_adjustment(pd, maturity)
        k = k_base * ma
    else:
        k = k_base

    rwa = k * 12.5 * ead * 1.06
    capital_req = rwa * 0.08

    return {"K": k, "RWA": rwa, "Capital": capital_req, "Correlation": rho}


# Apply to portfolio
data["LGD"] = 0.45
data["EAD"] = 100000
data["Maturity"] = 2.5

irb_results = []
for idx, row in data.iterrows():
    result = calculate_irb_capital(
        pd=row["PD_IRB"],
        lgd=row["LGD"],
        ead=row["EAD"],
        maturity=row["Maturity"],
        asset_class="corporate",
    )
    irb_results.append(result)

irb_df = pd.DataFrame(irb_results)
data = pd.concat([data, irb_df], axis=1)

print(f"\nPortfolio IRB Capital:")
print(f"   Total EAD: €{data['EAD'].sum()/1e6:.1f}M")
print(f"   Total RWA: €{data['RWA'].sum()/1e6:.1f}M")
print(f"   Total Capital Required: €{data['Capital'].sum()/1e6:.2f}M")
print(f"   Average Risk Weight: {(data['RWA'].sum()/data['EAD'].sum()):.2%}")
print(f"   Average Capital K: {data['K'].mean():.4%}")

# Standardized approach comparison
print("\n" + "=" * 70)
print("COMPARISON: IRB vs STANDARDIZED APPROACH")
print("=" * 70)

data["RWA_Standardized"] = data["EAD"] * 1.00
data["Capital_Standardized"] = data["RWA_Standardized"] * 0.08

print(f"\nStandardized Approach:")
print(f"   Total RWA: €{data['RWA_Standardized'].sum()/1e6:.1f}M")
print(f"   Total Capital: €{data['Capital_Standardized'].sum()/1e6:.2f}M")
print(f"   Risk Weight: 100% (BBB rating)")

# Basel III Output Floor
output_floor = 0.725
data["RWA_Floor"] = data["RWA_Standardized"] * output_floor
data["RWA_Final"] = np.maximum(data["RWA"], data["RWA_Floor"])
data["Capital_Final"] = data["RWA_Final"] * 0.08

print(f"\nWith Basel III Output Floor (72.5%):")
print(f"   Floor RWA: €{data['RWA_Floor'].sum()/1e6:.1f}M")
print(f"   Final RWA (max of IRB and Floor): €{data['RWA_Final'].sum()/1e6:.1f}M")
print(f"   Final Capital: €{data['Capital_Final'].sum()/1e6:.2f}M")
print(f"   Floor Binding: {(data['RWA_Final'] > data['RWA']).mean():.1%} of exposures")

capital_savings = data["Capital_Standardized"].sum() - data["Capital_Final"].sum()
print(
    f"\n   Capital Savings vs Standardized: €{capital_savings/1e6:.2f}M ({capital_savings/data['Capital_Standardized'].sum():.1%})"
)

# PD backtesting
print("\n" + "=" * 70)
print("PD MODEL BACKTESTING")
print("=" * 70)

data["PD_Bucket"] = pd.cut(
    data["PD_IRB"],
    bins=[0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0],
    labels=["<0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", ">10%"],
)

backtest_results = (
    data.groupby("PD_Bucket")
    .agg({"Default": ["count", "sum", "mean"], "PD_IRB": "mean"})
    .round(4)
)

backtest_results.columns = [
    "_".join(col).strip() for col in backtest_results.columns.values
]
backtest_results = backtest_results.rename(
    columns={
        "Default_count": "N_Obligors",
        "Default_sum": "Defaults",
        "Default_mean": "Realized_DR",
        "PD_IRB_mean": "Predicted_PD",
    }
)

print("\nBacktesting by PD Bucket:")
print(backtest_results)

# Binomial test for calibration
for bucket in backtest_results.index:
    n_obs = backtest_results.loc[bucket, "N_Obligors"]
    n_defaults = backtest_results.loc[bucket, "Defaults"]
    pred_pd = backtest_results.loc[bucket, "Predicted_PD"]

    p_value = stats.binom_test(n_defaults, n_obs, pred_pd, alternative="two-sided")
    backtest_results.loc[bucket, "P_Value"] = p_value
    backtest_results.loc[bucket, "Calibration"] = "Pass" if p_value > 0.05 else "Fail"

print("\nCalibration Test (Binomial):")
print(backtest_results[["Predicted_PD", "Realized_DR", "P_Value", "Calibration"]])

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("IRB approach implementation complete:")
print(
    f"• PD model AUC: {roc_auc_score(y_test, pd_model.predict_proba(X_test)[:, 1]):.3f}"
)
print(
    f"• IRB capital: €{data['Capital'].sum()/1e6:.1f}M vs Standardized: €{data['Capital_Standardized'].sum()/1e6:.1f}M"
)

if __name__ == "__main__":
    print("\nIRB approach execution complete.")
