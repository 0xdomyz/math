"""
Default Probability (PD) Modeling
Extracted from default_probability.md

Implements logistic regression PD models, calibration analysis, ROC curves,
and multi-year PD calculations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def main_default_probability():
    print("=== PD Model Performance ===")

    # Simulate borrower data
    n_borrowers = 5000
    data = pd.DataFrame(
        {
            "debt_to_income": np.random.normal(0.4, 0.2, n_borrowers),
            "credit_score": np.random.normal(700, 100, n_borrowers),
            "employment_years": np.random.exponential(8, n_borrowers),
            "age": np.random.normal(45, 15, n_borrowers),
            "loan_amount": np.random.exponential(200000, n_borrowers),
        }
    )

    # Generate synthetic defaults
    logit_score = (
        -3
        + 1.5 * (data["debt_to_income"] - 0.4)
        + -0.01 * (data["credit_score"] - 700)
        + -0.05 * (data["employment_years"] - 5)
        + 0.02 * (data["age"] - 45)
    )

    prob_default = 1 / (1 + np.exp(-logit_score))
    default = (np.random.rand(n_borrowers) < prob_default).astype(int)
    data["default"] = default

    # Split into train/test
    train_idx = np.random.rand(n_borrowers) < 0.7
    X_train = data[train_idx].drop("default", axis=1)
    y_train = data[train_idx]["default"]
    X_test = data[~train_idx].drop("default", axis=1)
    y_test = data[~train_idx]["default"]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    pd_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    pd_pred_test = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Training default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    print(f"Mean predicted PD (train): {pd_pred_train.mean():.2%}")
    print(f"Mean predicted PD (test): {pd_pred_test.mean():.2%}")

    train_auc = roc_auc_score(y_train, pd_pred_train)
    test_auc = roc_auc_score(y_test, pd_pred_test)
    print(f"Training AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")

    # Calibration check
    print("\n=== PD Calibration ===")
    pd_bins = np.linspace(0, 1, 11)

    print("PD Bin | Predicted | Actual | Calibration")
    for i in range(len(pd_bins) - 1):
        mask = (pd_pred_test >= pd_bins[i]) & (pd_pred_test < pd_bins[i + 1])
        if mask.sum() > 0:
            actual = y_test[mask].mean()
            predicted = pd_pred_test[mask].mean()
            calibrated = "✓" if abs(predicted - actual) < 0.02 else "✗"
            print(
                f"{pd_bins[i]:5.1%}-{pd_bins[i+1]:5.1%} | {predicted:8.2%} | {actual:6.2%} | {calibrated}"
            )

    # Feature importance
    print("\n=== PD Model Coefficients ===")
    feature_names = X_train.columns
    for feature, coef in zip(feature_names, model.coef_[0]):
        print(f"{feature:20s}: {coef:+.4f}")

    # Multi-year PD calculation
    print("\n=== Multi-Year PD ===")
    annual_pd = 0.01  # 1% annual
    years = [1, 2, 3, 5, 10]
    for year in years:
        survival_prob = (1 - annual_pd) ** year
        cumulative_pd = 1 - survival_prob
        print(f"{year:2d}-year PD: {cumulative_pd:.2%}")


if __name__ == "__main__":
    main_default_probability()
