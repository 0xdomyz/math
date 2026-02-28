# %% [markdown]
# # Treatment Effects Program Evaluation  Overview & Setup
# This notebook-style script builds a minimal end-to-end econometrics workflow and reports core diagnostics.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
print("Environment initialized for Treatment Effects Program Evaluation")

# %% [markdown]
# ## Data Generation
# Generate synthetic finance-style predictors and target with signal plus noise.

# %%
n = 600
x1 = np.random.normal(0, 1, n)
x2 = 0.6 * x1 + np.random.normal(0, 0.8, n)
noise = np.random.normal(0, 0.7, n)
y = 0.2 + 0.8 * x1 - 0.4 * x2 + noise

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
print(df.head())

# %% [markdown]
# ## Model Implementation
# Fit both OLS-style linear regression and a regularized benchmark.

# %%
X = df[["x1", "x2"]].values
y_vec = df["y"].values

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_vec[:split], y_vec[split:]

ols = LinearRegression()
ridge = Ridge(alpha=1.0)

ols.fit(X_train, y_train)
ridge.fit(X_train, y_train)
print("OLS coefficients:", ols.coef_, "intercept:", ols.intercept_)
print("Ridge coefficients:", ridge.coef_, "intercept:", ridge.intercept_)

# %% [markdown]
# ## Training & Evaluation
# Compare models using MAE, RMSE, and ^2$ on holdout data.

# %%
def evaluate(model, X_eval, y_eval, label):
    pred = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, pred)
    rmse = np.sqrt(mean_squared_error(y_eval, pred))
    r2 = r2_score(y_eval, pred)
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return pred

pred_ols = evaluate(ols, X_test, y_test, "OLS")
pred_ridge = evaluate(ridge, X_test, y_test, "Ridge")

# %% [markdown]
# ## Visualization & Interpretation
# Plot actual vs predicted and residual behavior to inspect fit quality.

# %%
residuals = y_test - pred_ols
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_test, pred_ols, alpha=0.7)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
axes[0].set_title("Actual vs Predicted (OLS)")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

axes[1].hist(residuals, bins=20, alpha=0.8)
axes[1].set_title("Residual Distribution (OLS)")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary & Deployment
# Key takeaways: OLS is interpretable and fast; Ridge can improve stability under collinearity.
# Deployment readiness checklist: data validation, drift monitoring, periodic retraining, and performance alerts.
