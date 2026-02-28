# %% [markdown]
# # Meta Topics And Foundations  Overview & Setup
# This interactive script demonstrates an end-to-end time-series workflow from synthetic data generation to evaluation and interpretation.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
print("Environment initialized for Meta Topics And Foundations")

# %% [markdown]
# ## Data Generation
# Create synthetic time-series predictors with trend, seasonality, and noise to emulate finance-style dynamics.

# %%
n = 720
t = np.arange(n)
trend = 0.0008 * t
season = 0.15 * np.sin(2 * np.pi * t / 30)
shock = np.random.normal(0, 0.10, n)
feature = 0.6 * np.roll(season, 1) + np.random.normal(0, 0.08, n)

y = 0.02 + trend + season + 0.35 * feature + shock

df = pd.DataFrame({"t": t, "feature": feature, "y": y})
print(df.head())
print("Data shape:", df.shape)

# %% [markdown]
# ## Model Implementation
# Build lagged features and fit baseline linear and ridge models for one-step-ahead forecasting.

# %%
df_model = df.copy()
df_model["y_lag1"] = df_model["y"].shift(1)
df_model["y_lag5_mean"] = df_model["y"].rolling(5).mean().shift(1)
df_model = df_model.dropna().reset_index(drop=True)

X = df_model[["feature", "y_lag1", "y_lag5_mean"]].values
y_vec = df_model["y"].values

split = int(0.8 * len(df_model))
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
# Evaluate holdout performance using MAE, RMSE, and $R^2$; also inspect directional accuracy.

# %%
def evaluate(model, X_eval, y_eval, label):
    pred = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, pred)
    rmse = np.sqrt(mean_squared_error(y_eval, pred))
    r2 = r2_score(y_eval, pred)
    direction = np.mean(np.sign(np.diff(y_eval)) == np.sign(np.diff(pred)))
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Directional Acc: {direction:.4f}")
    return pred

pred_ols = evaluate(ols, X_test, y_test, "OLS")
pred_ridge = evaluate(ridge, X_test, y_test, "Ridge")

# %% [markdown]
# ## Visualization & Interpretation
# Visualize actual vs predicted trajectories and residual distribution to assess fit stability.

# %%
residuals = y_test - pred_ols
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y_test, label="Actual", linewidth=1.5)
axes[0].plot(pred_ols, label="Predicted (OLS)", linewidth=1.2)
axes[0].set_title("Holdout Actual vs Predicted")
axes[0].legend()

axes[1].hist(residuals, bins=25, alpha=0.8)
axes[1].set_title("Residual Distribution (OLS)")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary & Deployment
# Key takeaways: lagged structure and regularization improve stability; diagnostics should be monitored with rolling windows.
# Deployment readiness checklist: point-in-time data controls, drift thresholds, challenger model, and automated alerting.
