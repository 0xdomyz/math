# Nonparametric Regression

## 1. Concept Skeleton
**Definition:** Flexible regression methods allowing complex relationships without assuming specific functional form  
**Purpose:** Model nonlinear patterns, avoid parametric bias, explore data-driven relationships  
**Prerequisites:** Linear regression, residual analysis, bias-variance tradeoff, smoothing concepts

## 2. Comparative Framing
| Method | Linear Regression | Polynomial Regression | LOWESS | Splines |
|--------|------------------|---------------------|--------|---------|
| **Form** | y = β₀ + β₁x | y = β₀ + β₁x + β₂x² + ... | Local weighted avg | Piecewise polynomials |
| **Flexibility** | Fixed slope | Moderate (fixed degree) | High (data-adaptive) | High (knot-controlled) |
| **Interpretability** | Easy (slope) | Moderate | Difficult | Moderate |
| **Overfitting Risk** | Low | High (large degree) | Tunable (bandwidth) | Tunable (knots, penalty) |

## 3. Examples + Counterexamples

**Simple Example:**  
Temperature vs ice cream sales: Linear misses summer peak. LOWESS captures curved relationship without specifying y = ax² + bx + c.

**Failure Case:**  
Extrapolation beyond data range. Nonparametric methods unreliable outside observed x values; no global model.

**Edge Case:**  
Sparse data regions. LOWESS/splines produce unstable estimates where few observations exist; smoothness parameters critical.

## 4. Layer Breakdown
```
Nonparametric Regression Toolkit:
├─ LOWESS (Locally Weighted Scatterplot Smoothing):
│   ├─ For each x, fit weighted regression to nearby points
│   ├─ Bandwidth (span): Controls locality (small = wiggly, large = smooth)
│   ├─ Weights: Tricube kernel (closer points → higher weight)
│   ├─ Robust variant: Iteratively downweight outliers
│   └─ Output: Smooth curve through data
├─ Splines (Piecewise Polynomial):
│   ├─ Divide x-range into intervals (knots)
│   ├─ Fit polynomial in each segment
│   ├─ Constrain: Continuous derivatives at knots
│   ├─ Types: Natural cubic spline, B-splines, smoothing splines
│   └─ Penalization: Regularize curvature (λ controls smoothness)
├─ Kernel Regression:
│   ├─ Similar to LOWESS but different weighting schemes
│   ├─ Nadaraya-Watson estimator: ŷ(x) = Σ K((x-xᵢ)/h) yᵢ / Σ K((x-xᵢ)/h)
│   └─ Bandwidth h: Key tuning parameter
└─ Selection:
    ├─ Cross-validation: Choose bandwidth/knots minimizing prediction error
    ├─ AIC/BIC: Balance fit vs complexity
    └─ Visual inspection: Check residual patterns
```

## 5. Mini-Project
Fit LOWESS and spline to nonlinear data:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate nonlinear data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = np.sin(x) + 0.5 * x  # True nonlinear function
y = y_true + np.random.normal(0, 0.3, 100)  # Add noise

# Linear regression (for comparison)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)
y_linear = lr.predict(x.reshape(-1, 1))

# LOWESS (multiple bandwidths)
lowess_fit_1 = lowess(y, x, frac=0.1)  # Small bandwidth (wiggly)
lowess_fit_2 = lowess(y, x, frac=0.3)  # Medium bandwidth
lowess_fit_3 = lowess(y, x, frac=0.6)  # Large bandwidth (smooth)

# Spline (smoothing spline)
spline = UnivariateSpline(x, y, s=5)  # s controls smoothness
y_spline = spline(x)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Linear vs True
axes[0, 0].scatter(x, y, alpha=0.3, label='Data')
axes[0, 0].plot(x, y_true, 'g-', linewidth=2, label='True function')
axes[0, 0].plot(x, y_linear, 'r--', linewidth=2, label='Linear fit')
axes[0, 0].set_title('Linear Regression (Parametric)')
axes[0, 0].legend()

# LOWESS variations
axes[0, 1].scatter(x, y, alpha=0.3, label='Data')
axes[0, 1].plot(lowess_fit_1[:, 0], lowess_fit_1[:, 1], label='LOWESS (frac=0.1)')
axes[0, 1].plot(lowess_fit_2[:, 0], lowess_fit_2[:, 1], label='LOWESS (frac=0.3)')
axes[0, 1].plot(lowess_fit_3[:, 0], lowess_fit_3[:, 1], label='LOWESS (frac=0.6)')
axes[0, 1].set_title('LOWESS with Different Bandwidths')
axes[0, 1].legend()

# Spline fit
axes[1, 0].scatter(x, y, alpha=0.3, label='Data')
axes[1, 0].plot(x, y_true, 'g-', linewidth=2, label='True function')
axes[1, 0].plot(x, y_spline, 'purple', linewidth=2, label='Smoothing spline')
axes[1, 0].set_title('Smoothing Spline')
axes[1, 0].legend()

# Residuals comparison
lowess_resid = y - lowess_fit_2[:, 1]
spline_resid = y - y_spline
axes[1, 1].scatter(lowess_fit_2[:, 1], lowess_resid, alpha=0.5, label='LOWESS residuals')
axes[1, 1].scatter(y_spline, spline_resid, alpha=0.5, label='Spline residuals')
axes[1, 1].axhline(0, color='black', linestyle='--')
axes[1, 1].set_xlabel('Fitted values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Diagnostics')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Performance metrics
from sklearn.metrics import mean_squared_error, r2_score
print("Mean Squared Error (MSE):")
print(f"  Linear: {mean_squared_error(y, y_linear):.4f}")
print(f"  LOWESS: {mean_squared_error(y, lowess_fit_2[:, 1]):.4f}")
print(f"  Spline: {mean_squared_error(y, y_spline):.4f}")
```

## 6. Challenge Round
When is nonparametric regression the wrong tool?
- Small sample size (insufficient data for flexible fitting)
- Need interpretable coefficients (β values for reporting)
- Extrapolation required (no parametric form outside data)
- High-dimensional predictors (curse of dimensionality)
- Theory-driven functional form known (use parametric)

## 7. Key References
- [LOWESS Cleveland 1979](https://www.jstor.org/stable/2286407)
- [Smoothing Splines Tutorial](https://en.wikipedia.org/wiki/Smoothing_spline)
- [Generalized Additive Models (GAM)](https://en.wikipedia.org/wiki/Generalized_additive_model)
- [Kernel Regression Explained](https://en.wikipedia.org/wiki/Kernel_regression)

---
**Status:** Flexible modeling framework | **Complements:** Linear Regression, GLM, Cross-Validation
