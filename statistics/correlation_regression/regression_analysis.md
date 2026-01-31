# Regression Analysis

## 9.1 Concept Skeleton
**Definition:** Statistical technique predicting outcome variable (y) from predictor(s) (x) via linear combination  
**Purpose:** Quantify relationships, predict future values, control for confounders  
**Prerequisites:** Linear algebra, correlation, hypothesis testing

## 9.2 Comparative Framing
| Method | Linear Regression | Logistic Regression | Multiple Regression |
|--------|------------------|-------------------|-------------------|
| **Outcome** | Continuous (unbounded) | Binary (0/1) | Multiple y or multiple x |
| **Assumptions** | Normality of residuals | Logistic link function | Extended linearity |
| **Interpretation** | Unit change in x → β change in y | Unit change in x → odds multiply by e^β | Holds other x constant |

## 9.3 Examples + Counterexamples

**Simple Example:**  
House price ~ square footage: For each +100 sq ft, price increases $50,000

**Failure Case:**  
Assuming linear relationship when true is quadratic (x²). Residuals show pattern → wrong model

**Edge Case:**  
Multicollinearity: Two predictors highly correlated → coefficients unstable, hard to interpret

## 9.4 Layer Breakdown
```
Regression Components:
├─ Model: y = β₀ + β₁x₁ + ... + βₖxₖ + ε
├─ β₀: Intercept (y when all x=0)
├─ βⱼ: Slope of xⱼ (effect holding others constant)
├─ ε: Error (residual variation)
├─ Assumptions:
│   ├─ Linearity: True relationship linear
│   ├─ Independence: Observations independent
│   ├─ Homoscedasticity: Constant variance of errors
│   ├─ Normality: Residuals normally distributed
│   └─ No multicollinearity: Predictors not too correlated
└─ Fit: R² (variance explained), F-test (overall significance)
```

## 9.5 Mini-Project
Fit and evaluate regression:
```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = 2 + 1.5*x.flatten() + np.random.normal(0, 1, 8)

# Fit model
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Evaluate
r2 = model.score(x, y)
residuals = y - y_pred

print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"R²: {r2:.3f}")

# Plot
plt.scatter(x, y, label='Actual')
plt.plot(x, y_pred, 'r-', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit')
plt.show()

# Diagnostic plots
fig, axes = plt.subplots(2, 2)
axes[0, 0].scatter(y_pred, residuals)
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 1].hist(residuals, bins=5)
axes[0, 1].set_title('Residual Distribution')
axes[1, 0].plot(residuals)
axes[1, 0].set_title('Residuals Over Time')
# Q-Q plot would go in axes[1,1]
plt.tight_layout()
plt.show()
```

## 9.6 Challenge Round
When is regression the wrong tool?
- Severely non-linear relationships (use splines, GAM)
- Clustered data (use mixed models, GEE)
- Classification outcome (use logistic)
- Time-series (use ARIMA, not naive regression)
- Extreme outliers (use robust regression)

## 9.7 Key References
- [Regression Assumptions Visual](https://en.wikipedia.org/wiki/Linear_regression)
- [Multicollinearity Diagnosis](https://stats.stackexchange.com/questions/tagged/multicollinearity)
- [Regression Diagnostics](https://www.r-bloggers.com/linear-regression-assumptions/)

---
**Status:** Core predictive method | **Complements:** Correlation, Causation, Hypothesis Testing
