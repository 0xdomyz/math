# Cross-Validation

## 1. Concept Skeleton
**Definition:** Model evaluation technique partitioning data into training and testing sets to assess generalization performance  
**Purpose:** Estimate out-of-sample prediction error, prevent overfitting, select optimal model complexity  
**Prerequisites:** Model training concepts, bias-variance tradeoff, statistical learning theory

## 2. Comparative Framing
| Method | k-Fold CV | Leave-One-Out (LOO) | Train-Test Split |
|--------|-----------|---------------------|------------------|
| **Iterations** | k times (typically 5-10) | n times | 1 time |
| **Computation** | Moderate | Expensive (n models) | Fast |
| **Variance** | Lower than LOO | High variance | Highest variance |
| **Bias** | Slight upward bias | Nearly unbiased | Depends on split |

## 3. Examples + Counterexamples

**Simple Example:**  
Polynomial degree selection: 5-fold CV shows degree 3 has lowest error, degree 7 overfits

**Failure Case:**  
Time series with standard k-fold: Trains on future, tests on past → leakage, use time series split

**Edge Case:**  
Imbalanced classes (1% positive): Random splits may omit rare class → use stratified CV

## 4. Layer Breakdown
```
Cross-Validation Process:
├─ Data Partitioning:
│   ├─ k-Fold: Split into k equal folds
│   ├─ Stratified: Maintain class proportions in each fold
│   ├─ Time Series: Sequential splits respecting temporal order
│   └─ Leave-P-Out: All combinations of leaving out p observations
├─ Training Loop (for each fold i):
│   ├─ Training Set: All folds except fold i
│   ├─ Validation Set: Fold i (held out)
│   ├─ Fit Model: Train on training set
│   └─ Evaluate: Predict on validation set, compute error Eᵢ
├─ Aggregation:
│   ├─ CV Score: Mean(E₁, E₂, ..., Eₖ)
│   ├─ Standard Error: SD(Eᵢ) / √k
│   └─ One-SE Rule: Choose simplest model within 1 SE of minimum
└─ Final Model: Retrain on full dataset with selected hyperparameters
```

**Interaction:** Split → Train → Validate → Rotate → Aggregate errors → Select model

## 5. Mini-Project
Compare model complexity using cross-validation:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate non-linear data with noise
np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
y = 2 + 0.5*x - 0.05*x**2 + np.random.normal(0, 1.5, n)

X = x.reshape(-1, 1)

# Test different polynomial degrees
degrees = range(1, 12)
cv_scores = []
loo_scores = []

# 5-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # k-Fold CV
    scores = cross_val_score(model, X, y, cv=kfold, 
                            scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())
    
    # Leave-One-Out (only for low degrees to save time)
    if degree <= 6:
        loo = LeaveOneOut()
        loo_score = cross_val_score(model, X, y, cv=loo,
                                   scoring='neg_mean_squared_error')
        loo_scores.append(-loo_score.mean())

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fits for different degrees
sample_degrees = [1, 3, 7, 11]
x_plot = np.linspace(0, 10, 200).reshape(-1, 1)

for deg in sample_degrees:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(x_plot)
    
    axes[0, 0].plot(x_plot, y_plot, label=f'Degree {deg}', linewidth=2)

axes[0, 0].scatter(x, y, color='black', alpha=0.5, s=30, label='Data')
axes[0, 0].set_title('Polynomial Fits')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].set_ylim(-5, 15)

# Plot 2: CV scores vs degree
axes[0, 1].plot(degrees, cv_scores, 'o-', linewidth=2, markersize=8)
best_degree = degrees[np.argmin(cv_scores)]
axes[0, 1].axvline(best_degree, color='r', linestyle='--', 
                   label=f'Best (degree {best_degree})')
axes[0, 1].set_title('5-Fold Cross-Validation Error')
axes[0, 1].set_xlabel('Polynomial Degree')
axes[0, 1].set_ylabel('Mean Squared Error')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

print(f"Best degree (5-fold CV): {best_degree}")
print(f"Minimum CV error: {min(cv_scores):.3f}")

# Plot 3: Compare k-fold vs LOO
axes[1, 0].plot(degrees[:6], cv_scores[:6], 'o-', label='5-Fold CV', linewidth=2)
axes[1, 0].plot(degrees[:6], loo_scores, 's-', label='Leave-One-Out', linewidth=2)
axes[1, 0].set_title('k-Fold vs Leave-One-Out')
axes[1, 0].set_xlabel('Polynomial Degree')
axes[1, 0].set_ylabel('Mean Squared Error')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Learning curve (training size vs error)
from sklearn.model_selection import learning_curve

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    make_pipeline(PolynomialFeatures(3), LinearRegression()),
    X, y, train_sizes=train_sizes, cv=5,
    scoring='neg_mean_squared_error', random_state=42
)

axes[1, 1].plot(train_sizes_abs, -train_scores.mean(axis=1), 
                'o-', label='Training error', linewidth=2)
axes[1, 1].plot(train_sizes_abs, -val_scores.mean(axis=1), 
                's-', label='Validation error', linewidth=2)
axes[1, 1].fill_between(train_sizes_abs, 
                        -val_scores.mean(axis=1) - val_scores.std(axis=1),
                        -val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.2)
axes[1, 1].set_title('Learning Curve (Degree 3)')
axes[1, 1].set_xlabel('Training Set Size')
axes[1, 1].set_ylabel('Mean Squared Error')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate stratified CV for classification
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

X_class, y_class = make_classification(n_samples=100, n_classes=2, 
                                      weights=[0.9, 0.1], random_state=42)

# Standard vs Stratified
kfold_standard = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

scores_standard = cross_val_score(clf, X_class, y_class, cv=kfold_standard)
scores_stratified = cross_val_score(clf, X_class, y_class, cv=kfold_stratified)

print(f"\nImbalanced Classification (10% positive class):")
print(f"Standard k-Fold: {scores_standard.mean():.3f} ± {scores_standard.std():.3f}")
print(f"Stratified k-Fold: {scores_stratified.mean():.3f} ± {scores_stratified.std():.3f}")
```

## 6. Challenge Round
When is cross-validation the wrong tool?
- Tiny datasets (n<30): High variance estimates, consider LOO or bootstrap
- Time series forecasting: Use time series split, not random k-fold
- Expensive model training: Each fold costly, consider single validation set
- Data leakage risks: Feature engineering must be inside CV loop
- Nested dependencies: Grouped observations (e.g., patients with multiple measures) need grouped CV

## 7. Key References
- [Cross-Validation Explained (scikit-learn)](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hastie et al., Elements of Statistical Learning (Chapter 7)](https://hastie.su.domains/ElemStatLearn/)
- [Time Series Split Visualization](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

---
**Status:** Core model evaluation method | **Complements:** Hypothesis Testing, Bootstrap, Bias-Variance Tradeoff
