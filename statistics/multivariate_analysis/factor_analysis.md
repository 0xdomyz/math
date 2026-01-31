# Factor Analysis

## 1. Concept Skeleton
**Definition:** Statistical method identifying unobserved latent variables (factors) explaining correlations among observed variables  
**Purpose:** Theory-building, construct validation, data simplification by discovering underlying structure  
**Prerequisites:** Covariance/correlation matrices, latent variable concepts, PCA background

## 2. Comparative Framing
| Method | Factor Analysis | PCA | Cluster Analysis |
|--------|----------------|-----|------------------|
| **Model** | Latent factors cause observations | Observations create components | Group similar observations |
| **Focus** | Explain correlations | Maximize variance | Minimize within-group distance |
| **Causality** | Factors → variables | Variables → components | No causal structure |
| **Unique Variance** | Modeled separately | Included in components | Not applicable |

## 3. Examples + Counterexamples

**Simple Example:**  
Survey: 20 questions about personality. FA identifies 5 factors (Big Five): extraversion, agreeableness, conscientiousness, neuroticism, openness

**Failure Case:**  
Variables uncorrelated: Factor model inappropriate, factors cannot explain relationships that don't exist

**Edge Case:**  
Two highly correlated items only: Factor perfectly determined, but adding more items reveals factor is unstable

## 4. Layer Breakdown
```
Factor Analysis Components:
├─ Model Equation:
│   X = Λ F + ε
│   ├─ X: Observed variables (p×1)
│   ├─ Λ: Factor loadings matrix (p×k)
│   ├─ F: Latent factors (k×1)
│   └─ ε: Unique/error variance (p×1)
├─ Variance Decomposition:
│   Var(Xᵢ) = Communality + Uniqueness
│   ├─ Communality: h²ᵢ = Σλ²ᵢⱼ (shared variance)
│   └─ Uniqueness: ψᵢ (specific + error variance)
├─ Estimation Methods:
│   ├─ Maximum Likelihood: Statistical inference, fit tests
│   ├─ Principal Axis: Iterative estimation
│   └─ Minimum Residual: Minimize off-diagonal residuals
├─ Factor Retention:
│   ├─ Kaiser criterion: Eigenvalues > 1
│   ├─ Scree plot: Elbow point
│   ├─ Parallel analysis: Compare to random data
│   └─ Theory-driven: A priori hypotheses
├─ Rotation (for interpretability):
│   ├─ Orthogonal: Varimax (maximize variance), Quartimax
│   └─ Oblique: Promax, Oblimin (allow correlation)
└─ Interpretation:
    ├─ Loadings: Correlation between variable and factor
    ├─ Communalities: Proportion explained by factors
    └─ Factor scores: Individual's values on factors
```

**Interaction:** Specify factors → Estimate loadings → Rotate → Interpret → Validate structure

## 5. Mini-Project
Exploratory Factor Analysis with rotation:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import seaborn as sns

# Generate simulated psychological test data
np.random.seed(42)
n = 300

# True factors
extraversion = np.random.normal(0, 1, n)
neuroticism = np.random.normal(0, 1, n)

# Observed variables (items)
# Extraversion items
item1 = 0.8 * extraversion + np.random.normal(0, 0.6, n)  # "I enjoy parties"
item2 = 0.7 * extraversion + np.random.normal(0, 0.7, n)  # "I talk a lot"
item3 = 0.6 * extraversion + np.random.normal(0, 0.8, n)  # "I meet new people"

# Neuroticism items
item4 = 0.75 * neuroticism + np.random.normal(0, 0.65, n)  # "I worry often"
item5 = 0.80 * neuroticism + np.random.normal(0, 0.6, n)   # "I get stressed"
item6 = 0.70 * neuroticism + np.random.normal(0, 0.7, n)   # "I feel anxious"

# Create dataset
data = pd.DataFrame({
    'sociable': item1,
    'talkative': item2,
    'outgoing': item3,
    'worried': item4,
    'stressed': item5,
    'anxious': item6
})

print("Data shape:", data.shape)
print("\nCorrelation Matrix:")
print(data.corr().round(2))

# Test factorability
chi_square, p_value = calculate_bartlett_sphericity(data)
print(f"\nBartlett's Test of Sphericity:")
print(f"  Chi-square: {chi_square:.2f}, p-value: {p_value:.6f}")
print(f"  Conclusion: {'Suitable for FA' if p_value < 0.05 else 'Not suitable'}")

kmo_all, kmo_model = calculate_kmo(data)
print(f"\nKaiser-Meyer-Olkin (KMO) Test:")
print(f"  Overall KMO: {kmo_model:.3f}")
print(f"  Interpretation: {'Excellent' if kmo_model > 0.9 else 'Good' if kmo_model > 0.8 else 'Mediocre' if kmo_model > 0.7 else 'Poor'}")

# Determine number of factors (scree plot)
fa_test = FactorAnalyzer(n_factors=6, rotation=None)
fa_test.fit(data)
ev, _ = fa_test.get_eigenvalues()

# Fit Factor Analysis with 2 factors
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(data)

# Extract results
loadings = fa.loadings_
communalities = fa.get_communalities()
uniqueness = 1 - communalities

print("\nFactor Loadings (after Varimax rotation):")
loadings_df = pd.DataFrame(
    loadings,
    index=data.columns,
    columns=['Factor 1', 'Factor 2']
)
print(loadings_df.round(3))

print("\nCommunalities (variance explained by factors):")
comm_df = pd.DataFrame({
    'Communality': communalities,
    'Uniqueness': uniqueness
}, index=data.columns)
print(comm_df.round(3))

# Variance explained
variance = fa.get_factor_variance()
print("\nVariance Explained:")
variance_df = pd.DataFrame(
    variance,
    index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
    columns=['Factor 1', 'Factor 2']
)
print(variance_df.round(3))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Scree plot
axes[0, 0].plot(range(1, len(ev)+1), ev, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axhline(1, color='r', linestyle='--', label='Kaiser criterion')
axes[0, 0].set_xlabel('Factor Number')
axes[0, 0].set_ylabel('Eigenvalue')
axes[0, 0].set_title('Scree Plot')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Factor loadings heatmap
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, ax=axes[0, 1],
            cbar_kws={'label': 'Loading'})
axes[0, 1].set_title('Factor Loadings (Varimax Rotation)')

# Plot 3: Factor plot (loading space)
axes[0, 2].scatter(loadings[:, 0], loadings[:, 1], s=100, alpha=0.6)
for i, var in enumerate(data.columns):
    axes[0, 2].annotate(var, (loadings[i, 0], loadings[i, 1]),
                       fontsize=9, ha='center')
    axes[0, 2].arrow(0, 0, loadings[i, 0]*0.9, loadings[i, 1]*0.9,
                    head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.5)

axes[0, 2].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[0, 2].axvline(0, color='k', linestyle='-', linewidth=0.5)
axes[0, 2].set_xlabel('Factor 1 (Extraversion)')
axes[0, 2].set_ylabel('Factor 2 (Neuroticism)')
axes[0, 2].set_title('Factor Loading Plot')
axes[0, 2].grid(alpha=0.3)
axes[0, 2].set_xlim(-1, 1)
axes[0, 2].set_ylim(-1, 1)

# Plot 4: Communalities
axes[1, 0].barh(data.columns, communalities, alpha=0.7)
axes[1, 0].set_xlabel('Communality (h²)')
axes[1, 0].set_title('Variance Explained by Factors')
axes[1, 0].set_xlim(0, 1)
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 5: Variance explained by factor
variance_explained = variance[1, :]
axes[1, 1].bar(['Factor 1', 'Factor 2'], variance_explained, alpha=0.7)
axes[1, 1].set_ylabel('Proportion of Variance')
axes[1, 1].set_title('Variance Explained by Each Factor')
axes[1, 1].set_ylim(0, 0.5)
axes[1, 1].grid(axis='y', alpha=0.3)

# Plot 6: Correlation matrix
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, ax=axes[1, 2], square=True,
            cbar_kws={'label': 'Correlation'})
axes[1, 2].set_title('Observed Variable Correlations')

plt.tight_layout()
plt.show()

# Factor scores
factor_scores = fa.transform(data)
print(f"\nFactor scores computed for {factor_scores.shape[0]} observations")
print("First 5 observations:")
print(pd.DataFrame(factor_scores[:5], columns=['Factor 1', 'Factor 2']).round(3))

# Compare with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(data)

print(f"\nPCA vs FA:")
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
print(f"FA variance explained: {variance[2, -1]:.3f}")
```

## 6. Challenge Round
When is Factor Analysis the wrong tool?
- Variables uncorrelated: No latent structure to extract
- Formative constructs: Factors should cause variables, not vice versa
- Small sample (n<100): Unstable estimates, use larger sample or fewer factors
- Prediction focus: Use PCA or regression instead
- Non-linear relationships: Standard FA assumes linearity

## 7. Key References
- [Factor Analysis Overview (Wikipedia)](https://en.wikipedia.org/wiki/Factor_analysis)
- [factor_analyzer Python Library](https://factor-analyzer.readthedocs.io/)
- [Exploratory vs Confirmatory FA](https://stats.idre.ucla.edu/spss/seminars/introduction-to-factor-analysis/)

---
**Status:** Core latent variable method | **Complements:** PCA, Structural Equation Modeling, Survey Design
