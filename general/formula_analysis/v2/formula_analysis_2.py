"""
Logistic Regression Probability Formula Deep Dive Analysis
Formula: p = 1/(1 + exp(-(β₀ + β·x)))
Run each cell interactively with Shift+Enter
"""

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %% PART 0: Linear Predictor (β₀ + β·x)
# Linear combination of intercept and weighted input
# Domain: (-∞, ∞), Range: (-∞, ∞), Shape: Linear

x_values = np.linspace(-10, 10, 500)
beta0 = 0  # Intercept (threshold)
beta = 1  # Slope (effect size)
z = beta0 + beta * x_values

# Properties: z represents the log-odds (logit) of the outcome
# β₀ = 0: No bias (p=0.5 when x=0)
# β > 0: Positive relationship (higher x → higher probability)
# β < 0: Negative relationship (higher x → lower probability)

plt.figure(figsize=(10, 6))
plt.plot(x_values, z, "b-", linewidth=2, label=f"z = {beta0} + {beta}·x")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="z=0 (threshold)")
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x (Predictor)", fontsize=11)
plt.ylabel("z (Log-odds)", fontsize=11)
plt.title("Part 0: Linear Predictor z = β₀ + β·x", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Motivation: Creates unbounded decision boundary that separates classes

# %% PART 1: Negation (-z)
# Flips the sign for exponential transformation
# Domain: (-∞, ∞), Range: (-∞, ∞), Shape: Linear (reflected)

neg_z = -z

# Properties: Negation prepares for exponential decay behavior
# When z is large positive → -z is large negative → exp(-z) near 0
# When z is large negative → -z is large positive → exp(-z) very large

plt.figure(figsize=(10, 6))
plt.plot(x_values, z, "b-", linewidth=2, alpha=0.6, label="z")
plt.plot(x_values, neg_z, "r-", linewidth=2, label="-z")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x", fontsize=11)
plt.ylabel("Value", fontsize=11)
plt.title("Part 1: Negation -z (Reflection)", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Motivation: Creates correct orientation for sigmoid (high x → high p)

# %% PART 2: Exponential (exp(-z))
# Transforms linear to exponential decay/growth
# Domain: (-∞, ∞), Range: (0, ∞), Shape: Exponential

exp_neg_z = np.exp(-z)

# Properties: Exponential creates nonlinear transformation
# z → +∞: exp(-z) → 0 (very small positive numbers)
# z = 0: exp(-z) = 1 (neutral point)
# z → -∞: exp(-z) → +∞ (very large positive numbers)

plt.figure(figsize=(10, 6))
plt.plot(x_values, exp_neg_z, "g-", linewidth=2, label="exp(-z)")
plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="exp(0) = 1")
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x", fontsize=11)
plt.ylabel("exp(-z)", fontsize=11)
plt.title("Part 2: Exponential Transformation exp(-z)", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([0, 10])
plt.show()

# Motivation: Converts linear scale to odds ratio (always positive)

# %% PART 3: Normalization (1 + exp(-z))
# Shifts exponential up by 1 for division stability
# Domain: (-∞, ∞), Range: (1, ∞), Shape: Exponential shifted

normalized = 1 + exp_neg_z

# Properties: Adding 1 ensures denominator always > 1
# z → +∞: 1 + exp(-z) → 1 (approaches minimum)
# z = 0: 1 + exp(-z) = 2 (midpoint)
# z → -∞: 1 + exp(-z) → +∞ (grows unbounded)

plt.figure(figsize=(10, 6))
plt.plot(x_values, normalized, "orange", linewidth=2, label="1 + exp(-z)")
plt.axhline(y=2, color="gray", linestyle="--", alpha=0.5, label="z=0 → 2")
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x", fontsize=11)
plt.ylabel("1 + exp(-z)", fontsize=11)
plt.title("Part 3: Normalization 1 + exp(-z)", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([0, 10])
plt.show()

# Motivation: Prepares for probability conversion (denominator in odds formula)

# %% PART 4: Logistic/Sigmoid (Complete Formula) p = 1/(1 + exp(-z))
# Final reciprocal creates bounded probability
# Domain: (-∞, ∞), Range: (0, 1), Shape: Sigmoid (S-curve)

p = 1 / (1 + exp_neg_z)

# Properties: Complete logistic function with classic S-shape
# x → -∞: p → 0 (very unlikely)
# x = 0 (z=0): p = 0.5 (50% probability, decision boundary)
# x → +∞: p → 1 (very likely)
# Symmetric around inflection point at p=0.5

# Derivative: p'(z) = p(1-p) (maximum at p=0.5)
derivative = p * (1 - p)

plt.figure(figsize=(12, 7))
plt.plot(x_values, p, "purple", linewidth=3, label="p = 1/(1+exp(-z))")
plt.axhline(
    y=0.5, color="gray", linestyle="-", alpha=0.6, label="Decision boundary (p=0.5)"
)
plt.axhline(y=0.25, color="red", linestyle="--", alpha=0.4)
plt.axhline(y=0.75, color="green", linestyle="--", alpha=0.4)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.fill_between(
    x_values, 0, 0.25, alpha=0.1, color="red", label="Low probability region"
)
plt.fill_between(
    x_values, 0.75, 1, alpha=0.1, color="green", label="High probability region"
)
plt.xlabel("x (Predictor)", fontsize=11)
plt.ylabel("p (Probability)", fontsize=11)
plt.title(
    "Part 4: Logistic Probability p = 1/(1+exp(-(β₀+β·x)))",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([-10, 10])
plt.ylim([0, 1])
plt.show()

# Motivation: Maps unbounded linear predictor to bounded [0,1] probability scale

# %% Overlay: All Transformations Together
# Visualize how each step transforms the signal (normalized for comparison)

plt.figure(figsize=(14, 8))
plt.plot(x_values, z / 20 + 0.5, "b-", alpha=0.5, linewidth=2, label="z (scaled)")
plt.plot(
    x_values,
    np.minimum(exp_neg_z / 10, 1),
    "g-",
    alpha=0.5,
    linewidth=2,
    label="exp(-z) (clamped)",
)
plt.plot(
    x_values,
    np.minimum(normalized / 10, 1),
    "orange",
    alpha=0.5,
    linewidth=2,
    label="1+exp(-z) (clamped)",
)
plt.plot(x_values, p, "purple", alpha=0.9, linewidth=3, label="p (final sigmoid)")
plt.axhline(y=0.5, color="black", linestyle=":", alpha=0.4)
plt.axvline(x=0, color="black", linestyle=":", alpha=0.4)
plt.xlabel("x", fontsize=11)
plt.ylabel("Normalized Output", fontsize=11)
plt.title(
    "All Transformations: z → -z → exp(-z) → 1+exp(-z) → 1/(1+exp(-z))",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xlim([-10, 10])
plt.ylim([0, 1])
plt.show()

# Key insight: Linear → Exponential → Reciprocal creates smooth S-curve

# %% Sensitivity Analysis: Sigmoid Derivative
# Shows where probability is most sensitive to predictor changes

plt.figure(figsize=(10, 6))
plt.plot(x_values, derivative, "orange", linewidth=2, label="p'(z) = p(1-p)")
plt.axvline(
    x=0, color="gray", linestyle="--", alpha=0.5, label="Max sensitivity at x=0"
)
plt.xlabel("x", fontsize=11)
plt.ylabel("dp/dx", fontsize=11)
plt.title(
    "Logistic Sensitivity: Maximum at Decision Boundary (p=0.5)",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([-10, 10])
plt.text(
    0,
    0.26,
    "Most sensitive\nat p=0.5",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
)
plt.show()

# Interpretation: Small x changes near decision boundary cause largest probability changes

# %% Effect of Beta (Slope) Parameter
# Shows how β controls steepness of sigmoid

plt.figure(figsize=(12, 7))
betas = [0.2, 0.5, 1, 2, 5]
for b in betas:
    z_beta = beta0 + b * x_values
    p_beta = 1 / (1 + np.exp(-z_beta))
    plt.plot(x_values, p_beta, linewidth=2, label=f"β={b}")

plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x", fontsize=11)
plt.ylabel("p", fontsize=11)
plt.title("Effect of β: Controls Steepness of Sigmoid", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([-10, 10])
plt.ylim([0, 1])
plt.show()

# Interpretation: Larger |β| → steeper curve → more decisive classification

# %% Effect of Beta0 (Intercept) Parameter
# Shows how β₀ controls horizontal position of sigmoid

plt.figure(figsize=(12, 7))
beta0_values = [-3, -1, 0, 1, 3]
for b0 in beta0_values:
    z_beta0 = b0 + beta * x_values
    p_beta0 = 1 / (1 + np.exp(-z_beta0))
    plt.plot(x_values, p_beta0, linewidth=2, label=f"β₀={b0}")

plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("x", fontsize=11)
plt.ylabel("p", fontsize=11)
plt.title(
    "Effect of β₀: Controls Decision Boundary Position", fontsize=13, fontweight="bold"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([-10, 10])
plt.ylim([0, 1])
plt.show()

# Interpretation: β₀ shifts curve left/right, changes x value where p=0.5

# %% 3D Visualization: Probability Surface (β₀ vs β vs p)

fig = plt.figure(figsize=(14, 6))

# Left plot: p as function of x and β
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
x_mesh = np.linspace(-10, 10, 80)
beta_mesh = np.linspace(0.2, 3, 80)
X_grid, B_grid = np.meshgrid(x_mesh, beta_mesh)
Z_grid = 0 + B_grid * X_grid
P_grid = 1 / (1 + np.exp(-Z_grid))

surf1 = ax1.plot_surface(X_grid, B_grid, P_grid, cmap="viridis", alpha=0.9)
ax1.plot_surface(X_grid, B_grid, np.ones_like(P_grid) * 0.5, alpha=0.15, color="gray")
ax1.set_xlabel("x (Predictor)")
ax1.set_ylabel("β (Slope)")
ax1.set_zlabel("p (Probability)")
ax1.set_title("Probability Surface: f(x, β)")
ax1.view_init(elev=25, azim=45)
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Right plot: p as function of x and β₀
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
beta0_mesh = np.linspace(-3, 3, 80)
X_grid2, B0_grid = np.meshgrid(x_mesh, beta0_mesh)
Z_grid2 = B0_grid + 1 * X_grid2
P_grid2 = 1 / (1 + np.exp(-Z_grid2))

surf2 = ax2.plot_surface(X_grid2, B0_grid, P_grid2, cmap="coolwarm", alpha=0.9)
ax2.plot_surface(
    X_grid2, B0_grid, np.ones_like(P_grid2) * 0.5, alpha=0.15, color="gray"
)
ax2.set_xlabel("x (Predictor)")
ax2.set_ylabel("β₀ (Intercept)")
ax2.set_zlabel("p (Probability)")
ax2.set_title("Probability Surface: f(x, β₀)")
ax2.view_init(elev=20, azim=135)
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.show()

# Shows how parameters shape the probability landscape

# %% Inverse: Logit Transformation
# Shows inverse relationship (log-odds from probability)

p_inv = np.linspace(0.01, 0.99, 500)
logit = np.log(p_inv / (1 - p_inv))

plt.figure(figsize=(10, 6))
plt.plot(p_inv, logit, "teal", linewidth=2, label="logit(p) = log(p/(1-p))")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="p=0.5 → logit=0")
plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Probability (p)", fontsize=11)
plt.ylabel("Log-odds (logit)", fontsize=11)
plt.title(
    "Inverse: Logit Function (Probability → Log-odds)", fontsize=12, fontweight="bold"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, 1])
plt.ylim([-5, 5])
plt.show()

# Interpretation: Logit is the inverse - converts probabilities back to linear scale

# %% Related Formulas (Summary)

related = {
    "Probit": "p = Φ(β₀ + β·x) - uses normal CDF instead of logistic",
    "Tanh": "tanh(x) = 2·sigmoid(2x) - 1 - centered at 0, range [-1,1]",
    "Softmax": "pᵢ = exp(zᵢ)/Σexp(zⱼ) - multiclass generalization",
    "Complementary log-log": "p = 1 - exp(-exp(β₀+β·x)) - asymmetric sigmoid",
    "Gompertz": "p = exp(-exp(-(β₀+β·x))) - asymmetric growth curve",
}

print("RELATED FORMULAS:")
print("=" * 70)
for name, desc in related.items():
    print(f"{name:20} : {desc}")

# Key: Logistic is most common sigmoid for binary classification

# %% Key Mathematical Properties Summary

print("\nKEY PROPERTIES OF LOGISTIC FUNCTION:")
print("=" * 70)
print("Domain:        x ∈ (-∞, ∞)")
print("Range:         p ∈ (0, 1)")
print("Midpoint:      x=0 (when β₀=0) → p=0.5 (decision boundary)")
print("Monotonic:     Strictly increasing (when β > 0)")
print("Symmetric:     Around inflection point at p=0.5")
print("Asymptotes:    p→0 as x→-∞, p→1 as x→+∞")
print("Derivative:    p'(z) = p(1-p) (max at p=0.5)")
print("Inflection:    At p=0.5 (maximum slope)")
print("\nPARAMETERS:")
print("  β₀ (intercept) → Shifts curve horizontally (bias/threshold)")
print("  β (slope)      → Controls steepness (effect size/confidence)")
print("\nINTERPRETATION:")
print("  p < 0.5  → Predict class 0 (negative)")
print("  p = 0.5  → Decision boundary (ambiguous)")
print("  p > 0.5  → Predict class 1 (positive)")

# %% Odds Ratio Interpretation

print("\nODDS RATIO INTERPRETATION:")
print("=" * 70)
print("Logistic regression models log-odds linearly:")
print("  log(p/(1-p)) = β₀ + β·x")
print("  Odds = p/(1-p) = exp(β₀ + β·x)")
print("\nMeaning of β:")
print("  • 1-unit increase in x multiplies odds by exp(β)")
print("  • β > 0: Increases probability (positive effect)")
print("  • β < 0: Decreases probability (negative effect)")
print("  • β = 0: No effect (x doesn't influence outcome)")
print("\nExample: If β=0.5, then exp(0.5)≈1.65")
print("  → 1-unit x increase makes outcome 65% more likely")

# %% Design Intuition

print("\nDESIGN PHILOSOPHY:")
print("=" * 70)
print("1. Linear predictor:    Unbounded decision boundary z = β₀ + β·x")
print("2. Negation:            Orients for correct sigmoid direction")
print("3. Exponential:         Converts to positive odds ratio")
print("4. Normalization:       Adds 1 for division stability")
print("5. Reciprocal:          Bounds to [0,1] probability scale")
print("\nRESULT: Linear separator → Smooth probabilistic predictions")
print("Created for binary classification with interpretable coefficients")
print("\nWHY LOGISTIC?")
print("• Natural link between linear model and probability")
print("• Odds ratios have intuitive interpretation")
print("• Maximum likelihood has closed-form derivative")
print("• Smooth gradients enable efficient optimization")
print("• Probabilistic output supports uncertainty quantification")

# Context: Logistic regression fundamental for classification tasks
# Thinking: Exponential-reciprocal composition creates bounded S-curve from linear input
