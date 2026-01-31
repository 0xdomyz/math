"""
RSI Formula Deep Dive Analysis
Formula: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
Run each cell interactively with Shift+Enter
"""

# %%

import matplotlib.pyplot as plt

# %% Setup
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %% PART 0: RS (Relative Strength)
# The fundamental ratio of gains to losses
# Domain: [0, ∞), Range: [0, ∞), Shape: Linear (identity)

rs_values = np.linspace(0.01, 10, 500)

# Properties: RS represents raw momentum - gains divided by losses
# RS > 1: more gains than losses (bullish)
# RS = 1: balanced (neutral)
# RS < 1: more losses than gains (bearish)

plt.figure(figsize=(10, 6))
plt.plot(rs_values, rs_values, "b-", linewidth=2)
plt.axhline(y=1, color="gray", linestyle="--", label="RS=1 (neutral)")
plt.axvline(x=1, color="gray", linestyle="--")
plt.xlabel("RS Input")
plt.ylabel("RS Output")
plt.title("Part 0: RS = Avg Gain / Avg Loss (Identity Function)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Motivation: Ratio ensures scale-invariance across different assets

# %% PART 1: Normalization (1 + RS)
# Shifts RS by 1 to prevent division by zero
# Domain: [0, ∞), Range: [1, ∞), Shape: Linear with y-intercept at 1

normalized = 1 + rs_values

# Properties: Adding 1 creates stability and centers the midpoint
# When RS = 0 → 1 (minimum, prevents division issues)
# When RS = 1 → 2 (creates natural balance point for reciprocal)

plt.figure(figsize=(10, 6))
plt.plot(rs_values, normalized, "g-", linewidth=2)
plt.axhline(y=2, color="gray", linestyle="--", label="RS=1 → 2")
plt.xlabel("RS")
plt.ylabel("1 + RS")
plt.title("Part 1: Normalization (1 + RS)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Motivation: Mathematical stability - ensures denominator never zero

# %% PART 2: Reciprocal (100 / (1 + RS))
# Inverts relationship and bounds output to [0, 100]
# Domain: [0, ∞), Range: (0, 100], Shape: Rectangular hyperbola (decreasing)

reciprocal = 100 / (1 + rs_values)

# Properties: Reciprocal compresses infinite RS into bounded range
# As RS → ∞: output → 0 (strong gains compress toward zero)
# As RS → 0: output → 100 (strong losses push toward 100)
# RS = 1 → 50 (balanced midpoint)

plt.figure(figsize=(10, 6))
plt.plot(rs_values, reciprocal, "r-", linewidth=2)
plt.axhline(y=50, color="gray", linestyle="--", label="RS=1 → 50")
plt.axvline(x=1, color="gray", linestyle="--")
plt.xlabel("RS")
plt.ylabel("100 / (1 + RS)")
plt.title("Part 2: Reciprocal Transformation (Inverts & Bounds)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([0, 105])
plt.show()

# Motivation: Creates logarithmic-like sensitivity - small changes at low RS matter more

# %% PART 3: RSI (Complete Formula) = 100 - (100 / (1 + RS))
# Final complement flips scale to make high RS → high RSI (intuitive)
# Domain: [0, ∞), Range: [0, 100), Shape: Sigmoid-like (S-curve, concave)

rsi = 100 - (100 / (1 + rs_values))

# Properties: Complete transformation with intuitive interpretation
# RS = 0 → RSI = 0 (extreme oversold)
# RS = 1 → RSI = 50 (neutral/balanced)
# RS = 9 → RSI = 90 (extreme overbought)
# RS → ∞ → RSI → 100 (approaches but never reaches)

# Derivative: RSI'(RS) = 100/(1+RS)² (always positive, decreasing)
# Most sensitive near RS=1, less sensitive at extremes

plt.figure(figsize=(12, 7))
plt.plot(rs_values, rsi, "purple", linewidth=3, label="RSI")
plt.axhline(y=30, color="red", linestyle="--", alpha=0.6, label="Oversold (30)")
plt.axhline(y=50, color="gray", linestyle="-", alpha=0.6, label="Neutral (50)")
plt.axhline(y=70, color="green", linestyle="--", alpha=0.6, label="Overbought (70)")
plt.fill_between(rs_values, 0, 30, alpha=0.1, color="red")
plt.fill_between(rs_values, 70, 100, alpha=0.1, color="green")
plt.xlabel("RS (Relative Strength)", fontsize=11)
plt.ylabel("RSI Value", fontsize=11)
plt.title(
    "Part 3: Complete RSI Formula = 100 - (100/(1+RS))", fontsize=13, fontweight="bold"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim([0, 10])
plt.ylim([0, 100])
plt.show()

# Motivation: Bounded [0,100] oscillator that's easy to interpret visually
# High values = overbought, low values = oversold, 50 = neutral

# %% Overlay: All Transformations Together
# Visualize how each step transforms the signal

plt.figure(figsize=(14, 8))
plt.plot(
    rs_values, rs_values * 10, "b-", alpha=0.5, linewidth=2, label="RS (scaled×10)"
)
plt.plot(
    rs_values,
    (1 + rs_values) * 10,
    "g-",
    alpha=0.5,
    linewidth=2,
    label="1+RS (scaled×10)",
)
plt.plot(rs_values, reciprocal, "r-", alpha=0.6, linewidth=2, label="100/(1+RS)")
plt.plot(rs_values, rsi, "purple", alpha=0.9, linewidth=3, label="RSI (final)")
plt.axhline(y=50, color="black", linestyle=":", alpha=0.4)
plt.axvline(x=1, color="black", linestyle=":", alpha=0.4)
plt.xlabel("RS", fontsize=11)
plt.ylabel("Output Value", fontsize=11)
plt.title(
    "All Transformations: RS → (1+RS) → Reciprocal → RSI",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xlim([0, 10])
plt.ylim([0, 100])
plt.show()

# Key insight: Each transformation serves a purpose in the composition

# %% Sensitivity Analysis: RSI Derivative
# Shows how sensitive RSI is to changes in RS at different points

derivative = 100 / ((1 + rs_values) ** 2)

plt.figure(figsize=(10, 6))
plt.plot(rs_values, derivative, "orange", linewidth=2)
plt.xlabel("RS", fontsize=11)
plt.ylabel("RSI'(RS) = 100/(1+RS)²", fontsize=11)
plt.title(
    "RSI Sensitivity: Rate of Change Decreases at Extremes",
    fontsize=13,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)
plt.xlim([0, 10])
plt.text(
    5,
    15,
    "Most sensitive\nnear RS=1",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
)
plt.show()

# Interpretation: Small RS changes near 1 cause bigger RSI changes than at extremes

# %% 3D Visualization: RSI Surface (Avg Gain vs Avg Loss)

fig = plt.figure(figsize=(14, 6))

# Left plot: RSI as function of gains and losses
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
avg_gain = np.linspace(0.1, 10, 80)
avg_loss = np.linspace(0.1, 10, 80)
AG, AL = np.meshgrid(avg_gain, avg_loss)
RS_grid = AG / AL
RSI_grid = 100 - (100 / (1 + RS_grid))

surf1 = ax1.plot_surface(AG, AL, RSI_grid, cmap="viridis", alpha=0.9)
ax1.plot_surface(AG, AL, np.ones_like(RSI_grid) * 50, alpha=0.15, color="gray")
ax1.set_xlabel("Avg Gain")
ax1.set_ylabel("Avg Loss")
ax1.set_zlabel("RSI")
ax1.set_title("RSI Surface: f(Gain, Loss)")
ax1.view_init(elev=25, azim=45)
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Right plot: RSI evolution with varying RS
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
rs_range = np.linspace(0.01, 12, 80)
time = np.linspace(0, 100, 80)
RS_mesh, T_mesh = np.meshgrid(rs_range, time)
RSI_mesh = 100 - (100 / (1 + RS_mesh))

surf2 = ax2.plot_surface(T_mesh, RS_mesh, RSI_mesh, cmap="coolwarm", alpha=0.9)
ax2.set_xlabel("Time/Period")
ax2.set_ylabel("RS")
ax2.set_zlabel("RSI")
ax2.set_title("RSI Evolution over RS Range")
ax2.view_init(elev=20, azim=135)
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.show()

# Shows RSI's nonlinear compression of RS space into bounded [0,100] range

# %% Related Formulas (Summary)

related = {
    "MFI": "Same formula, uses volume-weighted price (Money Flow Ratio)",
    "Stochastic": "%K = 100×(Close-Low)/(High-Low) - position in range",
    "Williams %R": "Similar to Stochastic, inverted scale [-100,0]",
    "Alternative RSI": "RSI = 100×RS/(1+RS) - mathematically equivalent!",
}

print("RELATED FORMULAS:")
print("=" * 60)
for name, desc in related.items():
    print(f"{name:15} : {desc}")

# Key: MFI has IDENTICAL math structure, just different input (price+volume)

# %% Key Mathematical Properties Summary

print("\nKEY PROPERTIES OF RSI:")
print("=" * 60)
print("Domain:      RS ∈ [0, ∞)")
print("Range:       RSI ∈ [0, 100)")
print("Midpoint:    RS=1 → RSI=50 (balanced)")
print("Monotonic:   Strictly increasing")
print("Concave:     Second derivative < 0")
print("Asymptote:   RSI→100 as RS→∞ (never reaches)")
print("Sensitivity: Highest at RS=1, decreases at extremes")
print("\nINTERPRETATION:")
print("  RSI < 30  → Oversold (potential buy signal)")
print("  RSI = 50  → Neutral (balanced market)")
print("  RSI > 70  → Overbought (potential sell signal)")

# %% Design Intuition

print("\nDESIGN PHILOSOPHY:")
print("=" * 60)
print("1. RS ratio:        Scale-invariant momentum measure")
print("2. +1 shift:        Mathematical stability (no division by zero)")
print("3. Reciprocal:      Compresses infinite range → [0,100]")
print("4. Complement:      Flips scale (high RS → high RSI, intuitive)")
print("\nRESULT: Unbounded noisy momentum → Bounded interpretable signal")
print("Created by J. Welles Wilder (1978) - elegant nonlinear transformation")

# Context: Workspace analysis based on workspace instructions
# Key thinking: Reciprocal function creates logarithmic sensitivity compression
