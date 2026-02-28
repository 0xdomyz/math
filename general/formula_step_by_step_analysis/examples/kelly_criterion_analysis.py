# %% Setup
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# %% [markdown]
# ### PART 0: Input Parameters
#
# **Kelly Criterion formula:** f* = (bp - q) / b = (p(b+1) - 1) / b
#
# **Where:** f* = optimal bet fraction, p = win probability, q = 1-p = loss probability, b = odds received (net odds), payout ratio

# %%
# Example: Betting on a biased coin
p = 0.55  # Win probability (55%)
q = 1 - p  # Loss probability (45%)
b = 1.0  # Odds: risk $1 to win $1 (even money, 1:1)

# Alternative parameterization: win/loss amounts
win_amount = b  # Net win per $1 bet
loss_amount = 1  # Loss per $1 bet (always 1 in fractional betting)

print(f"Win Probability (p): {p:.1%}")
print(f"Loss Probability (q): {q:.1%}")
print(f"Odds (b): {b}:1 (${b} won per $1 risked)")
print(f"Expected Value: ${p * b - q:.3f} per $1 bet")
print()

# %% [markdown]
# ### PART 1: Numerator Part 1 - Expected Win (bp)
#
# **Formula:** bp = Expected gain from wins (odds × win probability)
#
# **Domain:** p ∈ [0, 1] | **Range:** bp ∈ [0, b] | **Shape:** Linear in p

# %%
p_range = np.linspace(0, 1, 500)
expected_win = b * p_range

# %% [markdown]
# **Properties:** Profitable component of betting
# - bp increases linearly with win probability
# - bp = 0 when p = 0 (never win)
# - bp = b when p = 1 (always win)
# - Represents expected gain per $1 bet

# %%

plt.figure(figsize=(10, 6))
plt.plot(p_range, expected_win, "g-", linewidth=2.5, label="bp (Expected Win)")
plt.axhline(
    y=b * p, color="red", linestyle="--", alpha=0.7, label=f"Current: bp = {b*p:.3f}"
)
plt.axvline(x=p, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=q, color="orange", linestyle="--", alpha=0.5, label=f"Loss: q = {q:.3f}")
plt.fill_between(p_range, 0, expected_win, alpha=0.2, color="green")
plt.grid(alpha=0.3)
plt.xlabel("Win Probability (p)", fontsize=12)
plt.ylabel("Expected Win per $1", fontsize=12)
plt.title("Part 1: bp - Expected Gain from Wins", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Quantifies the winning side of the bet weighted by probability

# %% [markdown]
# ### PART 2: Complete Numerator - Edge (bp - q)
#
# **Formula:** Edge = bp - q = Expected value of bet
#
# **Domain:** ℝ | **Range:** Edge ∈ (-1, b) | **Shape:** Linear in p

# %%
edge = b * p_range - (1 - p_range)

# %% [markdown]
# **Properties:** Net advantage per $1 bet
# - Edge > 0: Profitable bet (required for Kelly)
# - Edge = 0: Fair bet (no Kelly bet)
# - Edge < 0: Unfavorable bet (Kelly = 0)
# - Edge = bp - q simplifies to p(b+1) - 1

# %%
current_edge = b * p - q

plt.figure(figsize=(10, 6))
plt.plot(p_range, edge, "b-", linewidth=2.5, label="Edge (bp - q)")
plt.axhline(
    y=0, color="gray", linestyle="--", alpha=0.7, linewidth=2, label="Zero Edge"
)
plt.axhline(
    y=current_edge,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Current Edge = {current_edge:.3f}",
)
plt.axvline(x=p, color="red", linestyle="--", alpha=0.5, label=f"p = {p:.2f}")
plt.axvline(
    x=1 / (b + 1),
    color="green",
    linestyle="--",
    alpha=0.5,
    label=f"Breakeven p = {1/(b+1):.3f}",
)
plt.fill_between(
    p_range, 0, edge, where=(edge >= 0), alpha=0.2, color="green", label="Positive Edge"
)
plt.fill_between(
    p_range, 0, edge, where=(edge < 0), alpha=0.2, color="red", label="Negative Edge"
)
plt.grid(alpha=0.3)
plt.xlabel("Win Probability (p)", fontsize=12)
plt.ylabel("Edge (Expected Value)", fontsize=12)
plt.title("Part 2: Edge = bp - q (Net Advantage)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Edge determines if betting is profitable; Kelly only applies when Edge > 0

# %% [markdown]
# ### PART 3: Denominator - Odds (b)
#
# **Formula:** b = Odds received on bet (payout ratio)
#
# **Domain:** (0, ∞) | **Range:** (0, ∞) | **Shape:** Linear

# %%
# Show how odds affect Kelly fraction
b_range = np.linspace(0.1, 5, 500)
edge_vs_odds = b_range * p - q

# %% [markdown]
# **Properties:** Payout structure
# - b = 1: Even money (1:1 odds)
# - b = 2: 2:1 odds (win $2 per $1)
# - b < 1: Negative odds (e.g., bet $2 to win $1)
# - Higher b means larger payoffs but affects Kelly fraction

# %%
plt.figure(figsize=(10, 6))
plt.plot(b_range, edge_vs_odds, "purple", linewidth=2.5, label="Edge vs Odds")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=b, color="red", linestyle="--", alpha=0.7, label=f"Current b = {b}")
plt.axhline(y=current_edge, color="red", linestyle="--", alpha=0.5)
plt.fill_between(
    b_range, 0, edge_vs_odds, where=(edge_vs_odds >= 0), alpha=0.2, color="green"
)
plt.grid(alpha=0.3)
plt.xlabel("Odds (b)", fontsize=12)
plt.ylabel("Edge", fontsize=12)
plt.title(f"Part 3: How Odds Affect Edge (p = {p} fixed)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Normalizes edge by payout structure to determine optimal bet size

# %% [markdown]
# ### PART 4a: Division Operation (Edge / b)
#
# **Formula:** f* = (bp - q) / b = Edge ÷ Odds
#
# **Domain:** Edge ∈ ℝ, b > 0 | **Range:** (-∞, ∞) | **Shape:** Hyperbolic in b

# %%
edge_current = b * p - q
kelly_current = edge_current / b

# Visualize the division as scaling operation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Left: Edge (numerator)
ax1.barh(["Edge"], [edge_current], color="blue", alpha=0.7, edgecolor="black")
ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
ax1.set_xlabel("Value", fontsize=11)
ax1.set_title(f"Numerator\nEdge = bp - q = {edge_current:.4f}", fontsize=12)
ax1.grid(alpha=0.3, axis="x")

# Middle: Odds (denominator)
ax2.barh(["Odds b"], [b], color="orange", alpha=0.7, edgecolor="black")
ax2.set_xlabel("Value", fontsize=11)
ax2.set_title(f"Denominator\nb = {b:.2f}", fontsize=12)
ax2.grid(alpha=0.3, axis="x")

# Right: Kelly fraction (quotient)
ax3.barh(["Kelly f*"], [kelly_current], color="green", alpha=0.7, edgecolor="black")
ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
ax3.axvline(
    x=0.5, color="purple", linestyle="--", alpha=0.5, label="f* = 0.5 (Half Kelly)"
)
ax3.set_xlabel("Value", fontsize=11)
ax3.set_title(f"Result\nf* = {kelly_current:.4f}", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

print(f"Kelly Fraction Calculation:")
print(f"  Edge (bp - q) = {edge_current:.4f}")
print(f"  Odds (b)      = {b:.2f}")
print(f"  Kelly f* = {edge_current:.4f} ÷ {b:.2f} = {kelly_current:.4f}")
print(f"  Bet {kelly_current*100:.2f}% of capital")

# %% [markdown]
# **Properties:** Division creates bet fraction
# - Edge normalized by payout ratio
# - Higher odds (b) → smaller fraction (more risk per unit bet)
# - Same edge with 2:1 odds requires half the bet of 1:1 odds
# - Division operation translates to risk adjustment

# %%
# Visualize how division by b scales the edge
b_viz = np.linspace(0.5, 5, 100)
edge_fixed = b * p - q  # Current edge
kelly_viz = edge_fixed / b_viz

plt.figure(figsize=(10, 6))
plt.plot(b_viz, kelly_viz, "b-", linewidth=2.5, label="f* = Edge / b")
plt.axhline(
    y=edge_fixed,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label=f"Edge = {edge_fixed:.3f} (unscaled)",
)
plt.axvline(x=b, color="red", linestyle="--", alpha=0.7, label=f"Current b = {b}")
plt.axhline(y=kelly_current, color="red", linestyle="--", alpha=0.5)
plt.plot(
    b, kelly_current, "ro", markersize=12, label=f"Current f* = {kelly_current:.3f}"
)

# Add annotation showing the scaling
plt.annotate(
    f"As odds increase,\nbet size decreases\n(hyperbolic relationship)",
    xy=(b, kelly_current),
    xytext=(b + 1, kelly_current + 0.03),
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
)

plt.grid(alpha=0.3)
plt.xlabel("Odds (b)", fontsize=12)
plt.ylabel("Kelly Fraction f*", fontsize=12)
plt.title(
    f"Part 4a: Division Operation - Edge/Odds (p={p}, Edge={edge_fixed:.3f})",
    fontsize=13,
)
plt.legend(fontsize=10)
plt.ylim(-0.05, max(kelly_viz) * 1.1)
plt.show()

# %% [markdown]
# **Motivation:** Larger payoffs require smaller bet sizes due to increased variance

# %% [markdown]
# ### PART 4b: Complete Kelly Fraction Formula
#
# **Formula:** f* = (bp - q) / b = Edge / Odds
#
# **Domain:** p ∈ [0, 1], b > 0 | **Range:** f* ∈ [0, 1] (when Edge > 0) | **Shape:** Linear in p, hyperbolic in b

# %%
f_kelly = np.maximum(
    (b * p_range - (1 - p_range)) / b, 0
)  # Set negative to 0 (don't bet)
current_kelly = (b * p - q) / b

# %% [markdown]
# **Properties:** Optimal bet fraction
# - f* ∈ [0, 1] for typical scenarios
# - f* = 0 when bp ≤ q (no edge)
# - f* approaches p as b → ∞
# - f* = (p - q) / 1 = 2p - 1 when b = 1

# %%
plt.figure(figsize=(10, 6))
plt.plot(p_range, f_kelly, "b-", linewidth=3, label="Kelly Fraction f*")
plt.axhline(
    y=current_kelly,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Current f* = {current_kelly:.3f}",
)
plt.axvline(x=p, color="red", linestyle="--", alpha=0.5, label=f"p = {p}")
plt.axvline(
    x=1 / (b + 1),
    color="green",
    linestyle="--",
    alpha=0.5,
    label=f"Breakeven p = {1/(b+1):.3f}",
)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axhline(y=1, color="orange", linestyle="--", alpha=0.5, label="100% Capital")
plt.fill_between(
    p_range,
    0,
    f_kelly,
    where=(f_kelly > 0),
    alpha=0.2,
    color="blue",
    label="Bet Region",
)
plt.grid(alpha=0.3)
plt.xlabel("Win Probability (p)", fontsize=12)
plt.ylabel("Optimal Bet Fraction f*", fontsize=12)
plt.title(f"Part 4b: Complete Kelly Criterion (b = {b})", fontsize=13)
plt.legend(fontsize=10)
plt.ylim(-0.05, 1.05)
plt.show()

# %% [markdown]
# **Motivation:** Maximizes long-run logarithmic growth rate of wealth

# %% [markdown]
# ### Overlay: Kelly vs Probability for Different Odds
#
# Compare Kelly fraction across different odds

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Multiple odds on same plot
for odds in [0.5, 1.0, 2.0, 3.0, 5.0]:
    f_temp = np.maximum((odds * p_range - (1 - p_range)) / odds, 0)
    ax1.plot(p_range, f_temp, linewidth=2.5, label=f"b = {odds}")

ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax1.grid(alpha=0.3)
ax1.set_xlabel("Win Probability (p)", fontsize=11)
ax1.set_ylabel("Kelly Fraction f*", fontsize=11)
ax1.set_title("Kelly Fraction vs Probability\n(Different Odds)", fontsize=12)
ax1.legend(fontsize=9)
ax1.set_ylim(-0.05, 1.05)

# Right: Kelly vs Odds for different probabilities
b_range2 = np.linspace(0.5, 5, 500)
for prob in [0.52, 0.55, 0.60, 0.65, 0.70]:
    f_temp2 = np.maximum((b_range2 * prob - (1 - prob)) / b_range2, 0)
    ax2.plot(b_range2, f_temp2, linewidth=2.5, label=f"p = {prob:.2f}")

ax2.grid(alpha=0.3)
ax2.set_xlabel("Odds (b)", fontsize=11)
ax2.set_ylabel("Kelly Fraction f*", fontsize=11)
ax2.set_title("Kelly Fraction vs Odds\n(Different Win Probabilities)", fontsize=12)
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# %% Sensitivity Analysis: Growth Rate Function
# G(f) = p·ln(1 + bf) + q·ln(1 - f)
# This is what Kelly maximizes

f_frac = np.linspace(0, 0.99, 500)
growth_rate = p * np.log(1 + b * f_frac) + q * np.log(1 - f_frac)

# Find maximum
max_idx = np.argmax(growth_rate)
f_max = f_frac[max_idx]
g_max = growth_rate[max_idx]

plt.figure(figsize=(10, 6))
plt.plot(f_frac, growth_rate, "purple", linewidth=2.5, label="Growth Rate G(f)")
plt.axvline(
    x=current_kelly,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Kelly f* = {current_kelly:.3f}",
)
plt.axhline(y=g_max, color="red", linestyle="--", alpha=0.5)
plt.plot(current_kelly, g_max, "ro", markersize=10, label=f"Maximum G = {g_max:.4f}")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel("Bet Fraction (f)", fontsize=12)
plt.ylabel("Expected Log Growth Rate", fontsize=12)
plt.title(
    f"Growth Rate Function G(f) = p·ln(1+bf) + q·ln(1-f)\n(p={p}, b={b})", fontsize=13
)
plt.legend(fontsize=10)
plt.show()

print(
    f"Kelly fraction f* = {current_kelly:.4f} maximizes growth rate at G = {g_max:.4f}"
)

# %% [markdown]
# **Motivation:** Shows Kelly maximizes geometric growth rate, not arithmetic expectation

# %% [markdown]
# ### Wealth Growth Simulation: Comparing Strategies
#
# Simulate N bets with different betting fractions

# %%
n_bets = 500
n_sims = 100
np.random.seed(42)

strategies = {
    "Kelly": current_kelly,
    "Half Kelly": current_kelly / 2,
    "Double Kelly": min(current_kelly * 2, 1.0),
    "Fixed 10%": 0.10,
    "Fixed 25%": 0.25,
}

plt.figure(figsize=(12, 8))

for strategy_name, frac in strategies.items():
    wealth_paths = np.zeros((n_sims, n_bets + 1))
    wealth_paths[:, 0] = 1.0  # Start with $1

    for sim in range(n_sims):
        outcomes = np.random.rand(n_bets) < p  # Win if random < p

        for i in range(n_bets):
            if outcomes[i]:  # Win
                wealth_paths[sim, i + 1] = wealth_paths[sim, i] * (1 + b * frac)
            else:  # Loss
                wealth_paths[sim, i + 1] = wealth_paths[sim, i] * (1 - frac)

    # Plot median path
    median_wealth = np.median(wealth_paths, axis=0)
    mean_wealth = np.mean(wealth_paths, axis=0)

    plt.plot(median_wealth, linewidth=2.5, label=f"{strategy_name} (f={frac:.3f})")

plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Initial Wealth")
plt.grid(alpha=0.3)
plt.xlabel("Number of Bets", fontsize=12)
plt.ylabel("Median Wealth (log scale)", fontsize=12)
plt.title(
    f"Wealth Growth Comparison: {n_sims} Simulations\n(p={p}, b={b})", fontsize=13
)
plt.legend(fontsize=10)
plt.yscale("log")
plt.show()

# %% Risk of Ruin Analysis
# Show probability of losing X% of capital

drawdown_threshold = 0.5  # 50% loss
n_trials = 1000
n_bets_ruin = 1000

strategies_ruin = {
    "Kelly": current_kelly,
    "Half Kelly": current_kelly / 2,
    "Double Kelly": min(current_kelly * 2, 0.99),
    "Fixed 20%": 0.20,
}

ruin_probs = {}

for strategy_name, frac in strategies_ruin.items():
    ruin_count = 0

    for _ in range(n_trials):
        wealth = 1.0
        outcomes = np.random.rand(n_bets_ruin) < p

        for outcome in outcomes:
            if outcome:
                wealth *= 1 + b * frac
            else:
                wealth *= 1 - frac

            if wealth < drawdown_threshold:
                ruin_count += 1
                break

    ruin_probs[strategy_name] = ruin_count / n_trials

# Plot
plt.figure(figsize=(10, 6))
strategies_list = list(ruin_probs.keys())
probs = list(ruin_probs.values())
colors = ["green", "blue", "red", "orange"]

bars = plt.bar(strategies_list, probs, color=colors, alpha=0.7, edgecolor="black")
plt.ylabel("Probability of 50% Drawdown", fontsize=12)
plt.title(
    f"Risk of Ruin: {n_bets_ruin} Bets, {n_trials} Simulations\n(p={p}, b={b})",
    fontsize=13,
)
plt.grid(alpha=0.3, axis="y")

# Add values on bars
for bar, prob in zip(bars, probs):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{prob:.1%}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.show()

print("\nRisk of 50% Drawdown:")
for name, prob in ruin_probs.items():
    print(f"  {name:15s}: {prob:.2%}")

# %% 3D Visualization: Kelly Surface
# Kelly fraction as function of p and b

p_3d = np.linspace(0.01, 0.99, 50)
b_3d = np.linspace(0.1, 5, 50)
p_mesh, b_mesh = np.meshgrid(p_3d, b_3d)

kelly_mesh = np.maximum((b_mesh * p_mesh - (1 - p_mesh)) / b_mesh, 0)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(p_mesh, b_mesh, kelly_mesh, cmap="viridis", alpha=0.9)
ax.set_xlabel("Win Probability (p)", fontsize=11)
ax.set_ylabel("Odds (b)", fontsize=11)
ax.set_zlabel("Kelly Fraction f*", fontsize=11)
ax.set_title("Kelly Criterion Surface: f*(p, b)", fontsize=13)
ax.set_zlim(0, 1)

# Mark current point
ax.scatter([p], [b], [current_kelly], color="red", s=100, label="Current")

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% 3D Visualization: Growth Rate Surface
# Expected growth rate as function of f and p

p_growth = np.linspace(0.51, 0.99, 50)
f_growth = np.linspace(0, 0.99, 50)
p_g_mesh, f_g_mesh = np.meshgrid(p_growth, f_growth)

growth_mesh = p_g_mesh * np.log(1 + b * f_g_mesh) + (1 - p_g_mesh) * np.log(
    1 - f_g_mesh
)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(p_g_mesh, f_g_mesh, growth_mesh, cmap="plasma", alpha=0.9)
ax.set_xlabel("Win Probability (p)", fontsize=11)
ax.set_ylabel("Bet Fraction (f)", fontsize=11)
ax.set_zlabel("Growth Rate G(f)", fontsize=11)
ax.set_title(f"Growth Rate Surface: G(f, p) with b={b}", fontsize=13)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% Related Formulas (Summary)

related = {
    "General Kelly": "f* = E[X] / Var[X] (for normal returns)",
    "Multi-Asset Kelly": "f* = Σ⁻¹ · μ (covariance matrix approach)",
    "Continuous Kelly": "f* = μ / σ² (for continuous outcomes)",
    "Fractional Kelly": "f = k · f*, k ∈ (0, 1) (reduced risk)",
    "Kelly + Utility": "f* = arg max E[U(W)] (general utility)",
    "Optimal Leverage": "L = μ / σ² (continuous version)",
    "Thorp Formula": "f* = (edge) / (variance of edge) (stock market)",
}

print("=" * 70)
print("RELATED POSITION SIZING FORMULAS")
print("=" * 70)
for name, formula in related.items():
    print(f"{name:22s}: {formula}")
print()

# %% Key Properties Summary

print("=" * 70)
print("KELLY CRITERION: KEY MATHEMATICAL PROPERTIES")
print("=" * 70)
print()
print(f"Structure: f* = (bp - q) / b = (p(b+1) - 1) / b")
print()
print("Edge (Numerator) Properties:")
print("  • Edge = bp - q = Expected value of $1 bet")
print("  • Domain: Edge ∈ (-1, b)")
print("  • Edge > 0 required for positive Kelly")
print("  • Linear in both p and b")
print("  • Breakeven: p = 1/(b+1)")
print()
print("Odds (Denominator) Properties:")
print("  • b = Payout ratio (net odds)")
print("  • Domain: b ∈ (0, ∞)")
print("  • b = 1: Even money")
print("  • b > 1: Favorable odds (win more than risked)")
print("  • b < 1: Unfavorable odds (win less than risked)")
print()
print("Kelly Fraction Properties:")
print("  • Domain: f* ∈ [0, 1] typically")
print("  • f* = 0 when Edge ≤ 0")
print("  • f* increases with p (more confidence → larger bet)")
print("  • f* has hyperbolic relationship with b")
print("  • When b = 1: f* = 2p - 1 = p - q (simple form)")
print("  • Maximum f* = 1 (all capital, risky)")
print()
print("Growth Rate Function:")
print("  • G(f) = p·ln(1 + bf) + q·ln(1 - f)")
print("  • Concave function (unique maximum)")
print("  • Maximized at f = f*")
print("  • G(0) = 0 (no bet, no growth)")
print("  • G(f) → -∞ as f → 1 (ruin)")
print()
print("Optimality Properties:")
print("  • Maximizes long-run geometric growth rate")
print("  • Minimizes expected time to reach wealth goal")
print("  • Asymptotically outperforms all other strategies")
print("  • Does NOT maximize expected wealth (maximizes median)")
print("  • Does NOT minimize risk of ruin in short term")
print()
print("Sensitivity:")
print("  • df*/dp = (b+1) / b > 0 (always positive)")
print("  • df*/db = -Edge / b² (negative when Edge > 0)")
print("  • Over-betting (f > f*) severely reduces growth")
print("  • Under-betting (f < f*) reduces growth mildly")
print()
print(f"Current Example (p={p}, b={b}):")
print(f"  • Edge = {current_edge:.4f}")
print(f"  • Kelly f* = {current_kelly:.4f} ({current_kelly*100:.2f}% of capital)")
print(f"  • Expected growth rate = {g_max:.4f} per bet")
print(f"  • Breakeven probability = {1/(b+1):.4f}")
print()

# %% Design Intuition

print("=" * 70)
print("DESIGN PHILOSOPHY & PRACTICAL CONSIDERATIONS")
print("=" * 70)
print()
print("What Kelly Maximizes:")
print("  • Logarithmic growth rate: E[ln(W_t/W_0)]")
print("  • Geometric mean wealth (not arithmetic mean)")
print("  • Almost surely outperforms any other strategy long-run")
print("  • Median wealth growth, not expected wealth")
print()
print("Why Logarithmic Utility?")
print("  • ln(W) exhibits diminishing marginal utility")
print("  • Balances greed (growth) with fear (ruin)")
print("  • Makes strategy scale-invariant")
print("  • Close to human intuitive risk preferences")
print("  • Mathematically tractable and elegant")
print()
print("Key Insights:")
print("  • Bet size proportional to edge and inversely to odds")
print("  • Small edge → small bet (even with high confidence)")
print("  • Large odds → smaller bet (variance concerns)")
print("  • Never bet more than fraction with positive edge")
print("  • Optimal strategy is uniquely determined")
print()
print("Advantages:")
print("  • Provably optimal for long-run growth")
print("  • Prevents over-betting and ruin")
print("  • Simple closed-form solution")
print("  • Adapts to edge and odds automatically")
print("  • Foundation for portfolio management")
print()
print("Limitations & Criticisms:")
print("  • Extremely aggressive (high volatility)")
print("  • Can experience severe drawdowns")
print("  • Assumes accurate edge estimation (hard!)")
print("  • Ignores transaction costs")
print("  • Assumes infinite time horizon")
print("  • Does not account for liquidity constraints")
print("  • Sensitive to probability mis-estimation")
print()
print("Practical Adjustments:")
print("  • Fractional Kelly: Use f = k·f*, k ∈ [0.25, 0.5]")
print("  • Half-Kelly: Reduces volatility ~50%, growth ~75%")
print("  • Monte Carlo: Test strategy with simulations")
print("  • Confidence intervals: Account for edge uncertainty")
print("  • Maximum bet: Cap at 10-20% even if Kelly higher")
print("  • Worst-case: Consider pessimistic edge estimates")
print()
print("When to Use:")
print("  • Known positive edge with accurate probabilities")
print("  • Independent repeated bets (sports betting, gambling)")
print("  • Long time horizon (decades)")
print("  • Can tolerate 20-30%+ drawdowns")
print("  • Single-asset position sizing")
print()
print("When to Modify:")
print("  • Uncertain edge → Use fractional Kelly")
print("  • Correlated bets → Multi-asset Kelly")
print("  • Short horizon → Risk-of-ruin constraints")
print("  • Fat tails → Expected shortfall adjustments")
print("  • Psychological limits → Reduce fraction")
print()
print("Historical Context:")
print("  • John L. Kelly Jr. (Bell Labs, 1956)")
print("  • Originally for information theory")
print("  • Popularized by Ed Thorp (blackjack, markets)")
print("  • Warren Buffett: 'Bet heavily when odds are in your favor'")
print("  • Renaissance Technologies: Rumored fractional Kelly user")
print("  • Foundation of modern quantitative betting/trading")
print()
print("Common Misconceptions:")
print("  • Kelly does NOT minimize risk")
print("  • Kelly does NOT maximize expected wealth")
print("  • Kelly CAN lead to large drawdowns")
print("  • Kelly assumes perfect edge knowledge (rarely true)")
print("  • Kelly is optimal only asymptotically (infinite bets)")
print()
print("Real-World Application:")
print("  • Professional gamblers: Use 1/4 to 1/2 Kelly")
print("  • Hedge funds: Often cap at 10-20% single position")
print("  • Options traders: Adjust for leverage and gamma")
print("  • Sports betting: Edge estimation is critical")
print("  • Portfolio management: Multi-asset generalization")
print("=" * 70)
