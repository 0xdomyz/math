# %% Setup
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# %% [markdown]
# ### PART 0: Input Parameters
#
# Define portfolio and benchmark returns
#
# **Sharpe Ratio formula:** (Rₚ - Rᶠ) / σₚ
#
# **Where:** Rₚ = portfolio return, Rᶠ = risk-free rate, σₚ = portfolio std dev

# %%
np.random.seed(42)
n_periods = 252  # Trading days in a year

# Generate sample returns
portfolio_returns = np.random.normal(0.0008, 0.015, n_periods)  # ~20% annual, 24% vol
Rp = np.mean(portfolio_returns) * 252  # Annualized return
Rf = 0.03  # Risk-free rate (3%)
sigma_p = np.std(portfolio_returns, ddof=1) * np.sqrt(252)  # Annualized volatility

print(f"Portfolio Annual Return: {Rp:.2%}")
print(f"Risk-Free Rate: {Rf:.2%}")
print(f"Portfolio Volatility: {sigma_p:.2%}")
print(f"Excess Return: {Rp - Rf:.2%}")

# %% [markdown]
# ### PART 1: Excess Return (Numerator)
#
# **Formula:** Excess Return = Rₚ - Rᶠ
#
# **Domain:** (-∞, ∞) | **Range:** (-∞, ∞) | **Shape:** Linear shift

# %%
# Visualize excess return as function of portfolio return
Rp_range = np.linspace(-0.2, 0.5, 500)
excess_return = Rp_range - Rf

# %% [markdown]
# **Properties:** Risk premium above risk-free rate
# - Excess > 0: Positive reward for risk
# - Excess = 0: No compensation for risk
# - Excess < 0: Losing to safe alternative

# %%

plt.figure(figsize=(10, 6))
plt.plot(Rp_range, excess_return, "b-", linewidth=2.5, label="Excess Return")
plt.plot(
    Rp_range,
    Rp_range,
    "gray",
    linestyle="--",
    alpha=0.5,
    label="Total Return",
    linewidth=1.5,
)
plt.axhline(
    y=0, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Zero Excess"
)
plt.axvline(
    x=Rf, color="green", linestyle="--", alpha=0.5, label=f"Risk-Free Rate {Rf:.1%}"
)
plt.fill_between(
    Rp_range,
    0,
    excess_return,
    where=(excess_return >= 0),
    alpha=0.2,
    color="green",
    label="Positive Premium",
)
plt.fill_between(
    Rp_range,
    0,
    excess_return,
    where=(excess_return < 0),
    alpha=0.2,
    color="red",
    label="Negative Premium",
)
plt.grid(alpha=0.3)
plt.xlabel("Portfolio Return Rₚ", fontsize=12)
plt.ylabel("Excess Return (Rₚ - Rᶠ)", fontsize=12)
plt.title("Part 1: Excess Return - Risk Premium", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Measures reward received beyond the risk-free alternative

# %% [markdown]
# ### PART 2: Standard Deviation (Denominator)
#
# **Formula:** σₚ = Standard Deviation of Portfolio Returns
#
# **Domain:** [0, ∞) | **Range:** [0, ∞) | **Shape:** Non-negative risk measure

# %%
# Show distribution of returns
plt.figure(figsize=(10, 6))
plt.hist(
    portfolio_returns, bins=30, density=True, alpha=0.7, color="blue", edgecolor="black"
)

# Overlay normal distribution
x_range = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 200)
normal_fit = stats.norm.pdf(
    x_range, np.mean(portfolio_returns), np.std(portfolio_returns, ddof=1)
)
plt.plot(
    x_range,
    normal_fit,
    "r-",
    linewidth=2.5,
    label=f"Normal Fit (σ={np.std(portfolio_returns, ddof=1):.4f})",
)

# Mark standard deviations
mean_ret = np.mean(portfolio_returns)
std_ret = np.std(portfolio_returns, ddof=1)
plt.axvline(
    mean_ret, color="green", linestyle="--", linewidth=2, label=f"Mean = {mean_ret:.4f}"
)
plt.axvline(mean_ret + std_ret, color="orange", linestyle="--", alpha=0.7, label=f"±1σ")
plt.axvline(mean_ret - std_ret, color="orange", linestyle="--", alpha=0.7)

# %% [markdown]
# **Properties:** Volatility quantifies total risk
# - Higher σₚ = Higher variability = Higher risk
# - Assumes normal distribution of returns
# - Treats upside and downside deviation equally
# - Daily σ scales by √T for longer periods

# %%
plt.grid(alpha=0.3)
plt.xlabel("Daily Return", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Part 2: Standard Deviation σₚ - Total Risk Measure", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Standard deviation normalizes excess return by risk taken

# %% [markdown]
# ### PART 3: Reciprocal of Volatility (1/σₚ)
#
# **Formula:** 1/σₚ = Inverse volatility scaling factor
#
# **Domain:** (0, ∞) | **Range:** (0, ∞) | **Shape:** Hyperbolic

# %%
sigma_range = np.linspace(0.05, 1.0, 500)
inv_sigma = 1 / sigma_range

# %% [markdown]
# **Properties:** Penalty for higher volatility
# - Hyperbolic decay: small σ → large 1/σ
# - As σ → 0, Sharpe → ∞ (unrealistic)
# - As σ → ∞, Sharpe → 0
# - Non-linear penalty: doubling σ halves Sharpe

# %%
plt.figure(figsize=(10, 6))
plt.plot(sigma_range, inv_sigma, "purple", linewidth=2.5, label="1/σₚ")
plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="1/σ = 1 (σ = 1)")
plt.axvline(
    x=sigma_p,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Portfolio σ = {sigma_p:.2%}",
)
plt.grid(alpha=0.3)
plt.xlabel("Volatility σₚ", fontsize=12)
plt.ylabel("1/σₚ", fontsize=12)
plt.title("Part 3: Inverse Volatility - Risk Penalty Factor", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Transforms risk into a penalty multiplier for excess return

# %% [markdown]
# ### PART 4a: Division Operation ((Rₚ - Rᶠ) × 1/σₚ)
#
# **Formula:** Visualizing (Rₚ - Rᶠ) / σₚ as (Rₚ - Rᶠ) × (1/σₚ)
#
# **Domain:** Rₚ ∈ ℝ, σₚ > 0 | **Range:** (-∞, ∞) | **Shape:** Multiplicative scaling

# %%
excess_current = Rp - Rf
inv_sigma_current = 1 / sigma_p
sharpe_current = excess_current * inv_sigma_current

# Visualize the multiplication
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Left: Excess return
ax1.barh(
    ["Excess Return"], [excess_current], color="blue", alpha=0.7, edgecolor="black"
)
ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
ax1.set_xlabel("Value", fontsize=11)
ax1.set_title(f"Numerator\nRₚ - Rᶠ = {excess_current:.4f}", fontsize=12)
ax1.grid(alpha=0.3, axis="x")

# Middle: Inverse volatility
ax2.barh(["1/σₚ"], [inv_sigma_current], color="purple", alpha=0.7, edgecolor="black")
ax2.set_xlabel("Value", fontsize=11)
ax2.set_title(f"Risk Penalty\n1/σₚ = {inv_sigma_current:.4f}", fontsize=12)
ax2.grid(alpha=0.3, axis="x")

# Right: Product (Sharpe ratio)
ax3.barh(
    ["Sharpe Ratio"], [sharpe_current], color="green", alpha=0.7, edgecolor="black"
)
ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
ax3.axvline(x=1, color="orange", linestyle="--", alpha=0.5, label="SR = 1")
ax3.set_xlabel("Value", fontsize=11)
ax3.set_title(f"Result\nSR = {sharpe_current:.4f}", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

print(f"Sharpe Ratio Calculation:")
print(f"  Excess Return (Rₚ - Rᶠ) = {excess_current:.4f}")
print(f"  Risk Penalty (1/σₚ)     = {inv_sigma_current:.4f}")
print(
    f"  Sharpe Ratio = {excess_current:.4f} × {inv_sigma_current:.4f} = {sharpe_current:.4f}"
)

# %% [markdown]
# **Properties:** Multiplicative combination
# - Excess return scaled by inverse volatility
# - Higher volatility → lower multiplier → lower Sharpe
# - Division operation translates to multiplication by reciprocal
# - Result is risk-adjusted return per unit volatility

# %%
# Visualize division as scaling operation
Rp_scale = np.linspace(0, 0.4, 100)
excess_scale = Rp_scale - Rf
sharpe_scale = excess_scale / sigma_p

plt.figure(figsize=(10, 6))
plt.plot(
    excess_scale, sharpe_scale, "b-", linewidth=2.5, label="Sharpe = Excess × (1/σₚ)"
)
plt.plot(
    excess_scale,
    excess_scale,
    "gray",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label="Before scaling (1:1)",
)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
plt.plot(
    excess_current,
    sharpe_current,
    "ro",
    markersize=12,
    label=f"Current Portfolio (SR={sharpe_current:.2f})",
)

# Add annotation showing the scaling factor
plt.annotate(
    f"Scaling factor: 1/σₚ = {inv_sigma_current:.2f}\n(slope of line)",
    xy=(excess_current, sharpe_current),
    xytext=(excess_current + 0.05, sharpe_current - 0.3),
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
)

plt.grid(alpha=0.3)
plt.xlabel("Excess Return (Rₚ - Rᶠ)", fontsize=12)
plt.ylabel("Sharpe Ratio", fontsize=12)
plt.title("Part 4a: Division Operation - Scaling Excess Return by 1/σₚ", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Division by volatility penalizes risk proportionally, creating risk-adjusted metric

# %% [markdown]
# ### PART 4b: Complete Sharpe Ratio Formula
#
# **Formula:** Sharpe Ratio = (Rₚ - Rᶠ) / σₚ
#
# **Domain:** Rₚ ∈ ℝ, σₚ > 0 | **Range:** (-∞, ∞) | **Shape:** Linear in Rₚ, hyperbolic in σₚ

# %%
# Hold volatility constant, vary return
sharpe_vs_return = (Rp_range - Rf) / sigma_p

# %% [markdown]
# **Properties:** Risk-adjusted performance metric
# - Sharpe > 1: Good risk-adjusted return
# - Sharpe > 2: Very good performance
# - Sharpe > 3: Excellent (rare for long-only strategies)
# - Sharpe < 0: Losing to risk-free rate

# %%
current_sharpe = (Rp - Rf) / sigma_p

plt.figure(figsize=(10, 6))
plt.plot(Rp_range, sharpe_vs_return, "b-", linewidth=2.5, label="Sharpe Ratio")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7, label="Sharpe = 0")
plt.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1 (Good)")
plt.axhline(
    y=2, color="orange", linestyle="--", alpha=0.5, label="Sharpe = 2 (Very Good)"
)
plt.axvline(x=Rf, color="red", linestyle="--", alpha=0.5, label=f"Rₚ = Rᶠ")
plt.plot(
    Rp,
    current_sharpe,
    "ro",
    markersize=10,
    label=f"Portfolio ({Rp:.1%}, SR={current_sharpe:.2f})",
)
plt.fill_between(
    Rp_range,
    0,
    sharpe_vs_return,
    where=(sharpe_vs_return >= 1),
    alpha=0.15,
    color="green",
)
plt.grid(alpha=0.3)
plt.xlabel("Portfolio Return Rₚ", fontsize=12)
plt.ylabel("Sharpe Ratio", fontsize=12)
plt.title("Part 4b: Complete Sharpe Ratio Formula", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Combines excess return and risk into single performance metric for comparison

# %% [markdown]
# ### Overlay: Return vs Volatility Efficient Frontier
#
# Show multiple portfolios in risk-return space

# %%
n_portfolios = 1000
returns = np.random.uniform(-0.1, 0.4, n_portfolios)
volatilities = np.random.uniform(0.05, 0.5, n_portfolios)
sharpe_ratios = (returns - Rf) / volatilities

# Create scatter plot colored by Sharpe ratio
plt.figure(figsize=(11, 7))
scatter = plt.scatter(
    volatilities,
    returns,
    c=sharpe_ratios,
    cmap="RdYlGn",
    s=30,
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5,
)
plt.colorbar(scatter, label="Sharpe Ratio")

# Add iso-Sharpe lines
for sr in [0, 0.5, 1.0, 1.5, 2.0]:
    vol_line = np.linspace(0.05, 0.5, 100)
    ret_line = Rf + sr * vol_line
    plt.plot(vol_line, ret_line, "k--", alpha=0.3, linewidth=1)
    plt.text(0.5, Rf + sr * 0.5, f"SR={sr}", fontsize=9, alpha=0.6)

# Mark risk-free rate and sample portfolio
plt.plot(0, Rf, "b*", markersize=15, label="Risk-Free Asset")
plt.plot(sigma_p, Rp, "ro", markersize=12, label=f"Portfolio (SR={current_sharpe:.2f})")

plt.xlabel("Volatility σₚ", fontsize=12)
plt.ylabel("Return Rₚ", fontsize=12)
plt.title("Overlay: Risk-Return Space with Iso-Sharpe Lines", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.xlim(0, 0.55)
plt.ylim(-0.15, 0.45)
plt.show()

# %% Sensitivity Analysis: Sharpe vs Volatility
# How Sharpe ratio changes with volatility (holding excess return constant)

excess = Rp - Rf
vol_range = np.linspace(0.05, 0.6, 500)
sharpe_vs_vol = excess / vol_range

plt.figure(figsize=(10, 6))
plt.plot(vol_range, sharpe_vs_vol, "purple", linewidth=2.5)
plt.axvline(
    x=sigma_p,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Current σ = {sigma_p:.2%}",
)
plt.axhline(y=current_sharpe, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="SR = 1")
plt.grid(alpha=0.3)
plt.xlabel("Volatility σₚ", fontsize=12)
plt.ylabel("Sharpe Ratio", fontsize=12)
plt.title(
    f"Sensitivity: Sharpe vs Volatility (Excess Return = {excess:.2%} fixed)",
    fontsize=13,
)
plt.legend(fontsize=10)
plt.show()

# Derivative: dSR/dσ = -(Rₚ - Rᶠ)/σ²
derivative_sr = -excess / vol_range**2
print(f"At σ = {sigma_p:.2%}, dSR/dσ = {-(excess / sigma_p**2):.2f}")
print(
    f"A 1% increase in volatility decreases Sharpe by ~{abs(0.01 * excess / sigma_p**2):.3f}"
)

# %% 3D Visualization: Sharpe Surface
# Sharpe ratio as function of return and volatility

ret_3d = np.linspace(-0.1, 0.4, 50)
vol_3d = np.linspace(0.05, 0.5, 50)
ret_mesh, vol_mesh = np.meshgrid(ret_3d, vol_3d)
sharpe_mesh = (ret_mesh - Rf) / vol_mesh

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(vol_mesh, ret_mesh, sharpe_mesh, cmap="viridis", alpha=0.9)
ax.set_xlabel("Volatility σₚ", fontsize=11)
ax.set_ylabel("Return Rₚ", fontsize=11)
ax.set_zlabel("Sharpe Ratio", fontsize=11)
ax.set_title("Sharpe Ratio Surface: SR(σₚ, Rₚ)", fontsize=13)
ax.set_zlim(-5, 5)

# Mark current portfolio
ax.scatter([sigma_p], [Rp], [current_sharpe], color="red", s=100, label="Portfolio")

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% 3D Visualization: Contour Map
# Top-down view of Sharpe ratio landscape

plt.figure(figsize=(11, 8))
contour = plt.contourf(vol_mesh, ret_mesh, sharpe_mesh, levels=20, cmap="RdYlGn")
plt.colorbar(contour, label="Sharpe Ratio")

# Add contour lines
contour_lines = plt.contour(
    vol_mesh,
    ret_mesh,
    sharpe_mesh,
    levels=[-1, 0, 0.5, 1, 1.5, 2, 2.5, 3],
    colors="black",
    alpha=0.4,
    linewidths=1,
)
plt.clabel(contour_lines, inline=True, fontsize=9)

# Mark current portfolio and risk-free rate
plt.plot(sigma_p, Rp, "ro", markersize=12, label=f"Portfolio (SR={current_sharpe:.2f})")
plt.plot(0, Rf, "b*", markersize=15, label="Risk-Free Asset")

plt.xlabel("Volatility σₚ", fontsize=12)
plt.ylabel("Return Rₚ", fontsize=12)
plt.title("Sharpe Ratio Contour Map", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# %% Time Series Analysis: Rolling Sharpe Ratio
# Calculate rolling Sharpe over different windows

window = 63  # Quarter (approx 3 months)
rolling_mean = np.convolve(portfolio_returns, np.ones(window) / window, mode="valid")
rolling_std = np.array(
    [
        np.std(portfolio_returns[i : i + window], ddof=1)
        for i in range(len(portfolio_returns) - window + 1)
    ]
)
rolling_sharpe = (rolling_mean * 252 - Rf) / (rolling_std * np.sqrt(252))

plt.figure(figsize=(12, 8))

# Subplot 1: Returns
plt.subplot(3, 1, 1)
plt.plot(portfolio_returns, "b-", alpha=0.7, linewidth=1)
plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
plt.ylabel("Daily Return", fontsize=10)
plt.title("Daily Portfolio Returns", fontsize=11)
plt.grid(alpha=0.3)

# Subplot 2: Rolling Volatility
plt.subplot(3, 1, 2)
plt.plot(rolling_std * np.sqrt(252), "orange", linewidth=2)
plt.axhline(
    y=sigma_p,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Full Period σ={sigma_p:.2%}",
)
plt.ylabel("Annualized Vol", fontsize=10)
plt.title(f"Rolling Volatility ({window}-day window)", fontsize=11)
plt.legend(fontsize=9)
plt.grid(alpha=0.3)

# Subplot 3: Rolling Sharpe
plt.subplot(3, 1, 3)
plt.plot(rolling_sharpe, "purple", linewidth=2)
plt.axhline(
    y=current_sharpe,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Full Period SR={current_sharpe:.2f}",
)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.axhline(y=1, color="green", linestyle="--", alpha=0.3)
plt.xlabel("Trading Day", fontsize=10)
plt.ylabel("Sharpe Ratio", fontsize=10)
plt.title(f"Rolling Sharpe Ratio ({window}-day window)", fontsize=11)
plt.legend(fontsize=9)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% Related Formulas (Summary)

related = {
    "Sortino Ratio": "(Rₚ - Rᶠ) / σ_downside (penalizes only downside vol)",
    "Calmar Ratio": "Annual Return / Max Drawdown",
    "Information Ratio": "(Rₚ - Rᵦ) / Tracking Error (vs benchmark)",
    "Treynor Ratio": "(Rₚ - Rᶠ) / βₚ (uses systematic risk)",
    "Adjusted Sharpe": "Sharpe × √(1 + Skew/6 - Kurtosis/24) (moment-adjusted)",
    "M² Measure": "Rᶠ + SR_p × σ_market (risk-adjusted to market vol)",
    "Omega Ratio": "∫P(R>τ) / ∫P(R<τ) (probability-weighted gains/losses)",
}

print("=" * 70)
print("RELATED RISK-ADJUSTED PERFORMANCE METRICS")
print("=" * 70)
for name, formula in related.items():
    print(f"{name:20s}: {formula}")
print()

# %% Key Properties Summary

print("=" * 70)
print("SHARPE RATIO: KEY MATHEMATICAL PROPERTIES")
print("=" * 70)
print()
print("Structure: SR = (Rₚ - Rᶠ) / σₚ")
print()
print("Excess Return Properties:")
print("  • Numerator: Risk premium = Rₚ - Rᶠ")
print("  • Domain: (-∞, ∞)")
print("  • Linear shift of return by risk-free rate")
print("  • Positive excess required for positive Sharpe")
print()
print("Standard Deviation Properties:")
print("  • Denominator: Total volatility σₚ")
print("  • Domain: (0, ∞), always positive")
print("  • Assumes normal distribution of returns")
print("  • Treats upside and downside volatility equally")
print("  • Time-scales: σ(T) = σ(1) × √T")
print()
print("Sharpe Ratio Properties:")
print("  • Domain: σₚ > 0, Rₚ ∈ ℝ")
print("  • Range: (-∞, ∞)")
print("  • Linear in excess return")
print("  • Hyperbolic (1/σₚ) in volatility")
print("  • Scale-invariant: SR(λX) = SR(X) for λ > 0")
print()
print("Interpretation Benchmarks:")
print("  • SR < 0: Underperforming risk-free rate")
print("  • SR = 0: Matching risk-free rate")
print("  • SR ∈ (0, 1): Positive but modest risk adjustment")
print("  • SR ∈ (1, 2): Good risk-adjusted performance")
print("  • SR > 2: Excellent (top-quartile strategies)")
print("  • SR > 3: Exceptional (rare without leverage)")
print()
print("Annualization:")
print("  • Daily to Annual: SR_annual = SR_daily × √252")
print("  • Monthly to Annual: SR_annual = SR_monthly × √12")
print("  • Square-root-of-time scaling assumes i.i.d. returns")
print()
print("Sensitivity:")
print("  • ∂SR/∂Rₚ = 1/σₚ (always positive)")
print("  • ∂SR/∂σₚ = -(Rₚ - Rᶠ)/σₚ² (always negative if Rₚ > Rᶠ)")
print("  • Volatility has quadratic impact on Sharpe")
print()

# %% Design Intuition

print("=" * 70)
print("DESIGN PHILOSOPHY & PRACTICAL CONSIDERATIONS")
print("=" * 70)
print()
print("Why This Formula?")
print("  • Normalizes excess return by total risk taken")
print("  • Enables comparison across different strategies")
print("  • Unit: 'return per unit risk' (dimensionless)")
print("  • Foundation: Mean-variance optimization (Markowitz 1952)")
print()
print("Advantages:")
print("  • Simple, intuitive, universally understood")
print("  • Scale-invariant (works for any position size)")
print("  • Symmetric treatment of volatility")
print("  • Directly linked to portfolio theory")
print("  • Easy to compute and track over time")
print()
print("Limitations:")
print("  • Assumes normal distribution (ignores fat tails)")
print("  • Penalizes upside volatility (good for investors)")
print("  • Sensitive to time period (estimation error)")
print("  • Can be manipulated (e.g., selling OTM options)")
print("  • Ignores higher moments (skewness, kurtosis)")
print("  • Not defined when σₚ = 0 (risk-free asset)")
print()
print("When to Use:")
print("  • Comparing similar strategies (same asset class)")
print("  • Long-only portfolios with symmetric risk")
print("  • Normally distributed or near-normal returns")
print("  • Ex-post performance evaluation")
print("  • Portfolio allocation between uncorrelated assets")
print()
print("When to Supplement:")
print("  • Non-normal returns → Use Sortino, Omega ratios")
print("  • Fat tails → Add CVaR, tail risk metrics")
print("  • Benchmark-relative → Use Information Ratio")
print("  • Leverage strategies → Use drawdown metrics")
print("  • Option strategies → Check higher moments")
print()
print("Practical Usage:")
print("  • Hedge funds often target SR > 1.5")
print("  • Long-only equity: SR ≈ 0.3-0.5 historically")
print("  • Market-neutral: SR ≈ 1.0-2.0 target")
print("  • Use rolling Sharpe to detect regime changes")
print("  • Combine with maximum drawdown for full picture")
print()
print("Historical Context:")
print("  • Developed by William F. Sharpe (1966)")
print("  • Originally called 'reward-to-variability ratio'")
print("  • Nobel Prize in Economics (1990)")
print("  • Most widely used risk-adjusted metric")
print("=" * 70)
print("=" * 70)
