# %% Setup
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# %% [markdown]
# ### PART 0: Input Parameters
# Define the Black-Scholes parameters for a call option
#
# **Parameters:** S₀ (spot), K (strike), r (risk-free rate), σ (volatility), T (time to maturity)

# %%
S0 = 100  # Current stock price
K = 100  # Strike price
r = 0.05  # Risk-free rate (5% annually)
sigma = 0.25  # Volatility (25% annually)
T = 1.0  # Time to maturity (1 year)

print(f"Parameters: S₀=${S0}, K=${K}, r={r:.1%}, σ={sigma:.1%}, T={T} year")
print(f"At-the-money option (S₀ = K)")

# %% [markdown]
# ### PART 1a: Moneyness Ratio (S/K)
#
# **Formula:** Moneyness = S/K
#
# **Domain:** (0, ∞) | **Range:** (0, ∞) | **Shape:** Linear

# %%
S_range = np.linspace(50, 150, 500)
moneyness = S_range / K

# %% [markdown]
# **Properties:** Relative stock price to strike
# - S/K = 1: At-the-money (ATM)
# - S/K > 1: In-the-money (ITM)
# - S/K < 1: Out-of-the-money (OTM)

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, moneyness, "b-", linewidth=2.5, label="S/K")
plt.axhline(y=1, color="gray", linestyle="--", alpha=0.7, label="S/K = 1 (ATM)")
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Moneyness S/K", fontsize=12)
plt.title("Part 1a: Moneyness Ratio", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Normalizes stock price relative to strike for comparison

# %% [markdown]
# ### PART 1b: Log Moneyness (ln(S/K))
#
# **Formula:** ln(S/K)
#
# **Domain:** (0, ∞) | **Range:** (-∞, ∞) | **Shape:** Logarithmic

# %%
log_moneyness = np.log(S_range / K)

# %% [markdown]
# **Properties:** Symmetric measure of relative price
# - ln(S/K) = 0 when S = K (ATM)
# - ln(S/K) > 0 when S > K (ITM)
# - ln(S/K) < 0 when S < K (OTM)
# - Symmetric: ln(2) = -ln(0.5) in magnitude

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, log_moneyness, "g-", linewidth=2.5, label="ln(S/K)")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7, label="ln(S/K) = 0")
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.fill_between(
    S_range,
    0,
    log_moneyness,
    where=(log_moneyness >= 0),
    alpha=0.2,
    color="green",
    label="ITM",
)
plt.fill_between(
    S_range,
    0,
    log_moneyness,
    where=(log_moneyness < 0),
    alpha=0.2,
    color="red",
    label="OTM",
)
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("ln(S/K)", fontsize=12)
plt.title("Part 1b: Log Moneyness", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Logarithm creates symmetric scale for multiplicative returns

# %% [markdown]
# ### PART 1c: Variance Adjustment (σ²/2)
#
# **Formula:** σ²/2
#
# **Domain:** [0, ∞) | **Range:** [0, ∞) | **Shape:** Quadratic in σ

# %%
variance_adj = sigma**2 / 2
sigma_viz = np.linspace(0.05, 0.6, 500)
var_adj_range = sigma_viz**2 / 2

# %% [markdown]
# **Properties:** Convexity adjustment for lognormal drift
# - Arises from Itô's lemma (dS/S follows GBM)
# - Half of variance corrects for Jensen's inequality
# - Larger for higher volatility
# - Current: σ²/2 = {:.4f}

# %%
plt.figure(figsize=(10, 6))
plt.plot(sigma_viz, var_adj_range, "purple", linewidth=2.5, label="σ²/2")
plt.axvline(
    x=sigma, color="red", linestyle="--", alpha=0.7, label=f"Current σ = {sigma:.2%}"
)
plt.axhline(y=variance_adj, color="red", linestyle="--", alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel("Volatility σ", fontsize=12)
plt.ylabel("σ²/2", fontsize=12)
plt.title("Part 1c: Variance Adjustment", fontsize=13)
plt.legend(fontsize=10)
plt.show()

print(f"Variance adjustment: σ²/2 = {variance_adj:.4f}")

# %% [markdown]
# **Motivation:** Accounts for convexity of lognormal distribution

# %% [markdown]
# ### PART 1d: Drift Term ((r + σ²/2)T)
#
# **Formula:** (r + σ²/2)T
#
# **Domain:** [0, ∞) | **Range:** [0, ∞) | **Shape:** Linear in T

# %%
drift_term = (r + sigma**2 / 2) * T
T_viz = np.linspace(0.1, 3, 500)
drift_range = (r + sigma**2 / 2) * T_viz

# %% [markdown]
# **Properties:** Expected growth adjustment
# - Combines risk-free rate and convexity adjustment
# - Increases linearly with time
# - Represents drift of log(S) under risk-neutral measure
# - Current: (r + σ²/2)T = {:.4f}

# %%
plt.figure(figsize=(10, 6))
plt.plot(T_viz, drift_range, "orange", linewidth=2.5, label="(r + σ²/2)T")
plt.plot(T_viz, r * T_viz, "b--", linewidth=2, alpha=0.7, label="rT only")
plt.plot(
    T_viz, (sigma**2 / 2) * T_viz, "g--", linewidth=2, alpha=0.7, label="(σ²/2)T only"
)
plt.axvline(x=T, color="red", linestyle="--", alpha=0.7, label=f"Current T = {T}")
plt.axhline(y=drift_term, color="red", linestyle="--", alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel("Time to Maturity T", fontsize=12)
plt.ylabel("Drift Term", fontsize=12)
plt.title("Part 1d: Drift Term (r + σ²/2)T", fontsize=13)
plt.legend(fontsize=10)
plt.show()

print(f"Drift term: (r + σ²/2)T = {drift_term:.4f}")

# %% [markdown]
# **Motivation:** Expected log-return over time horizon

# %% [markdown]
# ### PART 1e: Numerator (ln(S/K) + (r + σ²/2)T)
#
# **Formula:** ln(S/K) + (r + σ²/2)T
#
# **Domain:** (-∞, ∞) | **Range:** (-∞, ∞) | **Shape:** Linear in ln(S)

# %%
numerator_d1 = np.log(S_range / K) + (r + sigma**2 / 2) * T

# %% [markdown]
# **Properties:** Total expected log-moneyness
# - Combines current log moneyness with expected drift
# - Adjusted for both risk-free rate and volatility
# - Represents expected final position in log space

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, numerator_d1, "b-", linewidth=2.5, label="ln(S/K) + (r + σ²/2)T")
plt.plot(
    S_range, np.log(S_range / K), "g--", linewidth=2, alpha=0.7, label="ln(S/K) only"
)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axhline(
    y=drift_term,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"Drift = {drift_term:.3f}",
)
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Numerator", fontsize=12)
plt.title("Part 1e: d₁ Numerator", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Expected moneyness at maturity under risk-neutral measure

# %% [markdown]
# ### PART 1f: Denominator (σ√T)
#
# **Formula:** σ√T
#
# **Domain:** [0, ∞) | **Range:** [0, ∞) | **Shape:** Square root in T, linear in σ

# %%
denominator_d1 = sigma * np.sqrt(T)
T_denom_viz = np.linspace(0.1, 3, 500)
denom_range = sigma * np.sqrt(T_denom_viz)

# %% [markdown]
# **Properties:** Volatility scaling factor
# - Standard deviation of log(S) over time T
# - Scales as √T (due to Brownian motion)
# - Larger for higher volatility or longer time
# - Current: σ√T = {:.4f}

# %%
plt.figure(figsize=(10, 6))
plt.plot(T_denom_viz, denom_range, "purple", linewidth=2.5, label="σ√T")
plt.axvline(x=T, color="red", linestyle="--", alpha=0.7, label=f"Current T = {T}")
plt.axhline(y=denominator_d1, color="red", linestyle="--", alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel("Time to Maturity T", fontsize=12)
plt.ylabel("σ√T", fontsize=12)
plt.title("Part 1f: d₁ Denominator (Volatility Term)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

print(f"Denominator: σ√T = {denominator_d1:.4f}")

# %% [markdown]
# **Motivation:** Normalizes by total uncertainty over time period

# %% [markdown]
# ### PART 1g: Complete d₁ (Numerator / Denominator)
#
# **Formula:** d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
#
# **Domain:** (-∞, ∞) | **Range:** (-∞, ∞) | **Shape:** Linear in ln(S)

# %%
d1 = numerator_d1 / denominator_d1

# %% [markdown]
# **Properties:** Standardized adjusted moneyness
# - Z-score of log-moneyness at maturity
# - d₁ = 0 when S ≈ K (at-the-money adjusted)
# - d₁ > 0 indicates in-the-money likelihood
# - d₁ < 0 indicates out-of-the-money likelihood
# - Measures standard deviations from strike

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, d1, "b-", linewidth=2.5, label="d₁")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7, label="d₁ = 0")
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("d₁", fontsize=12)
plt.title("Part 1g: Complete d₁ - Adjusted Moneyness Measure", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** d₁ standardizes the distance from strike, accounting for expected growth and volatility

# %% [markdown]
# ### PART 2a: Volatility Term (σ√T)
#
# **Formula:** σ√T (same as d₁ denominator)
#
# **Domain:** [0, ∞) | **Range:** [0, ∞) | **Shape:** Square root in T

# %%
vol_term = sigma * np.sqrt(T)
vol_term_array = np.full_like(S_range, vol_term)

# %% [markdown]
# **Properties:** Risk adjustment factor
# - Represents total volatility over time period
# - Used to shift from d₁ to d₂
# - Independent of stock price S
# - Current: σ√T = {:.4f}

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, d1, "b--", linewidth=2, alpha=0.7, label="d₁")
plt.axhline(
    y=vol_term,
    color="orange",
    linestyle="-",
    linewidth=2.5,
    label=f"σ√T = {vol_term:.3f}",
)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.fill_between(
    S_range, 0, vol_term, alpha=0.15, color="orange", label="Subtraction amount"
)
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Part 2a: Volatility Term σ√T", fontsize=13)
plt.legend(fontsize=10)
plt.show()

print(f"Volatility term: σ√T = {vol_term:.4f}")

# %% [markdown]
# **Motivation:** Constant offset to adjust from physical to risk-neutral measure

# %% [markdown]
# ### PART 2b: Complete d₂ (d₁ - σ√T)
#
# **Formula:** d₂ = d₁ - σ√T
#
# **Domain:** (-∞, ∞) | **Range:** (-∞, ∞) | **Shape:** Shifted version of d₁

# %%
d2 = d1 - sigma * np.sqrt(T)

# %% [markdown]
# **Properties:** Risk-adjusted probability measure
# - Always less than d₁ by σ√T
# - Represents probability under risk-neutral measure
# - d₂ approaches d₁ as T→0 or σ→0
# - Parallel shift down from d₁

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, d1, "b-", linewidth=2.5, label="d₁")
plt.plot(S_range, d2, "g-", linewidth=2.5, label="d₂")
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.fill_between(
    S_range, d1, d2, alpha=0.2, color="orange", label=f"σ√T gap = {vol_term:.3f}"
)
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Part 2b: Complete d₂ = d₁ - σ√T (Risk-Neutral Adjustment)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** d₂ adjusts d₁ for risk-neutral valuation, separating drift from expected value

# %% [markdown]
# ### PART 3: N(d₁) - Cumulative Normal Distribution
#
# **Formula:** N(d₁): Probability that standard normal ≤ d₁
#
# **Domain:** (-∞, ∞) | **Range:** (0, 1) | **Shape:** Sigmoid (S-curve)

# %%
N_d1 = norm.cdf(d1)

# %% [markdown]
# **Properties:** Delta of the option (hedge ratio)
# - N(d₁) ∈ (0, 1), probability interpretation
# - N(d₁) → 1 as S → ∞ (deep in-the-money)
# - N(d₁) → 0 as S → 0 (deep out-of-the-money)
# - N(d₁) ≈ 0.5 when at-the-money

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, N_d1, "purple", linewidth=2.5, label="N(d₁)")
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="N(d₁) = 0.5")
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.fill_between(S_range, 0, N_d1, alpha=0.2, color="purple")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("N(d₁)", fontsize=12)
plt.title("Part 3: N(d₁) - Option Delta (Hedge Ratio)", fontsize=13)
plt.legend(fontsize=10)
plt.ylim(-0.05, 1.05)
plt.show()

# %% [markdown]
# **Motivation:** N(d₁) gives the delta hedge ratio - shares to hold per option to remain delta-neutral

# %% [markdown]
# ### PART 4: N(d₂) - Exercise Probability
#
# **Formula:** N(d₂): Risk-neutral probability of exercise
#
# **Domain:** (-∞, ∞) | **Range:** (0, 1) | **Shape:** Sigmoid

# %%
N_d2 = norm.cdf(d2)

# %% [markdown]
# **Properties:** Probability option expires in-the-money
# - Always ≤ N(d₁) due to d₂ < d₁
# - Used to discount the strike payment
# - Closer to 0.5 than N(d₁) at ATM

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, N_d1, "purple", linewidth=2.5, label="N(d₁) - Delta", alpha=0.7)
plt.plot(S_range, N_d2, "orange", linewidth=2.5, label="N(d₂) - Exercise Prob")
plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("Part 4: N(d₂) - Risk-Neutral Exercise Probability", fontsize=13)
plt.legend(fontsize=10)
plt.ylim(-0.05, 1.05)
plt.show()

# %% [markdown]
# **Motivation:** N(d₂) quantifies likelihood of exercise, needed to value the strike payment

# %% [markdown]
# ### PART 5a: Discount Factor (e^(-rT))
#
# **Formula:** e^(-rT)
#
# **Domain:** T ∈ [0, ∞) | **Range:** (0, 1] | **Shape:** Exponential decay

# %%
discount_factor = np.exp(-r * T)
T_discount = np.linspace(0, 5, 500)
discount_range = np.exp(-r * T_discount)

# %% [markdown]
# **Properties:** Present value multiplier
# - e^(-rT) = 1 when T = 0 (no discounting)
# - e^(-rT) < 1 for T > 0 (time value of money)
# - Decreases exponentially with time
# - Current: e^(-rT) = {:.4f}

# %%
plt.figure(figsize=(10, 6))
plt.plot(T_discount, discount_range, "purple", linewidth=2.5, label="e^(-rT)")
plt.axvline(x=T, color="red", linestyle="--", alpha=0.7, label=f"Current T = {T}")
plt.axhline(y=discount_factor, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No discount")
plt.grid(alpha=0.3)
plt.xlabel("Time to Maturity T", fontsize=12)
plt.ylabel("Discount Factor", fontsize=12)
plt.title("Part 5a: Discount Factor e^(-rT)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

print(f"Discount factor: e^(-rT) = {discount_factor:.4f}")

# %% [markdown]
# **Motivation:** Converts future cash flows to present value

# %% [markdown]
# ### PART 5b: Stock Receipt Term (S₀·N(d₁))
#
# **Formula:** S₀·N(d₁)
#
# **Domain:** (0, ∞) | **Range:** (0, S) | **Shape:** Sigmoid scaled by S

# %%
stock_receipt = S_range * N_d1

# %% [markdown]
# **Properties:** Expected value of receiving stock
# - Weighted by delta hedge ratio N(d₁)
# - Upper bound: S (100% probability)
# - Lower bound: 0 (0% probability)
# - Represents asset side of option payoff

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, stock_receipt, "b-", linewidth=2.5, label="S₀·N(d₁)")
plt.plot(
    S_range,
    S_range,
    "gray",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label="S₀ (upper bound)",
)
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.fill_between(S_range, 0, stock_receipt, alpha=0.2, color="blue")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Stock Receipt Value ($)", fontsize=12)
plt.title("Part 5b: Expected Stock Receipt S₀·N(d₁)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Value of stock conditionally received upon favorable outcome

# %% [markdown]
# ### PART 5c: Discounted Strike (K·e^(-rT))
#
# **Formula:** K·e^(-rT)
#
# **Domain:** Constant | **Range:** (0, K] | **Shape:** Constant in S

# %%
discounted_strike = K * np.exp(-r * T)
discounted_strike_array = np.full_like(S_range, discounted_strike)

# %% [markdown]
# **Properties:** Present value of strike payment
# - Fixed amount regardless of S
# - Always less than K (except T=0)
# - Represents maximum liability
# - Current: K·e^(-rT) = ${:.2f}

# %%
plt.figure(figsize=(10, 6))
plt.axhline(
    y=discounted_strike,
    color="orange",
    linestyle="-",
    linewidth=2.5,
    label=f"K·e^(-rT) = ${discounted_strike:.2f}",
)
plt.axhline(
    y=K,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label=f"K = ${K} (undiscounted)",
)
plt.fill_between(
    S_range, discounted_strike, K, alpha=0.15, color="orange", label="Interest savings"
)
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Present Value ($)", fontsize=12)
plt.title("Part 5c: Discounted Strike K·e^(-rT)", fontsize=13)
plt.legend(fontsize=10)
plt.xlim(S_range.min(), S_range.max())
plt.show()

print(f"Discounted strike: K·e^(-rT) = ${discounted_strike:.2f}")

# %% [markdown]
# **Motivation:** Strike must be paid only at maturity, so discount to present

# %% [markdown]
# ### PART 5d: Discounted Strike Payment (K·e^(-rT)·N(d₂))
#
# **Formula:** K·e^(-rT)·N(d₂)
#
# **Domain:** (0, ∞) | **Range:** (0, K·e^(-rT)] | **Shape:** Sigmoid

# %%
strike_payment = K * np.exp(-r * T) * N_d2

# %% [markdown]
# **Properties:** Expected present value of strike payment
# - Weighted by exercise probability N(d₂)
# - Upper bound: K·e^(-rT) (certain exercise)
# - Lower bound: 0 (never exercise)
# - Represents liability side of option payoff

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, strike_payment, "orange", linewidth=2.5, label="K·e^(-rT)·N(d₂)")
plt.axhline(
    y=discounted_strike,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=2,
    label=f"K·e^(-rT) = ${discounted_strike:.2f}",
)
plt.axvline(x=K, color="red", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.fill_between(S_range, 0, strike_payment, alpha=0.2, color="orange")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Expected Strike Payment ($)", fontsize=12)
plt.title("Part 5d: Expected Discounted Strike Payment K·e^(-rT)·N(d₂)", fontsize=13)
plt.legend(fontsize=10)
plt.show()

# %% [markdown]
# **Motivation:** Expected cost of exercising the option in present value terms

# %% [markdown]
# ### PART 5e: Complete Black-Scholes Formula (C = S₀·N(d₁) - K·e^(-rT)·N(d₂))
#
# **Formula:** C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
#
# **Domain:** S ∈ (0, ∞) | **Range:** C ∈ (0, S) | **Shape:** Convex, increasing

# %%
C = S_range * N_d1 - K * np.exp(-r * T) * N_d2
intrinsic = np.maximum(S_range - K, 0)

# %% [markdown]
# **Properties:** Call option value
# - Net value: Stock receipt minus strike payment
# - Lower bound: max(S - Ke^(-rT), 0)
# - Upper bound: S₀
# - Convex in S (positive gamma everywhere)
# - Time value = C - max(S-K, 0)

# %%
plt.figure(figsize=(10, 6))
plt.plot(S_range, C, "b-", linewidth=3, label="Black-Scholes Call Value C")
plt.plot(
    S_range, stock_receipt, "g--", linewidth=2, alpha=0.6, label="S₀·N(d₁) (asset)"
)
plt.plot(
    S_range,
    strike_payment,
    "r--",
    linewidth=2,
    alpha=0.6,
    label="K·e^(-rT)·N(d₂) (liability)",
)
plt.plot(
    S_range,
    intrinsic,
    "gray",
    linestyle="--",
    linewidth=2,
    label="Intrinsic Value",
    alpha=0.7,
)
plt.fill_between(
    S_range, strike_payment, stock_receipt, alpha=0.2, color="blue", label="Call Value"
)
plt.axvline(x=K, color="gray", linestyle="--", alpha=0.5, label=f"Strike K={K}")
plt.grid(alpha=0.3)
plt.xlabel("Stock Price S", fontsize=12)
plt.ylabel("Value ($)", fontsize=12)
plt.title("Part 5e: Complete Black-Scholes Call Option Value", fontsize=13)
plt.legend(fontsize=10, loc="upper left")
plt.show()

# %% [markdown]
# **Motivation:** Two-part structure: expected stock receipt minus expected discounted strike payment

# %% [markdown]
# ### Overlay: All Components Together
#
# Show value decomposition

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Top: d-terms
ax1.plot(S_range, d1, "b-", linewidth=2, label="d₁")
ax1.plot(S_range, d2, "g-", linewidth=2, label="d₂")
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax1.axvline(x=K, color="red", linestyle="--", alpha=0.5)
ax1.grid(alpha=0.3)
ax1.set_ylabel("Standardized Values", fontsize=11)
ax1.set_title("d₁ and d₂ Measures", fontsize=12)
ax1.legend()

# Bottom: Probabilities and Value
ax2_twin = ax2.twinx()
ax2.plot(S_range, N_d1, "purple", linewidth=2, label="N(d₁)", alpha=0.7)
ax2.plot(S_range, N_d2, "orange", linewidth=2, label="N(d₂)", alpha=0.7)
ax2.set_ylabel("Probability", fontsize=11, color="purple")
ax2.tick_params(axis="y", labelcolor="purple")
ax2.set_ylim(-0.05, 1.05)

ax2_twin.plot(S_range, C, "b-", linewidth=2.5, label="Call Value")
ax2_twin.set_ylabel("Option Value ($)", fontsize=11, color="blue")
ax2_twin.tick_params(axis="y", labelcolor="blue")

ax2.axvline(x=K, color="red", linestyle="--", alpha=0.5)
ax2.grid(alpha=0.3)
ax2.set_xlabel("Stock Price S", fontsize=11)
ax2.set_title("Probabilities (left) and Option Value (right)", fontsize=12)
ax2.legend(loc="upper left")
ax2_twin.legend(loc="lower right")

plt.tight_layout()
plt.show()

# %% Sensitivity Analysis: The Greeks
# Delta, Gamma, Vega, Theta, Rho

delta = N_d1
gamma = norm.pdf(d1) / (S_range * sigma * np.sqrt(T))
vega = S_range * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
theta = (
    -(S_range * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
) / 365  # Per day

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Delta
axes[0, 0].plot(S_range, delta, "b-", linewidth=2)
axes[0, 0].axvline(x=K, color="red", linestyle="--", alpha=0.5)
axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[0, 0].set_title("Delta (∂C/∂S)", fontsize=11)
axes[0, 0].set_ylabel("Delta", fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Gamma
axes[0, 1].plot(S_range, gamma, "g-", linewidth=2)
axes[0, 1].axvline(x=K, color="red", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Gamma (∂²C/∂S²)", fontsize=11)
axes[0, 1].set_ylabel("Gamma", fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Vega
axes[1, 0].plot(S_range, vega, "purple", linewidth=2)
axes[1, 0].axvline(x=K, color="red", linestyle="--", alpha=0.5)
axes[1, 0].set_title("Vega (∂C/∂σ)", fontsize=11)
axes[1, 0].set_ylabel("Vega (per 1%)", fontsize=10)
axes[1, 0].set_xlabel("Stock Price S", fontsize=10)
axes[1, 0].grid(alpha=0.3)

# Theta
axes[1, 1].plot(S_range, theta, "orange", linewidth=2)
axes[1, 1].axvline(x=K, color="red", linestyle="--", alpha=0.5)
axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 1].set_title("Theta (∂C/∂t)", fontsize=11)
axes[1, 1].set_ylabel("Theta (per day)", fontsize=10)
axes[1, 1].set_xlabel("Stock Price S", fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% 3D Visualization: Surface Plot
# Call value as function of S and T

S_3d = np.linspace(50, 150, 50)
T_3d = np.linspace(0.1, 2, 50)
S_mesh, T_mesh = np.meshgrid(S_3d, T_3d)

C_mesh = np.zeros_like(S_mesh)
for i in range(len(T_3d)):
    for j in range(len(S_3d)):
        d1_3d = (np.log(S_mesh[i, j] / K) + (r + sigma**2 / 2) * T_mesh[i, j]) / (
            sigma * np.sqrt(T_mesh[i, j])
        )
        d2_3d = d1_3d - sigma * np.sqrt(T_mesh[i, j])
        C_mesh[i, j] = S_mesh[i, j] * norm.cdf(d1_3d) - K * np.exp(
            -r * T_mesh[i, j]
        ) * norm.cdf(d2_3d)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(S_mesh, T_mesh, C_mesh, cmap="viridis", alpha=0.9)
ax.set_xlabel("Stock Price S", fontsize=11)
ax.set_ylabel("Time to Maturity T", fontsize=11)
ax.set_zlabel("Call Value C", fontsize=11)
ax.set_title("Black-Scholes Surface: C(S, T)", fontsize=13)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% 3D Visualization: Volatility Surface
# Call value as function of S and σ

sigma_3d = np.linspace(0.05, 0.6, 50)
S_vol_mesh, sigma_mesh = np.meshgrid(S_3d, sigma_3d)

C_vol_mesh = np.zeros_like(S_vol_mesh)
for i in range(len(sigma_3d)):
    for j in range(len(S_3d)):
        d1_vol = (
            np.log(S_vol_mesh[i, j] / K) + (r + sigma_mesh[i, j] ** 2 / 2) * T
        ) / (sigma_mesh[i, j] * np.sqrt(T))
        d2_vol = d1_vol - sigma_mesh[i, j] * np.sqrt(T)
        C_vol_mesh[i, j] = S_vol_mesh[i, j] * norm.cdf(d1_vol) - K * np.exp(
            -r * T
        ) * norm.cdf(d2_vol)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(S_vol_mesh, sigma_mesh, C_vol_mesh, cmap="plasma", alpha=0.9)
ax.set_xlabel("Stock Price S", fontsize=11)
ax.set_ylabel("Volatility σ", fontsize=11)
ax.set_zlabel("Call Value C", fontsize=11)
ax.set_title("Black-Scholes Volatility Surface: C(S, σ)", fontsize=13)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% Related Formulas (Summary)

related = {
    "Black-Scholes Put": "P = Ke^(-rT)·N(-d₂) - S₀·N(-d₁)",
    "Put-Call Parity": "C - P = S₀ - Ke^(-rT)",
    "Binary/Digital Option": "B = e^(-rT)·N(d₂)",
    "Asian Option": "Uses arithmetic/geometric mean of S",
    "Barrier Option": "Adds knock-in/knock-out conditions on S",
    "Merton Jump-Diffusion": "BS + Poisson jumps in S",
}

print("=" * 60)
print("RELATED OPTION PRICING FORMULAS")
print("=" * 60)
for name, formula in related.items():
    print(f"{name:25s}: {formula}")
print()

# %% Key Properties Summary

print("=" * 60)
print("BLACK-SCHOLES FORMULA: KEY MATHEMATICAL PROPERTIES")
print("=" * 60)
print()
print("Structure: C = S₀·N(d₁) - K·e^(-rT)·N(d₂)")
print()
print("d₁ Properties:")
print("  • Measures adjusted moneyness: ln(S/K) + drift")
print("  • Domain: ℝ, Range: ℝ")
print("  • Linear in ln(S), increases with S, T, σ, r")
print()
print("d₂ Properties:")
print("  • Risk-neutral measure: d₂ = d₁ - σ√T")
print("  • Always less than d₁ by volatility term")
print("  • Converges to d₁ as T→0 or σ→0")
print()
print("N(d₁) Properties:")
print("  • Delta hedge ratio ∈ (0, 1)")
print("  • Monotonic increasing sigmoid")
print("  • Physical measure weight on stock")
print()
print("N(d₂) Properties:")
print("  • Risk-neutral exercise probability")
print("  • Always ≤ N(d₁)")
print("  • Weight on discounted strike payment")
print()
print("Call Value Properties:")
print("  • Bounds: max(S - Ke^(-rT), 0) ≤ C ≤ S")
print("  • Convex in S (positive gamma everywhere)")
print("  • Increases with S, T, σ, r; decreases with K")
print("  • Time decay: negative theta (exception: deep ITM)")
print()
print("Greeks:")
print("  • Delta = N(d₁) ∈ (0, 1)")
print("  • Gamma = φ(d₁)/(S·σ√T) ≥ 0")
print("  • Vega = S·φ(d₁)·√T ≥ 0")
print("  • Theta < 0 (except deep ITM European calls)")
print("  • Rho = K·T·e^(-rT)·N(d₂) ≥ 0")
print()

# %% Design Intuition

print("=" * 60)
print("DESIGN PHILOSOPHY & INTERPRETATION")
print("=" * 60)
print()
print("Economic Interpretation:")
print("  The Black-Scholes formula values a call as the difference between:")
print("  1. Expected stock receipt: S₀·N(d₁)")
print("  2. Expected cash outflow: K·e^(-rT)·N(d₂)")
print()
print("Why Two Different Probabilities?")
print("  • N(d₁): Accounts for delta hedging under physical measure")
print("  • N(d₂): Risk-neutral probability of exercise")
print("  • Gap σ√T arises from change of measure (Girsanov theorem)")
print()
print("Key Insights:")
print("  • Volatility increases value (uncertainty = opportunity for calls)")
print("  • Time value peaks at-the-money (maximum uncertainty)")
print("  • Delta approaches 1 for deep ITM (acts like stock)")
print("  • Gamma peaks ATM (maximum convexity benefit)")
print("  • Formula assumes continuous trading and no arbitrage")
print()
print("Mathematical Elegance:")
print("  • Closed-form solution from Ito's lemma")
print("  • Lognormal stock price assumption")
print("  • Risk-neutral valuation eliminates drift μ")
print("  • Complete market: option replicable by stock + bond")
print()
print("Historical Significance:")
print("  • Published 1973 by Black, Scholes, and Merton")
print("  • Revolutionized derivatives trading")
print("  • Foundation for modern quantitative finance")
print("=" * 60)
