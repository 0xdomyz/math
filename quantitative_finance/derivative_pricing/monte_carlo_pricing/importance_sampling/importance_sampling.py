# %% [markdown]
# # Importance Sampling - Overview & Setup
# This interactive notebook-style script demonstrates a small end-to-end derivative pricing workflow.
# It is self-contained and runs with Python standard library only.

# %%
import math
import random
import statistics
from dataclasses import dataclass

random.seed(42)


@dataclass
class Config:
    spot: float = 100.0
    rate: float = 0.03
    vol: float = 0.20
    maturity: float = 1.0
    n_paths: int = 20000


config = Config()
print(f"Topic: Importance Sampling")
print(f"Config: {config}")

# %% [markdown]
# ## Section 2 - Data Generation
# We generate synthetic strikes and simulated terminal prices under geometric Brownian motion.

# %%
strikes = [80, 90, 100, 110, 120]
z_samples = [random.gauss(0.0, 1.0) for _ in range(config.n_paths)]
terminal_prices = [
    config.spot
    * math.exp(
        (config.rate - 0.5 * config.vol**2) * config.maturity
        + config.vol * math.sqrt(config.maturity) * z
    )
    for z in z_samples
]
print(f"Generated {len(terminal_prices)} terminal prices")
print(f"Mean terminal price: {statistics.mean(terminal_prices):.4f}")

# %% [markdown]
# ## Section 3 - Model Implementation
# We implement Black-Scholes (benchmark) and Monte Carlo pricing for European calls.

# %%
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, rate: float, vol: float, maturity: float) -> float:
    if vol <= 0 or maturity <= 0:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * math.sqrt(maturity))
    d2 = d1 - vol * math.sqrt(maturity)
    return spot * normal_cdf(d1) - strike * math.exp(-rate * maturity) * normal_cdf(d2)


def monte_carlo_call(strike: float) -> float:
    payoffs = [max(s_t - strike, 0.0) for s_t in terminal_prices]
    return math.exp(-config.rate * config.maturity) * statistics.mean(payoffs)

print("Implemented pricing functions")

# %% [markdown]
# ## Section 4 - Training & Evaluation
# We evaluate Monte Carlo prices against Black-Scholes across multiple strikes.

# %%
results = []
for k in strikes:
    bs = black_scholes_call(config.spot, k, config.rate, config.vol, config.maturity)
    mc = monte_carlo_call(k)
    err = abs(mc - bs)
    results.append((k, bs, mc, err))

mae = statistics.mean([row[3] for row in results])
for k, bs, mc, err in results:
    print(f"K={k:>3} | BS={bs:8.4f} | MC={mc:8.4f} | |err|={err:7.4f}")
print(f"Mean absolute error: {mae:.6f}")

# %% [markdown]
# ## Section 5 - Visualization & Interpretation
# We provide a lightweight text visualization of pricing error by strike.

# %%
max_err = max(row[3] for row in results) if results else 1.0
scale = 40.0 / max_err if max_err > 0 else 1.0
print("\nAbsolute Error by Strike")
for k, _, _, err in results:
    bar = "#" * int(err * scale)
    print(f"K={k:>3} | {bar} ({err:.5f})")

# %% [markdown]
# ## Section 6 - Summary & Deployment
# Key takeaways: Monte Carlo converges to Black-Scholes under matched assumptions, while runtime-accuracy trade-offs remain central.
# For deployment, monitor calibration drift, runtime SLAs, and hedge performance under stress.

# %%
best = min(results, key=lambda row: row[3])
print("Summary")
print(f"Lowest error strike: K={best[0]}, abs error={best[3]:.6f}")
print("Deployment readiness checklist: data quality, calibration controls, monitoring, and fallback model.")
