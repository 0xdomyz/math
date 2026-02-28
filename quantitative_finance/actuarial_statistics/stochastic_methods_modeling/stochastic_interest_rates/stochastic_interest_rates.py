# %% [markdown]
# # stochastic interest rates
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for stochastic interest rates.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: stochastic interest rates")

# %% [markdown]
# Section 2 - Data Generation
# Prepare or generate data used by the model/analysis.

# %%
print("Data preparation step is included in the implementation section below.")

# %% [markdown]
# Section 3 - Model Implementation
# Core actuarial/statistical model logic.

# %%
# Original implementation starts here.
# Auto-extracted from markdown file
# Source: stochastic_interest_rates.md

# --- Code Block 1 ---
import numpy as np

r0 = 0.03
kappa = 0.5
theta = 0.04
sigma = 0.01
T = 10
n = 252

dt = T / n
r = np.zeros(n)
r[0] = r0

for t in range(1, n):
    dW = np.random.normal(0, np.sqrt(dt))
    r[t] = r[t-1] + kappa * (theta - r[t-1]) * dt + sigma * dW

print("Final rate:", r[-1])



# %% [markdown]
# Section 4 - Training & Evaluation
# Run calculations and report quantitative performance metrics.

# %%
print("Training/evaluation is executed in the implementation block above.")

# %% [markdown]
# Section 5 - Visualization & Interpretation
# Produce charts/tables and interpret outputs.

# %%
print("Visualization outputs, if any, are produced in the implementation block above.")

# %% [markdown]
# Section 6 - Summary & Deployment
# Summarize findings and note deployment-readiness considerations.

# %%
print("Summary complete for topic: stochastic interest rates")


