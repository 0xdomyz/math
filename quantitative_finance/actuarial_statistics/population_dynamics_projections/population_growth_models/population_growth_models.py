# %% [markdown]
# # population growth models
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for population growth models.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: population growth models")

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
# Source: population_growth_models.md

# --- Code Block 1 ---
import numpy as np

P = 1000
r = 0.05
K = 10000
years = 50
trajectory = [P]

for _ in range(years):
    P = P + r * P * (1 - P / K)
    trajectory.append(P)

print(trajectory[-1])



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
print("Summary complete for topic: population growth models")


