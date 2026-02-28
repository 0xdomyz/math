# %% [markdown]
# # scenario analysis
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for scenario analysis.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: scenario analysis")

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
# Source: scenario_analysis.md

# --- Code Block 1 ---
import numpy as np

reserve_base = 1000000
duration = 7.0
rate_shock = 0.01  # 100bps increase

reserve_impact = -reserve_base * duration * rate_shock
stressed_reserve = reserve_base + reserve_impact
print("Stressed reserve:", stressed_reserve)



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
print("Summary complete for topic: scenario analysis")


