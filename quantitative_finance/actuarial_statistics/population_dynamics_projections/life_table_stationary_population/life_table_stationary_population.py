# %% [markdown]
# # life table stationary population
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for life table stationary population.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: life table stationary population")

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
# Source: life_table_stationary_population.md

# --- Code Block 1 ---
import numpy as np

Lx = np.array([100000, 95000, 90000, 80000, 50000])
T_x = np.cumsum(Lx[::-1])[::-1]
Cx = Lx / T_x[0]
print("Proportion in each age group:", Cx)



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
print("Summary complete for topic: life table stationary population")


