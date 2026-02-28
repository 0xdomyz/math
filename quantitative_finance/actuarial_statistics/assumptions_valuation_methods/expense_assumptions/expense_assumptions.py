# %% [markdown]
# # expense assumptions
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for expense assumptions.

# %%
import warnings

import numpy as np

warnings.filterwarnings("ignore")

print("Starting topic workflow: expense assumptions")

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
# Source: expense_assumptions.md

# --- Code Block 1 ---
premium = 1000
renewal_years = 9
acquisition = 200
renewal_exp = 50 * (1.03 ** np.arange(renewal_years))

total = acquisition + renewal_exp.sum()
print("Total expenses:", total)


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
print("Summary complete for topic: expense assumptions")
