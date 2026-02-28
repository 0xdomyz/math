# %% [markdown]
# # longevity risk
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for longevity risk.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: longevity risk")

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
# Source: longevity_risk.md

# --- Code Block 1 ---
import numpy as np

# simple longevity stress: reduce mortality by 10%
qx = np.array([0.010, 0.012, 0.014, 0.016])
qx_stress = 0.9 * qx
px = 1 - qx
px_stress = 1 - qx_stress
print(px, px_stress)



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
print("Summary complete for topic: longevity risk")


