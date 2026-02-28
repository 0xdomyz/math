# %% [markdown]
# # interest rate risk
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for interest rate risk.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: interest rate risk")

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
# Source: interest_rate_risk.md

# --- Code Block 1 ---
import numpy as np

asset_dur = np.array([3.0, 7.0])
asset_weights = np.array([0.6, 0.4])
liability_dur = 5.5

port_dur = (asset_dur * asset_weights).sum()
print("Duration gap:", port_dur - liability_dur)



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
print("Summary complete for topic: interest rate risk")


