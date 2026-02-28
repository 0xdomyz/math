# %% [markdown]
# # graduation of mortality data
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for graduation of mortality data.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: graduation of mortality data")

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
# Source: graduation_of_mortality_data.md

# --- Code Block 1 ---
import numpy as np

qx = np.array([0.01, 0.012, 0.02, 0.018, 0.022])
window = 3
smooth = np.convolve(qx, np.ones(window)/window, mode='valid')
print(smooth)



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
print("Summary complete for topic: graduation of mortality data")


