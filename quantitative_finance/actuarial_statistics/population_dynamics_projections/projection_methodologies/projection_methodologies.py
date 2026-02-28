# %% [markdown]
# # projection methodologies
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for projection methodologies.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: projection methodologies")

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
# Source: projection_methodologies.md

# --- Code Block 1 ---
import numpy as np

pop = np.array([10000, 8000, 5000, 2000])
asfr = np.array([0.08, 0.15, 0.10, 0.02])
lx = np.array([0.95, 0.92, 0.85, 0.50])

births = (pop[1:-1] * asfr[1:-1]).sum() * 0.5  # half female
next_pop = np.zeros(4)
next_pop[0] = births
next_pop[1:] = pop[:-1] * lx[:-1]
print(next_pop)



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
print("Summary complete for topic: projection methodologies")


