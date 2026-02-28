# %% [markdown]
# # nelson aalen estimator
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for nelson aalen estimator.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: nelson aalen estimator")

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
# Source: nelson_aalen_estimator.md

# --- Code Block 1 ---
import numpy as np

times = np.array([2, 3, 5, 5, 8])
status = np.array([1, 1, 1, 0, 0])

unique_times = np.unique(times[status == 1])
H = 0.0
for t in unique_times:
    at_risk = np.sum(times >= t)
    events = np.sum((times == t) & (status == 1))
    H += events / at_risk
    print(t, H)



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
print("Summary complete for topic: nelson aalen estimator")


