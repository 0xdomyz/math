# %% [markdown]
# # actuarial cost methods
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for actuarial cost methods.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: actuarial cost methods")

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
# Source: actuarial_cost_methods.md

# --- Code Block 1 ---
pbos = [50000, 60000, 70000]
salary = [40000, 50000, 60000]
costs = []
for pbo, sal in zip(pbos, salary):
    cost_pct = pbo / sal
    costs.append(cost_pct)
print(costs)



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
print("Summary complete for topic: actuarial cost methods")


