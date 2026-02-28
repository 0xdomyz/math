# %% [markdown]
# # non proportional reinsurance
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for non proportional reinsurance.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: non proportional reinsurance")

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
# Source: non_proportional_reinsurance.md

# --- Code Block 1 ---
loss = 3000000
attachment = 1000000
limit = 4000000

reinsured_loss = max(0, min(loss - attachment, limit))
print("Reinsurer pays:", reinsured_loss)



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
print("Summary complete for topic: non proportional reinsurance")


