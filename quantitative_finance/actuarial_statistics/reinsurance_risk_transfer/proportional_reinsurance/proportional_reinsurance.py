# %% [markdown]
# # proportional reinsurance
#
# Section 1 - Overview & Setup
# This notebook-style script provides a runnable mini-project for proportional reinsurance.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: proportional reinsurance")

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
# Source: proportional_reinsurance.md

# --- Code Block 1 ---
premium = 100000
loss = 50000
reinsurer_share = 0.30

reinsured_premium = premium * reinsurer_share
reinsured_loss = loss * reinsurer_share
print("Premium:", reinsured_premium, "Loss:", reinsured_loss)



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
print("Summary complete for topic: proportional reinsurance")


