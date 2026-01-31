# Auto-extracted from markdown file
# Source: cox_proportional_hazards.md

# --- Code Block 1 ---
# requires lifelines package
from lifelines import CoxPHFitter
import pandas as pd

# df columns: duration, event, covariates...
# cph = CoxPHFitter().fit(df, duration_col='duration', event_col='event')
# cph.summary

