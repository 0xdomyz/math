# Auto-extracted from markdown file
# Source: long_term_care.md

# --- Code Block 1 ---
import numpy as np

prob_indep_to_assist = 0.05
prob_assist_to_nursing = 0.10
prob_die = 0.02

prob_stay = 1 - prob_indep_to_assist - prob_die
print("Transition probs:", prob_stay, prob_indep_to_assist, prob_die)

