# Auto-extracted from markdown file
# Source: morbidity_claim_rates.md

# --- Code Block 1 ---
import numpy as np

claims = 500
members = 10000
member_months = members * 12
claim_rate = claims / member_months
print("Claim rate per member-month:", claim_rate)

