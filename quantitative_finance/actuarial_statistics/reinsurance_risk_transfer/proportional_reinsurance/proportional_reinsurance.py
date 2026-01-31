# Auto-extracted from markdown file
# Source: proportional_reinsurance.md

# --- Code Block 1 ---
premium = 100000
loss = 50000
reinsurer_share = 0.30

reinsured_premium = premium * reinsurer_share
reinsured_loss = loss * reinsurer_share
print("Premium:", reinsured_premium, "Loss:", reinsured_loss)

