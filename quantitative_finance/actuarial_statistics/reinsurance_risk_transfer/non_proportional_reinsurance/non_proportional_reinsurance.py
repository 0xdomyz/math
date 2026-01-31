# Auto-extracted from markdown file
# Source: non_proportional_reinsurance.md

# --- Code Block 1 ---
loss = 3000000
attachment = 1000000
limit = 4000000

reinsured_loss = max(0, min(loss - attachment, limit))
print("Reinsurer pays:", reinsured_loss)

