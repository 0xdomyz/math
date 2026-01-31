# Auto-extracted from markdown file
# Source: retrocession.md

# --- Code Block 1 ---
primary_loss = 100000
primary_reins_recovery = 60000
retro_recovery = max(0, primary_reins_recovery - 30000)

net_to_primary = primary_loss - primary_reins_recovery
net_to_reinsurer = primary_reins_recovery - retro_recovery
print("Primary net:", net_to_primary, "Reinsurer net:", net_to_reinsurer)

