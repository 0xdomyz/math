# Auto-extracted from markdown file
# Source: migration.md

# --- Code Block 1 ---
import numpy as np

population = 1000000
births = 20000
deaths = 15000
net_migration = 5000
new_pop = population + births - deaths + net_migration
print("New population:", new_pop)

