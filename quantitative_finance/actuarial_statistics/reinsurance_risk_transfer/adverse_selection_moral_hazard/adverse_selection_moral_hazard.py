# Auto-extracted from markdown file
# Source: adverse_selection_moral_hazard.md

# --- Code Block 1 ---
selected_claims = 120
selected_count = 800
population_claims = 100
population_count = 10000

selected_rate = selected_claims / selected_count
pop_rate = population_claims / population_count
print("Selection ratio:", selected_rate / pop_rate)

